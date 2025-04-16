from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Any
import re
from bson import ObjectId
from bson.errors import InvalidId
from pydantic import BaseModel
from ..models.wine import Wine  # Your existing Pydantic Wine model
from ..db import db          # Your Motor-connected database instance
import os

router = APIRouter(prefix="/api/wines", tags=["wines"])

def convert_to_str(value):
    """Convert a value to a string; if None, return an empty string."""
    return "" if value is None else str(value)

def wine_helper(wine) -> dict:
    """
    Convert a MongoDB wine document into a dictionary matching the Wine model.
    This function extracts a full set of fields from the document.
    """
    return {
        "id": str(wine["_id"]),
        "wine_id": convert_to_str(wine.get("wine_id")),
        "winery": convert_to_str(wine.get("Winery")),
        "wine_name": convert_to_str(wine.get("Wine Name")),
        "country": convert_to_str(wine.get("Country")),
        "region": convert_to_str(wine.get("Region")),
        "wine_type": convert_to_str(wine.get("Wine Type")),
        "grape_type_list": convert_to_str(wine.get("Grape Type List")),
        "price": wine.get("Price"),
        "rating": wine.get("Rating"),
        "wine_description_1": convert_to_str(wine.get("Wine Description 1")),
        "wine_description_2": convert_to_str(wine.get("Wine Description 2")),
        "food_pairing": convert_to_str(wine.get("Food Pairing")),
        "alcohol_content": wine.get("Alcohol Content (%)"),
        "allergens": convert_to_str(wine.get("Allergens")),
        "bottle_closure": convert_to_str(wine.get("Bottle Closure")),
        "light_to_bold": convert_to_str(wine.get("Light to Bold")),
        "smooth_to_tannic": convert_to_str(wine.get("Smooth to Tannic")),
        "dry_to_sweet": convert_to_str(wine.get("Dry to Sweet")),
        "soft_to_acidic": convert_to_str(wine.get("Soft to Acidic")),
        "gentle_to_fizzy": convert_to_str(wine.get("Gentle to Fizzy")),
        "vintage": wine.get("Vintage"),
        "primary_type": convert_to_str(wine.get("primary type")),
        "image_url": convert_to_str(wine.get("Image URL"))
    }

# --------------------------------------------------
# NEW: Dynamic Search Endpoint Using MongoDB Text Search / Atlas Search
# --------------------------------------------------
@router.get("/search", response_model=List[Any])
async def dynamic_search(
    q: str = Query(..., description="Search term for wines"),
    skip: int = Query(0, ge=0, description="Number of results to skip"),
    limit: int = Query(20, gt=0, description="Maximum number of results to return")
):
    """
    Perform a dynamic search on the wines collection.
    - For queries with fewer than 3 characters, uses regex search across:
      "Wine Name", "Winery", "Country", "Region", "Grape Type", and "primary type".
    - For queries of 3 or more characters, uses MongoDB Atlas Search via an aggregation pipeline,
      leveraging the $search stage with the "search" index and fuzzy matching.
    """
    q = q.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Search term cannot be empty")
    
    if len(q) < 3:
        # Use regex search for very short queries.
        regex = re.compile(f".*{re.escape(q)}.*", re.IGNORECASE)
        mongo_query = {
            "$or": [
                {"Wine Name": regex},
                {"Winery": regex},
                {"Country": regex},
                {"Region": regex},
                {"Grape Type": regex},
                {"primary type": regex}
            ]
        }
        try:
            cursor = db.wines.find(mongo_query).skip(skip).limit(limit)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"MongoDB query failed: {e}")
    else:
        # Use Atlas Search via the $search aggregation stage.
        pipeline = [
            {"$search": {
                "index": "search",  # Ensure this matches your Atlas Search index name
                "text": {
                    "query": q,
                    "path": [
                        "Wine Name",
                        "Winery",
                        "Country",
                        "Region",
                        "Grape Type",
                        "primary type"
                    ],
                    "fuzzy": {}  # Enables fuzzy matching
                }
            }},
            {"$skip": skip},
            {"$limit": limit}
        ]
        try:
            cursor = db.wines.aggregate(pipeline)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"MongoDB Atlas Search query failed: {e}")
    
    results = await cursor.to_list(length=limit)
    return [wine_helper(wine) for wine in results]


# --------------------------------------------------
# Featured Wines Endpoint (Unchanged)
# --------------------------------------------------
class FeaturedWine(BaseModel):
    id: str
    wine_name: str
    winery: str
    price: Optional[float]
    description: str
    image_url: str

def featured_wine_helper(wine) -> dict:
    """
    Helper function to extract only the necessary fields for a featured wine.
    """
    return {
        "id": str(wine["_id"]),
        "wine_name": convert_to_str(wine.get("Wine Name")),
        "winery": convert_to_str(wine.get("Winery")),
        "price": wine.get("Price"),
        "description": convert_to_str(wine.get("Wine Description 1")),
        "image_url": convert_to_str(wine.get("Image URL"))
    }

@router.get("/featured", response_model=List[FeaturedWine])
async def get_featured_wine():
    """
    Return a list of featured wines selected from different price ranges.
    """
    price_ranges = [
        {"min": 50, "max": 100, "size": 2},
        {"min": 100, "max": 150, "size": 2},
        {"min": 150, "max": 250, "size": 2},
        {"min": 250, "max": 350, "size": 2},
        {"min": 350, "max": None, "size": 4},
    ]
    
    featured_wines = []
    for pr in price_ranges:
        if pr["max"] is None:
            query = {"Price": {"$gte": pr["min"]}}
        else:
            query = {"Price": {"$gte": pr["min"], "$lt": pr["max"]}}
            
        pipeline = [
            {"$match": query},
            {"$group": {
                "_id": None,
                "maxRating": {"$max": "$Rating"},
                "wines": {"$push": "$$ROOT"}
            }},
            {"$project": {
                "wines": {
                    "$filter": {
                        "input": "$wines",
                        "as": "wine",
                        "cond": {"$eq": ["$$wine.Rating", "$maxRating"]}
                    }
                }
            }},
            {"$unwind": "$wines"},
            {"$replaceRoot": {"newRoot": "$wines"}},
            {"$sort": {"_id": 1}},
            {"$limit": pr["size"]}
        ]
        
        cursor = db.wines.aggregate(pipeline)
        async for wine in cursor:
            featured_wines.append(featured_wine_helper(wine))
            
    return featured_wines

# --------------------------------------------------
# Existing Endpoints (List, Count, Detail)
# --------------------------------------------------
@router.get("/", response_model=List[Wine])
async def get_wines(
    region: Optional[str] = Query(None),
    wine_type: Optional[str] = Query(None),
    min_price: Optional[float] = Query(None),
    max_price: Optional[float] = Query(None),
    limit: Optional[int] = Query(None),
    page: Optional[int] = Query(1)
):
    query = {}
    if region:
        query["Region"] = region
    if wine_type:
        query["Wine Type"] = wine_type
    if min_price is not None or max_price is not None:
        price_query = {}
        if min_price is not None:
            price_query["$gte"] = min_price
        if max_price is not None:
            price_query["$lte"] = max_price
        query["Price"] = price_query

    wines = []
    cursor = db.wines.find(query)
    if limit is not None:
        skip = (page - 1) * limit if page and page > 0 else 0
        cursor = cursor.skip(skip).limit(limit)
    async for wine in cursor:
        wines.append(wine_helper(wine))
    return wines

@router.get("/count", response_model=dict)
async def get_wines_count(
    region: Optional[str] = Query(None),
    wine_type: Optional[str] = Query(None),
    min_price: Optional[float] = Query(None),
    max_price: Optional[float] = Query(None)
):
    query = {}
    if region:
        query["Region"] = region
    if wine_type:
        query["Wine Type"] = wine_type
    if min_price is not None or max_price is not None:
        price_query = {}
        if min_price is not None:
            price_query["$gte"] = min_price
        if max_price is not None:
            price_query["$lte"] = max_price
        query["Price"] = price_query

    count = await db.wines.count_documents(query)
    return {"count": count}

@router.get("/{wine_id}", response_model=Wine)
async def get_wine_detail(wine_id: str):
    wine = None
    try:
        wine = await db.wines.find_one({"_id": ObjectId(wine_id)})
    except InvalidId:
        wine = await db.wines.find_one({"wine_id": wine_id})
    
    if wine:
        return wine_helper(wine)
    
    raise HTTPException(status_code=404, detail="Wine not found")

@router.get("/", response_model=List[Wine])
async def get_wines(
    types: Optional[str] = Query(None, description="Comma-separated wine types"),
    min_price: Optional[float] = Query(None),
    max_price: Optional[float] = Query(None),
    min_rating: Optional[float] = Query(None, description="Minimum Vivino average rating"),
    grapes: Optional[str] = Query(None, description="Comma-separated grape types"),
    regions: Optional[str] = Query(None, description="Comma-separated regions"),
    countries: Optional[str] = Query(None, description="Comma-separated countries"),
    styles: Optional[str] = Query(None, description="Comma-separated wine styles (primary_type)"),
    pairings: Optional[str] = Query(None, description="Comma-separated food pairings"),
    limit: Optional[int] = Query(20, gt=0),
    page: Optional[int] = Query(1, ge=1)
):
    query: dict = {}

    # Wine Types
    if types:
        type_list = [t.strip() for t in types.split(",") if t.strip()]
        query["Wine Type"] = {"$in": type_list}

    # Price range
    if min_price is not None or max_price is not None:
        price_q: dict = {}
        if min_price is not None:
            price_q["$gte"] = min_price
        if max_price is not None:
            price_q["$lte"] = max_price
        query["Price"] = price_q

    # Minimum rating
    if min_rating is not None:
        query["Rating"] = {"$gte": min_rating}

    # Grapes
    if grapes:
        grape_list = [g.strip() for g in grapes.split(",") if g.strip()]
        query["Grape Type List"] = {"$in": grape_list}

    # Regions
    if regions:
        region_list = [r.strip() for r in regions.split(",") if r.strip()]
        query["Region"] = {"$in": region_list}

    # Countries
    if countries:
        country_list = [c.strip() for c in countries.split(",") if c.strip()]
        query["Country"] = {"$in": country_list}

    # Styles (primary_type)
    if styles:
        style_list = [s.strip() for s in styles.split(",") if s.strip()]
        query["primary type"] = {"$in": style_list}

    # Food pairings
    if pairings:
        pairing_list = [p.strip() for p in pairings.split(",") if p.strip()]
        query["Food Pairing"] = {"$in": pairing_list}

    # Pagination
    skip = (page - 1) * limit

    try:
        cursor = db.wines.find(query).skip(skip).limit(limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    results = []
    async for wine in cursor:
        results.append(wine_helper(wine))
    return results