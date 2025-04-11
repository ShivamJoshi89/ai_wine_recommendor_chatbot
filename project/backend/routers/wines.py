from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from bson import ObjectId
from bson.errors import InvalidId
from pydantic import BaseModel
from ..models.wine import Wine
from ..db import db

router = APIRouter(prefix="/api/wines", tags=["wines"])

def convert_to_str(value):
    """Convert a value to a string; if None, return an empty string."""
    if value is None:
        return ""
    return str(value)

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

# Define a dedicated Pydantic model for featured wines.
class FeaturedWine(BaseModel):
    id: str
    wine_name: str
    winery: str
    price: Optional[float]
    description: str
    image_url: str

def featured_wine_helper(wine) -> dict:
    """
    Helper function to extract only the necessary fields for a featured wine,
    including the unique identifier and the image URL.
    """
    return {
        "id": str(wine["_id"]),
        "wine_name": convert_to_str(wine.get("Wine Name")),
        "winery": convert_to_str(wine.get("Winery")),
        "price": wine.get("Price"),
        "description": convert_to_str(wine.get("Wine Description 1")),
        "image_url": convert_to_str(wine.get("Image URL")),
    }

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
        # Calculate how many documents to skip based on the page number.
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

@router.get("/featured", response_model=List[FeaturedWine])
async def get_featured_wine():
    """
    Return a list of featured wines selected from different price ranges.
    For each price range:
      - Filter wines by price.
      - Determine the maximum rating within that group.
      - Filter the wines to only those with that maximum rating.
      - Deterministically select the desired number of bottles by sorting by _id and limiting.
      
    Price Ranges:
      1. $50 - $100    -> 2 bottles
      2. $100 - $150   -> 2 bottles
      3. $150 - $250   -> 2 bottles
      4. $250 - $350   -> 2 bottles
      5. $350 & above  -> 4 bottles

    Each featured wine includes id, wine_name, winery, price, description, and image_url.
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
        # Build the price query:
        if pr["max"] is None:
            query = {"Price": {"$gte": pr["min"]}}
        else:
            query = {"Price": {"$gte": pr["min"], "$lt": pr["max"]}}
            
        # Aggregation pipeline:
        # 1. $match by price.
        # 2. $group to get the maximum "Rating" in this group,
        #    and push all matching documents into an array.
        # 3. $project to filter that array to only include wines with the max rating.
        # 4. $unwind and $replaceRoot to output individual wines.
        # 5. $sort by _id (deterministically) and $limit to the desired number.
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

@router.get("/{wine_id}", response_model=Wine)
async def get_wine_detail(wine_id: str):
    wine = None
    try:
        # Try to interpret wine_id as a MongoDB ObjectId.
        wine = await db.wines.find_one({"_id": ObjectId(wine_id)})
    except InvalidId:
        # If it's not a valid ObjectId, search using the custom wine_id field.
        wine = await db.wines.find_one({"wine_id": wine_id})
    
    if wine:
        return wine_helper(wine)
    
    raise HTTPException(status_code=404, detail="Wine not found")
