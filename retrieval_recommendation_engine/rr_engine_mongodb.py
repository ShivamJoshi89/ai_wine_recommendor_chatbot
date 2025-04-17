import os
import sys
import math
import json
import logging
import numpy as np
import pandas as pd
import faiss
import re
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
from urllib.parse import quote_plus
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from bson import ObjectId

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)

# Constants
RESULT_LIMIT = 15
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

###############################################
# Price Filtering Settings
###############################################
AFFORDABLE_KEYWORDS = ["affordable", "budget", "cheap", "inexpensive", "low cost"]
MID_PRICE_KEYWORDS = ["mid-priced", "moderate", "moderately priced", "average", "reasonable", "value"]
EXPENSIVE_KEYWORDS = ["expensive", "premium", "high end", "luxury", "dear"]

DEFAULT_MAX_PRICE_AFFORDABLE = 50.0
DEFAULT_MIN_PRICE_MID = 50.0
DEFAULT_MAX_PRICE_MID = 100.0
DEFAULT_MIN_PRICE_EXPENSIVE = 100.0

def extract_price_preferences(query):
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in AFFORDABLE_KEYWORDS):
        return (None, DEFAULT_MAX_PRICE_AFFORDABLE)
    elif any(keyword in query_lower for keyword in MID_PRICE_KEYWORDS):
        return (DEFAULT_MIN_PRICE_MID, DEFAULT_MAX_PRICE_MID)
    elif any(keyword in query_lower for keyword in EXPENSIVE_KEYWORDS):
        return (DEFAULT_MIN_PRICE_EXPENSIVE, None)
    else:
        return (None, None)

###############################################
# Data Loading and Cleaning
###############################################
def load_wines_from_mongo():
    """
    Connects to MongoDB Atlas and loads wine data from the 'wines' collection
    in the WineRecommendationProject database. Falls back to a CSV file if needed.
    """
    try:
        # Hardcoded connection string – ensure this is correct for your Atlas cluster.
        MONGO_URI = "mongodb+srv://shivamjoshi89us:DEuNYRPbElRfml0e@cluster0.twkupzt.mongodb.net/WineRecommendationProject?retryWrites=true&w=majority&appName=Cluster0"
        client = MongoClient(MONGO_URI)
        db = client["WineRecommendationProject"]
        wines = []
        cursor = db["wines"].find({}, {'_id': 0})
        for wine in cursor:
            wine.setdefault("Wine Name", "Unknown")
            wine.setdefault("Wine Description 1", "")
            wine.setdefault("Price", 0)
            wine.setdefault("Region", "Unknown")
            wine.setdefault("Food Pairing", "")
            wine.setdefault("Alcohol Content (%)", "")
            wine.setdefault("Primary Type", "")
            wine.setdefault("Country", "Unknown")
            wines.append(wine)
        return wines
    except Exception as e:
        logger.error(f"MongoDB load failed: {e}")
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        csv_path = os.path.join(data_dir, "final_wine_data.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            column_mapping = {
                "Wine": "Wine Name",
                "Description": "Wine Description 1",
                "Alcohol": "Alcohol Content (%)",
                "Type": "Primary Type"
            }
            df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
            if "Country" not in df.columns:
                df["Country"] = "Unknown"
            return df.to_dict('records')
        raise Exception("No data source available")

def clean_value(v):
    if isinstance(v, float) and math.isnan(v):
        return None
    if pd.isna(v):
        return None
    return str(v) if not isinstance(v, str) else v.strip()

def clean_dataframe(df):
    return df.applymap(clean_value)

###############################################
# MongoDB Atlas Search Functions
###############################################
def search_keyword_mongodb_atlas(query, size=RESULT_LIMIT):
    """
    Performs a full-text search using MongoDB Atlas Search via an aggregation pipeline.
    Assumes that a search index named "search_rr" has been created on your wines collection.
    The search is conducted across: "Wine Name", "Region", "Grape Type", "Winery", and "primary type".
    """
    try:
        MONGO_URI = "mongodb+srv://shivamjoshi89us:DEuNYRPbElRfml0e@cluster0.twkupzt.mongodb.net/WineRecommendationProject?retryWrites=true&w=majority&appName=Cluster0"
        client = MongoClient(MONGO_URI)
        db = client["WineRecommendationProject"]
        wines_collection = db.wines
        pipeline = [
            {
                "$search": {
                    "index": "search_rr",  # Your Atlas Search index name.
                    "text": {
                        "query": query,
                        "path": ["Wine Name", "Region", "Grape Type", "Winery", "primary type"]
                    }
                }
            },
            {"$addFields": {"score": {"$meta": "searchScore"}}},
            {"$limit": size}
        ]
        results = list(wines_collection.aggregate(pipeline))
        # Remove _id if present to prevent serialization issues.
        for doc in results:
            if "_id" in doc:
                del doc["_id"]
        return results
    except Exception as e:
        logger.error(f"MongoDB Atlas text search failed: {e}")
        return []

###############################################
# Embeddings and FAISS Indexing
###############################################
def compute_embeddings(dataframe, model):
    try:
        texts = (dataframe["Wine Description 1"].fillna("") + " " + dataframe["Wine Name"].fillna("")).tolist()
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return normalize(embeddings, norm="l2")
    except Exception as e:
        logger.error(f"Embedding computation failed: {e}")
        raise

def build_faiss_index(embeddings):
    try:
        num_embeddings, dim = embeddings.shape
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        logger.info(f"Built FAISS index with {index.ntotal} vectors")
        return index
    except Exception as e:
        logger.error(f"FAISS index build failed: {e}")
        raise

def search_semantic(query, model, faiss_index, wine_df, k=RESULT_LIMIT):
    try:
        query_embedding = model.encode([query], convert_to_numpy=True)
        query_embedding = normalize(query_embedding, norm="l2")
        distances, indices = faiss_index.search(query_embedding, k)
        valid_indices = [idx for idx in indices[0] if 0 <= idx < len(wine_df)]
        if not valid_indices:
            logger.warning("No valid indices found in semantic search")
            return []
        results = wine_df.iloc[valid_indices].to_dict(orient="records")
        for result, score in zip(results, distances[0][:len(valid_indices)]):
            result["similarity"] = float(score)
        return results
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return []

###############################################
# Hybrid Search Functions (Using MongoDB Atlas Search)
###############################################
def extract_wine_type(nlu_result):
    for ent in nlu_result.get("entities_pipeline", []):
        if ent["label"].upper() == "WINE_TYPE":
            return ent["word"]
    return None

# Replace your existing extract_price_filter(...) with:

_price_re = re.compile(r'\$\s*(\d+(?:\.\d+)?)')

def extract_price_filter(nlu_result):
    # 1) Check pipeline‐extracted entities
    for ent in nlu_result.get("entities_pipeline", []):
        if ent["label"].upper() == "PRICE_RANGE":
            m = _price_re.search(ent["word"])
            if m:
                return float(m.group(1))
    # 2) Fallback to custom entities
    for ent in nlu_result.get("entities_custom", []):
        if ent["label"].upper() == "PRICE_RANGE":
            m = _price_re.search(ent["word"])
            if m:
                return float(m.group(1))
    return None


def extract_country(nlu_result):
    for ent in nlu_result.get("entities_pipeline", []):
        if ent["label"].upper() == "COUNTRY":
            return ent["word"]
    return None

def search_keyword(query, nlu_result, size=RESULT_LIMIT):
    """
    Uses MongoDB Atlas Search to perform keyword retrieval.
    """
    return search_keyword_mongodb_atlas(query, size=size)

def combine_and_rank(keyword_results, semantic_results, alpha=0.4):
    try:
        combined = {}
        for res in keyword_results:
            # Use "Product Link" if available as a unique identifier,
            # otherwise hash the record.
            product_id = res.get("Product Link", str(hash(json.dumps(res, sort_keys=True))))
            combined[product_id] = {
                "data": res,
                "keyword_score": res.get("score", 0.5),
                "semantic_score": 0
            }
        for res in semantic_results:
            product_id = res.get("Product Link", str(hash(json.dumps(res, sort_keys=True))))
            if product_id in combined:
                combined[product_id]["semantic_score"] = max(
                    combined[product_id]["semantic_score"],
                    res.get("similarity", 0)
                )
            else:
                combined[product_id] = {
                    "data": res,
                    "keyword_score": 0,
                    "semantic_score": res.get("similarity", 0)
                }
        for product_id in combined:
            combined[product_id]["combined_score"] = (
                alpha * combined[product_id]["keyword_score"] +
                (1 - alpha) * combined[product_id]["semantic_score"]
            )
        return sorted(
            [item["data"] for item in combined.values()],
            key=lambda x: x.get("combined_score", 0),
            reverse=True
        )
    except Exception as e:
        logger.error(f"Result combination failed: {e}")
        return keyword_results + semantic_results

def hybrid_search(query, nlu_result, model, faiss_index, wine_df, k=RESULT_LIMIT):
    """
    Hybrid search using:
      - MongoDB Atlas Search for keyword retrieval.
      - FAISS for semantic similarity.
    Designed to work for any user query.
    """
    try:
        if wine_df.empty:
            logger.warning("Empty wine dataframe provided")
            return []
        keyword_results = search_keyword(query, nlu_result, size=k)
        semantic_results = search_semantic(query, model, faiss_index, wine_df, k=k)
        if not keyword_results and not semantic_results:
            logger.warning("No results found from either search method")
            return []
        combined_results = combine_and_rank(keyword_results, semantic_results, alpha=0.4)
        return combined_results[:k]
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        return []

###############################################
# Custom JSON Serializer
###############################################
def default_serializer(obj):
    # Convert NumPy float32 to native float.
    if isinstance(obj, np.float32):
        return float(obj)
    # Convert NumPy int32 to native int.
    if isinstance(obj, np.int32):
        return int(obj)
    # Convert BSON ObjectId to string.
    if isinstance(obj, ObjectId):
        return str(obj)
    # Try __str__ if available.
    if hasattr(obj, '__str__'):
        return str(obj)
    return None

###############################################
# Example Test Run
###############################################
if __name__ == "__main__":
    try:
        # Load wine data
        wines = load_wines_from_mongo()
        wine_df = pd.DataFrame(wines)
        logger.info(f"Loaded {len(wine_df)} wines")
        
        # Initialize SentenceTransformer for semantic embeddings.
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        # Compute embeddings and build FAISS index.
        embeddings = compute_embeddings(wine_df, model)
        faiss_index = build_faiss_index(embeddings)
        
        # Define a sample query.
        sample_query = "recommend me a red wine from napa valley which is above 150 dollars"
        
        # Import and use the NLU pipeline from ner.py to extract intent and entities.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ner_path = os.path.join(current_dir, "..", "nlu")
        if ner_path not in sys.path:
            sys.path.append(ner_path)
        from ner import process_query
        nlu_result = process_query(sample_query)
        logger.info("NLU result: " + json.dumps(nlu_result, indent=2, default=default_serializer))
        
        # Execute the hybrid search.
        results = hybrid_search(sample_query, nlu_result, model, faiss_index, wine_df, k=RESULT_LIMIT)
        
        # Print out search results using custom JSON serialization.
        print("Search Results:")
        print(json.dumps(results, indent=2, default=default_serializer))
        
    except Exception as ex:
        logger.error(f"An error occurred during the test run: {ex}")
