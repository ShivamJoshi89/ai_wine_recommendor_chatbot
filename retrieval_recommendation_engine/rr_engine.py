import os
import sys
import math
import json
import logging
import numpy as np
import pandas as pd
import faiss
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import re
import asyncio
from datetime import datetime
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)

# Constants
ES_INDEX = "winerecommendor"
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
    try:
        from pymongo import MongoClient
        from urllib.parse import quote_plus
        
        username = quote_plus(os.getenv("MONGO_USER", ""))
        password = quote_plus(os.getenv("MONGO_PASS", ""))
        uri = f"mongodb+srv://{username}:{password}@your-cluster.mongodb.net/"
        
        client = MongoClient(uri, retryWrites=True, w="majority")
        db = client.your_database
        
        wines = []
        cursor = db.wines.find({}, {'_id': 0})
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
# Elasticsearch Client and Indexing
###############################################
def get_es_client():
    from elasticsearch import Elasticsearch
    try:
        return Elasticsearch(
            hosts=[f"https://127.0.0.1:9200"],
            basic_auth=(os.getenv("ES_USERNAME", "elastic"), os.getenv("ES_PASSWORD", "password")),
            verify_certs=False
        )
    except Exception as e:
        logger.error(f"Failed to create Elasticsearch client: {e}")
        raise

def index_data_elasticsearch(dataframe, index_name=ES_INDEX):
    try:
        es = get_es_client()
        required_columns = {"Wine Name", "Primary Type", "Grape Type List", 
                            "Wine Description 1", "Food Pairing", "Price", "Region", "Country"}
        missing_cols = required_columns - set(dataframe.columns)
        if missing_cols:
            logger.warning(f"Missing columns in dataframe: {missing_cols}")
            for col in missing_cols:
                dataframe[col] = ""
        mapping = {
            "mappings": {
                "properties": {
                    "Wine Name": {"type": "text"},
                    "Primary Type": {"type": "text"},
                    "Grape Type List": {"type": "text"},
                    "Wine Description 1": {"type": "text"},
                    "Food Pairing": {"type": "text"},
                    "Price": {"type": "float"},
                    "Region": {"type": "text"},
                    "Country": {"type": "text"}
                }
            }
        }
        if es.indices.exists(index=index_name):
            es.indices.delete(index=index_name, ignore=[400, 404])
        es.indices.create(index=index_name, body=mapping)
        dataframe = clean_dataframe(dataframe)
        from elasticsearch import helpers
        actions = [
            {
                "_index": index_name,
                "_id": idx,
                "_source": {k: clean_value(v) for k, v in row.to_dict().items()}
            }
            for idx, row in dataframe.iterrows()
        ]
        success_count = 0
        for ok, _ in helpers.streaming_bulk(es, actions, raise_on_error=False):
            if ok:
                success_count += 1
        logger.info(f"Successfully indexed {success_count}/{len(actions)} documents")
        return success_count == len(actions)
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        return False

###############################################
# Embeddings and FAISS Indexing
###############################################
def compute_embeddings(dataframe, model):
    try:
        texts = (dataframe["Wine Description 1"].fillna("") + " " + dataframe["Wine Name"].fillna("")).tolist()
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        from sklearn.preprocessing import normalize
        return normalize(embeddings, norm="l2")
    except Exception as e:
        logger.error(f"Embedding computation failed: {e}")
        raise

def build_faiss_index(embeddings):
    try:
        num_embeddings, dim = embeddings.shape
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        logger.info(f"Built Faiss index with {index.ntotal} vectors")
        return index
    except Exception as e:
        logger.error(f"Faiss index build failed: {e}")
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
# Hybrid Search Functions
###############################################
def extract_wine_type(nlu_result):
    for ent in nlu_result.get("entities_pipeline", []):
        if ent["label"].upper() == "WINE_TYPE":
            return ent["word"]
    return None

def extract_food_pairing(nlu_result):
    for ent in nlu_result.get("entities_pipeline", []):
        if ent["label"].upper() == "FOOD_PAIRING":
            return ent["word"].replace("pairs well with", "").replace("well with", "").strip()
    return None

def extract_price_filter(nlu_result):
    for ent in nlu_result.get("entities_pipeline", []):
        if ent["label"].upper() == "PRICE_RANGE":
            match = re.search(r'\d+', ent["word"])
            if match:
                return float(match.group())
    return None

def extract_region(nlu_result):
    for ent in nlu_result.get("entities_pipeline", []):
        if ent["label"].upper() == "REGION":
            return ent["word"]
    return None

def extract_country(nlu_result):
    for ent in nlu_result.get("entities_pipeline", []):
        if ent["label"].upper() == "COUNTRY":
            return ent["word"]
    return None

def search_keyword(query, nlu_result, index_name=ES_INDEX, size=RESULT_LIMIT, price_filter=None):
    try:
        es = get_es_client()
        wine_type = extract_wine_type(nlu_result)
        food_pairing = extract_food_pairing(nlu_result)
        region = extract_region(nlu_result)
        country = extract_country(nlu_result)
        if not country and "italy" in query.lower():
            country = "italy"
        pref_min, pref_max = extract_price_preferences(query)
        ent_price = extract_price_filter(nlu_result)
        if ent_price is not None:
            price_min = None
            price_max = ent_price
        else:
            price_min = pref_min
            price_max = pref_max
        multi_fields = ["Wine Name", "Wine Description 1", "Food Pairing"]
        must_clause = [{"multi_match": {"query": query, "fields": multi_fields}}]
        if region:
            must_clause.append({"match": {"Region": {"query": region, "boost": 3}}})
        if country:
            must_clause.append({"match": {"Country": {"query": country, "boost": 3}}})
        field_queries = []
        if wine_type:
            field_queries.extend([
                {"match": {"Primary Type": {"query": wine_type, "boost": 5}}},
                {"match": {"Grape Type List": {"query": wine_type, "boost": 3}}},
                {"match": {"Wine Name": {"query": wine_type, "boost": 2}}}
            ])
        if food_pairing:
            field_queries.append({"match": {"Food Pairing": {"query": food_pairing, "boost": 4}}})
        if field_queries:
            must_clause.append({"bool": {"should": field_queries, "minimum_should_match": 1}})
        if price_min is not None or price_max is not None:
            range_clause = {}
            if price_min is not None:
                range_clause["gte"] = price_min
            if price_max is not None:
                range_clause["lte"] = price_max
            must_clause.append({"range": {"Price": range_clause}})
        body = {"query": {"bool": {"must": must_clause}}, "size": size}
        res = es.search(index=index_name, body=body)
        return [hit["_source"] for hit in res["hits"]["hits"]]
    except Exception as e:
        logger.error(f"Keyword search failed: {e}")
        return []

def combine_and_rank(keyword_results, semantic_results, alpha=0.4):
    try:
        combined = {}
        for res in keyword_results:
            product_id = res.get("Product Link", str(hash(json.dumps(res, sort_keys=True))))
            combined[product_id] = {
                "data": res,
                "keyword_score": res.get("_score", 0.5),
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

def hybrid_search(query, nlu_result, es_index, model, faiss_index, wine_df, k=RESULT_LIMIT, price_filter=None):
    try:
        if wine_df.empty:
            logger.warning("Empty wine dataframe provided")
            return []
        keyword_results = search_keyword(query, nlu_result, index_name=es_index, size=k, price_filter=price_filter)
        semantic_results = search_semantic(query, model, faiss_index, wine_df, k=k)
        if not keyword_results and not semantic_results:
            logger.warning("No results found from either search method")
            return []
        combined_results = combine_and_rank(keyword_results, semantic_results, alpha=0.4)
        return combined_results[:k]
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        return []