import os
import sys
import math
import json
import warnings
import numpy as np
import pandas as pd
import faiss
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from transformers import pipeline
from datasets import Dataset
import re

###############################################
# PRODUCTION SECURITY SETTINGS
###############################################
IS_PROD = os.environ.get("IS_PROD", "False").lower() in ("true", "1", "yes")
VERIFY_CERTS = True if IS_PROD else False
ES_CA_CERTS = os.environ.get("ES_CA_CERTS", None)

###############################################
# Adjective-based Price Filtering Settings
###############################################
AFFORDABLE_KEYWORDS = ["affordable", "budget", "cheap", "inexpensive", "low cost"]
DEFAULT_MAX_PRICE = 50.0  # Set this value based on your domain

def extract_price_from_adjectives(query):
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in AFFORDABLE_KEYWORDS):
        return DEFAULT_MAX_PRICE
    return None

###############################################
# BOOST VALUES FOR FIELD WEIGHTING
###############################################
BOOST_VALUES = {
    "primary": 5,
    "grape": 3,
    "wine_name": 2,
    "food_pairing": 4,
    "region": 3
}

###############################################
# Add the "nlu" folder to PYTHONPATH.
###############################################
current_dir = os.path.dirname(os.path.abspath(__file__))
nlu_path = os.path.join(current_dir, "..", "nlu")
if nlu_path not in sys.path:
    sys.path.append(nlu_path)

from ner import load_full_ner_pipeline

###############################################
# CONFIGURATION
###############################################
# Instead of a hard-coded absolute path, build the path relative to this script.
data_dir = os.path.join(current_dir, "..", "data")
WINE_DATA_CSV = os.path.join(data_dir, "final_wine_data.csv")

# Elasticsearch configuration details
ES_HOST = "127.0.0.1"
ES_PORT = 9200
ES_SCHEME = "https"
ES_USERNAME = "elastic"
ES_PASSWORD = "AyydP4KMT*gZT+YvtdMq"  # Change for production!
ES_INDEX = "winerecommendor"

# Semantic search configuration.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # SentenceTransformer model
EMBEDDING_DIM = 384

# Recommendation limit
RESULT_LIMIT = 15

###############################################
# ELASTICSEARCH CLIENT SETUP
###############################################
def get_es_client():
    return Elasticsearch(
        hosts=[f"{ES_SCHEME}://{ES_HOST}:{ES_PORT}"],
        basic_auth=(ES_USERNAME, ES_PASSWORD),
        verify_certs=VERIFY_CERTS,
        ca_certs=ES_CA_CERTS
    )

###############################################
# DATA CLEANING FOR INDEXING
###############################################
def clean_value(v):
    if isinstance(v, float) and math.isnan(v):
        return None
    if pd.isna(v):
        return None
    return v

def clean_dataframe(df):
    return df.applymap(lambda x: None if pd.isna(x) else x)

###############################################
# PART 1: INDEXING INTO ELASTICSEARCH
###############################################
def index_data_elasticsearch(dataframe, index_name=ES_INDEX):
    es = get_es_client()

    mapping = {
        "mappings": {
            "properties": {
                "Cleaned Wine Name": {"type": "text"},
                "Primary Type": {"type": "text"},
                "Grape Type List": {"type": "text"},
                "Wine Name": {"type": "text"},
                "Wine Description 1": {"type": "text"},
                "Food Pairing": {"type": "text"},
                "Price": {"type": "float"},
                "Region": {"type": "text"}
            }
        }
    }

    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name, ignore=[400, 404])
    es.indices.create(index=index_name, body=mapping)

    dataframe = clean_dataframe(dataframe)

    actions = []
    for idx, row in dataframe.iterrows():
        doc = {k: clean_value(v) for k, v in row.to_dict().items()}
        actions.append({
            "_index": index_name,
            "_id": idx,
            "_source": doc
        })

    errors = []
    for ok, item in helpers.streaming_bulk(es, actions, raise_on_error=False):
        if not ok:
            errors.append(item)
    if errors:
        print(f"{len(errors)} document(s) failed to index:")
        for error in errors:
            print(error)
    else:
        print(f"Indexed {len(actions)} documents successfully into index '{index_name}'.")

###############################################
# PART 2: SEMANTIC SEARCH WITH FAISS
###############################################
def compute_embeddings(dataframe, model):
    texts = (dataframe["Wine Description 1"].fillna("") + " " + dataframe["Wine Name"].fillna("")).tolist()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    embeddings = normalize(embeddings, norm="l2")
    return embeddings

def build_faiss_index(embeddings):
    num_embeddings, dim = embeddings.shape
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"Faiss index built with {index.ntotal} vectors.")
    return index

def search_semantic(query, model, faiss_index, wine_df, k=RESULT_LIMIT):
    query_embedding = model.encode([query], convert_to_numpy=True)
    query_embedding = normalize(query_embedding, norm="l2")
    distances, indices = faiss_index.search(query_embedding, k)
    results = wine_df.iloc[indices[0]].to_dict(orient="records")
    for result, score in zip(results, distances[0]):
        result["similarity"] = float(score)
    return results

###############################################
# PART 3: HYBRID SEARCH: COMBINING RESULTS
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

def search_keyword(query, nlu_result, index_name=ES_INDEX, size=RESULT_LIMIT, price_filter=None):
    es = get_es_client()
    
    wine_type = extract_wine_type(nlu_result)
    food_pairing = extract_food_pairing(nlu_result)
    region = extract_region(nlu_result)
    if price_filter is None:
        price_filter = extract_price_filter(nlu_result)
    if price_filter is None:
        price_filter = extract_price_from_adjectives(query)
    
    field_queries = []
    if wine_type:
        field_queries.extend([
            {"match": {"Primary Type": {"query": wine_type, "boost": BOOST_VALUES["primary"]}}},
            {"match": {"Grape Type List": {"query": wine_type, "boost": BOOST_VALUES["grape"]}}},
            {"match": {"Wine Name": {"query": wine_type, "boost": BOOST_VALUES["wine_name"]}}}
        ])
    if food_pairing:
        field_queries.append({"match": {"Food Pairing": {"query": food_pairing, "boost": BOOST_VALUES["food_pairing"]}}})
    if region:
        field_queries.append({"match": {"Region": {"query": region, "boost": BOOST_VALUES["region"]}}})
    
    multi_fields = [
        "Cleaned Wine Name",
        "Wine Description 1",
        "Food Pairing"
    ]
    multi_match = {"multi_match": {"query": query, "fields": multi_fields}}
    
    must_clause = []
    if field_queries:
        must_clause.append({
            "bool": {
                "should": field_queries,
                "minimum_should_match": 1
            }
        })
    must_clause.append(multi_match)
    
    if price_filter is not None:
        must_clause.append({"range": {"Price": {"lte": price_filter}}})
    
    body = {"query": {"bool": {"must": must_clause}}, "size": size}
    res = es.search(index=index_name, body=body)
    hits = res["hits"]["hits"]
    return [hit["_source"] for hit in hits]

def combine_and_rank(keyword_results, semantic_results, alpha=0.4):
    combined = {}
    
    for res in keyword_results:
        product_id = res.get("Product Link")
        kw_score = res.get("keyword_score", 0.5)
        combined[product_id] = {
            "data": res,
            "keyword_score": kw_score,
            "semantic_score": 0
        }
    
    for res in semantic_results:
        product_id = res.get("Product Link")
        sem_score = res.get("similarity", 0)
        if product_id in combined:
            combined[product_id]["semantic_score"] = max(combined[product_id]["semantic_score"], sem_score)
        else:
            combined[product_id] = {
                "data": res,
                "keyword_score": 0,
                "semantic_score": sem_score
            }
    
    for product_id, values in combined.items():
        values["combined_score"] = alpha * values["keyword_score"] + (1 - alpha) * values["semantic_score"]
    
    ranked_results = sorted(combined.values(), key=lambda x: x["combined_score"], reverse=True)
    return [item["data"] for item in ranked_results]

def hybrid_search(query, nlu_result, es_index, model, faiss_index, wine_df, k=RESULT_LIMIT, price_filter=None):
    keyword_results = search_keyword(query, nlu_result, index_name=es_index, size=k, price_filter=price_filter)
    semantic_results = search_semantic(query, model, faiss_index, wine_df, k=k)
    combined_results = combine_and_rank(keyword_results, semantic_results, alpha=0.4)
    return combined_results

###############################################
# PART 4: INTENT & ENTITY EXTRACTION (NLU)
###############################################
from transformers import AutoTokenizer, AutoModelForSequenceClassification

intent_labels = ["recommend_wine", "food_pairing", "product_details"]
intent_label2id = {label: idx for idx, label in enumerate(intent_labels)}
intent_id2label = {idx: label for label, idx in intent_label2id.items()}
intent_model_checkpoint = "distilbert-base-uncased"

intent_tokenizer = AutoTokenizer.from_pretrained("./intent_model")
intent_model = AutoModelForSequenceClassification.from_pretrained(
    "./intent_model",
    num_labels=len(intent_labels),
    id2label=intent_id2label,
    label2id=intent_label2id
)

intent_classifier = pipeline(
    "text-classification",
    model="./intent_model",
    tokenizer="./intent_model"
)

def classify_intent(text):
    result = intent_classifier(text)
    return result[0]['label']

entity_extraction_pipeline = load_full_ner_pipeline()

def process_query(query):
    intent_result = intent_classifier(query)
    intent = intent_result[0]['label']
    entities_pipeline = entity_extraction_pipeline(query)
    return {
        "intent": intent,
        "entities_pipeline": entities_pipeline
    }

###############################################
# MAIN: LOAD DATA, BUILD INDEXES, PROCESS QUERIES, AND PERFORM HYBRID SEARCH
###############################################
if __name__ == "__main__":
    wine_df = pd.read_csv(WINE_DATA_CSV)
    
    # --- Part 1: Index data into Elasticsearch --- 
    index_data_elasticsearch(wine_df, index_name=ES_INDEX)

    # --- Part 2: Build Faiss Index for Semantic Search ---
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = compute_embeddings(wine_df, embedder)
    faiss_index = build_faiss_index(embeddings)

    # --- Part 3: Process a batch of 15 sample queries ---
    sample_queries = [
        "Looking for an affordable Pinot Noir wine.",
        "Suggest a white wine made from Chenin Blanc from France.",
        "Can you recommend a red wine that pairs well with steak?",
        "Can you recommend a fortified wine that pairs well with seafood?",
        "What wine pairs well with seafood?",
        "What wine pairs well with poultry?",
        "Can you recommend a red wine that pairs well with salad?",
        "What wine pairs well with steak?",
        "I need a premium wine from Oakville.",
        "What is a good Cabernet Sauvignon wine under $162?",
        "Find me a premium red wine from Dry Creek Valley.",
        "Suggest a white wine made from Chardonnay from France.",
        "Can you recommend a sparkling wine that pairs well with cheese?",
        "What wine pairs well with dessert?",
        "Can you recommend a red wine that pairs well with cheese?"
    ]

    query_dataset = Dataset.from_dict({"query": sample_queries})
    
    def process_query_batch(batch):
        outputs = [process_query(q) for q in batch["query"]]
        return {"nlu": outputs}
    
    processed_dataset = query_dataset.map(process_query_batch, batched=True)
    
    results_dict = {f"Query {i+1}": {"query": q, "nlu": out} 
                    for i, (q, out) in enumerate(zip(query_dataset["query"], processed_dataset["nlu"]))}
    
    print(json.dumps(results_dict, indent=2, default=lambda x: float(x) if isinstance(x, np.float32) else x))
    
    full_ner = load_full_ner_pipeline()
    processed_entities = full_ner(sample_queries[0])
    print("\nProcessed NER Output from loaded pipeline for Query 1:")
    print(json.dumps(processed_entities, indent=2, default=lambda x: float(x) if isinstance(x, np.float32) else x))
    
    # --- Part 4: Perform Hybrid Search using the refined query from NLU ---
    nlu_result_q1 = results_dict["Query 1"]["nlu"]
    refined_query = sample_queries[0]
    hybrid_results = hybrid_search(refined_query, nlu_result_q1, ES_INDEX, embedder, faiss_index, wine_df, k=RESULT_LIMIT)
    
    print("\nHybrid Search Results:")
    print(json.dumps(hybrid_results, indent=2, default=lambda x: float(x) if isinstance(x, np.float32) else x))
