# project/backend/index_wines.py
import asyncio
from elasticsearch import Elasticsearch, helpers
from motor.motor_asyncio import AsyncIOMotorClient
from .es_client import get_es_client
from config import MONGODB_URI, DATABASE_NAME  # Adjust these as needed

ES_INDEX = "winerecommendor"

# Create the Elasticsearch client and MongoDB client
es = get_es_client()
mongo_client = AsyncIOMotorClient(MONGODB_URI)
db = mongo_client[DATABASE_NAME]

# Define an index mapping for your wines in Elasticsearch.
INDEX_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "wine_id": {"type": "keyword"},
            "wine_name": {"type": "text"},
            "winery": {"type": "text"},
            "country": {"type": "text"},
            "region": {"type": "text"},
            "wine_type": {"type": "keyword"},
            "grape_type_list": {"type": "text"},
            "price": {"type": "float"},
            "rating": {"type": "float"},
            "wine_description_1": {"type": "text"},
            "primary_type": {"type": "text"},
            "image_url": {"type": "keyword"}
        }
    }
}

def wine_doc_mapper(wine: dict) -> dict:
    """
    Map a MongoDB wine document to an Elasticsearch document.
    Adjust based on the actual CSV/MongoDB field names.
    """
    return {
        "wine_id": wine.get("wine_id"),
        "wine_name": wine.get("Wine Name") or "",
        "winery": wine.get("Winery") or "",
        "country": wine.get("Country") or "",
        "region": wine.get("Region") or "",
        "wine_type": wine.get("Wine Type") or "",
        "grape_type_list": wine.get("Grape Type List") or "",
        "price": wine.get("Price"),
        "rating": wine.get("Rating"),
        "wine_description_1": wine.get("Wine Description 1") or "",
        "primary_type": wine.get("primary type") or "",
        "image_url": wine.get("Image URL") or ""
    }

async def index_wines():
    # Create the ES index with mapping if it doesn't exist.
    if not es.indices.exists(index=ES_INDEX):
        es.indices.create(index=ES_INDEX, body=INDEX_MAPPING)
        print(f"Created index '{ES_INDEX}'.")

    # Retrieve all wine documents from MongoDB.
    cursor = db.wines.find({})
    actions = []
    count = 0
    async for wine in cursor:
        doc = wine_doc_mapper(wine)
        doc_id = wine.get("wine_id") or str(wine["_id"])
        action = {
            "_index": ES_INDEX,
            "_id": doc_id,
            "_source": doc
        }
        actions.append(action)
        count += 1
        if len(actions) >= 500:
            helpers.bulk(es, actions)
            actions = []
    if actions:
        helpers.bulk(es, actions)
    print(f"Indexed {count} wines into Elasticsearch.")
    mongo_client.close()

if __name__ == "__main__":
    asyncio.run(index_wines())
