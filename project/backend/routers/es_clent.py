# project/backend/es_client.py
import os
import logging
from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)

def get_es_client():
    """
    Create and return an Elasticsearch client using environment variables
    for credentials, with fallback defaults.
    """
    es_username = os.getenv("ES_USERNAME", "elastic")
    es_password = os.getenv("ES_PASSWORD", "AyydP4KMT*gZT+YvtdMq")
    try:
        client = Elasticsearch(
            hosts=["https://127.0.0.1:9200"],
            basic_auth=(es_username, es_password),
            verify_certs=False  # For local testing; in production, enable cert verification
        )
        if not client.ping():
            logger.error("Elasticsearch cluster is not available!")
            raise Exception("Elasticsearch cluster is not available!")
        return client
    except Exception as e:
        logger.error(f"Failed to create Elasticsearch client: {e}")
        raise
