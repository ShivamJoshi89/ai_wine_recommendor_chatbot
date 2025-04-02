# project/backend/db.py
from motor.motor_asyncio import AsyncIOMotorClient
from .config import MONGODB_URI, DATABASE_NAME

client = AsyncIOMotorClient(MONGODB_URI)
db = client[DATABASE_NAME]