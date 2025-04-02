# backend/load_sample_data.py
import os
import pandas as pd
import asyncio
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

# Load configuration from .env and config.py
load_dotenv()
from config import MONGODB_URI, DATABASE_NAME

async def insert_data(collection_name: str, data: list):
    client = AsyncIOMotorClient(MONGODB_URI)
    db = client[DATABASE_NAME]
    collection = db[collection_name]
    if data:
        result = await collection.insert_many(data)
        print(f"Inserted {len(result.inserted_ids)} documents into collection '{collection_name}'.")
    else:
        print(f"No data found for collection '{collection_name}'.")
    client.close()

async def process_users():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "data", "Users_RealNames.xlsx")
    df = pd.read_excel(file_path)
    # Optionally, do any preprocessing on df (e.g., convert date strings to datetime)
    users_data = df.to_dict(orient="records")
    await insert_data("users", users_data)

async def process_chat_messages():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "data", "ChatMessages_RealNames.xlsx")
    df = pd.read_excel(file_path)
    chat_data = df.to_dict(orient="records")
    await insert_data("chat_history", chat_data)

async def process_recommendations():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "data", "Recommendations_RealNames.xlsx")
    df = pd.read_excel(file_path)
    recommendations_data = df.to_dict(orient="records")
    await insert_data("recommendations", recommendations_data)

async def process_wine_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "data", "final_wine_data.csv")
    df = pd.read_csv(file_path)
    # Fill any NaN values if needed
    wine_data = df.fillna("").to_dict(orient="records")
    await insert_data("wines", wine_data)

async def main():
    print("Loading Users data...")
    await process_users()
    print("Loading Chat Messages data...")
    await process_chat_messages()
    print("Loading Recommendations data...")
    await process_recommendations()
    print("Loading Wine data...")
    await process_wine_data()

if __name__ == "__main__":
    asyncio.run(main())