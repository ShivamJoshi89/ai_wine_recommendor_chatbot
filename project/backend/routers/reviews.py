# project/backend/routers/reviews.py
from fastapi import APIRouter, HTTPException, Depends
from typing import List
from datetime import datetime
from bson import ObjectId
from ..models.review import Review
from ..db import db
from ..routers.users import get_current_user  # This dependency ensures the user is authenticated
from ..models.user import User

router = APIRouter(prefix="/api/reviews", tags=["reviews"])

# Helper function to format the MongoDB review document
def review_helper(review) -> dict:
    return {
        "id": str(review["_id"]),
        "wine_id": review.get("wine_id"),
        "user_id": review.get("user_id"),
        "rating": review.get("rating"),
        "review_text": review.get("review_text"),
        "created_at": review.get("created_at")
    }

# Endpoint to fetch reviews for a specific wine
@router.get("/", response_model=List[Review])
async def get_reviews(wine_id: str):
    reviews = []
    cursor = db.reviews.find({"wine_id": wine_id})
    async for review in cursor:
        reviews.append(review_helper(review))
    return reviews

# Endpoint to submit a review (requires authentication)
@router.post("/", response_model=Review)
async def submit_review(review: Review, current_user: User = Depends(get_current_user)):
    review.user_id = current_user.id  # Attach the authenticated user's id
    review.created_at = datetime.utcnow()
    review_dict = review.dict()
    result = await db.reviews.insert_one(review_dict)
    review.id = str(result.inserted_id)
    return review