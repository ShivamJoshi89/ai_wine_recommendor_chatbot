# project/backend/models/review.py
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class Review(BaseModel):
    id: Optional[str] = None         # This will hold the MongoDB _id as a string
    wine_id: str                     # ID of the wine being reviewed
    user_id: Optional[str] = None    # ID of the user who submitted the review
    rating: float                    # e.g., a rating from 0 to 5
    review_text: str                 # The text of the review
    created_at: Optional[datetime] = None  # When the review was created
