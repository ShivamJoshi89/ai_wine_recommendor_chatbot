# backend/models/user.py
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

# Schema for user creation (registration)
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

# Schema for user stored in the DB (do not return password!)
class UserInDB(BaseModel):
    id: Optional[str] = None  # MongoDB document ID as a string
    username: str
    email: EmailStr
    hashed_password: str
    created_at: datetime

# Schema for user response (public info)
class User(BaseModel):
    id: str
    username: str
    email: EmailStr