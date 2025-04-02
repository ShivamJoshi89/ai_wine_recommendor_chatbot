# backend/models/chat.py
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class ChatMessage(BaseModel):
    sender: str  # "user" or "assistant"
    message: str
    timestamp: datetime

class ChatHistory(BaseModel):
    user_id: str
    messages: List[ChatMessage] = []
