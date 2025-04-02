# project/backend/routers/chat.py
from fastapi import APIRouter, Depends, HTTPException
from ..models.chat import ChatHistory, ChatMessage
from datetime import datetime
from typing import List
from .users import get_current_user
from ..models.user import User
from ..db import db
from ..response_generator_openai import ResponseGeneratorOpenAI  # Import your response generator

router = APIRouter(prefix="/api/chat", tags=["chat"])

# Instantiate a global response generator instance.
# (Note: generate_response() is a blocking call; if needed, consider running it in a thread executor.)
response_generator = ResponseGeneratorOpenAI()

@router.post("/", response_model=dict)
async def send_chat_message(message: dict, current_user: User = Depends(get_current_user)):
    """
    Expected request body:
    {
      "message": "Your query here"
    }
    """
    user_query = message.get("message")
    if not user_query:
        raise HTTPException(status_code=400, detail="Message is required")
    
    # Generate a response using your OpenAI response generator
    assistant_reply = response_generator.generate_response(user_query)
    
    # Create chat message instances for both user and assistant
    chat_msg_user = ChatMessage(
        sender="user",
        message=user_query,
        timestamp=datetime.utcnow()
    )
    chat_msg_assistant = ChatMessage(
        sender="assistant",
        message=assistant_reply,
        timestamp=datetime.utcnow()
    )
    
    # Save the messages in the chat_history collection
    chat_doc = await db.chat_history.find_one({"user_id": current_user.id})
    if chat_doc:
        # Append to the existing chat history
        await db.chat_history.update_one(
            {"user_id": current_user.id},
            {"$push": {"messages": {"$each": [chat_msg_user.dict(), chat_msg_assistant.dict()]}}}
        )
    else:
        # Create a new chat history document
        new_chat = ChatHistory(user_id=current_user.id, messages=[chat_msg_user, chat_msg_assistant])
        await db.chat_history.insert_one(new_chat.dict())
    
    return {"response": assistant_reply}