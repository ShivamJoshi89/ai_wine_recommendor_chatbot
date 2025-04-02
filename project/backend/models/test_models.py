# backend/models/test_models.py
from user import UserCreate, UserInDB, User
from chat import ChatMessage, ChatHistory
from datetime import datetime, timezone

# Create a sample user
new_user = UserCreate(username="shivamj", email="shivam.joshi89.us@gmail.com", password="ballu")
print(new_user)

# Create a sample chat message using timezone-aware datetime
message = ChatMessage(
    sender="user",
    message="Hello, recommend me a wine worth 30 dollars from italy",
    timestamp=datetime.now(timezone.utc)
)
print(message)

# Create a chat history instance
chat_history = ChatHistory(user_id="123456", messages=[message])
print(chat_history)