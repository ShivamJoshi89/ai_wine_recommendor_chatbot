# project/backend/main.py
from fastapi import FastAPI
from .routers import auth, users, chat
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Wine Recommendation API",
    description="APIs for user authentication, chat, and more.",
    version="1.0.0"
)

# Configure CORS so that your React app can make requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust as necessary
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Wine Recommendation API!"}

# Include routers
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(chat.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)