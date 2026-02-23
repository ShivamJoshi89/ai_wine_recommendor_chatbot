# project/backend/main.py
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import auth, users, chat, wines, reviews

app = FastAPI(
    title="Wine Recommendation API",
    description="APIs for user authentication, chat, wines, and reviews.",
    version="1.0.0",
)

# CORS: set via env for local + deployment
origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Wine Recommendation API!"}

@app.get("/health")
async def health():
    return {"status": "ok"}

# Include all routers
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(chat.router)
app.include_router(wines.router)
app.include_router(reviews.router)