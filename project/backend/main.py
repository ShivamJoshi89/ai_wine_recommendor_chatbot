# project/backend/main.py
from fastapi import FastAPI
from .routers import auth, users, chat, wines, reviews
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Wine Recommendation API",
    description="APIs for user authentication, chat, wines, and reviews.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://192.168.56.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
async def read_root():
    return {"message": "Welcome to the Wine Recommendation API!"}

# Include all routers.
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(chat.router)
app.include_router(wines.router)
app.include_router(reviews.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
