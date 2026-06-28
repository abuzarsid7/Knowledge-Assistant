import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.api.health import router as health_router

app = FastAPI(
    title="Enterprise Knowledge Assistant API",
    description="RAG-powered API for answering questions over ingested documents.",
    version="1.0.0"
)

# Set up CORS middleware
# This is necessary because the Streamlit frontend usually runs on a different port (e.g., 8501)
# than the FastAPI backend (e.g., 8000). Without this, browser requests would be blocked.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, lock this down to the specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include our modular routers
app.include_router(health_router, tags=["System"])
app.include_router(api_router, prefix="/api/v1", tags=["API"])

@app.get("/", tags=["Root"])
def root():
    return {"message": "Knowledge Assistant API is running"}

if __name__ == "__main__":
    # Provides a simple way to launch the dev server by running `python app/main.py`
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)