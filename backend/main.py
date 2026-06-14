"""
FastAPI backend for MPR Chatbot.
Provides /chat endpoint for chatbot queries.
Integrates ML model from chatbot_pipeline_v1 for intelligent response generation.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
from datetime import datetime
import sys
from pathlib import Path

# Add chatbot module to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "chatbot"))

try:
    from chatbot.chatbot_pipeline_v1 import ask as pipeline_ask
    ML_MODEL_LOADED = True
except Exception as e:
    print(f"Warning: Could not load ML model: {e}")
    from dummy_responses import get_dummy_response, get_suggested_questions
    ML_MODEL_LOADED = False

# ===========================
# App Setup
# ===========================
app = FastAPI(
    title="MPR Chatbot API",
    description="NLP Chatbot API for Ministry of Justice BiH",
    version="0.1.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================
# Data Models
# ===========================
class ChatQuery(BaseModel):
    """User chat query."""
    query: str
    session_id: Optional[str] = None
    language: Optional[str] = "bs"  # bs=Bosnian, hr=Croatian, sr=Serbian, en=English


class ChatResponse(BaseModel):
    """Chatbot response."""
    response: str
    confidence: float
    category: str
    sources: List[Dict[str, Any]] = []
    timestamp: Optional[str] = None
    query: str
    intent: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    message: str


# ===========================
# Endpoints
# ===========================
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - API info."""
    return HealthResponse(
        status="ok",
        message="MPR Chatbot API is running. Visit /docs for API documentation."
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        message="API is operational"
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(chat_query: ChatQuery):
    """
    Process a user chat query and return a response.
    Uses integrated ML model for intelligent response generation.
    
    Args:
        chat_query: ChatQuery object with user's question
    
    Returns:
        ChatResponse with answer, confidence score, category, sources and intent
    """
    if not chat_query.query or len(chat_query.query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if len(chat_query.query) > 5000:
        raise HTTPException(status_code=400, detail="Query too long (max 5000 chars)")
    
    try:
        if ML_MODEL_LOADED:
            # Use ML pipeline for response
            pipeline_result = pipeline_ask(chat_query.query)
            
            # Extract main source for category and source fields
            main_source = ""
            main_category = ""
            if pipeline_result.get("sources"):
                main_source = pipeline_result["sources"][0].get("url", "")
                main_category = pipeline_result["sources"][0].get("page_type", "")
            
            return ChatResponse(
                response=pipeline_result.get("answer", ""),
                confidence=float(pipeline_result.get("confidence", 0.0)),
                category=main_category or pipeline_result.get("intent", "unknown"),
                sources=pipeline_result.get("sources", []),
                timestamp=datetime.utcnow().isoformat(),
                query=chat_query.query,
                intent=pipeline_result.get("intent", "")
            )
        else:
            # Fallback to dummy responses if ML model not loaded
            dummy_result = get_dummy_response(chat_query.query)
            
            return ChatResponse(
                response=dummy_result["response"],
                confidence=dummy_result["confidence"],
                category=dummy_result["category"],
                sources=[],
                timestamp=datetime.utcnow().isoformat(),
                query=chat_query.query
            )
    except Exception as e:
        # Fallback if pipeline errors
        print(f"Error in pipeline: {e}")
        return ChatResponse(
            response="Došlo je do greške pri obrada vašeg pitanja. Molimo pokušajte ponovo.",
            confidence=0.0,
            category="error",
            sources=[],
            timestamp=datetime.utcnow().isoformat(),
            query=chat_query.query
        )


@app.get("/suggested-questions", response_model=List[str])
async def suggested_questions():
    """Get list of suggested questions to guide users."""
    if ML_MODEL_LOADED:
        # Use curated questions from the system
        return [
            "Kako da osnovam udruženje?",
            "Što su potrebni dokumenti za registraciju?",
            "Kako da se prijavim na pravosudni ispit?",
            "Gdje mogu dobiti pravnu pomoć?",
            "Koji su troškovi registracije?",
            "Kako da promenim podatke o svojoj organizaciji?",
            "Što trebam znati o strućnom upravnom ispitu?",
            "Kako da registrujem fondaciju?"
        ]
    else:
        # Fallback to dummy suggestions
        return get_suggested_questions()


@app.get("/info", response_model=dict)
async def get_info():
    """Get API info and status."""
    return {
        "name": "MPR Chatbot API",
        "version": "0.2.0",
        "status": "production" if ML_MODEL_LOADED else "development",
        "language": "Python + FastAPI",
        "ml_model": "chatbot_pipeline_v1" if ML_MODEL_LOADED else "dummy_responses",
        "features": {
            "intelligent_retrieval": ML_MODEL_LOADED,
            "semantic_search": ML_MODEL_LOADED,
            "intent_classification": ML_MODEL_LOADED,
            "source_tracking": ML_MODEL_LOADED
        },
        "note": "ML model is live" if ML_MODEL_LOADED else "Using fallback dummy responses"
    }


# ===========================
# Error Handlers
# ===========================
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return {
        "error": exc.detail,
        "status_code": exc.status_code
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
