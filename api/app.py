from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import SentimentPredictor

# Initialize FastAPI
app = FastAPI(
    title="Enhanced Sentiment Analysis API",
    description="Advanced DistilBERT model with LoRA adapters for sentiment classification",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor (adjust path as needed)
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model_sst2.pt')
predictor = SentimentPredictor(MODEL_PATH, device='cpu')

# Request models
class TextInput(BaseModel):
    text: str = Field(..., example="This movie was absolutely fantastic!")

class BatchInput(BaseModel):
    texts: List[str] = Field(..., example=["Great movie!", "Terrible film."])

class PredictionResponse(BaseModel):
    text: str
    sentiment: str
    label: int
    confidence: float
    probabilities: dict
    inference_time_ms: float

# Routes
@app.get("/")
def root():
    """API information"""
    model_info = predictor.get_model_info()
    return {
        "message": "Enhanced Sentiment Analysis API",
        "model": "DistilBERT with LoRA adapters",
        "status": "operational",
        "model_info": model_info,
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "model_info": "/info",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(input: TextInput):
    """Predict sentiment for a single text"""
    try:
        if not input.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        result = predictor.predict(input.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch")
def predict_batch(input: BatchInput):
    """Predict sentiment for multiple texts"""
    try:
        if not input.texts:
            raise HTTPException(status_code=400, detail="Texts list cannot be empty")
        
        if len(input.texts) > 100:
            raise HTTPException(status_code=400, detail="Maximum 100 texts per request")
        
        results = predictor.predict_batch(input.texts)
        return {
            "count": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/info")
def model_info():
    """Get model information"""
    return predictor.get_model_info()

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": predictor.device
    }

@app.get("/examples")
def get_examples():
    """Get example predictions"""
    examples = [
        "This movie was absolutely fantastic!",
        "Worst film I've ever seen.",
        "It was okay, nothing special.",
        "Amazing performance by the cast!",
        "Complete waste of time and money."
    ]
    
    results = predictor.predict_batch(examples)
    return {"examples": results}