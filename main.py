from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from schemas import TextInput, BatchTextInput, SentimentResponse, BatchSentimentResponse, HealthResponse
from model import analyze_sentiment
 
app = FastAPI(
    title="Sentiment Analysis API",
    description="Analyze any text and return an emotion score in real time using a pre-trained NLP model.",
    version="1.0.0",
)
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
 
@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse("index.html")
 
 
@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Confirm the API is running."""
    return HealthResponse(status="ok", message="Sentiment Analysis API is running.")
 
 
@app.post("/analyze", response_model=SentimentResponse, tags=["Sentiment"])
def analyze(input: TextInput):
    """
    Analyze the sentiment of a single text string.
 
    - Returns POSITIVE, NEGATIVE, or NEUTRAL label
    - Returns a confidence score between 0 and 1
    - Rejects empty strings
    """
    result = analyze_sentiment(input.text)
    return SentimentResponse(**result)
 
 
@app.post("/analyze/batch", response_model=BatchSentimentResponse, tags=["Sentiment"])
def analyze_batch(input: BatchTextInput):
    """
    Analyze the sentiment of a list of text strings in one request.
 
    - Accepts up to 32 texts at once
    - Returns POSITIVE, NEGATIVE, or NEUTRAL labels with confidence scores
    """
    results = [analyze_sentiment(text) for text in input.texts]
    return BatchSentimentResponse(results=results)