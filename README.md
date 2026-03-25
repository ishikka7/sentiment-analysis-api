# Sentiment Analysis API

> Analyze any text and return an emotion score in real time using a pre-trained NLP model.

## What You Will Learn
- How to use pre-trained NLP models from HuggingFace
- How to build and document a REST API with FastAPI
- How to deploy a Python backend on Render for free

## Tech Stack
- Python
- FastAPI
- HuggingFace Transformers
- Uvicorn
- Render (deployment)

## Getting Started

### Prerequisites
- Python 3.9+ installed
- pip installed
- A free Render account

### Installation
```bash
git clone https://github.com/thanos.zip/sentiment-analysis-api
cd sentiment-analysis-api
pip install -r requirements.txt
uvicorn main:app --reload
```

## Project Structure
```
sentiment-analysis-api/
├── main.py              # Entry point, API routes live here
├── model.py             # Load and run the HuggingFace model here
├── schemas.py           # Define request and response shapes here
├── requirements.txt     # All dependencies
└── README.md
```

## What To Build
- Load a pre-trained sentiment model from HuggingFace (hint: pipeline("sentiment-analysis"))
- Create a POST /analyze endpoint that accepts a text string
- Return the label (POSITIVE/NEGATIVE) and confidence score
- Add input validation — reject empty strings
- Add a GET /health endpoint to confirm the API is running
- Deploy to Render and test your live URL

## Stretch Goals
- Support batch analysis — accept a list of texts in one request
- Add a NEUTRAL label for low confidence scores
- Build a minimal React frontend to demo the API

## Resources
- https://huggingface.co/docs/transformers/quicktour
- https://fastapi.tiangolo.com/tutorial/
- https://render.com/docs/deploy-fastapi

---
Built for [@thanos.zip](https://instagram.com/thanos.zip) followers.
