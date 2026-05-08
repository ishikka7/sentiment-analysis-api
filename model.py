from transformers import pipeline

_sentiment_pipeline = pipeline("sentiment-analysis")
NEUTRAL_THRESHOLD = 0.65

def analyze_sentiment(text: str) -> dict:
    raw = _sentiment_pipeline(text)[0]
    label = raw["label"]
    score = raw["score"]
    if score < NEUTRAL_THRESHOLD:
        label = "NEUTRAL"
    return {"text": text, "label": label, "score": round(score, 4)}