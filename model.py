from transformers import pipeline

# Load the model once at startup — this avoids reloading on every request.
# The default model is distilbert-base-uncased-finetuned-sst-2-english,
# a lightweight but accurate sentiment classifier from HuggingFace.
_sentiment_pipeline = pipeline("sentiment-analysis")

NEUTRAL_THRESHOLD = 0.65  # Scores below this become NEUTRAL


def analyze_sentiment(text: str) -> dict:
    """
    Run inference on a single text string.

    Returns a dict with:
      - text:  the original input
      - label: POSITIVE | NEGATIVE | NEUTRAL
      - score: confidence float (0–1)
    """
    raw = _sentiment_pipeline(text)[0]
    label: str = raw["label"]        # "POSITIVE" or "NEGATIVE"
    score: float = raw["score"]      # confidence for the predicted label

    # Stretch goal: relabel low-confidence predictions as NEUTRAL
    if score < NEUTRAL_THRESHOLD:
        label = "NEUTRAL"

    return {"text": text, "label": label, "score": round(score, 4)}
