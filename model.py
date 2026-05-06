from textblob import TextBlob

def analyze_sentiment(text: str) -> dict:
    blob = TextBlob(text)
    score = blob.sentiment.polarity  # -1 to 1

    if score > 0.1:
        label = "POSITIVE"
        confidence = round((score + 1) / 2, 4)
    elif score < -0.1:
        label = "NEGATIVE"
        confidence = round((1 - score) / 2, 4)
    else:
        label = "NEUTRAL"
        confidence = round(1 - abs(score), 4)

    return {"text": text, "label": label, "score": confidence}