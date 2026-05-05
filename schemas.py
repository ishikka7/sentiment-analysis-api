from pydantic import BaseModel, Field, field_validator
from typing import List, Literal


# ── Request schemas ───────────────────────────────────────────────────────────

class TextInput(BaseModel):
    text: str = Field(..., description="Text to analyze.", examples=["I love this product!"])

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("text must not be empty or whitespace.")
        return v.strip()


class BatchTextInput(BaseModel):
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=32,
        description="List of texts to analyze (max 32).",
        examples=[["Great experience!", "Terrible service.", "It was okay."]],
    )

    @field_validator("texts", mode="before")
    @classmethod
    def texts_must_not_contain_empty(cls, v: List[str]) -> List[str]:
        for i, text in enumerate(v):
            if not text or not text.strip():
                raise ValueError(f"texts[{i}] must not be empty or whitespace.")
        return [t.strip() for t in v]


# ── Response schemas ──────────────────────────────────────────────────────────

class SentimentResponse(BaseModel):
    text: str = Field(..., description="The original input text.")
    label: Literal["POSITIVE", "NEGATIVE", "NEUTRAL"] = Field(
        ..., description="Predicted sentiment label."
    )
    score: float = Field(..., description="Model confidence (0–1).", ge=0.0, le=1.0)


class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse]


class HealthResponse(BaseModel):
    status: str
    message: str
