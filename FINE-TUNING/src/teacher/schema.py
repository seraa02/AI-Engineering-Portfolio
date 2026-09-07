"""
Data schema for the distillation pipeline.

The task: extract named entities from SEC 10-K filing excerpts.
Teacher (Claude) produces labeled examples; student (fine-tuned small LM) learns to replicate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EntityType(str, Enum):
    COMPANY = "COMPANY"
    PRODUCT = "PRODUCT"
    PERSON = "PERSON"
    REGULATION = "REGULATION"
    METRIC = "METRIC"          # Financial metrics (revenue, EPS, etc.)
    LOCATION = "LOCATION"


@dataclass
class Entity:
    text: str           # Surface form as it appears in the text
    entity_type: str    # One of EntityType values
    start: int          # Character offset in source text
    end: int            # Character offset in source text
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "entity_type": self.entity_type,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Entity":
        return cls(
            text=d["text"],
            entity_type=d["entity_type"],
            start=d.get("start", 0),
            end=d.get("end", 0),
            confidence=d.get("confidence", 1.0),
        )


@dataclass
class ExtractionExample:
    """One labeled example: text → entities."""
    id: str
    text: str
    entities: list[Entity] = field(default_factory=list)
    source: str = "teacher"          # "teacher" | "validated" | "synthetic"
    difficulty: str = "medium"       # "easy" | "medium" | "hard"
    token_count: int = 0
    teacher_cost_usd: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "entities": [e.to_dict() for e in self.entities],
            "source": self.source,
            "difficulty": self.difficulty,
            "token_count": self.token_count,
            "teacher_cost_usd": self.teacher_cost_usd,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExtractionExample":
        return cls(
            id=d["id"],
            text=d["text"],
            entities=[Entity.from_dict(e) for e in d.get("entities", [])],
            source=d.get("source", "teacher"),
            difficulty=d.get("difficulty", "medium"),
            token_count=d.get("token_count", 0),
            teacher_cost_usd=d.get("teacher_cost_usd", 0.0),
        )
