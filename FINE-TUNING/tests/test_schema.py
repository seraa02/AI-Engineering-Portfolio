"""Tests for data schema."""
import pytest
from src.teacher.schema import Entity, EntityType, ExtractionExample


class TestEntity:
    def test_to_dict(self):
        e = Entity(text="NVIDIA", entity_type="COMPANY", start=0, end=6)
        d = e.to_dict()
        assert d["text"] == "NVIDIA"
        assert d["entity_type"] == "COMPANY"
        assert d["start"] == 0
        assert d["end"] == 6

    def test_from_dict(self):
        d = {"text": "NVIDIA", "entity_type": "COMPANY", "start": 0, "end": 6}
        e = Entity.from_dict(d)
        assert e.text == "NVIDIA"
        assert e.entity_type == "COMPANY"

    def test_from_dict_defaults(self):
        e = Entity.from_dict({"text": "test", "entity_type": "PERSON"})
        assert e.start == 0
        assert e.confidence == 1.0

    def test_roundtrip(self):
        e = Entity(text="$5B", entity_type="METRIC", start=10, end=13, confidence=0.9)
        assert Entity.from_dict(e.to_dict()).text == "$5B"
        assert Entity.from_dict(e.to_dict()).confidence == 0.9


class TestExtractionExample:
    def test_to_dict(self):
        ex = ExtractionExample(
            id="ex_001",
            text="NVIDIA reported $5B revenue.",
            entities=[Entity(text="NVIDIA", entity_type="COMPANY", start=0, end=6)],
            difficulty="easy",
        )
        d = ex.to_dict()
        assert d["id"] == "ex_001"
        assert len(d["entities"]) == 1
        assert d["difficulty"] == "easy"

    def test_from_dict_roundtrip(self):
        ex = ExtractionExample(
            id="ex_002",
            text="AMD GPU",
            entities=[Entity(text="AMD", entity_type="COMPANY", start=0, end=3)],
        )
        restored = ExtractionExample.from_dict(ex.to_dict())
        assert restored.id == "ex_002"
        assert len(restored.entities) == 1
        assert restored.entities[0].text == "AMD"

    def test_empty_entities(self):
        ex = ExtractionExample(id="empty", text="no entities here")
        assert ex.entities == []
