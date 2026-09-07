"""Tests for the deferrable request queue."""
import pytest

from src.queue import DeferrableQueue


@pytest.fixture
def queue(fake_redis):
    return DeferrableQueue(fake_redis, key="test:queue")


def test_enqueue_and_dequeue(queue):
    payload = {"request_id": "req-1", "messages": [{"role": "user", "content": "hello"}]}
    queue.enqueue(payload)
    result = queue.dequeue()
    assert result == payload


def test_fifo_order(queue):
    for i in range(5):
        queue.enqueue({"i": i})
    for i in range(5):
        item = queue.dequeue()
        assert item["i"] == i


def test_dequeue_empty_returns_none(queue):
    assert queue.dequeue() is None


def test_depth(queue):
    assert queue.depth() == 0
    queue.enqueue({"a": 1})
    queue.enqueue({"b": 2})
    assert queue.depth() == 2
    queue.dequeue()
    assert queue.depth() == 1


def test_drain(queue):
    for i in range(3):
        queue.enqueue({"i": i})
    items = queue.drain()
    assert len(items) == 3
    assert queue.depth() == 0
