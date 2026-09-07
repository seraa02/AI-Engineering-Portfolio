"""Redis queue for deferrable requests during provider degradation."""
from __future__ import annotations

import json
from typing import Optional

import redis

from src.metrics import gateway_queue_depth


class DeferrableQueue:
    def __init__(self, redis_client: redis.Redis, key: str = "gateway:deferrable"):
        self.redis = redis_client
        self.key = key

    def enqueue(self, request_payload: dict) -> int:
        """Push a request to the deferrable queue. Returns new queue length."""
        self.redis.rpush(self.key, json.dumps(request_payload))
        depth = self.depth()
        gateway_queue_depth.labels(queue_type="deferrable").set(depth)
        return depth

    def dequeue(self) -> Optional[dict]:
        """Pop the oldest request (FIFO). Returns None if empty."""
        raw = self.redis.lpop(self.key)
        if raw is None:
            return None
        depth = self.depth()
        gateway_queue_depth.labels(queue_type="deferrable").set(depth)
        return json.loads(raw)

    def depth(self) -> int:
        return self.redis.llen(self.key)

    def drain(self) -> list[dict]:
        """Return all items and clear the queue (for testing)."""
        items = []
        while True:
            item = self.dequeue()
            if item is None:
                break
            items.append(item)
        return items
