"""Prometheus metrics for the LLM gateway."""
from __future__ import annotations
from prometheus_client import Counter, Histogram, Gauge

# Per-provider request counter
gateway_requests_total = Counter(
    "gateway_requests_total",
    "Total gateway requests",
    ["provider", "status"],
)

# Per-provider latency histogram (p50, p95, p99 derived from this)
gateway_request_duration_seconds = Histogram(
    "gateway_request_duration_seconds",
    "Request duration in seconds",
    ["provider"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

# Per-provider error type counter
gateway_error_total = Counter(
    "gateway_error_total",
    "Gateway errors by type",
    ["provider", "error_type"],
)

# Circuit breaker state (0=CLOSED, 1=OPEN, 2=HALF_OPEN)
circuit_breaker_state = Gauge(
    "circuit_breaker_state",
    "Circuit breaker state per provider (0=CLOSED,1=OPEN,2=HALF_OPEN)",
    ["provider"],
)

# Failover events
gateway_failover_total = Counter(
    "gateway_failover_total",
    "Failover events between providers",
    ["from_provider", "to_provider"],
)

# Queue depth
gateway_queue_depth = Gauge(
    "gateway_queue_depth",
    "Depth of deferrable request queue",
    ["queue_type"],
)

# Cost tracking
gateway_cost_usd_total = Counter(
    "gateway_cost_usd_total",
    "Cumulative cost in USD",
    ["tenant", "feature", "provider"],
)

# Hedge overhead
gateway_hedge_overhead_total = Counter(
    "gateway_hedge_overhead_total",
    "Count of hedged requests (extra provider calls)",
)
