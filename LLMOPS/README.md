# Self-Healing LLM Gateway

**Demonstrates: CLOSED → OPEN → HALF_OPEN → CLOSED circuit breaker cycle under controlled chaos.**

The gateway handles provider outages transparently: when the primary provider exceeds the error rate or latency threshold, its circuit breaker trips OPEN, traffic automatically fails over to the next available provider, and the breaker recovers once the provider is healthy again.

Run `python scripts/chaos_demo.py` to simulate a complete provider outage cycle against a live gateway.

---

## Architecture

```mermaid
graph LR
    Client -->|POST /v1/chat/completions| Gateway
    Gateway --> Router
    Router -->|preference list| CB_Anthropic[CB: Anthropic]
    Router -->|fallback| CB_OpenAI[CB: OpenAI]
    Router -->|fallback| CB_Mock[CB: Mock]
    CB_Anthropic -->|CLOSED| Anthropic[Anthropic API]
    CB_OpenAI -->|CLOSED| OpenAI[OpenAI API]
    CB_Mock -->|always available| MockProvider[Mock Provider]
    Gateway -->|health state| Redis[(Redis)]
    Gateway -->|metrics| Prometheus[(Prometheus)]
    Prometheus --> Grafana[Grafana]
    Gateway -->|deferrable| Queue[Redis Queue]
    Gateway <-->|chaos inject| ChaosAPI[/chaos/provider]
```

---

## Circuit Breaker States

| State | Behavior | Transition |
|-------|----------|------------|
| `CLOSED` | Normal operation, all requests pass through | → OPEN when error_rate > 50% OR p95 > 5s (min 5 requests) |
| `OPEN` | All requests rejected immediately | → HALF_OPEN after 30s |
| `HALF_OPEN` | Probe requests allowed through | → CLOSED after 3 consecutive successes; → OPEN on any failure |

---

## Key Design Decisions

**Why 3 providers?** The gateway needs at least 3 to demonstrate real failover chains (primary → secondary → tertiary). The mock provider ensures the system never fully degrades during development/testing.

**Why Redis for health state?** Health state (error windows, circuit breaker state) must survive process restarts. An in-process counter would reset on every restart, causing the system to "forget" a provider was degraded. Redis also enables multi-instance gateway deployments to share health state.

**Why hedged requests?** For interactive requests where latency matters most, sending to two providers simultaneously and using the first response reduces tail latency. The overhead (extra API call) is only incurred when explicitly requested via `X-Hedge: true`.

**Why deferrable queue?** Not all requests need immediate responses. During provider degradation, deferrable tasks (batch processing, async workflows) can wait in a Redis queue rather than failing. The queue drains automatically when providers recover.

**Chaos injection** targets the mock provider by default so real API quotas are never consumed during testing. The chaos state is ephemeral (in-process) and isolated from Redis health state.

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start Redis (or use Docker)
redis-server

# Or start everything with Docker
docker compose up

# Run the gateway
uvicorn src.api.main:app --host 0.0.0.0 --port 8001

# Run tests (no live APIs required)
pytest tests/ -v
```

---

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI-compatible chat completion |
| `/health` | GET | Gateway health check |
| `/metrics` | GET | Prometheus metrics |
| `/breakers` | GET | Circuit breaker state per provider |
| `/breakers/{provider}` | DELETE | Reset breaker to CLOSED |
| `/chaos/{provider}` | POST | Inject latency/errors |
| `/chaos/{provider}` | DELETE | Remove chaos injection |
| `/queue/depth` | GET | Deferrable queue depth |

### Required Headers

| Header | Description | Example |
|--------|-------------|---------|
| `X-Tenant` | Tenant ID for cost tracking | `my-company` |
| `X-Feature` | Feature name for cost tracking | `search-widget` |
| `X-Request-Id` | Idempotency key | `req-abc-123` |
| `X-Request-Class` | `interactive` or `deferrable` | `interactive` |
| `X-Hedge` | `true` to enable hedged requests | `true` |

---

## Chaos Demo

```bash
# Start gateway and Redis first
docker compose up -d redis gateway

# Run the complete circuit breaker demonstration
python scripts/chaos_demo.py --gateway-url http://localhost:8001
```

The demo:
1. Sends 5 healthy requests (all succeed via mock provider)
2. Injects 100% error rate on the mock provider
3. Sends 15 requests — mock circuit trips OPEN, traffic fails over
4. Removes chaos injection (provider "recovers")
5. Resets circuit breaker manually
6. Sends 5 requests confirming traffic returns to primary

---

## Observability

**Prometheus metrics available at `/metrics`:**
- `gateway_requests_total{provider, status}` — request volume
- `gateway_request_duration_seconds{provider}` — latency histograms (p50/p95/p99)
- `gateway_error_total{provider, error_type}` — errors by taxonomy
- `circuit_breaker_state{provider}` — 0=CLOSED, 1=OPEN, 2=HALF_OPEN
- `gateway_failover_total{from_provider, to_provider}` — failover events
- `gateway_queue_depth{queue_type}` — deferrable queue depth
- `gateway_cost_usd_total{tenant, feature, provider}` — cumulative cost

**Grafana dashboard** provisioned automatically via `dashboards/gateway.json`.

---

## Known Limitations

The mock provider is hardcoded to return deterministic responses. In production, the provider abstraction layer would use LiteLLM's unified interface to call real providers. The circuit breaker and failover logic is identical regardless of which provider implementation is used.

Idempotency cache is in-process (dict) in this implementation. A production deployment would use Redis TTL-based caching keyed by `X-Request-Id`.
