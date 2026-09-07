"""
Self-Healing LLM Gateway — FastAPI entry point.

OpenAI-compatible POST /v1/chat/completions endpoint with:
- Per-request tenant, feature, request_id headers
- Circuit breaker per provider
- Automatic failover
- Hedged requests (X-Hedge: true)
- Deferrable queue during degradation
- Prometheus metrics on /metrics
- Chaos injection on /chaos/{provider}
"""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

import redis
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from src import chaos as chaos_store
from src.circuit_breaker import CircuitBreaker
from src.config import get_settings
from src.health import HealthStore
from src.metrics import (
    gateway_cost_usd_total,
    gateway_error_total,
    gateway_request_duration_seconds,
    gateway_requests_total,
)
from src.providers import PROVIDERS, ProviderError, ProviderName
from src.queue import DeferrableQueue
from src.retry import RetryExhausted, retry_with_backoff
from src.router import GatewayRouter, NoAvailableProvider

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    r = redis.from_url(settings.redis_url, decode_responses=True)
    store = HealthStore(r, window_size=settings.cb_window_size)
    cb = CircuitBreaker(
        store=store,
        error_rate_threshold=settings.cb_error_rate_threshold,
        p95_latency_threshold_ms=settings.cb_p95_latency_threshold_ms,
        open_duration_seconds=settings.cb_open_duration_seconds,
        half_open_probe_count=settings.cb_half_open_probes,
    )
    _state["redis"] = r
    _state["store"] = store
    _state["cb"] = cb
    _state["router"] = GatewayRouter(cb)
    _state["queue"] = DeferrableQueue(r, settings.deferrable_queue_key)
    _state["idempotency_cache"] = {}  # request_id -> response (in-process; use Redis in prod)
    yield
    r.close()


app = FastAPI(title="Self-Healing LLM Gateway", lifespan=lifespan)


# ---------------------------------------------------------------------------
# OpenAI-compatible models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "claude-haiku-4-5"
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 1024
    stream: bool = False


# ---------------------------------------------------------------------------
# Mock provider implementation (no external API required)
# ---------------------------------------------------------------------------

def _call_mock_provider(messages: list[ChatMessage], model: str) -> dict:
    """Return a deterministic mock response for testing / fallback."""
    # Simulate chaos
    extra_latency = chaos_store.get_extra_latency_ms("mock")
    if extra_latency:
        time.sleep(extra_latency / 1000)

    fail, err_type = chaos_store.should_fail("mock")
    if fail:
        raise RuntimeError(f"Chaos-injected error: {err_type}")

    last_user_msg = next(
        (m.content for m in reversed(messages) if m.role == "user"),
        "Hello"
    )
    content = f"[Mock response to: {last_user_msg[:80]}]"
    return {
        "id": f"mock-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 15, "total_tokens": 35},
        "provider": "mock",
    }


def _call_litellm_provider(provider: str, messages: list[ChatMessage], model: str) -> dict:
    """Normalize all real provider calls through LiteLLM for a unified interface.

    LiteLLM translates the OpenAI message format to each provider's native SDK,
    so adding a new provider (Cohere, Bedrock, Gemini, etc.) only requires
    updating the model string — no SDK-specific code.
    """
    import litellm  # pip install litellm

    settings = get_settings()

    extra_latency = chaos_store.get_extra_latency_ms(provider)
    if extra_latency:
        time.sleep(extra_latency / 1000)

    fail, err_type = chaos_store.should_fail(provider)
    if fail:
        raise RuntimeError(f"Chaos-injected error: {err_type}")

    # Build litellm model string — format: "provider/model-name"
    if provider == ProviderName.ANTHROPIC:
        litellm_model = f"anthropic/{model}"
        litellm.api_key = settings.anthropic_api_key
    elif provider == ProviderName.OPENAI:
        litellm_model = f"openai/{model}"
        litellm.openai_key = settings.openai_api_key
    else:
        raise ValueError(f"Provider {provider!r} not supported via LiteLLM path")

    # LiteLLM accepts OpenAI-format messages for all providers
    litellm_messages = [{"role": m.role, "content": m.content} for m in messages]

    response = litellm.completion(
        model=litellm_model,
        messages=litellm_messages,
        max_tokens=1024,
    )

    return {
        "id": getattr(response, "id", f"{provider}-{uuid.uuid4().hex[:8]}"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response.choices[0].message.content,
                },
                "finish_reason": response.choices[0].finish_reason or "stop",
            }
        ],
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
        "provider": provider,
    }


def _call_provider(provider: str, messages: list[ChatMessage], model: str) -> dict:
    """Dispatch to the right provider implementation.

    Real providers (Anthropic, OpenAI) are normalized through LiteLLM so that
    adding a new provider requires no new SDK-specific code — only a new model
    string in providers.py. Mock provider is handled locally for testing.
    """
    settings = get_settings()

    if provider == ProviderName.MOCK:
        return _call_mock_provider(messages, model)
    elif provider == ProviderName.ANTHROPIC:
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not configured")
        return _call_litellm_provider(provider, messages, model)
    elif provider == ProviderName.OPENAI:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not configured")
        return _call_litellm_provider(provider, messages, model)
    else:
        # Default fallback to mock
        return _call_mock_provider(messages, model)


# ---------------------------------------------------------------------------
# Main chat completion endpoint
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(
    request_body: ChatCompletionRequest,
    x_tenant: str = Header(default="default", alias="X-Tenant"),
    x_feature: str = Header(default="default", alias="X-Feature"),
    x_request_id: str = Header(default="", alias="X-Request-Id"),
    x_request_class: str = Header(default="interactive", alias="X-Request-Class"),
    x_hedge: str = Header(default="false", alias="X-Hedge"),
) -> JSONResponse:
    if not x_request_id:
        x_request_id = str(uuid.uuid4())

    cb: CircuitBreaker = _state["cb"]
    router: GatewayRouter = _state["router"]
    queue: DeferrableQueue = _state["queue"]
    idem_cache: dict = _state["idempotency_cache"]

    # Idempotency check
    if x_request_id in idem_cache:
        return JSONResponse(content=idem_cache[x_request_id])

    do_hedge = x_hedge.lower() in ("true", "1", "yes")

    try:
        primary_provider = router.select_provider(x_request_class)
    except NoAvailableProvider:
        # All providers down — queue if deferrable, else 503
        if x_request_class == "deferrable":
            queue.enqueue({
                "request_id": x_request_id,
                "tenant": x_tenant,
                "feature": x_feature,
                "messages": [m.model_dump() for m in request_body.messages],
            })
            return JSONResponse({"queued": True, "request_id": x_request_id}, status_code=202)
        raise HTTPException(503, "All providers are currently unavailable")

    def _try_provider(provider: str) -> dict:
        start = time.monotonic()
        try:
            result = _call_provider(provider, request_body.messages, request_body.model)
            latency_ms = (time.monotonic() - start) * 1000
            cb.record_success(provider)
            gateway_requests_total.labels(provider=provider, status="success").inc()
            gateway_request_duration_seconds.labels(provider=provider).observe(latency_ms / 1000)

            # Cost tracking
            tokens = result.get("usage", {})
            prov_config = PROVIDERS.get(provider)
            if prov_config:
                cost = (
                    tokens.get("prompt_tokens", 0) * prov_config.cost_per_input_token
                    + tokens.get("completion_tokens", 0) * prov_config.cost_per_output_token
                )
                gateway_cost_usd_total.labels(
                    tenant=x_tenant, feature=x_feature, provider=provider
                ).inc(cost)

            return result
        except Exception as exc:
            latency_ms = (time.monotonic() - start) * 1000
            cb.record_failure(provider, latency_ms)
            # Classify error
            err_str = str(exc).lower()
            if "rate" in err_str or "429" in err_str:
                err_type = ProviderError.RATE_LIMIT
            elif "timeout" in err_str:
                err_type = ProviderError.TIMEOUT
            elif "auth" in err_str or "401" in err_str:
                err_type = ProviderError.AUTHENTICATION_FAILURE
            elif "content" in err_str or "filter" in err_str:
                err_type = ProviderError.CONTENT_FILTER
            else:
                err_type = ProviderError.SERVER_ERROR
            gateway_error_total.labels(provider=provider, error_type=err_type).inc()
            gateway_requests_total.labels(provider=provider, status="error").inc()
            raise

    response = None
    failed_providers: list[str] = []

    if do_hedge:
        # Hedge: pick two providers and race them
        try:
            secondary_provider = router.select_with_fallback(x_request_class, [primary_provider])
            from src.hedge import hedged_call
            response = hedged_call(
                lambda: _try_provider(primary_provider),
                lambda: _try_provider(secondary_provider),
            )
        except Exception:
            do_hedge = False  # Fall through to normal path

    if response is None:
        # Normal path with fallback
        providers_to_try = [primary_provider] + [
            p for p in [ProviderName.OPENAI, ProviderName.ANTHROPIC, ProviderName.MOCK]
            if p != primary_provider and cb.allow_request(p)
        ]

        for provider in providers_to_try:
            try:
                response = _try_provider(provider)
                break
            except Exception:
                failed_providers.append(provider)
                continue

    if response is None:
        raise HTTPException(502, f"All providers failed: {failed_providers}")

    idem_cache[x_request_id] = response
    return JSONResponse(content=response)


# ---------------------------------------------------------------------------
# Health + metrics
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    from starlette.responses import Response
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------------------
# Chaos management
# ---------------------------------------------------------------------------

class ChaosRequest(BaseModel):
    latency_ms: float = 0.0
    error_rate: float = 0.0
    error_type: Optional[str] = None


@app.post("/chaos/{provider}")
def inject_chaos(provider: str, cfg: ChaosRequest) -> dict:
    chaos_store.inject(
        provider=provider,
        latency_ms=cfg.latency_ms,
        error_rate=cfg.error_rate,
        error_type=cfg.error_type,
    )
    return {"provider": provider, "chaos": cfg.model_dump()}


@app.delete("/chaos/{provider}")
def reset_chaos(provider: str) -> dict:
    chaos_store.reset(provider)
    return {"provider": provider, "chaos": "reset"}


@app.delete("/chaos")
def reset_all_chaos() -> dict:
    chaos_store.reset_all()
    return {"chaos": "all_reset"}


# ---------------------------------------------------------------------------
# Circuit breaker management
# ---------------------------------------------------------------------------

@app.get("/breakers")
def get_breakers() -> dict:
    cb: CircuitBreaker = _state["cb"]
    results = {}
    for provider in [ProviderName.ANTHROPIC, ProviderName.OPENAI, ProviderName.MOCK]:
        state = cb.get_state(provider)
        health = cb.store.load(provider)
        results[provider] = {
            "state": state,
            "error_rate": health.error_rate(),
            "p95_latency_ms": health.p95_latency_ms(),
            "window_size": len(health.window),
        }
    return results


@app.delete("/breakers/{provider}")
def reset_breaker(provider: str) -> dict:
    store: HealthStore = _state["store"]
    store.reset(provider)
    return {"provider": provider, "state": "CLOSED"}


# ---------------------------------------------------------------------------
# Queue management
# ---------------------------------------------------------------------------

@app.get("/queue/depth")
def queue_depth() -> dict:
    queue: DeferrableQueue = _state["queue"]
    return {"depth": queue.depth()}
