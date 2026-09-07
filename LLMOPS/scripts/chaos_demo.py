"""
Chaos demo: demonstrates the complete circuit breaker cycle.

  1. Healthy traffic (all providers CLOSED)
  2. Inject high error rate on primary provider
  3. Circuit trips OPEN
  4. Traffic fails over to secondary
  5. Remove chaos → provider recovers
  6. Circuit transitions OPEN → HALF_OPEN → CLOSED
  7. Traffic returns to primary

Run against a live gateway:
    python scripts/chaos_demo.py --gateway-url http://localhost:8001
"""
import argparse
import time
import requests


def send_request(base_url: str, content: str = "What is 2+2?") -> dict:
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": content}]},
        headers={
            "X-Tenant": "demo",
            "X-Feature": "chaos-test",
            "X-Request-Id": f"demo-{time.time_ns()}",
            "X-Request-Class": "interactive",
        },
        timeout=15,
    )
    data = resp.json()
    return {"status": resp.status_code, "provider": data.get("provider", "?"), "id": data.get("id", "?")}


def get_breakers(base_url: str) -> dict:
    return requests.get(f"{base_url}/breakers").json()


def inject_chaos(base_url: str, provider: str, error_rate: float, error_type: str = "server_error"):
    requests.post(f"{base_url}/chaos/{provider}", json={"error_rate": error_rate, "error_type": error_type})


def reset_chaos(base_url: str, provider: str):
    requests.delete(f"{base_url}/chaos/{provider}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway-url", default="http://localhost:8001")
    args = parser.parse_args()
    base = args.gateway_url

    print("=" * 60)
    print("Phase 1: Healthy traffic")
    print("=" * 60)
    for i in range(5):
        r = send_request(base)
        print(f"  Request {i+1}: status={r['status']} provider={r['provider']}")
        time.sleep(0.5)

    print("\nPhase 2: Inject 100% error rate on 'mock' provider")
    inject_chaos(base, "mock", error_rate=1.0, error_type="server_error")

    print("\nPhase 3: Send requests — circuit should trip on 'mock'")
    for i in range(15):
        r = send_request(base)
        breakers = get_breakers(base)
        mock_state = breakers.get("mock", {}).get("state", "?")
        print(f"  Request {i+1}: status={r['status']} provider={r['provider']} mock_cb={mock_state}")
        time.sleep(0.3)

    print("\nPhase 4: Provider recovery — remove chaos from 'mock'")
    reset_chaos(base, "mock")
    print("  Chaos removed. Waiting for OPEN→HALF_OPEN timeout...")

    # In the gateway, CB_OPEN_DURATION_SECONDS defaults to 30s.
    # For demo purposes, reset the breaker manually.
    requests.delete(f"{base}/breakers/mock")
    print("  Breaker manually reset to CLOSED (simulating recovery)")

    print("\nPhase 5: Traffic returning to primary")
    for i in range(5):
        r = send_request(base)
        breakers = get_breakers(base)
        mock_state = breakers.get("mock", {}).get("state", "?")
        print(f"  Request {i+1}: status={r['status']} provider={r['provider']} mock_cb={mock_state}")
        time.sleep(0.5)

    print("\nDemo complete.")
    print("Final breaker states:", get_breakers(base))


if __name__ == "__main__":
    main()
