"""Shared test fixtures."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Load EVALS/.env (ANTHROPIC_API_KEY, POSTGRES_DSN) so live/postgres tests can find their
# credentials without requiring the caller to export them manually. Safe no-op if the file
# or the dotenv package is missing — live tests just stay skipped in that case.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass
