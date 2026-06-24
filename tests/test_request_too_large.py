"""A 413 "request too large" must not be treated as a retryable rate-limit.

Groq's free tier rejects a single request that exceeds its per-minute token
budget with HTTP 413 and "rate limit" wording. Retrying the identical request
can never succeed, so it must be excluded from the rate-limit class (which
would otherwise requeue the task with minute-scale backoffs) and instead fall
back to a higher-capacity provider.
"""

from __future__ import annotations

import httpx

from forven import ai


def _http_error(status: int, message: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status, text=message, request=request)
    return httpx.HTTPStatusError(message, request=request, response=response)


GROQ_413 = (
    "Request too large for model `llama-3.3-70b-versatile` in organization "
    "org_x service tier `on_demand` on tokens per minute (TPM): Limit 12000, "
    "Requested 13409, please reduce your message size and try again. "
    "Code: rate_limit_exceeded"
)


def test_request_too_large_detected_by_status():
    assert ai._is_request_too_large(_http_error(413, "anything"))


def test_request_too_large_detected_by_message():
    assert ai._is_request_too_large(_http_error(413, GROQ_413))
    # Even without the 413 status, the wording alone is enough.
    assert ai._is_request_too_large(RuntimeError(GROQ_413))


def test_request_too_large_not_classified_as_rate_limit():
    # The Groq 413 message contains "rate limit" / "rate_limit", but it must NOT
    # be treated as a retryable rate-limit.
    err = _http_error(413, GROQ_413)
    assert ai._is_request_too_large(err) is True
    assert ai._is_rate_limit_exception(err) is False


def test_genuine_rate_limit_still_classified():
    # A real 429 with no "too large" wording stays a retryable rate-limit.
    err = _http_error(429, "Rate limit reached for requests. Please try again later.")
    assert ai._is_request_too_large(err) is False
    assert ai._is_rate_limit_exception(err) is True


GEMINI_SPEND_CAP = (
    "Error code: 429 - [{'error': {'code': 429, 'message': 'Your project has "
    "exceeded its monthly spending cap. Please go to AI Studio at "
    "https://ai.studio/spend to manage your project spend cap.', "
    "'status': 'RESOURCE_EXHAUSTED'}}]"
)


def test_quota_exhausted_detects_spend_cap():
    err = _http_error(429, GEMINI_SPEND_CAP)
    assert ai._is_quota_exhausted(err) is True


def test_quota_exhausted_detects_out_of_credits_and_insufficient_quota():
    assert ai._is_quota_exhausted(RuntimeError("You're out of credits"))
    assert ai._is_quota_exhausted(RuntimeError("error code: insufficient_quota"))


def test_generic_rate_limit_is_not_quota_exhausted():
    # A transient per-minute throttle must NOT be treated as persistent
    # exhaustion (which would apply a 30+ minute backoff).
    err = _http_error(429, "Rate limit reached for requests. Please try again in 1s.")
    assert ai._is_quota_exhausted(err) is False
    assert ai._is_rate_limit_exception(err) is True


def test_spend_cap_is_not_request_too_large():
    err = _http_error(429, GEMINI_SPEND_CAP)
    assert ai._is_request_too_large(err) is False
    assert ai._is_quota_exhausted(err) is True


def test_groq_fallback_chain_is_self_only():
    from forven.model_routing import get_fallback_chain

    chain = get_fallback_chain("groq")
    providers = [entry[0] if isinstance(entry, tuple) else entry.get("provider") for entry in chain]
    # Fail-closed default: no auto cross-provider fallback (groq used to degrade
    # to Gemini). Operators opt into a fallback per-slot in the Routing tab.
    assert providers == ["groq"]


def test_provider_quota_alert_dedupes(forven_db):
    """A persistent-exhaustion alert is raised once per provider per cooldown."""
    from forven.agents.runner import _emit_provider_quota_alert
    from forven.db import get_db

    _emit_provider_quota_alert("gemini", "spend cap exceeded")
    _emit_provider_quota_alert("gemini", "spend cap exceeded again")  # within cooldown

    with get_db() as conn:
        count = conn.execute(
            "SELECT COUNT(*) AS c FROM activity_log WHERE source = 'provider:gemini'"
        ).fetchone()["c"]
    assert count == 1  # second call deduped, not flooding

    # A different provider is tracked independently.
    _emit_provider_quota_alert("groq", "out of credits")
    with get_db() as conn:
        groq_count = conn.execute(
            "SELECT COUNT(*) AS c FROM activity_log WHERE source = 'provider:groq'"
        ).fetchone()["c"]
    assert groq_count == 1
