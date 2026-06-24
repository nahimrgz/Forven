"""Tests for Z.AI (GLM) provider integration."""

import asyncio
from copy import deepcopy
from unittest.mock import AsyncMock, patch, MagicMock

from forven.model_routing import (
    _SUPPORTED_PROVIDERS,
    _DEFAULT_MODEL_ROUTING,
    get_default_model_for_provider,
)
from forven.ai import (
    normalize_provider_and_model,
    _normalize_provider,
    ENDPOINTS,
)


def _mock_kv_get(key, default=None):
    """Return default so tests don't need a real database."""
    return default


# Task 1 tests
def test_zai_in_supported_providers():
    assert "zai" in _SUPPORTED_PROVIDERS


def test_zai_default_model():
    assert _DEFAULT_MODEL_ROUTING["default_models"]["zai"] == "glm-5.1"


def test_zai_in_provider_priority():
    assert "zai" in _DEFAULT_MODEL_ROUTING["provider_priority"]


def test_zai_is_default_primary_provider():
    assert _DEFAULT_MODEL_ROUTING["provider_priority"][0] == "zai"


def test_get_default_model_for_zai():
    with patch("forven.model_routing.kv_get", side_effect=_mock_kv_get):
        model = get_default_model_for_provider("zai")
    assert model == "glm-5.1"


def test_zai_fallback_chain_exists():
    assert "zai" in _DEFAULT_MODEL_ROUTING["fallback_chains"]
    chain = _DEFAULT_MODEL_ROUTING["fallback_chains"]["zai"]
    providers_in_chain = [entry["provider"] for entry in chain]
    # Self-only (fail-closed) by default — no auto cross-provider fallback.
    assert providers_in_chain == ["zai"]


# Task 2 tests
def test_zai_provider_aliases():
    assert _normalize_provider("zai") == "zai"
    assert _normalize_provider("z.ai") == "zai"
    assert _normalize_provider("z-ai") == "zai"
    assert _normalize_provider("ZAI") == "zai"


def test_zai_endpoint_exists():
    assert "zai" in ENDPOINTS


def test_normalize_zai_provider_and_model():
    provider, model = normalize_provider_and_model("zai", "glm-5.1")
    assert provider == "zai"
    assert model == "glm-5.1"


def test_normalize_zai_default_model():
    with patch("forven.model_routing.kv_get", side_effect=_mock_kv_get):
        provider, model = normalize_provider_and_model("zai", None)
    assert provider == "zai"
    assert model == "glm-5.1"


def test_normalize_missing_provider_and_model_uses_primary_routing():
    with patch("forven.model_routing.kv_get", side_effect=_mock_kv_get):
        provider, model = normalize_provider_and_model(None, None)
    assert provider == "zai"
    assert model == "glm-5.1"


def test_normalize_glm_model_infers_zai():
    provider, model = normalize_provider_and_model(None, "glm-5.1")
    assert provider == "zai"
    assert model == "glm-5.1"


def test_normalize_glm_prefix_infers_zai():
    provider, model = normalize_provider_and_model("openai", "glm-4.7-flash")
    assert provider == "zai"
    assert model == "glm-4.7-flash"


def test_auth_store_reads_anthropic_env_for_zai():
    from forven.auth.store import get_profile

    with patch.dict(
        "os.environ",
        {
            "ANTHROPIC_AUTH_TOKEN": "env-zai-token",
            "ANTHROPIC_BASE_URL": "https://api.z.ai/api/anthropic",
        },
        clear=False,
    ):
        with patch("forven.auth.store.load_auth", return_value={"version": 1, "profiles": {}}):
            profile = get_profile("zai")

    assert profile["access"] == "env-zai-token"
    assert profile["base_url"] == "https://api.z.ai/api/anthropic"


# Task 3 tests
def test_call_single_dispatches_to_zai():
    from forven.ai import _call_single

    mock_response = "Hello from GLM"

    async def _run():
        with patch("forven.ai._call_zai", new_callable=AsyncMock, return_value=mock_response) as mock_call:
            with patch("forven.ai.get_token", return_value="fake-zai-token"):
                result = await _call_single("zai", "glm-5.1", [{"role": "user", "content": "hi"}], 100, 0.7, None)
        return result, mock_call

    result, mock_call = asyncio.run(_run())
    assert result == mock_response
    mock_call.assert_called_once()


def test_call_zai_uses_reasoning_content_when_content_empty():
    from forven.ai import _call_zai

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "reasoning_content": "OK",
                }
            }
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 1},
    }

    async def _run():
        with patch("forven.ai.get_profile", return_value={"base_url": "https://api.z.ai/api/paas/v4"}):
            with patch("forven.ai.httpx.AsyncClient") as MockClient:
                mock_client = MagicMock()
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client.post = AsyncMock(return_value=mock_response)
                MockClient.return_value = mock_client
                return await _call_zai(
                    "fake-token",
                    "glm-5.1",
                    [{"role": "user", "content": "hi"}],
                    32,
                    0.0,
                    None,
                )

    assert asyncio.run(_run()) == "OK"


def test_call_zai_supports_anthropic_compatible_base_url():
    from forven.ai import _call_zai

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "content": [{"type": "text", "text": "OK"}],
        "usage": {"input_tokens": 4, "output_tokens": 1},
    }
    captured: dict[str, object] = {}

    async def _post(url, json=None, headers=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return mock_response

    async def _run():
        with patch("forven.ai.get_profile", return_value={"base_url": "https://api.z.ai/api/anthropic"}):
            with patch("forven.ai.httpx.AsyncClient") as MockClient:
                mock_client = MagicMock()
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client.post = AsyncMock(side_effect=_post)
                MockClient.return_value = mock_client
                return await _call_zai(
                    "fake-token",
                    "glm-5.1",
                    [{"role": "user", "content": "hi"}],
                    32,
                    0.0,
                    "system prompt",
                )

    result = asyncio.run(_run())
    assert result == "OK"
    assert captured["url"] == "https://api.z.ai/api/anthropic/v1/messages"
    assert captured["headers"]["x-api-key"] == "fake-token"
    assert captured["json"]["system"] == "system prompt"


def test_agent_provider_factory_returns_zai_provider():
    from forven.agents.providers import ZAIProvider, get_provider

    provider = get_provider("zai")
    assert isinstance(provider, ZAIProvider)


def test_agent_zai_provider_uses_openai_compatible_endpoint():
    from forven.agents.providers import ZAIProvider

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "reasoning_content": "OK",
                    "tool_calls": [],
                }
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1},
    }
    captured: dict[str, object] = {}

    async def _post(url, json=None, headers=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return mock_response

    async def _run():
        provider = ZAIProvider()
        with patch("forven.agents.providers.get_profile", return_value={"base_url": "https://api.z.ai/api/paas/v4"}):
            with patch("forven.agents.providers.httpx.AsyncClient") as MockClient:
                mock_client = MagicMock()
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client.post = AsyncMock(side_effect=_post)
                MockClient.return_value = mock_client
                return await provider.call(
                    "glm-5.1",
                    [{"role": "user", "content": "hi"}],
                    "system prompt",
                    [{"name": "demo_tool", "description": "Demo", "input_schema": {"type": "object", "properties": {}}}],
                    "fake-token",
                )

    result = asyncio.run(_run())
    assert result.text == "OK"
    assert captured["url"] == "https://api.z.ai/api/paas/v4/chat/completions"
    assert captured["json"]["tools"][0]["type"] == "function"


def test_agent_zai_provider_uses_anthropic_compatible_endpoint():
    from forven.agents.providers import ZAIProvider

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "content": [
            {"type": "text", "text": "OK"},
            {"type": "tool_use", "id": "tool-1", "name": "demo_tool", "input": {"x": 1}},
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 3, "output_tokens": 1},
    }
    captured: dict[str, object] = {}

    async def _post(url, json=None, headers=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return mock_response

    async def _run():
        provider = ZAIProvider()
        with patch("forven.agents.providers.get_profile", return_value={"base_url": "https://api.z.ai/api/anthropic"}):
            with patch("forven.agents.providers.httpx.AsyncClient") as MockClient:
                mock_client = MagicMock()
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client.post = AsyncMock(side_effect=_post)
                MockClient.return_value = mock_client
                return await provider.call(
                    "glm-5.1",
                    [{"role": "user", "content": "hi"}],
                    "system prompt",
                    [{"name": "demo_tool", "description": "Demo", "input_schema": {"type": "object", "properties": {}}}],
                    "fake-token",
                )

    result = asyncio.run(_run())
    assert result.text == "OK"
    assert captured["url"] == "https://api.z.ai/api/anthropic/v1/messages"
    assert captured["headers"]["x-api-key"] == "fake-token"
    assert result.tool_calls[0].name == "demo_tool"


# Task 4 tests
def test_zai_in_auth_store_supported():
    from forven.auth.store import _SUPPORTED_AUTH_PROVIDERS
    assert "zai" in _SUPPORTED_AUTH_PROVIDERS


# Task 5 tests
def test_zai_in_api_core_supported():
    from forven.api_core import _SUPPORTED_AUTH_PROVIDERS
    assert "zai" in _SUPPORTED_AUTH_PROVIDERS


def test_zai_env_var_registered():
    from forven.api_core import _AUTH_PROVIDER_ENV_VARS
    assert "zai" in _AUTH_PROVIDER_ENV_VARS
    assert _AUTH_PROVIDER_ENV_VARS["zai"] == "ZAI_API_KEY"


def test_zai_display_name():
    from forven.api_core import _MODEL_PROVIDER_DISPLAY_NAMES
    assert _MODEL_PROVIDER_DISPLAY_NAMES["zai"] == "Z.AI"


def test_zai_catalog_entries():
    from forven.api_core import _AGENT_MODEL_CATALOG
    zai_models = [e for e in _AGENT_MODEL_CATALOG if e["provider"] == "zai"]
    model_ids = [e["model_id"] for e in zai_models]
    assert "glm-5.1" in model_ids
    assert "glm-5" in model_ids
    assert "glm-4.7" in model_ids
    assert len(zai_models) >= 10


# Task 6 tests
def test_zai_discovery_endpoints_registered():
    from forven.api_core import _MODEL_DISCOVERY_ALT_ENDPOINTS
    assert "zai" in _MODEL_DISCOVERY_ALT_ENDPOINTS
    endpoints = _MODEL_DISCOVERY_ALT_ENDPOINTS["zai"]
    assert any("z.ai" in ep or "bigmodel.cn" in ep for ep in endpoints)


def test_zai_discovery_headers_registered():
    from forven.api_core import _MODEL_DISCOVERY_HEADERS
    assert "zai" in _MODEL_DISCOVERY_HEADERS
    assert "Authorization" in _MODEL_DISCOVERY_HEADERS["zai"]


def test_zai_model_should_belong():
    from forven.api_core import _discovery_model_should_belong
    assert _discovery_model_should_belong("zai", "glm-5.1") is True
    assert _discovery_model_should_belong("zai", "glm-4.7-flash") is True
    assert _discovery_model_should_belong("zai", "gpt-5.2") is False


def test_zai_oauth_not_supported():
    from forven.api_core import _provider_supports_oauth
    assert _provider_supports_oauth("zai") is False


def test_zai_requires_token():
    from forven.api_core import _provider_requires_token
    assert _provider_requires_token("zai") is True


# Task 7 tests
def test_zai_endpoint_detection_success():
    from forven.api_core import _detect_zai_endpoint
    from unittest.mock import patch, MagicMock

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "pong"}}]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("forven.api_core.httpx.Client") as MockClient:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        MockClient.return_value = mock_client

        result = _detect_zai_endpoint("fake-token")

    assert result["ok"] is True
    assert result["base_url"]
    assert result["endpoint_id"]


def test_zai_endpoint_detection_all_fail():
    from forven.api_core import _detect_zai_endpoint
    from unittest.mock import patch, MagicMock

    with patch("forven.api_core.httpx.Client") as MockClient:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = Exception("connection refused")
        MockClient.return_value = mock_client

        result = _detect_zai_endpoint("fake-token")

    assert result["ok"] is False
    assert "error" in result


# Task 8 tests
def test_build_auth_provider_payload_zai():
    from forven.api_core import _build_auth_provider_payload
    from unittest.mock import patch

    with patch("forven.api_core.get_profile", return_value=None):
        payload = _build_auth_provider_payload("zai")

    assert payload["provider"] == "zai"
    assert payload["supports_oauth"] is False
    assert payload["requires_token"] is True
    assert payload["configured"] is False


# Task 11 – integration round-trip tests
def test_zai_full_provider_round_trip():
    """Verify zai appears in auth providers, model catalog, and model policy."""
    from forven.api_core import (
        _get_auth_providers_compat,
        _get_model_policy_compat,
        _AGENT_MODEL_CATALOG,
    )

    with patch("forven.api_core.get_profile", return_value=None), \
         patch("forven.model_routing.kv_get", side_effect=_mock_kv_get):
        # Auth providers include zai
        auth = _get_auth_providers_compat()
        provider_ids = [p["provider"] for p in auth["providers"]]
        assert "zai" in provider_ids

        # Model catalog includes GLM models
        zai_catalog = [e for e in _AGENT_MODEL_CATALOG if e["provider"] == "zai"]
        assert len(zai_catalog) >= 10
        assert any(e["model_id"] == "glm-5.1" for e in zai_catalog)

        # Model policy includes zai defaults
        policy = _get_model_policy_compat()
        assert "zai" in policy["default_models"]
        assert policy["default_models"]["zai"] == "glm-5.1"


def test_model_routing_migrates_legacy_priority_when_anthropic_env_present():
    from forven.model_routing import get_model_routing

    legacy_policy = deepcopy(_DEFAULT_MODEL_ROUTING)
    legacy_policy["provider_priority"] = ["openai", "minimax", "lmstudio", "zai"]
    captured: dict[str, object] = {}

    with patch.dict("os.environ", {"ANTHROPIC_AUTH_TOKEN": "env-zai-token"}, clear=False):
        with patch("forven.model_routing.kv_get", return_value=legacy_policy):
            with patch("forven.model_routing.kv_set", side_effect=lambda key, value: captured.update({"key": key, "value": value})):
                migrated = get_model_routing()

    assert migrated["provider_priority"][0] == "zai"
    assert captured["key"] == "forven:model-routing"
    assert captured["value"]["provider_priority"][0] == "zai"


def test_zai_normalize_provider_and_model_all_aliases():
    """All provider aliases resolve to zai with correct model."""
    from forven.ai import normalize_provider_and_model

    for alias in ["zai", "z.ai", "z-ai", "ZAI", "Z.AI"]:
        provider, model = normalize_provider_and_model(alias, "glm-5.1")
        assert provider == "zai", f"alias {alias!r} resolved to {provider!r}"
        assert model == "glm-5.1"


def test_zai_model_inference_routes_correctly():
    """GLM model IDs infer zai provider regardless of stated provider."""
    from forven.ai import normalize_provider_and_model

    test_cases = [
        ("glm-5.1", "zai"),
        ("glm-4.7-flash", "zai"),
        ("glm-4.5", "zai"),
        ("glm-5-turbo", "zai"),
    ]
    for model_id, expected_provider in test_cases:
        provider, model = normalize_provider_and_model(None, model_id)
        assert provider == expected_provider, f"model {model_id!r} resolved to {provider!r}"
        assert model == model_id
