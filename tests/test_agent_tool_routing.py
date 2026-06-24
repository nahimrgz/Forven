from forven.agents import runner


def test_tool_call_chain_is_self_only_without_explicit_fallbacks(monkeypatch):
    # Fail-closed by default: with no operator-configured per-slot fallbacks the
    # chain is just the requested model — never an auto cross-provider fallback.
    monkeypatch.setattr(runner, "normalize_provider_and_model", lambda provider, model: (provider, model))

    assert runner._resolve_tool_call_chain("minimax", "MiniMax-M2.7") == [
        ("minimax", "MiniMax-M2.7"),
    ]


def test_tool_call_chain_appends_explicit_agent_fallbacks(monkeypatch):
    # The agent's OWN operator-configured fallbacks (Routing tab -> agent:<id>)
    # are appended after the requested model — explicit opt-in only.
    monkeypatch.setattr(runner, "normalize_provider_and_model", lambda provider, model: (provider, model))
    monkeypatch.setattr(
        "forven.model_selection._policy_slot_fallbacks",
        lambda slot: [("minimax", "MiniMax-M2.5"), ("openai", "gpt-5.2")],
    )

    assert runner._resolve_tool_call_chain("minimax", "MiniMax-M2.7", agent_id="dev") == [
        ("minimax", "MiniMax-M2.7"),
        ("minimax", "MiniMax-M2.5"),
        ("openai", "gpt-5.2"),
    ]


def test_tool_call_chain_dedupes_configured_primary(monkeypatch):
    monkeypatch.setattr(runner, "normalize_provider_and_model", lambda provider, model: (provider, model))
    monkeypatch.setattr(
        "forven.model_selection._policy_slot_fallbacks",
        lambda slot: [("minimax", "MiniMax-M2.5"), ("openai", "gpt-5.2")],
    )

    assert runner._resolve_tool_call_chain("minimax", "MiniMax-M2.5", agent_id="dev") == [
        ("minimax", "MiniMax-M2.5"),
        ("openai", "gpt-5.2"),
    ]
