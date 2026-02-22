from fastapi.testclient import TestClient

import backend.app.main as main
import asyncio


client = TestClient(main.app)


def _profiled_config() -> main.AppConfig:
    return main.AppConfig(
        default_profile="cost",
        profiles={
            "cost": main.ProfileConfig(
                agents=[
                    main.AgentConfig(agent="A", provider="openai", model="gpt-4o-mini"),
                    main.AgentConfig(agent="B", provider="anthropic", model="claude-haiku-4-5-20251001"),
                    main.AgentConfig(agent="C", provider="gemini", model="gemini-2.5-flash"),
                ],
                consensus=main.ConsensusConfig(strategy="peer_vote", min_ok_results=2, rounds=1),
                timeout_seconds=20,
            ),
            "performance": main.ProfileConfig(
                agents=[
                    main.AgentConfig(agent="A", provider="openai", model="gpt-4.1-mini"),
                    main.AgentConfig(agent="B", provider="anthropic", model="claude-sonnet-4-20250514"),
                    main.AgentConfig(agent="C", provider="gemini", model="gemini-2.5-flash"),
                ],
                consensus=main.ConsensusConfig(
                    strategy="peer_vote",
                    min_ok_results=2,
                    rounds=2,
                    debate_mode="strict",
                    min_criticisms=2,
                ),
                timeout_seconds=30,
            ),
        },
    )


def _profiled_config_with_router(enabled: bool = True, min_confidence: int = 75) -> main.AppConfig:
    cfg = _profiled_config()
    cfg.request_router = main.RequestRouterConfig(
        enabled=enabled,
        provider="ollama",
        model="qwen2.5:7b-instruct-q4_K_M",
        timeout_seconds=4,
        min_confidence=min_confidence,
    )
    cfg.router_rules = main.RouterRulesConfig(
        default_profile="balance",
        routes=[
            main.RouteRule(
                when_intents_any=["coding", "analysis", "research"],
                when_complexity_any=["high"],
                profile="performance",
            ),
            main.RouteRule(
                when_intents_any=["qa", "creative", "other", "translation"],
                when_complexity_any=["low", "medium"],
                profile="cost",
            ),
        ],
    )
    cfg.default_profile = "cost"
    return cfg


def test_profiles_endpoint_returns_profile_list(monkeypatch) -> None:
    monkeypatch.setattr(main, "load_config", _profiled_config)
    response = client.get("/api/magi/profiles")
    assert response.status_code == 200
    body = response.json()
    assert body["default_profile"] == "cost"
    assert body["profiles"] == ["cost", "performance"]


def test_run_rejects_empty_prompt() -> None:
    response = client.post("/api/magi/run", json={"prompt": "   "})
    assert response.status_code == 400
    assert response.json()["detail"] == "prompt must not be empty"


def test_run_rejects_too_long_prompt() -> None:
    response = client.post("/api/magi/run", json={"prompt": "a" * 4001})
    assert response.status_code == 400
    assert response.json()["detail"] == "prompt must be 4000 characters or fewer"


def test_run_returns_partial_failure_without_failing_request(monkeypatch) -> None:
    monkeypatch.setattr(main, "load_config", _profiled_config)
    monkeypatch.setattr(main, "_save_run_history", lambda *args, **kwargs: None)

    async def fake_runner(agent_config, prompt, timeout_seconds):
        if agent_config.agent == "B":
            return main.AgentResult(
                agent="B",
                provider=agent_config.provider,
                model=agent_config.model,
                text="",
                status="ERROR",
                latency_ms=300,
                error_message="provider request failed",
            )

        return main.AgentResult(
            agent=agent_config.agent,
            provider=agent_config.provider,
            model=agent_config.model,
            text=f"ok-{agent_config.agent}",
            status="OK",
            latency_ms=100,
        )

    monkeypatch.setattr(main, "_run_single_agent", fake_runner)

    response = client.post("/api/magi/run", json={"prompt": "hello", "profile": "performance"})
    assert response.status_code == 200

    body = response.json()
    assert isinstance(body.get("run_id"), str) and body["run_id"]
    assert body["profile"] == "performance"
    assert len(body["results"]) == 3

    by_agent = {item["agent"]: item for item in body["results"]}
    assert by_agent["A"]["status"] == "OK"
    assert by_agent["B"]["status"] == "ERROR"
    assert by_agent["B"]["error_message"] == "provider request failed"
    assert by_agent["C"]["status"] == "OK"
    assert body["consensus"]["status"] == "LOADING"
    assert body["consensus"]["text"] == ""


def test_retry_rejects_empty_prompt() -> None:
    response = client.post("/api/magi/retry", json={"prompt": "   ", "agent": "A"})
    assert response.status_code == 400
    assert response.json()["detail"] == "prompt must not be empty"


def test_retry_rejects_invalid_agent() -> None:
    response = client.post("/api/magi/retry", json={"prompt": "hello", "agent": "D"})
    assert response.status_code == 422


def test_retry_runs_only_target_agent(monkeypatch) -> None:
    monkeypatch.setattr(main, "load_config", _profiled_config)

    async def fake_runner(agent_config, prompt, timeout_seconds):
        return main.AgentResult(
            agent=agent_config.agent,
            provider=agent_config.provider,
            model=agent_config.model,
            text=f"retry-{agent_config.agent}",
            status="OK",
            latency_ms=123,
        )

    monkeypatch.setattr(main, "_run_single_agent", fake_runner)

    response = client.post("/api/magi/retry", json={"prompt": "hello", "agent": "B", "profile": "performance"})
    assert response.status_code == 200

    body = response.json()
    assert isinstance(body.get("run_id"), str) and body["run_id"]
    assert body["profile"] == "performance"
    assert body["result"]["agent"] == "B"
    assert body["result"]["status"] == "OK"
    assert body["result"]["text"] == "retry-B"


def test_consensus_endpoint_recalculates(monkeypatch) -> None:
    monkeypatch.setattr(main, "load_config", _profiled_config)
    async def fake_consensus(config_obj, prompt, results):
        return main.ConsensusResult(
            provider=config_obj.consensus.provider or "magi",
            model=config_obj.consensus.model or "peer_vote_v1",
            text="recalc-consensus",
            status="OK",
            latency_ms=55,
        )

    monkeypatch.setattr(main, "_run_consensus", fake_consensus)

    payload = {
        "prompt": "hello",
        "results": [
            {
                "agent": "A",
                "provider": "openai",
                "model": "gpt-4.1-mini",
                "text": "a",
                "status": "OK",
                "latency_ms": 10,
                "error_message": None,
            },
            {
                "agent": "B",
                "provider": "anthropic",
                "model": "claude-sonnet-4-20250514",
                "text": "b",
                "status": "OK",
                "latency_ms": 12,
                "error_message": None,
            },
            {
                "agent": "C",
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "text": "",
                "status": "ERROR",
                "latency_ms": 13,
                "error_message": "provider request failed",
            },
        ],
    }

    payload["profile"] = "performance"

    response = client.post("/api/magi/consensus", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["profile"] == "performance"
    assert body["consensus"]["status"] == "OK"
    assert body["consensus"]["text"] == "recalc-consensus"


def test_consensus_endpoint_updates_saved_run(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MAGI_DB_PATH", str(tmp_path / "magi-consensus-update.db"))
    main._init_db()
    monkeypatch.setattr(main, "load_config", _profiled_config)

    ok_results = [
        main.AgentResult(agent="A", provider="openai", model="m1", text="a", status="OK", latency_ms=10),
        main.AgentResult(agent="B", provider="anthropic", model="m2", text="b", status="OK", latency_ms=10),
        main.AgentResult(agent="C", provider="gemini", model="m3", text="c", status="OK", latency_ms=10),
    ]
    pending = main.ConsensusResult(provider="magi", model="peer_vote_r1", text="", status="LOADING", latency_ms=0)
    main._save_run_history("run-update", "cost", "hello", ok_results, pending, thread_id="thread-x", turn_index=3)

    async def fake_consensus(config_obj, prompt, results):
        return main.ConsensusResult(
            provider="magi",
            model="peer_vote_r1",
            text="finalized-consensus",
            status="OK",
            latency_ms=44,
        )

    monkeypatch.setattr(main, "_run_consensus", fake_consensus)

    response = client.post(
        "/api/magi/consensus",
        json={
            "run_id": "run-update",
            "prompt": "hello",
            "profile": "cost",
            "results": [item.model_dump() for item in ok_results],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == "run-update"
    assert body["thread_id"] == "thread-x"
    assert body["turn_index"] == 3
    assert body["consensus"]["status"] == "OK"

    item_response = client.get("/api/magi/history/run-update")
    assert item_response.status_code == 200
    item = item_response.json()
    assert item["consensus"]["status"] == "OK"
    assert item["consensus"]["text"] == "finalized-consensus"


def test_history_list_and_get_item(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MAGI_DB_PATH", str(tmp_path / "magi-test.db"))
    main._init_db()
    monkeypatch.setattr(main, "load_config", _profiled_config)

    async def fake_runner(agent_config, prompt, timeout_seconds):
        return main.AgentResult(
            agent=agent_config.agent,
            provider=agent_config.provider,
            model=agent_config.model,
            text=f"ok-{agent_config.agent}",
            status="OK",
            latency_ms=111,
        )

    monkeypatch.setattr(main, "_run_single_agent", fake_runner)

    run_response = client.post("/api/magi/run", json={"prompt": "persist me", "profile": "cost"})
    assert run_response.status_code == 200
    run_id = run_response.json()["run_id"]

    list_response = client.get("/api/magi/history?limit=20&offset=0")
    assert list_response.status_code == 200
    list_body = list_response.json()
    assert list_body["total"] >= 1
    assert len(list_body["items"]) >= 1
    assert list_body["items"][0]["run_id"] == run_id
    assert isinstance(list_body["items"][0]["thread_id"], str) and list_body["items"][0]["thread_id"]
    assert int(list_body["items"][0]["turn_index"]) >= 1
    assert list_body["items"][0]["prompt"] == "persist me"

    item_response = client.get(f"/api/magi/history/{run_id}")
    assert item_response.status_code == 200
    item_body = item_response.json()
    assert item_body["run_id"] == run_id
    assert isinstance(item_body["thread_id"], str) and item_body["thread_id"]
    assert int(item_body["turn_index"]) >= 1
    assert item_body["consensus"]["status"] == "LOADING"
    assert len(item_body["results"]) == 3


def test_run_assigns_thread_id_and_turn_index(monkeypatch) -> None:
    monkeypatch.setattr(main, "load_config", _profiled_config)
    monkeypatch.setattr(main, "_save_run_history", lambda *args, **kwargs: None)

    async def fake_runner(agent_config, prompt, timeout_seconds):
        return main.AgentResult(
            agent=agent_config.agent,
            provider=agent_config.provider,
            model=agent_config.model,
            text=f"ok-{agent_config.agent}",
            status="OK",
            latency_ms=100,
        )

    monkeypatch.setattr(main, "_run_single_agent", fake_runner)

    response = client.post("/api/magi/run", json={"prompt": "hello", "profile": "cost"})
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["thread_id"], str) and body["thread_id"]
    assert body["turn_index"] == 1


def test_run_with_same_thread_injects_thread_context(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MAGI_DB_PATH", str(tmp_path / "magi-thread.db"))
    main._init_db()
    monkeypatch.setattr(main, "load_config", _profiled_config)

    async def fake_effective(prompt: str, fresh_mode: bool) -> str:
        return prompt

    async def fake_history(prompt: str, base_prompt: str, app_config=None) -> str:
        return base_prompt

    captured_prompts: list[str] = []

    async def fake_runner(agent_config, prompt, timeout_seconds):
        captured_prompts.append(prompt)
        return main.AgentResult(
            agent=agent_config.agent,
            provider=agent_config.provider,
            model=agent_config.model,
            text=f"ok-{agent_config.agent}",
            status="OK",
            latency_ms=100,
        )

    monkeypatch.setattr(main, "_build_effective_prompt", fake_effective)
    monkeypatch.setattr(main, "_build_prompt_with_history", fake_history)
    monkeypatch.setattr(main, "_run_single_agent", fake_runner)

    first = client.post("/api/magi/run", json={"prompt": "What is Auth.js?", "profile": "cost"})
    assert first.status_code == 200
    thread_id = first.json()["thread_id"]
    assert first.json()["turn_index"] == 1

    second = client.post(
        "/api/magi/run",
        json={"prompt": "それってどういうこと？", "profile": "cost", "thread_id": thread_id},
    )
    assert second.status_code == 200
    assert second.json()["thread_id"] == thread_id
    assert second.json()["turn_index"] == 2
    assert any("[Thread Conversation Context]" in value for value in captured_prompts[-3:])


def test_history_get_returns_404_for_unknown_run(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MAGI_DB_PATH", str(tmp_path / "magi-empty.db"))
    main._init_db()
    response = client.get("/api/magi/history/not-found")
    assert response.status_code == 404


def test_delete_thread_history_removes_only_target_thread(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MAGI_DB_PATH", str(tmp_path / "magi-delete-thread.db"))
    main._init_db()

    ok_results = [
        main.AgentResult(agent="A", provider="openai", model="m1", text="a", status="OK", latency_ms=10),
        main.AgentResult(agent="B", provider="anthropic", model="m2", text="b", status="OK", latency_ms=10),
        main.AgentResult(agent="C", provider="gemini", model="m3", text="c", status="OK", latency_ms=10),
    ]
    consensus = main.ConsensusResult(provider="magi", model="peer_vote_v1", text="answer", status="OK", latency_ms=5)
    main._save_run_history("run-a1", "cost", "p1", ok_results, consensus, thread_id="thread-a", turn_index=1)
    main._save_run_history("run-a2", "cost", "p2", ok_results, consensus, thread_id="thread-a", turn_index=2)
    main._save_run_history("run-b1", "cost", "p3", ok_results, consensus, thread_id="thread-b", turn_index=1)

    response = client.delete("/api/magi/history/thread/thread-a")
    assert response.status_code == 200
    assert response.json()["deleted_runs"] == 2

    list_response = client.get("/api/magi/history?limit=20&offset=0")
    assert list_response.status_code == 200
    thread_ids = {item["thread_id"] for item in list_response.json()["items"]}
    assert "thread-a" not in thread_ids
    assert "thread-b" in thread_ids


def test_history_list_rejects_invalid_query_params() -> None:
    response = client.get("/api/magi/history?limit=0&offset=-1")
    assert response.status_code == 400


def test_history_persistence_failure_returns_500(monkeypatch) -> None:
    monkeypatch.setattr(main, "load_config", _profiled_config)

    async def fake_runner(agent_config, prompt, timeout_seconds):
        return main.AgentResult(
            agent=agent_config.agent,
            provider=agent_config.provider,
            model=agent_config.model,
            text=f"ok-{agent_config.agent}",
            status="OK",
            latency_ms=100,
        )

    async def fake_consensus(config_obj, prompt, results):
        return main.ConsensusResult(
            provider="magi",
            model="peer_vote_v1",
            text="consensus-ok",
            status="OK",
            latency_ms=10,
        )

    def fail_save(*args, **kwargs):
        raise RuntimeError("db failure")

    monkeypatch.setattr(main, "_run_single_agent", fake_runner)
    monkeypatch.setattr(main, "_run_consensus", fake_consensus)
    monkeypatch.setattr(main, "_save_run_history", fail_save)

    response = client.post("/api/magi/run", json={"prompt": "hello", "profile": "cost"})
    assert response.status_code == 500
    assert "failed to persist history" in response.json()["detail"]


def test_parse_deliberation_turn_strict_requires_criticisms() -> None:
    cfg = main.ConsensusConfig(strategy="peer_vote", debate_mode="strict", min_criticisms=2)
    text = '{"revised_answer":"x","preferred_agent":"A","reason":"ok","confidence":80,"criticisms":["only one"]}'
    turn = main._parse_deliberation_turn(
        text=text,
        agent_config=main.AgentConfig(agent="A", provider="openai", model="gpt-4o-mini"),
        latency_ms=10,
        consensus_config=cfg,
    )
    assert turn.status == "ERROR"
    assert "at least 2 criticisms" in (turn.error_message or "")


def test_parse_deliberation_turn_strict_accepts_valid_criticisms() -> None:
    cfg = main.ConsensusConfig(strategy="peer_vote", debate_mode="strict", min_criticisms=2)
    text = (
        '{"revised_answer":"x","preferred_agent":"B","reason":"ok","confidence":81,'
        '"criticisms":["A misses assumptions","C lacks evidence"]}'
    )
    turn = main._parse_deliberation_turn(
        text=text,
        agent_config=main.AgentConfig(agent="B", provider="anthropic", model="claude"),
        latency_ms=10,
        consensus_config=cfg,
    )
    assert turn.status == "OK"
    assert turn.preferred_agent == "B"
    assert len(turn.criticisms) == 2


def test_run_with_fresh_mode_uses_effective_prompt(monkeypatch) -> None:
    monkeypatch.setattr(main, "load_config", _profiled_config)
    monkeypatch.setattr(main, "_save_run_history", lambda *args, **kwargs: None)

    async def fake_effective(prompt: str, fresh_mode: bool) -> str:
        assert fresh_mode is True
        return f"fresh::{prompt}"

    async def fake_runner(agent_config, prompt, timeout_seconds):
        assert prompt.startswith("fresh::")
        return main.AgentResult(
            agent=agent_config.agent,
            provider=agent_config.provider,
            model=agent_config.model,
            text=f"ok-{agent_config.agent}",
            status="OK",
            latency_ms=100,
        )

    monkeypatch.setattr(main, "_build_effective_prompt", fake_effective)
    monkeypatch.setattr(main, "_run_single_agent", fake_runner)

    response = client.post("/api/magi/run", json={"prompt": "hello", "profile": "cost", "fresh_mode": True})
    assert response.status_code == 200
    assert response.json()["consensus"]["status"] == "LOADING"


def test_run_uses_request_router_when_profile_not_provided(monkeypatch) -> None:
    monkeypatch.setattr(main, "load_config", lambda: _profiled_config_with_router(enabled=True))
    monkeypatch.setattr(main, "_save_run_history", lambda *args, **kwargs: None)

    async def fake_route_profile(config, prompt):
        return "performance"

    async def fake_runner(agent_config, prompt, timeout_seconds):
        return main.AgentResult(
            agent=agent_config.agent,
            provider=agent_config.provider,
            model=agent_config.model,
            text=f"ok-{agent_config.agent}",
            status="OK",
            latency_ms=100,
        )

    monkeypatch.setattr(main, "_route_profile", fake_route_profile)
    monkeypatch.setattr(main, "_run_single_agent", fake_runner)

    response = client.post("/api/magi/run", json={"prompt": "deep architecture review"})
    assert response.status_code == 200
    body = response.json()
    assert body["profile"] == "performance"


def test_run_skips_request_router_when_profile_is_explicit(monkeypatch) -> None:
    monkeypatch.setattr(main, "load_config", lambda: _profiled_config_with_router(enabled=True))
    monkeypatch.setattr(main, "_save_run_history", lambda *args, **kwargs: None)

    async def fail_route_profile(config, prompt):
        raise AssertionError("router should not be called when profile is explicit")

    async def fake_runner(agent_config, prompt, timeout_seconds):
        return main.AgentResult(
            agent=agent_config.agent,
            provider=agent_config.provider,
            model=agent_config.model,
            text=f"ok-{agent_config.agent}",
            status="OK",
            latency_ms=100,
        )

    monkeypatch.setattr(main, "_route_profile", fail_route_profile)
    monkeypatch.setattr(main, "_run_single_agent", fake_runner)

    response = client.post("/api/magi/run", json={"prompt": "hello", "profile": "cost"})
    assert response.status_code == 200
    assert response.json()["profile"] == "cost"


def test_route_profile_falls_back_when_confidence_is_low(monkeypatch) -> None:
    config = _profiled_config_with_router(enabled=True, min_confidence=80)

    async def fake_call(full_model: str, prompt: str, timeout_seconds: int):
        return '{"intent":"coding","complexity":"high","profile":"performance","confidence":42,"reason":"uncertain"}', 12

    monkeypatch.setattr(main, "_call_model_text", fake_call)
    selected = asyncio.run(main._route_profile(config, "refactor this service"))
    assert selected == "cost"


def test_run_auto_enables_fresh_mode_for_time_sensitive_prompt(monkeypatch) -> None:
    monkeypatch.setattr(main, "load_config", _profiled_config)
    monkeypatch.setattr(main, "_save_run_history", lambda *args, **kwargs: None)
    monkeypatch.setenv("FRESH_AUTO_MODE", "1")

    async def fake_effective(prompt: str, fresh_mode: bool) -> str:
        assert fresh_mode is True
        return f"fresh::{prompt}"

    async def fake_runner(agent_config, prompt, timeout_seconds):
        assert prompt.startswith("fresh::")
        return main.AgentResult(
            agent=agent_config.agent,
            provider=agent_config.provider,
            model=agent_config.model,
            text=f"ok-{agent_config.agent}",
            status="OK",
            latency_ms=100,
        )

    monkeypatch.setattr(main, "_build_effective_prompt", fake_effective)
    monkeypatch.setattr(main, "_run_single_agent", fake_runner)

    response = client.post("/api/magi/run", json={"prompt": "latest gdp numbers", "profile": "cost", "fresh_mode": False})
    assert response.status_code == 200


def test_resolve_fresh_mode_respects_auto_toggle(monkeypatch) -> None:
    monkeypatch.setenv("FRESH_AUTO_MODE", "0")
    assert main._resolve_fresh_mode("latest ai news", False) is False
    monkeypatch.setenv("FRESH_AUTO_MODE", "1")
    assert main._resolve_fresh_mode("latest ai news", False) is True


def test_build_effective_prompt_falls_back_without_tavily_key(monkeypatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    prompt = "latest policy update"
    effective = asyncio.run(main._build_effective_prompt(prompt, True))
    assert effective == prompt


def test_find_similar_history_returns_ranked_matches(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MAGI_DB_PATH", str(tmp_path / "magi-similar.db"))
    monkeypatch.setenv("HISTORY_SIMILARITY_THRESHOLD", "0.20")
    main._init_db()

    ok_results = [
        main.AgentResult(agent="A", provider="openai", model="m1", text="a", status="OK", latency_ms=10),
        main.AgentResult(agent="B", provider="anthropic", model="m2", text="b", status="OK", latency_ms=10),
        main.AgentResult(agent="C", provider="gemini", model="m3", text="c", status="OK", latency_ms=10),
    ]
    consensus = main.ConsensusResult(provider="magi", model="peer_vote_v1", text="previous answer", status="OK", latency_ms=5)

    main._save_run_history("run-1", "cost", "how to sort a python list", ok_results, consensus)
    main._save_run_history("run-2", "cost", "python list sorting ascending order", ok_results, consensus)
    main._save_run_history("run-3", "cost", "tokyo weather today", ok_results, consensus)

    matches = main._find_similar_history("python list sort", 2)
    assert len(matches) == 2
    matched_run_ids = {str(item["run_id"]) for item in matches}
    assert "run-1" in matched_run_ids
    assert "run-2" in matched_run_ids


def test_build_prompt_with_history_falls_back_on_lookup_error(monkeypatch) -> None:
    def fail_candidates():
        raise RuntimeError("lookup failure")

    monkeypatch.setattr(main, "_load_history_candidates", fail_candidates)
    base = "hello"
    assert asyncio.run(main._build_prompt_with_history("hello", base)) == base


def test_build_prompt_with_history_uses_embedding_strategy(monkeypatch) -> None:
    app_cfg = main.AppConfig(
        history_context=main.HistoryContextConfig(
            strategy="embedding",
            provider="openai",
            model="text-embedding-3-small",
            timeout_seconds=10,
            batch_size=8,
        )
    )
    row = {
        "run_id": "r1",
        "prompt": "python list sort",
        "created_at": "2026-01-01T00:00:00+00:00",
        "consensus_text": "old answer",
        "validity_state": "active",
        "superseded_by": None,
        "superseded_at": None,
    }

    monkeypatch.setattr(main, "_load_history_candidates", lambda: [row])

    async def fake_embedding_rank(prompt, rows, threshold, limit, provider, model, timeout_seconds, batch_size, app_config):
        assert provider == "openai"
        assert model == "text-embedding-3-small"
        return [
            {
                "run_id": "r1",
                "prompt": "python list sort",
                "created_at": "2026-01-01T00:00:00+00:00",
                "consensus_text": "old answer",
                "similarity": 0.91,
                "raw_similarity": 0.91,
                "validity_state": "active",
                "superseded_by": "",
            }
        ]

    monkeypatch.setattr(main, "_rank_history_matches_embedding", fake_embedding_rank)
    enriched = asyncio.run(main._build_prompt_with_history("python sort", "python sort", app_cfg))
    assert "[Similar Past Runs]" in enriched


def test_run_with_history_context_uses_enriched_prompt(monkeypatch) -> None:
    monkeypatch.setattr(main, "load_config", _profiled_config)
    monkeypatch.setattr(main, "_save_run_history", lambda *args, **kwargs: None)

    async def fake_effective(prompt: str, fresh_mode: bool) -> str:
        return prompt

    async def fake_history(original_prompt: str, base_prompt: str, app_config=None) -> str:
        return f"{base_prompt}\n\n[Similar Past Runs]\n[H1] sample"

    async def fake_runner(agent_config, prompt, timeout_seconds):
        assert "[Similar Past Runs]" in prompt
        return main.AgentResult(
            agent=agent_config.agent,
            provider=agent_config.provider,
            model=agent_config.model,
            text=f"ok-{agent_config.agent}",
            status="OK",
            latency_ms=100,
        )

    monkeypatch.setattr(main, "_build_effective_prompt", fake_effective)
    monkeypatch.setattr(main, "_build_prompt_with_history", fake_history)
    monkeypatch.setattr(main, "_run_single_agent", fake_runner)

    response = client.post("/api/magi/run", json={"prompt": "hello", "profile": "cost"})
    assert response.status_code == 200
    assert response.json()["consensus"]["status"] == "LOADING"


def test_save_run_history_marks_legacy_as_superseded(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MAGI_DB_PATH", str(tmp_path / "magi-supersede.db"))
    main._init_db()

    cfg = main.AppConfig(
        history_context=main.HistoryContextConfig(
            deprecations=[
                main.HistoryDeprecationRule(
                    id="authjs-migration",
                    legacy_terms=["nextauth", "next-auth"],
                    current_terms=["auth.js", "authjs"],
                )
            ]
        )
    )

    ok_results = [
        main.AgentResult(agent="A", provider="openai", model="m1", text="a", status="OK", latency_ms=10),
        main.AgentResult(agent="B", provider="anthropic", model="m2", text="b", status="OK", latency_ms=10),
        main.AgentResult(agent="C", provider="gemini", model="m3", text="c", status="OK", latency_ms=10),
    ]
    consensus = main.ConsensusResult(provider="magi", model="peer_vote_v1", text="answer", status="OK", latency_ms=5)

    main._save_run_history("old", "cost", "How to use NextAuth in app router?", ok_results, consensus, cfg)
    main._save_run_history("new", "cost", "How to migrate to Auth.js?", ok_results, consensus, cfg)

    with main._get_db_connection() as conn:
        row = conn.execute("SELECT validity_state, superseded_by FROM runs WHERE run_id='old'").fetchone()
    assert row is not None
    assert row["validity_state"] == "superseded"
    assert row["superseded_by"] == "authjs-migration"


def test_rank_history_matches_lexical_downweights_superseded() -> None:
    rows = [
        {
            "run_id": "active",
            "prompt": "python list sort",
            "created_at": "2026-01-01T00:00:00+00:00",
            "consensus_text": "x",
            "validity_state": "active",
            "superseded_by": None,
            "superseded_at": None,
        },
        {
            "run_id": "superseded",
            "prompt": "python list sort",
            "created_at": "2026-01-01T00:00:00+00:00",
            "consensus_text": "x",
            "validity_state": "superseded",
            "superseded_by": "migration",
            "superseded_at": "2026-01-02T00:00:00+00:00",
        },
    ]
    ranked = main._rank_history_matches_lexical("python list sort", rows, 0.0, 2)
    assert ranked[0]["run_id"] == "active"
    assert ranked[1]["run_id"] == "superseded"


def test_fresh_query_attempts_prioritizes_general_and_expansion() -> None:
    attempts = main._fresh_query_attempts("最新のff14零式4層を解説して", "general")
    assert attempts[0] == ("最新のff14零式4層を解説して", "general")
    assert ("最新のff14零式4層を解説して", "news") in attempts
    assert ("ffxiv savage floor 4 guide latest", "general") in attempts


def test_fresh_query_attempts_news_primary_keeps_general_fallback() -> None:
    attempts = main._fresh_query_attempts("latest ai release", "news")
    assert attempts[0] == ("latest ai release", "news")
    assert ("latest ai release", "general") in attempts


def test_run_single_agent_retries_once_for_openai_timeout(monkeypatch) -> None:
    calls = {"count": 0}

    async def fake_call_model_text(full_model: str, prompt: str, timeout_seconds: int):
        calls["count"] += 1
        if calls["count"] == 1:
            raise asyncio.TimeoutError()
        return "ok-after-retry", 123

    monkeypatch.setattr(main, "_call_model_text", fake_call_model_text)

    result = asyncio.run(
        main._run_single_agent(
            agent_config=main.AgentConfig(agent="A", provider="openai", model="gpt-5.1"),
            prompt="hello",
            timeout_seconds=1,
        )
    )

    assert calls["count"] == 2
    assert result.status == "OK"
    assert result.text == "ok-after-retry"


def test_run_single_agent_does_not_retry_non_openai_timeout(monkeypatch) -> None:
    calls = {"count": 0}

    async def fake_call_model_text(full_model: str, prompt: str, timeout_seconds: int):
        calls["count"] += 1
        raise asyncio.TimeoutError()

    monkeypatch.setattr(main, "_call_model_text", fake_call_model_text)

    result = asyncio.run(
        main._run_single_agent(
            agent_config=main.AgentConfig(agent="B", provider="anthropic", model="claude-sonnet-4-20250514"),
            prompt="hello",
            timeout_seconds=1,
        )
    )

    assert calls["count"] == 1
    assert result.status == "ERROR"
    assert result.error_message == "timeout"
