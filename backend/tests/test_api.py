from fastapi.testclient import TestClient

import backend.app.main as main
import asyncio
import uuid
from datetime import datetime, timedelta, timezone


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
    cfg.profiles["local_only"] = main.ProfileConfig(
        agents=[
            main.AgentConfig(agent="A", provider="ollama", model="qwen2.5:7b-instruct-q4_K_M"),
        ],
        consensus=main.ConsensusConfig(strategy="peer_vote", min_ok_results=1, rounds=1),
        timeout_seconds=20,
    )
    cfg.request_router = main.RequestRouterConfig(
        enabled=enabled,
        provider="ollama",
        model="qwen2.5:7b-instruct-q4_K_M",
        timeout_seconds=4,
        min_confidence=min_confidence,
    )
    cfg.router_rules = main.RouterRulesConfig(
        default_profile="cost",
        routes=[
            main.RouteRule(
                when_intents_any=["translation", "rewrite", "summarize_short"],
                when_complexity_any=["low"],
                when_safety_any=["low"],
                when_execution_tiers_any=["local"],
                profile="local_only",
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


def test_chat_endpoint_returns_unified_reply(monkeypatch) -> None:
    monkeypatch.setattr(main, "load_config", _profiled_config)
    monkeypatch.setattr(main, "_save_run_history", lambda *args, **kwargs: None)

    async def fake_runner(agent_config, prompt, timeout_seconds):
        return main.AgentResult(
            agent=agent_config.agent,
            provider=agent_config.provider,
            model=agent_config.model,
            text=f"answer-{agent_config.agent}",
            status="OK",
            latency_ms=50,
        )

    async def fake_consensus(config_obj, prompt, results):
        return main.ConsensusResult(
            provider="magi",
            model="peer_vote_r1",
            text=(
                "Consensus winner: Agent B (2/3 votes).\n\n"
                "Final answer:\nこれは統合した最終回答です。\n\n"
                "Vote details:\n- Agent A voted B"
            ),
            status="OK",
            latency_ms=120,
        )

    async def force_insufficient(prompt, draft, judge_agent, timeout_seconds):
        return False, "test_force_escalate"

    monkeypatch.setattr(main, "_run_single_agent", fake_runner)
    monkeypatch.setattr(main, "_run_consensus", fake_consensus)
    monkeypatch.setattr(main, "_is_local_draft_sufficient", force_insufficient)

    response = client.post(
        "/api/magi/chat",
        json={"prompt": "自然な会話で答えて", "profile": "performance"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["profile"] == "performance"
    assert body["consensus"]["status"] == "OK"
    assert body["reply"] == "これは統合した最終回答です。"
    assert len(body["results"]) == 3


def test_chat_endpoint_uses_local_only_result_when_consensus_is_immediate(monkeypatch) -> None:
    monkeypatch.setattr(main, "load_config", _profiled_config_with_router)
    monkeypatch.setattr(main, "_save_run_history", lambda *args, **kwargs: None)

    async def fake_runner(agent_config, prompt, timeout_seconds):
        return main.AgentResult(
            agent=agent_config.agent,
            provider=agent_config.provider,
            model=agent_config.model,
            text="承知しました。こちらが回答です。",
            status="OK",
            latency_ms=20,
        )

    async def force_sufficient(prompt, draft, judge_agent, timeout_seconds):
        return True, "test_force_local_accept"

    async def fail_consensus(*args, **kwargs):
        raise AssertionError("consensus should not run when local draft is sufficient")

    monkeypatch.setattr(main, "_run_single_agent", fake_runner)
    monkeypatch.setattr(main, "_is_local_draft_sufficient", force_sufficient)
    monkeypatch.setattr(main, "_run_consensus", fail_consensus)

    response = client.post(
        "/api/magi/chat",
        json={"prompt": "この文章を丁寧にして", "profile": "local_only"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["profile"] == "local_only"
    assert body["consensus"]["status"] == "OK"
    assert body["reply"] == "承知しました。こちらが回答です。"


def test_chat_first_turn_skips_history_context(monkeypatch) -> None:
    monkeypatch.setattr(main, "load_config", _profiled_config_with_router)
    monkeypatch.setattr(main, "_save_run_history", lambda *args, **kwargs: None)

    async def fail_history(*args, **kwargs):
        raise AssertionError("history context should be skipped on first chat turn")

    async def fake_runner(agent_config, prompt, timeout_seconds):
        return main.AgentResult(
            agent=agent_config.agent,
            provider=agent_config.provider,
            model=agent_config.model,
            text="こんにちは。",
            status="OK",
            latency_ms=15,
        )

    async def force_sufficient(prompt, draft, judge_agent, timeout_seconds):
        return True, "test_force_local_accept"

    monkeypatch.setattr(main, "_build_prompt_with_history", fail_history)
    monkeypatch.setattr(main, "_run_single_agent", fake_runner)
    monkeypatch.setattr(main, "_is_local_draft_sufficient", force_sufficient)

    response = client.post("/api/magi/chat", json={"prompt": "はじめまして"})
    assert response.status_code == 200


def test_chat_smalltalk_skips_history_context_even_with_existing_turns(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MAGI_DB_PATH", str(tmp_path / "magi-chat-smalltalk.db"))
    main._init_db()
    monkeypatch.setattr(main, "load_config", _profiled_config_with_router)
    monkeypatch.setattr(main, "_save_run_history", lambda *args, **kwargs: None)

    ok_results = [
        main.AgentResult(agent="A", provider="openai", model="m1", text="old", status="OK", latency_ms=10),
    ]
    consensus = main.ConsensusResult(provider="magi", model="peer_vote_r1", text="old", status="OK", latency_ms=10)
    thread_id = str(uuid.uuid4())
    main._save_run_history("run-old", "local_only", "old prompt", ok_results, consensus, thread_id=thread_id, turn_index=1)

    async def fail_history(*args, **kwargs):
        raise AssertionError("history context should be skipped for smalltalk")

    async def fake_runner(agent_config, prompt, timeout_seconds):
        return main.AgentResult(
            agent=agent_config.agent,
            provider=agent_config.provider,
            model=agent_config.model,
            text="やあ。",
            status="OK",
            latency_ms=15,
        )

    async def force_sufficient(prompt, draft, judge_agent, timeout_seconds):
        return True, "test_force_local_accept"

    monkeypatch.setattr(main, "_build_prompt_with_history", fail_history)
    monkeypatch.setattr(main, "_run_single_agent", fake_runner)
    monkeypatch.setattr(main, "_is_local_draft_sufficient", force_sufficient)

    response = client.post("/api/magi/chat", json={"prompt": "やあ", "thread_id": thread_id})
    assert response.status_code == 200


def test_chat_skips_history_context_when_prompt_contains_url(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MAGI_DB_PATH", str(tmp_path / "magi-chat-url-anchor.db"))
    main._init_db()
    monkeypatch.setattr(main, "load_config", _profiled_config_with_router)
    monkeypatch.setattr(main, "_save_run_history", lambda *args, **kwargs: None)

    ok_results = [
        main.AgentResult(agent="A", provider="openai", model="m1", text="old", status="OK", latency_ms=10),
    ]
    consensus = main.ConsensusResult(provider="magi", model="peer_vote_r1", text="old", status="OK", latency_ms=10)
    thread_id = str(uuid.uuid4())
    main._save_run_history("run-old", "cost", "old prompt", ok_results, consensus, thread_id=thread_id, turn_index=1)

    async def fail_history(*args, **kwargs):
        raise AssertionError("history context should be skipped for URL-anchored chat request")

    async def fake_runner(agent_config, prompt, timeout_seconds):
        return main.AgentResult(
            agent=agent_config.agent,
            provider=agent_config.provider,
            model=agent_config.model,
            text="URLを確認しました。",
            status="OK",
            latency_ms=15,
        )

    async def force_sufficient(prompt, draft, judge_agent, timeout_seconds):
        return True, "test_force_local_accept"

    monkeypatch.setattr(main, "_build_prompt_with_history", fail_history)
    monkeypatch.setattr(main, "_run_single_agent", fake_runner)
    monkeypatch.setattr(main, "_is_local_draft_sufficient", force_sufficient)

    response = client.post(
        "/api/magi/chat",
        json={"prompt": "このURL見て https://example.com", "thread_id": thread_id},
    )
    assert response.status_code == 200


def test_chat_returns_partial_failure_without_failing_request(monkeypatch) -> None:
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
                latency_ms=120,
                error_message="provider request failed",
            )
        return main.AgentResult(
            agent=agent_config.agent,
            provider=agent_config.provider,
            model=agent_config.model,
            text=f"ok-{agent_config.agent}",
            status="OK",
            latency_ms=80,
        )

    async def fake_consensus(config_obj, prompt, results):
        return main.ConsensusResult(
            provider="magi",
            model="peer_vote_r1",
            text="Final answer: fallback",
            status="OK",
            latency_ms=50,
        )

    async def force_insufficient(prompt, draft, judge_agent, timeout_seconds):
        return False, "test_force_escalate"

    monkeypatch.setattr(main, "_run_single_agent", fake_runner)
    monkeypatch.setattr(main, "_run_consensus", fake_consensus)
    monkeypatch.setattr(main, "_is_local_draft_sufficient", force_insufficient)

    response = client.post("/api/magi/chat", json={"prompt": "説明して", "profile": "performance"})
    assert response.status_code == 200
    body = response.json()
    by_agent = {item["agent"]: item for item in body["results"]}
    assert by_agent["A"]["status"] == "OK"
    assert by_agent["B"]["status"] == "ERROR"
    assert by_agent["B"]["error_message"] == "provider request failed"
    assert by_agent["C"]["status"] == "OK"
    assert body["consensus"]["status"] == "OK"


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


def test_run_ignores_non_uuid_thread_id_and_generates_uuid(monkeypatch) -> None:
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

    response = client.post("/api/magi/run", json={"prompt": "hello", "profile": "cost", "thread_id": "thread-custom"})
    assert response.status_code == 200
    body = response.json()
    assert body["thread_id"] != "thread-custom"
    assert str(uuid.UUID(body["thread_id"])) == body["thread_id"]


def test_run_with_same_thread_injects_thread_context(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MAGI_DB_PATH", str(tmp_path / "magi-thread.db"))
    main._init_db()
    monkeypatch.setattr(main, "load_config", _profiled_config)

    async def fake_effective(prompt: str, fresh_mode: bool, source_urls: list[str] | None = None) -> str:
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

    async def fake_effective(prompt: str, fresh_mode: bool, source_urls: list[str] | None = None) -> str:
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


def test_run_passes_source_urls_to_effective_prompt(monkeypatch) -> None:
    monkeypatch.setattr(main, "load_config", _profiled_config)
    monkeypatch.setattr(main, "_save_run_history", lambda *args, **kwargs: None)

    async def fake_effective(prompt: str, fresh_mode: bool, source_urls: list[str] | None = None) -> str:
        assert source_urls == ["https://example.com/a", "https://example.com/b"]
        return f"src::{prompt}"

    async def fake_runner(agent_config, prompt, timeout_seconds):
        assert prompt.startswith("src::")
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

    response = client.post(
        "/api/magi/run",
        json={
            "prompt": "hello",
            "profile": "cost",
            "source_urls": ["https://example.com/a", "https://example.com/b"],
        },
    )
    assert response.status_code == 200


def test_run_rejects_invalid_source_url(monkeypatch) -> None:
    monkeypatch.setattr(main, "load_config", _profiled_config)
    monkeypatch.setattr(main, "_save_run_history", lambda *args, **kwargs: None)

    response = client.post(
        "/api/magi/run",
        json={"prompt": "hello", "profile": "cost", "source_urls": ["file:///etc/passwd"]},
    )
    assert response.status_code == 400
    assert "invalid source url" in response.json()["detail"]


def test_build_effective_prompt_auto_extracts_url_from_prompt(monkeypatch) -> None:
    captured: dict[str, list[str]] = {}

    async def fake_fetch_direct_sources(urls: list[str]):
        captured["urls"] = urls
        return [
            main.DirectSourceFetchResult(
                url=urls[0],
                status="OK",
                snippet="source snippet",
                content_type="text/html",
            )
        ]

    monkeypatch.setattr(main, "_fetch_direct_sources", fake_fetch_direct_sources)

    prompt = "https://example.com/guide を読んで要点を説明して"
    effective = asyncio.run(main._build_effective_prompt(prompt, False))

    assert captured["urls"] == ["https://example.com/guide"]
    assert "[Direct URL Evidence]" in effective
    assert "url=https://example.com/guide" in effective


def test_run_uses_request_router_when_profile_not_provided(monkeypatch) -> None:
    monkeypatch.setattr(main, "load_config", lambda: _profiled_config_with_router(enabled=True))
    monkeypatch.setattr(main, "_save_run_history", lambda *args, **kwargs: None)

    async def fake_route_profile(config, prompt):
        return (
            "local_only",
            {"intent": "rewrite", "complexity": "low", "safety": "low", "execution_tier": "local", "language": "ja", "prompt_length": len(prompt)},
            {"chosen_profile": "local_only", "candidates": {"local_only": {"base_score": 1.0, "policy_weight": 0.0, "final_score": 1.0}}, "policy_key": "intent=rewrite|complexity=low|lang=ja"},
        )

    async def fake_runner(agent_config, prompt, timeout_seconds):
        return main.AgentResult(
            agent=agent_config.agent,
            provider=agent_config.provider,
            model=agent_config.model,
            text=f"ok-{agent_config.agent}",
            status="OK",
            latency_ms=100,
        )

    monkeypatch.setattr(main, "_route_profile_with_trace", fake_route_profile)
    monkeypatch.setattr(main, "_run_single_agent", fake_runner)

    response = client.post("/api/magi/run", json={"prompt": "deep architecture review"})
    assert response.status_code == 200
    body = response.json()
    assert body["profile"] == "local_only"
    assert len(body["results"]) == 1


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

    monkeypatch.setattr(main, "_route_profile_with_trace", fail_route_profile)
    monkeypatch.setattr(main, "_run_single_agent", fake_runner)

    response = client.post("/api/magi/run", json={"prompt": "hello", "profile": "cost"})
    assert response.status_code == 200
    assert response.json()["profile"] == "cost"


def test_route_profile_falls_back_when_confidence_is_low(monkeypatch) -> None:
    config = _profiled_config_with_router(enabled=True, min_confidence=80)

    async def fake_call(full_model: str, prompt: str, timeout_seconds: int, max_tokens: int | None = None):
        return '{"intent":"coding","complexity":"high","profile":"performance","confidence":42,"reason":"uncertain"}', 12, {}

    monkeypatch.setattr(main, "_call_model_text", fake_call)
    selected = asyncio.run(main._route_profile(config, "refactor this service"))
    assert selected == "cost"


def test_route_profile_uses_local_only_for_low_risk_translation(monkeypatch) -> None:
    config = _profiled_config_with_router(enabled=True, min_confidence=75)

    async def fake_call(full_model: str, prompt: str, timeout_seconds: int, max_tokens: int | None = None):
        return (
            '{"intent":"translation","complexity":"low","safety":"low","execution_tier":"local",'
            '"profile":"cost","confidence":92,"reason":"safe text rewrite"}',
            10,
            {},
        )

    monkeypatch.setattr(main, "_call_model_text", fake_call)
    selected = asyncio.run(main._route_profile(config, "日本語の敬語で言い換えて"))
    assert selected == "local_only"


def test_route_profile_fast_path_skips_router_call_for_simple_rewrite(monkeypatch) -> None:
    config = _profiled_config_with_router(enabled=True, min_confidence=75)

    async def fail_call(*args, **kwargs):
        raise AssertionError("router llm call should be skipped by fast_path")

    monkeypatch.setattr(main, "_call_model_text", fail_call)
    selected = asyncio.run(main._route_profile(config, "日本語の敬語で、丁寧に言い換えて"))
    assert selected == "local_only"


def test_route_profile_fast_path_uses_local_only_for_smalltalk(monkeypatch) -> None:
    config = _profiled_config_with_router(enabled=True, min_confidence=75)

    async def fail_call(*args, **kwargs):
        raise AssertionError("router llm call should be skipped by fast_path")

    monkeypatch.setattr(main, "_call_model_text", fail_call)
    selected = asyncio.run(main._route_profile(config, "やあ"))
    assert selected == "local_only"


def test_route_profile_does_not_use_local_only_when_execution_tier_is_cloud(monkeypatch) -> None:
    config = _profiled_config_with_router(enabled=True, min_confidence=75)

    async def fake_call(full_model: str, prompt: str, timeout_seconds: int, max_tokens: int | None = None):
        return (
            '{"intent":"translation","complexity":"low","safety":"low","execution_tier":"cloud",'
            '"profile":"local_only","confidence":95,"reason":"classified as cloud"}',
            10,
            {},
        )

    monkeypatch.setattr(main, "_call_model_text", fake_call)
    selected = asyncio.run(main._route_profile(config, "最新の米国金利動向を踏まえて、投資戦略を提案して"))
    assert selected == "cost"


def test_run_local_only_skips_history_context(monkeypatch) -> None:
    monkeypatch.setattr(main, "load_config", lambda: _profiled_config_with_router(enabled=False))
    monkeypatch.setattr(main, "_save_run_history", lambda *args, **kwargs: None)

    async def fail_history(*args, **kwargs):
        raise AssertionError("history context should be skipped for local_only")

    def fail_thread_context(*args, **kwargs):
        raise AssertionError("thread context should be skipped for local_only")

    async def fake_effective(prompt: str, fresh_mode: bool, source_urls: list[str] | None = None) -> str:
        return prompt

    async def fake_runner(agent_config, prompt, timeout_seconds):
        return main.AgentResult(
            agent=agent_config.agent,
            provider=agent_config.provider,
            model=agent_config.model,
            text="ok-local",
            status="OK",
            latency_ms=120,
        )

    monkeypatch.setattr(main, "_build_effective_prompt", fake_effective)
    monkeypatch.setattr(main, "_build_prompt_with_history", fail_history)
    monkeypatch.setattr(main, "_build_prompt_with_thread_context", fail_thread_context)
    monkeypatch.setattr(main, "_run_single_agent", fake_runner)

    response = client.post("/api/magi/run", json={"prompt": "敬語に言い換えて", "profile": "local_only"})
    assert response.status_code == 200
    body = response.json()
    assert body["profile"] == "local_only"
    assert len(body["results"]) == 1
    assert body["consensus"]["status"] == "OK"
    assert body["consensus"]["text"] == "ok-local"


def test_run_skips_history_context_when_prompt_contains_url(monkeypatch) -> None:
    monkeypatch.setattr(main, "load_config", _profiled_config)
    monkeypatch.setattr(main, "_save_run_history", lambda *args, **kwargs: None)

    async def fail_history(*args, **kwargs):
        raise AssertionError("history context should be skipped for URL-anchored prompts")

    async def fake_effective(prompt: str, fresh_mode: bool, source_urls: list[str] | None = None) -> str:
        return prompt

    async def fake_runner(agent_config, prompt, timeout_seconds):
        return main.AgentResult(
            agent=agent_config.agent,
            provider=agent_config.provider,
            model=agent_config.model,
            text=f"ok-{agent_config.agent}",
            status="OK",
            latency_ms=100,
        )

    monkeypatch.setattr(main, "_build_prompt_with_history", fail_history)
    monkeypatch.setattr(main, "_build_effective_prompt", fake_effective)
    monkeypatch.setattr(main, "_run_single_agent", fake_runner)

    response = client.post(
        "/api/magi/run",
        json={"prompt": "https://game8.jp/ff14/757988 この攻略記事を解説して", "profile": "cost"},
    )
    assert response.status_code == 200


def test_run_local_only_normalizes_verbose_rewrite_output(monkeypatch) -> None:
    monkeypatch.setattr(main, "load_config", lambda: _profiled_config_with_router(enabled=False))
    monkeypatch.setattr(main, "_save_run_history", lambda *args, **kwargs: None)

    async def fake_effective(prompt: str, fresh_mode: bool, source_urls: list[str] | None = None) -> str:
        return prompt

    async def fake_runner(agent_config, prompt, timeout_seconds):
        return main.AgentResult(
            agent=agent_config.agent,
            provider=agent_config.provider,
            model=agent_config.model,
            text=(
                "日本語の敬語で丁寧に言い換えると、"
                "「明日は伺えません」"
                "です。"
            ),
            status="OK",
            latency_ms=120,
        )

    monkeypatch.setattr(main, "_build_effective_prompt", fake_effective)
    monkeypatch.setattr(main, "_run_single_agent", fake_runner)

    response = client.post("/api/magi/run", json={"prompt": "日本語の敬語で、以下を丁寧に言い換えて：『明日行けない』", "profile": "local_only"})
    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["text"] == "明日は伺えません"
    assert body["consensus"]["text"] == "明日は伺えません"


def test_run_local_only_skips_source_quote_when_normalizing(monkeypatch) -> None:
    monkeypatch.setattr(main, "load_config", lambda: _profiled_config_with_router(enabled=False))
    monkeypatch.setattr(main, "_save_run_history", lambda *args, **kwargs: None)

    async def fake_effective(prompt: str, fresh_mode: bool, source_urls: list[str] | None = None) -> str:
        return prompt

    async def fake_runner(agent_config, prompt, timeout_seconds):
        return main.AgentResult(
            agent=agent_config.agent,
            provider=agent_config.provider,
            model=agent_config.model,
            text=(
                "丁寧に言い換えると、"
                "「明日行けない」ではなく"
                "「明日は伺えません」"
                "です。"
            ),
            status="OK",
            latency_ms=120,
        )

    monkeypatch.setattr(main, "_build_effective_prompt", fake_effective)
    monkeypatch.setattr(main, "_run_single_agent", fake_runner)

    response = client.post("/api/magi/run", json={"prompt": "日本語の敬語で、以下を丁寧に言い換えて：『明日行けない』", "profile": "local_only"})
    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["text"] == "明日は伺えません"
    assert body["consensus"]["text"] == "明日は伺えません"


def test_run_local_only_rewrites_when_output_equals_source(monkeypatch) -> None:
    monkeypatch.setattr(main, "load_config", lambda: _profiled_config_with_router(enabled=False))
    monkeypatch.setattr(main, "_save_run_history", lambda *args, **kwargs: None)

    async def fake_effective(prompt: str, fresh_mode: bool, source_urls: list[str] | None = None) -> str:
        return prompt

    async def fake_runner(agent_config, prompt, timeout_seconds):
        return main.AgentResult(
            agent=agent_config.agent,
            provider=agent_config.provider,
            model=agent_config.model,
            text="明日行けない",
            status="OK",
            latency_ms=120,
        )

    monkeypatch.setattr(main, "_build_effective_prompt", fake_effective)
    monkeypatch.setattr(main, "_run_single_agent", fake_runner)

    response = client.post("/api/magi/run", json={"prompt": "日本語の敬語で、以下を丁寧に言い換えて：『明日行けない』", "profile": "local_only"})
    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["text"] == "明日は行けません"
    assert body["consensus"]["text"] == "明日は行けません"


def test_run_local_only_rewrites_when_output_is_unstable(monkeypatch) -> None:
    monkeypatch.setattr(main, "load_config", lambda: _profiled_config_with_router(enabled=False))
    monkeypatch.setattr(main, "_save_run_history", lambda *args, **kwargs: None)

    async def fake_effective(prompt: str, fresh_mode: bool, source_urls: list[str] | None = None) -> str:
        return prompt

    async def fake_runner(agent_config, prompt, timeout_seconds):
        return main.AgentResult(
            agent=agent_config.agent,
            provider=agent_config.provider,
            model=agent_config.model,
            text="明日はご同行ができないことがございます。",
            status="OK",
            latency_ms=120,
        )

    monkeypatch.setattr(main, "_build_effective_prompt", fake_effective)
    monkeypatch.setattr(main, "_run_single_agent", fake_runner)

    response = client.post("/api/magi/run", json={"prompt": "日本語の敬語で、以下を丁寧に言い換えて：『明日行けない』", "profile": "local_only"})
    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["text"] == "明日は行けません"
    assert body["consensus"]["text"] == "明日は行けません"


def test_consensus_local_only_passthrough(monkeypatch) -> None:
    monkeypatch.setattr(main, "load_config", lambda: _profiled_config_with_router(enabled=False))

    async def fail_consensus(*args, **kwargs):
        raise AssertionError("local_only should not run consensus deliberation")

    monkeypatch.setattr(main, "_run_consensus", fail_consensus)

    payload = {
        "prompt": "敬語に言い換えて",
        "profile": "local_only",
        "results": [
            {
                "agent": "A",
                "provider": "ollama",
                "model": "qwen2.5:7b-instruct-q4_K_M",
                "text": "明日は伺えません。",
                "status": "OK",
                "latency_ms": 1111,
                "error_message": None,
            }
        ],
    }
    response = client.post("/api/magi/consensus", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["profile"] == "local_only"
    assert body["consensus"]["status"] == "OK"
    assert body["consensus"]["text"] == "明日は伺えません。"


def test_run_auto_enables_fresh_mode_for_time_sensitive_prompt(monkeypatch) -> None:
    monkeypatch.setattr(main, "load_config", _profiled_config)
    monkeypatch.setattr(main, "_save_run_history", lambda *args, **kwargs: None)
    monkeypatch.setenv("FRESH_AUTO_MODE", "1")

    async def fake_effective(prompt: str, fresh_mode: bool, source_urls: list[str] | None = None) -> str:
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
    assert main._resolve_fresh_mode("YouTube攻略動画も参考にして", False) is True


def test_build_effective_prompt_falls_back_without_tavily_key(monkeypatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    prompt = "latest policy update"
    effective = asyncio.run(main._build_effective_prompt(prompt, True))
    assert effective == prompt


def test_routing_feedback_rejects_non_uuid_thread_id() -> None:
    response = client.post(
        "/api/magi/routing/feedback",
        json={"thread_id": "thread-abc", "request_id": "req-1", "rating": 1},
    )
    assert response.status_code == 400
    assert "thread_id must be a UUID" in response.json()["detail"]


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

    async def fake_effective(prompt: str, fresh_mode: bool, source_urls: list[str] | None = None) -> str:
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


def test_resolve_history_deprecations_merges_remote_rules(monkeypatch) -> None:
    main._DEPRECATIONS_SOURCE_CACHE.clear()
    cfg = main.HistoryContextConfig(
        deprecations=[
            main.HistoryDeprecationRule(
                id="authjs-migration",
                legacy_terms=["nextauth"],
                current_terms=["auth.js"],
            ),
            main.HistoryDeprecationRule(
                id="react-query-rename",
                legacy_terms=["react-query"],
                current_terms=["tanstack query"],
            ),
        ],
        deprecations_source=main.DeprecationsSourceConfig(
            enabled=True,
            url="https://example.com/deprecations.json",
            mode="merge",
            refresh_interval_seconds=3600,
        ),
    )

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "deprecations": [
                    {
                        "id": "authjs-migration",
                        "legacy_terms": ["nextauth", "next-auth"],
                        "current_terms": ["auth.js", "authjs"],
                    },
                    {
                        "id": "next-font-migration",
                        "legacy_terms": ["@next/font"],
                        "current_terms": ["next/font"],
                    },
                ]
            }

    monkeypatch.setattr(main.httpx, "get", lambda *args, **kwargs: _FakeResponse())

    merged = main._resolve_history_deprecations(cfg)
    by_id = {item.id: item for item in merged}
    assert set(by_id.keys()) == {"authjs-migration", "react-query-rename", "next-font-migration"}
    assert by_id["authjs-migration"].legacy_terms == ["nextauth", "next-auth"]


def test_resolve_history_deprecations_falls_back_to_local_when_fetch_fails(monkeypatch) -> None:
    main._DEPRECATIONS_SOURCE_CACHE.clear()
    cfg = main.HistoryContextConfig(
        deprecations=[
            main.HistoryDeprecationRule(
                id="authjs-migration",
                legacy_terms=["nextauth"],
                current_terms=["auth.js"],
            )
        ],
        deprecations_source=main.DeprecationsSourceConfig(
            enabled=True,
            url="https://example.com/deprecations.json",
            mode="merge",
            refresh_interval_seconds=3600,
        ),
    )

    def _raise_get(*args, **kwargs):
        raise RuntimeError("network error")

    monkeypatch.setattr(main.httpx, "get", _raise_get)

    merged = main._resolve_history_deprecations(cfg)
    assert len(merged) == 1
    assert merged[0].id == "authjs-migration"


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

    async def fake_call_model_text(
        full_model: str,
        prompt: str,
        timeout_seconds: int,
        system_prompt: str | None = None,
    ):
        calls["count"] += 1
        if calls["count"] == 1:
            raise asyncio.TimeoutError()
        return "ok-after-retry", 123, {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16, "cost_estimate_usd": 0.000123}

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
    assert result.total_tokens == 16
    assert result.cost_estimate_usd == 0.000123


def test_call_model_text_includes_system_prompt_when_provided(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        }

    monkeypatch.setattr(main, "acompletion", fake_acompletion)

    text, _, _ = asyncio.run(
        main._call_model_text(
            "ollama/qwen2.5:7b-instruct-q4_K_M",
            "hello",
            timeout_seconds=3,
            system_prompt="system-rule",
        )
    )

    assert text == "ok"
    messages = captured["messages"]
    assert isinstance(messages, list)
    assert messages[0] == {"role": "system", "content": "system-rule"}
    assert messages[1] == {"role": "user", "content": "hello"}


def test_call_model_text_keeps_user_only_when_system_prompt_not_set(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        }

    monkeypatch.setattr(main, "acompletion", fake_acompletion)

    text, _, _ = asyncio.run(
        main._call_model_text(
            "ollama/qwen2.5:7b-instruct-q4_K_M",
            "hello",
            timeout_seconds=3,
        )
    )

    assert text == "ok"
    messages = captured["messages"]
    assert isinstance(messages, list)
    assert messages == [{"role": "user", "content": "hello"}]


def test_run_single_agent_retries_once_for_gemini_timeout(monkeypatch) -> None:
    calls = {"count": 0}

    async def fake_call_model_text(
        full_model: str,
        prompt: str,
        timeout_seconds: int,
        system_prompt: str | None = None,
    ):
        calls["count"] += 1
        if calls["count"] == 1:
            raise asyncio.TimeoutError()
        return "ok-after-retry", 123, {"prompt_tokens": 9, "completion_tokens": 5, "total_tokens": 14}

    monkeypatch.setattr(main, "_call_model_text", fake_call_model_text)

    result = asyncio.run(
        main._run_single_agent(
            agent_config=main.AgentConfig(agent="C", provider="gemini", model="gemini-2.5-flash"),
            prompt="hello",
            timeout_seconds=1,
        )
    )

    assert calls["count"] == 2
    assert result.status == "OK"
    assert result.text == "ok-after-retry"


def test_run_single_agent_does_not_retry_non_openai_timeout(monkeypatch) -> None:
    calls = {"count": 0}

    async def fake_call_model_text(
        full_model: str,
        prompt: str,
        timeout_seconds: int,
        system_prompt: str | None = None,
    ):
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


def test_public_error_message_maps_503_high_demand() -> None:
    exc = RuntimeError('ServiceUnavailableError: {"error":{"code":503,"message":"high demand. try again later"}}')
    message = main._public_error_message(exc)
    assert "503 high demand" in message


def test_public_error_message_includes_original_exception_text() -> None:
    exc = RuntimeError("connection reset by peer on upstream")
    message = main._public_error_message(exc)
    assert "RuntimeError:" in message
    assert "connection reset by peer" in message


def test_routing_policy_key_is_stable() -> None:
    router_input = {
        "intent": "translation",
        "complexity": "low",
        "language": "ja",
    }
    key1 = main._routing_policy_key_from_input(router_input)
    key2 = main._routing_policy_key_from_input(dict(router_input))
    assert key1 == key2 == "intent=translation|complexity=low|lang=ja"


def test_policy_update_weight_increase_and_decrease(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MAGI_DB_PATH", str(tmp_path / "magi-routing-policy.db"))
    main._init_db()
    config = main.AppConfig(
        default_profile="cost",
        routing_learning=main.RoutingLearningConfig(alpha=0.5, weight_min=-2.0, weight_max=2.0),
    )
    main._save_routing_event(
        thread_id="thread-1",
        request_id="req-1",
        router_input={"intent": "qa", "complexity": "high", "language": "en"},
        router_output={
            "chosen_profile": "cost",
            "policy_key": "intent=qa|complexity=high|lang=en",
            "candidates": {},
        },
    )

    main._update_routing_event_execution(
        "req-1",
        {
            "profile": "cost",
            "latency_ms": 100,
            "error": False,
            "cost_estimate": 0.1,
        },
        config,
    )
    policy_after_run = main._get_routing_policy("intent=qa|complexity=high|lang=en")
    assert policy_after_run["weights"]["cost"] == 0.0

    main._update_routing_event_feedback("thread-1", "req-1", 1, "good", config)
    policy_after_positive = main._get_routing_policy("intent=qa|complexity=high|lang=en")
    assert policy_after_positive["weights"]["cost"] > 0.0

    main._update_routing_event_feedback("thread-1", "req-1", -1, "bad", config)
    policy_after_negative = main._get_routing_policy("intent=qa|complexity=high|lang=en")
    assert policy_after_negative["weights"]["cost"] < policy_after_positive["weights"]["cost"]


def test_policy_update_clamps_weight(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MAGI_DB_PATH", str(tmp_path / "magi-routing-clamp.db"))
    main._init_db()
    config = main.AppConfig(
        default_profile="cost",
        routing_learning=main.RoutingLearningConfig(alpha=1.0, weight_min=-0.1, weight_max=0.1),
    )
    main._save_routing_event(
        thread_id="thread-1",
        request_id="req-clamp",
        router_input={"intent": "qa", "complexity": "high", "language": "en"},
        router_output={
            "chosen_profile": "cost",
            "policy_key": "intent=qa|complexity=high|lang=en",
            "candidates": {},
        },
    )
    main._update_routing_event_execution(
        "req-clamp",
        {
            "profile": "cost",
            "latency_ms": 100,
            "error": False,
            "cost_estimate": 0.1,
        },
        config,
    )
    for _ in range(4):
        main._update_routing_event_feedback("thread-1", "req-clamp", 1, "good", config)
    positive = main._get_routing_policy("intent=qa|complexity=high|lang=en")
    assert positive["weights"]["cost"] == 0.1

    for _ in range(4):
        main._update_routing_event_feedback("thread-1", "req-clamp", -1, "bad", config)
    negative = main._get_routing_policy("intent=qa|complexity=high|lang=en")
    assert negative["weights"]["cost"] == -0.1


def test_policy_update_applies_time_decay(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MAGI_DB_PATH", str(tmp_path / "magi-routing-decay.db"))
    main._init_db()
    config = main.AppConfig(
        default_profile="cost",
        routing_learning=main.RoutingLearningConfig(
            alpha=0.0,
            weight_min=-2.0,
            weight_max=2.0,
            decay_lambda_per_day=1.0,
            stats_ema_beta=0.0,
        ),
    )
    key = "intent=qa|complexity=high|lang=en"
    main._save_routing_event(
        thread_id="thread-1",
        request_id="req-decay",
        router_input={"intent": "qa", "complexity": "high", "language": "en"},
        router_output={"chosen_profile": "cost", "policy_key": key, "candidates": {}},
    )
    main._upsert_routing_policy(key, {"cost": 1.0}, {"n": 0, "avg_reward": 0.0})
    stale_time = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    with main._get_db_connection() as conn:
        conn.execute("UPDATE routing_policy SET updated_at = ? WHERE key = ?", (stale_time, key))

    main._update_routing_event_execution(
        "req-decay",
        {"profile": "cost", "latency_ms": 100, "error": False, "cost_estimate": 0.1},
        config,
    )
    updated = main._get_routing_policy(key)
    decayed = float(updated["weights"]["cost"])
    assert 0.3 < decayed < 0.5


def test_policy_stats_ema_is_used_when_configured() -> None:
    config = main.AppConfig(
        default_profile="cost",
        routing_learning=main.RoutingLearningConfig(stats_ema_beta=0.2),
    )
    updated = main._next_policy_stats({"n": 2, "avg_reward": 0.5}, reward=-1.0, config=config)
    assert updated["n"] == 3
    assert abs(float(updated["avg_reward"]) - 0.2) < 1e-9


def test_routing_feedback_affects_next_auto_route(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MAGI_DB_PATH", str(tmp_path / "magi-routing-integration.db"))
    main._init_db()

    config = _profiled_config_with_router(enabled=True)
    config.profiles["balance"] = main.ProfileConfig(
        agents=[
            main.AgentConfig(agent="A", provider="openai", model="gpt-4.1"),
            main.AgentConfig(agent="B", provider="anthropic", model="claude-sonnet-4-20250514"),
            main.AgentConfig(agent="C", provider="gemini", model="gemini-2.5-flash"),
        ],
        consensus=main.ConsensusConfig(strategy="peer_vote", min_ok_results=2, rounds=1),
        timeout_seconds=20,
    )
    config.default_profile = "cost"
    config.routing_learning = main.RoutingLearningConfig(alpha=1.5, weight_min=-2.0, weight_max=2.0)
    monkeypatch.setattr(main, "load_config", lambda: config)

    async def fake_route_llm(full_model: str, prompt: str, timeout_seconds: int, max_tokens: int | None = None):
        return (
            '{"intent":"qa","complexity":"high","safety":"low","execution_tier":"cloud","profile":"cost","confidence":95,"reason":"default"}',
            10,
            {},
        )

    async def fake_runner(agent_config, prompt, timeout_seconds):
        return main.AgentResult(
            agent=agent_config.agent,
            provider=agent_config.provider,
            model=agent_config.model,
            text=f"ok-{agent_config.agent}",
            status="OK",
            latency_ms=100,
        )

    monkeypatch.setattr(main, "_call_model_text", fake_route_llm)
    monkeypatch.setattr(main, "_run_single_agent", fake_runner)

    first = client.post("/api/magi/run", json={"prompt": "design a backend architecture"})
    assert first.status_code == 200
    first_body = first.json()
    assert first_body["profile"] == "cost"

    feedback = client.post(
        "/api/magi/routing/feedback",
        json={
            "thread_id": first_body["thread_id"],
            "request_id": first_body["run_id"],
            "rating": -1,
            "reason": "too expensive",
        },
    )
    assert feedback.status_code == 200

    second = client.post("/api/magi/run", json={"prompt": "design a backend architecture"})
    assert second.status_code == 200
    second_body = second.json()
    assert second_body["profile"] == "balance"


def test_routing_signal_updates_implicit_reward(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MAGI_DB_PATH", str(tmp_path / "magi-routing-signal.db"))
    main._init_db()

    config = _profiled_config_with_router(enabled=False)
    monkeypatch.setattr(main, "load_config", lambda: config)

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

    run = client.post("/api/magi/run", json={"prompt": "hello", "profile": "cost"})
    assert run.status_code == 200
    run_body = run.json()

    signal = client.post(
        "/api/magi/routing/signal",
        json={
            "thread_id": run_body["thread_id"],
            "request_id": run_body["run_id"],
            "signal": "copy_result",
        },
    )
    assert signal.status_code == 200
    signal_body = signal.json()
    assert signal_body["implicit_reward"] > 0

    events = client.get(f"/api/magi/routing/events?thread_id={run_body['thread_id']}&limit=5")
    assert events.status_code == 200
    items = events.json()["items"]
    assert items
    assert items[0]["implicit_signals"].get("copy_result") == 1


def test_single_model_consensus_sets_error_code_for_insufficient_results() -> None:
    cfg = main.ConsensusConfig(
        strategy="single_model",
        provider="openai",
        model="gpt-5-mini",
        min_ok_results=2,
    )
    results = [
        main.AgentResult(agent="A", provider="openai", model="m1", text="ok", status="OK", latency_ms=10),
        main.AgentResult(agent="B", provider="anthropic", model="m2", text="", status="ERROR", latency_ms=10, error_message="timeout"),
    ]

    consensus = asyncio.run(main._run_single_model_consensus(cfg, "hello", results, timeout_seconds=3))
    assert consensus.status == "ERROR"
    assert consensus.error_code == "INSUFFICIENT_OK_RESULTS"
