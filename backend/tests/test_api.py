from fastapi.testclient import TestClient

import backend.app.main as main


client = TestClient(main.app)


def test_run_rejects_empty_prompt() -> None:
    response = client.post("/api/magi/run", json={"prompt": "   "})
    assert response.status_code == 400
    assert response.json()["detail"] == "prompt must not be empty"


def test_run_rejects_too_long_prompt() -> None:
    response = client.post("/api/magi/run", json={"prompt": "a" * 4001})
    assert response.status_code == 400
    assert response.json()["detail"] == "prompt must be 4000 characters or fewer"


def test_run_returns_partial_failure_without_failing_request(monkeypatch) -> None:
    config = main.AppConfig(
        agents=[
            main.AgentConfig(agent="A", provider="openai", model="gpt-4o-mini"),
            main.AgentConfig(agent="B", provider="anthropic", model="claude-haiku-4-5-20251001"),
            main.AgentConfig(agent="C", provider="gemini", model="gemini-2.5-flash"),
        ],
        consensus=main.ConsensusConfig(provider="openai", model="gpt-4o-mini", min_ok_results=2),
        timeout_seconds=20,
    )

    monkeypatch.setattr(main, "load_config", lambda: config)

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
    async def fake_consensus(config_obj, prompt, results):
        return main.ConsensusResult(
            provider=config_obj.consensus.provider or "magi",
            model=config_obj.consensus.model or "peer_vote_v1",
            text="consensus-ok",
            status="OK",
            latency_ms=99,
        )

    monkeypatch.setattr(main, "_run_consensus", fake_consensus)

    response = client.post("/api/magi/run", json={"prompt": "hello"})
    assert response.status_code == 200

    body = response.json()
    assert isinstance(body.get("run_id"), str) and body["run_id"]
    assert len(body["results"]) == 3

    by_agent = {item["agent"]: item for item in body["results"]}
    assert by_agent["A"]["status"] == "OK"
    assert by_agent["B"]["status"] == "ERROR"
    assert by_agent["B"]["error_message"] == "provider request failed"
    assert by_agent["C"]["status"] == "OK"
    assert body["consensus"]["status"] == "OK"
    assert body["consensus"]["text"] == "consensus-ok"


def test_retry_rejects_empty_prompt() -> None:
    response = client.post("/api/magi/retry", json={"prompt": "   ", "agent": "A"})
    assert response.status_code == 400
    assert response.json()["detail"] == "prompt must not be empty"


def test_retry_rejects_invalid_agent() -> None:
    response = client.post("/api/magi/retry", json={"prompt": "hello", "agent": "D"})
    assert response.status_code == 422


def test_retry_runs_only_target_agent(monkeypatch) -> None:
    config = main.AppConfig(
        agents=[
            main.AgentConfig(agent="A", provider="openai", model="gpt-4.1-mini"),
            main.AgentConfig(agent="B", provider="anthropic", model="claude-sonnet-4-20250514"),
            main.AgentConfig(agent="C", provider="gemini", model="gemini-2.5-flash"),
        ],
        consensus=main.ConsensusConfig(provider="openai", model="gpt-4.1-mini", min_ok_results=2),
        timeout_seconds=20,
    )

    monkeypatch.setattr(main, "load_config", lambda: config)

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

    response = client.post("/api/magi/retry", json={"prompt": "hello", "agent": "B"})
    assert response.status_code == 200

    body = response.json()
    assert isinstance(body.get("run_id"), str) and body["run_id"]
    assert body["result"]["agent"] == "B"
    assert body["result"]["status"] == "OK"
    assert body["result"]["text"] == "retry-B"


def test_consensus_endpoint_recalculates(monkeypatch) -> None:
    config = main.AppConfig(
        agents=[
            main.AgentConfig(agent="A", provider="openai", model="gpt-4.1-mini"),
            main.AgentConfig(agent="B", provider="anthropic", model="claude-sonnet-4-20250514"),
            main.AgentConfig(agent="C", provider="gemini", model="gemini-2.5-flash"),
        ],
        consensus=main.ConsensusConfig(provider="openai", model="gpt-4.1-mini", min_ok_results=2),
        timeout_seconds=20,
    )

    monkeypatch.setattr(main, "load_config", lambda: config)
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

    response = client.post("/api/magi/consensus", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["consensus"]["status"] == "OK"
    assert body["consensus"]["text"] == "recalc-consensus"
