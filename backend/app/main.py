import asyncio
import json
import os
import uuid
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from litellm import acompletion
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
# LiteLLM Gemini adapter expects GEMINI_API_KEY in many environments.
if not os.getenv("GEMINI_API_KEY") and os.getenv("GOOGLE_API_KEY"):
    os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]


class AgentConfig(BaseModel):
    agent: str
    provider: str
    model: str


class AppConfig(BaseModel):
    agents: list[AgentConfig]
    timeout_seconds: int = 20


class RunRequest(BaseModel):
    prompt: str


class AgentResult(BaseModel):
    agent: str
    provider: str
    model: str
    text: str
    status: Literal["OK", "ERROR"]
    latency_ms: int
    error_message: str | None = None


class RunResponse(BaseModel):
    run_id: str
    results: list[AgentResult]


app = FastAPI(title="MAGI v0 Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_config() -> AppConfig:
    config_path = Path(__file__).resolve().parents[1] / "config.json"
    with config_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    config = AppConfig.model_validate(data)
    if len(config.agents) != 3:
        raise ValueError("config.json must define exactly 3 agents")
    return config


def _extract_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")
    if not choices:
        return ""

    choice = choices[0]
    message = getattr(choice, "message", None)
    if message is None and isinstance(choice, dict):
        message = choice.get("message")

    content = getattr(message, "content", None) if message is not None else None
    if content is None and isinstance(message, dict):
        content = message.get("content")

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return "" if content is None else str(content)


def _public_error_message(exc: Exception) -> str:
    message = str(exc).lower()
    if "credit balance is too low" in message:
        return "provider account has insufficient credits"
    if "api key" in message or "authentication" in message:
        return "provider authentication failed (check API key)"
    if "not found" in message and "model" in message:
        return "model not found (check config.json model name)"
    if "rate limit" in message or "too many requests" in message:
        return "provider rate limit reached"
    return "provider request failed"


async def _run_single_agent(agent_config: AgentConfig, prompt: str, timeout_seconds: int) -> AgentResult:
    full_model = f"{agent_config.provider}/{agent_config.model}"
    start = perf_counter()
    print(f"[magi] agent={agent_config.agent} start model={full_model}")

    try:
        response = await asyncio.wait_for(
            acompletion(
                model=full_model,
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=timeout_seconds,
        )
        text = _extract_text(response)
        latency_ms = int((perf_counter() - start) * 1000)
        print(f"[magi] agent={agent_config.agent} success latency_ms={latency_ms}")
        return AgentResult(
            agent=agent_config.agent,
            provider=agent_config.provider,
            model=agent_config.model,
            text=text,
            status="OK",
            latency_ms=latency_ms,
        )
    except asyncio.TimeoutError:
        latency_ms = int((perf_counter() - start) * 1000)
        print(f"[magi] agent={agent_config.agent} error=timeout latency_ms={latency_ms}")
        return AgentResult(
            agent=agent_config.agent,
            provider=agent_config.provider,
            model=agent_config.model,
            text="",
            status="ERROR",
            latency_ms=latency_ms,
            error_message="timeout",
        )
    except Exception as exc:  # noqa: BLE001
        latency_ms = int((perf_counter() - start) * 1000)
        print(f"[magi] agent={agent_config.agent} error={type(exc).__name__}: {exc} latency_ms={latency_ms}")
        return AgentResult(
            agent=agent_config.agent,
            provider=agent_config.provider,
            model=agent_config.model,
            text="",
            status="ERROR",
            latency_ms=latency_ms,
            error_message=_public_error_message(exc),
        )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/magi/run", response_model=RunResponse)
async def run_magi(payload: RunRequest) -> RunResponse:
    prompt = payload.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt must not be empty")
    if len(prompt) > 4000:
        raise HTTPException(status_code=400, detail="prompt must be 4000 characters or fewer")

    try:
        config = load_config()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"invalid config: {exc}") from exc

    tasks = [_run_single_agent(agent, prompt, config.timeout_seconds) for agent in config.agents]
    results = await asyncio.gather(*tasks)

    return RunResponse(run_id=str(uuid.uuid4()), results=results)
