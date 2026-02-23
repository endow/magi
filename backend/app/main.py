import asyncio
import json
import math
import os
import re
import sqlite3
import uuid
import warnings
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from litellm import acompletion, aembedding
from pydantic import BaseModel, Field

warnings.filterwarnings(
    "ignore",
    message="You seem to already have a custom sys.excepthook handler installed.*",
    category=RuntimeWarning,
    module="trio._core._multierror",
)

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


class AgentConfig(BaseModel):
    agent: str
    provider: str
    model: str


class ConsensusConfig(BaseModel):
    strategy: Literal["peer_vote", "single_model"] = "peer_vote"
    provider: str | None = None
    model: str | None = None
    min_ok_results: int = 2
    rounds: int = 1
    debate_mode: Literal["normal", "strict"] = "normal"
    min_criticisms: int = 0


class ProfileConfig(BaseModel):
    agents: list[AgentConfig]
    consensus: ConsensusConfig
    timeout_seconds: int = 20


class HistoryContextConfig(BaseModel):
    strategy: Literal["lexical", "embedding"] = "lexical"
    provider: str | None = None
    model: str | None = None
    timeout_seconds: int = 12
    batch_size: int = 24
    freshness_half_life_days: int = 180
    stale_weight: float = 0.55
    superseded_weight: float = 0.20
    deprecations: list["HistoryDeprecationRule"] = Field(default_factory=list)


class HistoryDeprecationRule(BaseModel):
    id: str
    legacy_terms: list[str]
    current_terms: list[str]


class RequestRouterConfig(BaseModel):
    enabled: bool = False
    provider: str = "ollama"
    model: str = "qwen2.5:7b-instruct-q4_K_M"
    timeout_seconds: int = 4
    min_confidence: int = 75


class RouteRule(BaseModel):
    when_intents_any: list[str] = Field(default_factory=list)
    when_complexity_any: list[str] = Field(default_factory=list)
    when_safety_any: list[str] = Field(default_factory=list)
    when_execution_tiers_any: list[str] = Field(default_factory=list)
    profile: str


class RouterRulesConfig(BaseModel):
    default_profile: str | None = None
    routes: list[RouteRule] = Field(default_factory=list)


class RoutingLearningConfig(BaseModel):
    enabled: bool = True
    alpha: float = 0.05
    weight_min: float = -2.0
    weight_max: float = 2.0
    latency_threshold_ms: int = 8000
    cost_threshold: float = 2.0
    decay_lambda_per_day: float = 0.0
    stats_ema_beta: float = 0.0


class RouteDecision(BaseModel):
    profile: str
    confidence: int
    reason: str | None = None
    intent: str | None = None
    complexity: str | None = None
    safety: str | None = None
    execution_tier: str | None = None


class AppConfig(BaseModel):
    default_profile: str | None = None
    profiles: dict[str, ProfileConfig] | None = None
    agents: list[AgentConfig] | None = None
    consensus: ConsensusConfig | None = None
    timeout_seconds: int | None = None
    history_context: HistoryContextConfig | None = None
    request_router: RequestRouterConfig | None = None
    router_rules: RouterRulesConfig | None = None
    routing_learning: RoutingLearningConfig | None = None


HistoryContextConfig.model_rebuild()


class RunRequest(BaseModel):
    prompt: str
    profile: str | None = None
    fresh_mode: bool = False
    thread_id: str | None = None


class RetryRequest(BaseModel):
    prompt: str
    agent: Literal["A", "B", "C"]
    profile: str | None = None
    fresh_mode: bool = False
    thread_id: str | None = None


class AgentResult(BaseModel):
    agent: str
    provider: str
    model: str
    text: str
    status: Literal["OK", "ERROR"]
    latency_ms: int
    error_message: str | None = None


class ConsensusRequest(BaseModel):
    prompt: str
    results: list[AgentResult]
    profile: str | None = None
    fresh_mode: bool = False
    thread_id: str | None = None
    run_id: str | None = None


class FreshSource(BaseModel):
    title: str
    url: str
    snippet: str
    published_date: str | None = None


class ConsensusResult(BaseModel):
    provider: str
    model: str
    text: str
    status: Literal["OK", "ERROR", "LOADING"]
    latency_ms: int
    error_message: str | None = None


class DeliberationTurn(BaseModel):
    agent: str
    provider: str
    model: str
    status: Literal["OK", "ERROR"]
    latency_ms: int
    raw_text: str
    revised_answer: str
    preferred_agent: Literal["A", "B", "C"] | None = None
    reason: str | None = None
    confidence: int | None = None
    criticisms: list[str] = Field(default_factory=list)
    error_message: str | None = None


class RunResponse(BaseModel):
    run_id: str
    thread_id: str
    turn_index: int
    profile: str
    results: list[AgentResult]
    consensus: ConsensusResult
    routing: "RoutingDecisionInfo | None" = None


class RetryResponse(BaseModel):
    run_id: str
    thread_id: str
    turn_index: int
    profile: str
    result: AgentResult


class ConsensusResponse(BaseModel):
    run_id: str
    thread_id: str
    turn_index: int
    profile: str
    consensus: ConsensusResult


class ProfilesResponse(BaseModel):
    default_profile: str
    profiles: list[str]
    profile_agents: dict[str, list[AgentConfig]]


class HistoryItem(BaseModel):
    run_id: str
    thread_id: str
    turn_index: int
    profile: str
    prompt: str
    created_at: str
    results: list[AgentResult]
    consensus: ConsensusResult | None = None


class HistoryListResponse(BaseModel):
    total: int
    items: list[HistoryItem]


class RoutingFeedbackRequest(BaseModel):
    thread_id: str
    request_id: str
    rating: Literal[-1, 0, 1]
    reason: str | None = None


class RoutingFeedbackResponse(BaseModel):
    thread_id: str
    request_id: str
    rating: int
    policy_key: str | None = None


class RoutingPolicyResponse(BaseModel):
    key: str
    weights: dict[str, float]
    stats: dict[str, float | int]
    updated_at: str | None = None


class RoutingEventItem(BaseModel):
    id: int
    thread_id: str
    request_id: str
    created_at: str
    router_input: dict[str, Any]
    router_output: dict[str, Any]
    execution_result: dict[str, Any]
    user_rating: int
    user_reason: str | None = None


class RoutingEventsResponse(BaseModel):
    total: int
    items: list[RoutingEventItem]


class RoutingDecisionInfo(BaseModel):
    profile: str
    reason: str | None = None
    intent: str | None = None
    complexity: str | None = None
    safety: str | None = None
    execution_tier: str | None = None
    policy_key: str | None = None


RunResponse.model_rebuild()


@asynccontextmanager
async def app_lifespan(_: FastAPI):
    _init_db()
    yield


app = FastAPI(title="MAGI v1.0 Backend", lifespan=app_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://host.docker.internal:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_FRESH_CACHE: dict[str, tuple[float, list[FreshSource]]] = {}


def load_config() -> AppConfig:
    config_path = Path(__file__).resolve().parents[1] / "config.json"
    with config_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    return AppConfig.model_validate(data)


def _validate_profile(profile: ProfileConfig) -> None:
    if len(profile.agents) < 1:
        raise ValueError("each profile must define at least 1 agent")


def _resolve_profile(config: AppConfig, requested_profile: str | None) -> tuple[str, ProfileConfig]:
    if config.profiles:
        if not config.default_profile:
            raise ValueError("default_profile is required when profiles are configured")

        profile_key = requested_profile or config.default_profile
        profile = config.profiles.get(profile_key)
        if profile is None:
            raise HTTPException(status_code=400, detail=f"unknown profile: {profile_key}")
        _validate_profile(profile)
        return profile_key, profile

    # Backward compatibility for legacy single-profile config.json
    if not config.agents or not config.consensus:
        raise ValueError("config.json must define either profiles or legacy agents/consensus")
    profile = ProfileConfig(
        agents=config.agents,
        consensus=config.consensus,
        timeout_seconds=config.timeout_seconds or 20,
    )
    _validate_profile(profile)
    return "default", profile


def _normalize_confidence(value: Any) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0
    if 0.0 <= numeric <= 1.0:
        numeric *= 100.0
    return max(0, min(100, int(round(numeric))))


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip().lower() for item in value if str(item).strip()]
    if isinstance(value, str):
        cleaned = value.strip().lower()
        return [cleaned] if cleaned else []
    return []


def _build_routing_reason_display(
    selected_profile: str,
    router_input: dict[str, Any],
    raw_reason: str | None,
) -> str:
    intent = _safe_str(router_input.get("intent")) or "other"
    complexity = _safe_str(router_input.get("complexity")) or "medium"
    safety = _safe_str(router_input.get("safety")) or "low"
    tier = _safe_str(router_input.get("execution_tier")) or "cloud"
    base = f"intent={intent}, complexity={complexity}, safety={safety}, tier={tier} の判定で {selected_profile} を選択"
    if not raw_reason:
        return base
    # Keep original reason only when it includes Japanese kana/ASCII text; otherwise prefer deterministic display.
    if re.search(r"[ぁ-んァ-ンA-Za-z]", raw_reason):
        return f"{base} ({raw_reason})"
    return base


def _routing_learning_config(config: AppConfig | None = None) -> RoutingLearningConfig:
    if config and config.routing_learning:
        return config.routing_learning
    return RoutingLearningConfig()


def _detect_language(prompt: str) -> str:
    text = prompt.strip()
    if not text:
        return "unknown"
    if re.search(r"[ぁ-んァ-ン一-龥]", text):
        return "ja"
    if re.search(r"[A-Za-z]", text):
        return "en"
    return "other"


def _build_router_input_snapshot(
    prompt: str,
    decision: RouteDecision | None = None,
    fallback_profile: str | None = None,
) -> dict[str, Any]:
    intent = _safe_str(decision.intent if decision else None).lower() or "other"
    complexity = _safe_str(decision.complexity if decision else None).lower() or "medium"
    safety = _safe_str(decision.safety if decision else None).lower() or ("low" if _is_low_safety_prompt(prompt) else "medium")
    execution_tier = _safe_str(decision.execution_tier if decision else None).lower() or "cloud"
    return {
        "intent": intent,
        "complexity": complexity,
        "safety": safety,
        "execution_tier": execution_tier,
        "language": _detect_language(prompt),
        "prompt_length": len(prompt.strip()),
        "fallback_profile": fallback_profile or "",
    }


def _routing_policy_key_from_input(router_input: dict[str, Any]) -> str:
    intent = _safe_str(router_input.get("intent")).lower() or "other"
    complexity = _safe_str(router_input.get("complexity")).lower() or "medium"
    lang = _safe_str(router_input.get("language")).lower() or "unknown"
    return f"intent={intent}|complexity={complexity}|lang={lang}"


def _select_profile_with_policy(
    available_profiles: set[str],
    base_selected: str,
    router_input: dict[str, Any],
) -> tuple[str, dict[str, dict[str, float]], str]:
    profile_names = sorted(available_profiles)
    base_scores = {name: (1.0 if name == base_selected else 0.0) for name in profile_names}
    policy_key = _routing_policy_key_from_input(router_input)
    policy = _get_routing_policy(policy_key)
    weights = policy.get("weights", {})
    candidates: dict[str, dict[str, float]] = {}
    for name in profile_names:
        policy_weight = float(weights.get(name, 0.0))
        final_score = float(base_scores[name] + policy_weight)
        candidates[name] = {
            "base_score": float(base_scores[name]),
            "policy_weight": policy_weight,
            "final_score": final_score,
        }
    selected = profile_names[0]
    for name in profile_names[1:]:
        current = candidates[selected]
        contender = candidates[name]
        if contender["final_score"] > current["final_score"]:
            selected = name
            continue
        if contender["final_score"] == current["final_score"] and contender["base_score"] > current["base_score"]:
            selected = name
    return selected, candidates, policy_key


def _router_enabled(config: AppConfig, requested_profile: str | None) -> bool:
    if requested_profile:
        return False
    return bool(config.request_router and config.request_router.enabled and config.profiles)


def _is_low_safety_prompt(prompt: str) -> bool:
    lowered = prompt.strip().lower()
    risky_terms = (
        "kill",
        "suicide",
        "bomb",
        "explosive",
        "weapon",
        "hack",
        "malware",
        "ransomware",
        "self-harm",
        "自殺",
        "爆弾",
        "武器",
        "不正アクセス",
        "ハッキング",
    )
    return not any(term in lowered for term in risky_terms)


def _local_intent_fast_path(prompt: str) -> str | None:
    lowered = prompt.strip().lower()
    if not lowered:
        return None

    # Keep heuristic intentionally strict: only short, low-risk text operations.
    compact = re.sub(r"\s+", " ", lowered)
    if len(compact) > 240:
        return None

    if any(token in compact for token in ("要約", "summarize", "summary", "短くまとめ", "short summary")):
        return "summarize_short"
    if any(token in compact for token in ("言い換え", "言い直", "rephrase", "paraphrase", "rewrite")):
        return "rewrite"
    if any(token in compact for token in ("翻訳", "translate", "敬語", "丁寧に")):
        return "translation"
    return None


def _should_apply_history_context(profile_name: str) -> bool:
    return profile_name != "local_only"


def _should_apply_thread_context(profile_name: str) -> bool:
    return profile_name != "local_only"


def _is_rewrite_or_translation_prompt(prompt: str) -> bool:
    lowered = prompt.strip().lower()
    return any(token in lowered for token in ("敬語", "丁寧に", "言い換え", "言い直", "rephrase", "rewrite", "translate", "翻訳"))


def _extract_prompt_quoted_text(prompt: str) -> str | None:
    match = re.search(r"[「『]([^「」『』\n]{1,120})[」』]", prompt.strip())
    if not match:
        return None
    value = match.group(1).strip()
    return value or None


def _extract_local_single_answer(text: str, source_text: str | None = None) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped
    quoted = re.findall(r"[「『]([^「」『』\n]{1,120})[」』]", stripped)
    source = (source_text or "").strip()
    for candidate in quoted:
        value = candidate.strip()
        if not value:
            continue
        if source and value == source:
            continue
        if len(value) <= 2:
            continue
        if value:
            return value
    return stripped


def _heuristic_polite_rewrite(source_text: str) -> str:
    value = source_text.strip().strip("。")
    if not value:
        return source_text
    if value.startswith("明日") and not value.startswith("明日は"):
        value = f"明日は{value[2:]}"
    replaced = value
    replaced = replaced.replace("行けない", "行けません")
    replaced = replaced.replace("いけない", "いけません")
    replaced = replaced.replace("できない", "できません")
    replaced = replaced.replace("無理", "難しいです")
    if replaced == value and replaced.endswith("ない"):
        replaced = f"{replaced[:-2]}ません"
    if not re.search(r"(です|ます|ません|でした|ください)$", replaced):
        replaced = f"{replaced}です"
    return replaced


def _should_fallback_local_rewrite(text: str) -> bool:
    value = text.strip()
    if not value:
        return True
    if "\n" in value:
        return True
    if any(token in value for token in ("または", "以下", "これら", "言い換えると", "丁寧に言い換え")):
        return True
    if any(token in value for token in ("ご同行", "ご案内できかねる", "お mee", "参り難い", "参り难い")):
        return True
    if re.search(r"[A-Za-z]{2,}", value) and re.search(r"[ぁ-んァ-ン一-龥]", value):
        return True
    return False


def _postprocess_local_only_result(prompt: str, result: AgentResult) -> AgentResult:
    if result.status != "OK":
        return result
    if not _is_rewrite_or_translation_prompt(prompt):
        return result
    source_text = _extract_prompt_quoted_text(prompt)
    extracted = _extract_local_single_answer(result.text, source_text=source_text)
    if source_text and extracted.strip().strip("。") == source_text.strip().strip("。"):
        fallback = _heuristic_polite_rewrite(source_text)
        print("[magi] local_only_output fallback rewrite")
        return result.model_copy(update={"text": fallback})
    if source_text and _should_fallback_local_rewrite(extracted):
        fallback = _heuristic_polite_rewrite(source_text)
        print("[magi] local_only_output fallback rewrite")
        return result.model_copy(update={"text": fallback})
    if extracted == result.text:
        return result
    print("[magi] local_only_output normalized")
    return result.model_copy(update={"text": extracted})


def _build_local_only_consensus(results: list[AgentResult]) -> ConsensusResult:
    primary = results[0] if results else None
    if primary is None:
        return ConsensusResult(
            provider="magi",
            model="local_passthrough",
            text="",
            status="ERROR",
            latency_ms=0,
            error_message="no result for local_only consensus",
        )
    if primary.status != "OK":
        return ConsensusResult(
            provider=primary.provider,
            model=primary.model,
            text="",
            status="ERROR",
            latency_ms=primary.latency_ms,
            error_message=primary.error_message or "local_only agent failed",
        )
    return ConsensusResult(
        provider=primary.provider,
        model=primary.model,
        text=primary.text,
        status="OK",
        latency_ms=primary.latency_ms,
    )


def _route_rule_matches(
    rule: RouteRule,
    intents: list[str],
    complexity: str,
    safety: str,
    execution_tier: str,
) -> bool:
    normalized_intents = {item.strip().lower() for item in intents if item.strip()}
    rule_intents = {item.strip().lower() for item in rule.when_intents_any if item.strip()}
    if rule_intents and not (normalized_intents & rule_intents):
        return False

    rule_complexities = {item.strip().lower() for item in rule.when_complexity_any if item.strip()}
    if rule_complexities and complexity.strip().lower() not in rule_complexities:
        return False

    rule_safety = {item.strip().lower() for item in rule.when_safety_any if item.strip()}
    if rule_safety and safety.strip().lower() not in rule_safety:
        return False

    rule_execution_tiers = {item.strip().lower() for item in rule.when_execution_tiers_any if item.strip()}
    if rule_execution_tiers and execution_tier.strip().lower() not in rule_execution_tiers:
        return False
    return True


def _extract_route_decision(text: str, fallback_profile: str) -> RouteDecision:
    payload: dict[str, Any] = {}
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    candidate = match.group(0) if match else text
    try:
        loaded = json.loads(candidate)
        if isinstance(loaded, dict):
            payload = loaded
    except json.JSONDecodeError:
        payload = {}

    profile = _safe_str(payload.get("profile")) or fallback_profile
    confidence = _normalize_confidence(payload.get("confidence"))
    reason = _safe_str(payload.get("reason")) or None
    intent = _safe_str(payload.get("intent")) or None
    complexity = _safe_str(payload.get("complexity")) or None
    safety = _safe_str(payload.get("safety")) or None
    execution_tier = _safe_str(payload.get("execution_tier")) or None
    return RouteDecision(
        profile=profile,
        confidence=confidence,
        reason=reason,
        intent=intent,
        complexity=complexity,
        safety=safety,
        execution_tier=execution_tier,
    )


def _pick_profile_by_rules(
    route_decision: RouteDecision,
    rules: RouterRulesConfig | None,
    default_profile: str,
    available_profiles: set[str],
) -> str:
    candidate_profile = route_decision.profile.strip()
    intents = _safe_str_list(route_decision.intent)
    complexity = _safe_str(route_decision.complexity).lower()
    safety = _safe_str(route_decision.safety).lower()
    execution_tier = _safe_str(route_decision.execution_tier).lower()

    if rules:
        for rule in rules.routes:
            if _route_rule_matches(rule, intents, complexity, safety, execution_tier):
                mapped = rule.profile.strip()
                if mapped in available_profiles:
                    return mapped

        if rules.default_profile and rules.default_profile in available_profiles:
            return rules.default_profile

    if candidate_profile in available_profiles:
        return candidate_profile

    return default_profile


async def _route_profile_with_trace(
    config: AppConfig,
    prompt: str,
) -> tuple[str | None, dict[str, Any], dict[str, Any]]:
    if not config.profiles or not config.default_profile:
        router_input = _build_router_input_snapshot(prompt, None, None)
        return None, router_input, {"chosen_profile": None, "candidates": {}, "policy_key": _routing_policy_key_from_input(router_input)}
    router = config.request_router
    if not router or not router.enabled:
        router_input = _build_router_input_snapshot(prompt, None, config.default_profile)
        chosen, candidates, policy_key = _select_profile_with_policy(
            set(config.profiles.keys()),
            config.default_profile,
            router_input,
        )
        return chosen, router_input, {"chosen_profile": chosen, "candidates": candidates, "policy_key": policy_key}

    fallback_profile = config.default_profile
    full_model = f"{router.provider}/{router.model}"
    available_profiles = set(config.profiles.keys())

    fast_intent = _local_intent_fast_path(prompt)
    if (
        fast_intent in {"translation", "rewrite", "summarize_short"}
        and _is_low_safety_prompt(prompt)
        and "local_only" in available_profiles
    ):
        decision = RouteDecision(
            profile="local_only",
            confidence=100,
            reason="fast_path",
            intent=fast_intent,
            complexity="low",
            safety="low",
            execution_tier="local",
        )
        router_input = _build_router_input_snapshot(prompt, decision, fallback_profile)
        selected_profile, candidates, policy_key = _select_profile_with_policy(
            available_profiles,
            "local_only",
            router_input,
        )
        print(
            "[magi] request_router fast_path "
            f"intent={fast_intent} complexity=low safety=low tier=local profile={selected_profile}"
        )
        return (
            selected_profile,
            router_input,
            {
                "chosen_profile": selected_profile,
                "candidates": candidates,
                "policy_key": policy_key,
                "reason": "fast_path",
            },
        )

    routing_prompt = (
        "You are a lightweight request router. "
        "Classify the user request and output strict JSON only.\n"
        "Allowed values:\n"
        '- intent: one of ["qa","coding","research","creative","translation","rewrite","summarize_short","analysis","other"]\n'
        '- complexity: one of ["low","medium","high"]\n'
        '- safety: one of ["low","medium","high"]\n'
        '- execution_tier: one of ["local","cloud"]\n'
        f'- profile: one of {sorted(available_profiles)}\n'
        "- confidence: integer 0-100 (not decimal)\n"
        "Return exactly this shape:\n"
        '{"intent":"qa","complexity":"low","safety":"low","execution_tier":"local","profile":"cost","confidence":85,"reason":"..."}\n'
        "If classification is clear, confidence should be >= 80.\n"
        "No markdown, no extra keys.\n\n"
        f"User request:\n{prompt}"
    )

    max_attempts = 2
    for attempt in range(1, max_attempts + 1):
        try:
            raw_text, latency_ms = await _call_model_text(
                full_model,
                routing_prompt,
                router.timeout_seconds,
                max_tokens=120,
            )
            decision = _extract_route_decision(raw_text, fallback_profile)
            base_profile = _pick_profile_by_rules(
                decision,
                config.router_rules,
                fallback_profile,
                available_profiles,
            )
            router_input = _build_router_input_snapshot(prompt, decision, fallback_profile)
            if decision.confidence < router.min_confidence:
                base_profile = fallback_profile
                reason = "fallback_by_confidence"
            else:
                reason = decision.reason or "router"
            selected_profile, candidates, policy_key = _select_profile_with_policy(
                available_profiles,
                base_profile,
                router_input,
            )

            if decision.confidence < router.min_confidence:
                print(
                    "[magi] request_router fallback_by_confidence "
                    f"model={full_model} confidence={decision.confidence} threshold={router.min_confidence} "
                    f"profile={selected_profile}"
                )
                return (
                    selected_profile,
                    router_input,
                    {
                        "chosen_profile": selected_profile,
                        "candidates": candidates,
                        "policy_key": policy_key,
                        "reason": reason,
                    },
                )

            print(
                "[magi] request_router success "
                f"model={full_model} latency_ms={latency_ms} "
                f"intent={decision.intent or '-'} complexity={decision.complexity or '-'} "
                f"safety={decision.safety or '-'} tier={decision.execution_tier or '-'} "
                f"profile={selected_profile} confidence={decision.confidence}"
            )
            return (
                selected_profile,
                router_input,
                {
                    "chosen_profile": selected_profile,
                    "candidates": candidates,
                    "policy_key": policy_key,
                    "reason": reason,
                },
            )
        except asyncio.TimeoutError:
            router_input = _build_router_input_snapshot(prompt, None, fallback_profile)
            selected_profile, candidates, policy_key = _select_profile_with_policy(
                available_profiles,
                fallback_profile,
                router_input,
            )
            if attempt < max_attempts:
                print(f"[magi] request_router timeout attempt={attempt}; retrying once")
                continue
            print(f"[magi] request_router failed TimeoutError: ; fallback={fallback_profile}")
            return (
                selected_profile,
                router_input,
                {
                    "chosen_profile": selected_profile,
                    "candidates": candidates,
                    "policy_key": policy_key,
                    "reason": "timeout_fallback",
                },
            )
        except Exception as exc:  # noqa: BLE001
            router_input = _build_router_input_snapshot(prompt, None, fallback_profile)
            selected_profile, candidates, policy_key = _select_profile_with_policy(
                available_profiles,
                fallback_profile,
                router_input,
            )
            print(f"[magi] request_router failed {type(exc).__name__}: {exc}; fallback={fallback_profile}")
            return (
                selected_profile,
                router_input,
                {
                    "chosen_profile": selected_profile,
                    "candidates": candidates,
                    "policy_key": policy_key,
                    "reason": "error_fallback",
                },
            )

    router_input = _build_router_input_snapshot(prompt, None, fallback_profile)
    selected_profile, candidates, policy_key = _select_profile_with_policy(
        available_profiles,
        fallback_profile,
        router_input,
    )
    return (
        selected_profile,
        router_input,
        {"chosen_profile": selected_profile, "candidates": candidates, "policy_key": policy_key, "reason": "fallback"},
    )


async def _route_profile(config: AppConfig, prompt: str) -> str | None:
    selected, _, _ = await _route_profile_with_trace(config, prompt)
    return selected


async def _resolve_profile_with_router(
    config: AppConfig,
    requested_profile: str | None,
    prompt: str,
) -> tuple[str, ProfileConfig]:
    if not _router_enabled(config, requested_profile):
        return _resolve_profile(config, requested_profile)

    routed_profile = await _route_profile(config, prompt)
    return _resolve_profile(config, routed_profile or requested_profile)


async def _resolve_profile_with_router_trace(
    config: AppConfig,
    requested_profile: str | None,
    prompt: str,
) -> tuple[str, ProfileConfig, dict[str, Any], dict[str, Any]]:
    if not _router_enabled(config, requested_profile):
        profile_name, profile = _resolve_profile(config, requested_profile)
        available_profiles = set(config.profiles.keys()) if config.profiles else {profile_name}
        router_input = _build_router_input_snapshot(prompt, None, profile_name)
        selected_profile, candidates, policy_key = _select_profile_with_policy(
            available_profiles,
            profile_name,
            router_input,
        )
        # Explicit profile selection must remain authoritative.
        if requested_profile:
            selected_profile = profile_name
            candidates = {
                key: {"base_score": (1.0 if key == profile_name else 0.0), "policy_weight": 0.0, "final_score": (1.0 if key == profile_name else 0.0)}
                for key in sorted(available_profiles)
            }
        return selected_profile, profile, router_input, {
            "chosen_profile": selected_profile,
            "candidates": candidates,
            "policy_key": policy_key,
            "reason": "manual" if requested_profile else "default",
        }

    routed_profile, router_input, router_output = await _route_profile_with_trace(config, prompt)
    resolved_profile_name, profile = _resolve_profile(config, routed_profile or requested_profile)
    router_output = {**router_output, "chosen_profile": resolved_profile_name}
    return resolved_profile_name, profile, router_input, router_output


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
    if "quota exceeded" in message or "resource_exhausted" in message or "429" in message:
        return "provider quota exceeded (try later or upgrade plan)"
    if "api key" in message or "authentication" in message:
        return "provider authentication failed (check API key)"
    if "not found" in message and "model" in message:
        return "model not found (check config.json model name)"
    if "rate limit" in message or "too many requests" in message:
        return "provider rate limit reached"
    return "provider request failed"


def _validate_prompt(raw_prompt: str) -> str:
    prompt = raw_prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt must not be empty")
    if len(prompt) > 4000:
        raise HTTPException(status_code=400, detail="prompt must be 4000 characters or fewer")
    return prompt


def _fresh_max_results() -> int:
    raw = os.getenv("FRESH_MAX_RESULTS", "3")
    try:
        return max(1, min(10, int(raw)))
    except ValueError:
        return 5


def _fresh_cache_ttl_seconds() -> int:
    raw = os.getenv("FRESH_CACHE_TTL_SECONDS", "1800")
    try:
        return max(0, int(raw))
    except ValueError:
        return 1800


def _fresh_search_depth() -> Literal["basic", "advanced"]:
    raw = os.getenv("FRESH_SEARCH_DEPTH", "basic").strip().lower()
    if raw in {"basic", "advanced"}:
        return raw  # type: ignore[return-value]
    return "basic"


def _fresh_primary_topic() -> Literal["general", "news"]:
    raw = os.getenv("FRESH_PRIMARY_TOPIC", "general").strip().lower()
    if raw in {"general", "news"}:
        return raw  # type: ignore[return-value]
    return "general"


def _fresh_auto_enabled() -> bool:
    raw = os.getenv("FRESH_AUTO_MODE", "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _is_fresh_sensitive_prompt(prompt: str) -> bool:
    lowered = prompt.strip().lower()
    if not lowered:
        return False

    # Keep auto mode conservative: only enable when user explicitly requests recency.
    latest_keywords = (
        "latest",
        "today",
        "current",
        "now",
        "as of",
        "breaking",
        "news",
        "price",
        "stock",
        "weather",
        "release date",
        "version",
        "速報",
        "最新",
        "今日",
        "現在",
        "ニュース",
        "株価",
        "為替",
        "価格",
        "天気",
        "直近",
        "アップデート",
        "リリース",
    )
    if any(token in lowered for token in latest_keywords):
        return True
    return False


def _resolve_fresh_mode(prompt: str, requested_fresh_mode: bool) -> bool:
    if requested_fresh_mode:
        return True
    if not _fresh_auto_enabled():
        return False
    auto_enabled = _is_fresh_sensitive_prompt(prompt)
    if auto_enabled:
        print("[magi] fresh_mode auto-enabled by prompt pattern")
    return auto_enabled


def _cached_fresh_sources(query: str) -> list[FreshSource] | None:
    cached = _FRESH_CACHE.get(query)
    if not cached:
        return None
    created, sources = cached
    ttl_seconds = _fresh_cache_ttl_seconds()
    if ttl_seconds <= 0:
        return None
    age_seconds = perf_counter() - created
    if age_seconds > ttl_seconds:
        _FRESH_CACHE.pop(query, None)
        return None
    return sources


def _cache_fresh_sources(query: str, sources: list[FreshSource]) -> None:
    _FRESH_CACHE[query] = (perf_counter(), sources)


def _fresh_query_attempts(query: str, primary_topic: Literal["general", "news"]) -> list[tuple[str, Literal["general", "news"]]]:
    q = query.strip()
    attempts: list[tuple[str, Literal["general", "news"]]] = []

    if primary_topic == "general":
        attempts.extend(
            [
                (q, "general"),
                (f"{q} guide walkthrough strategy 解説 攻略", "general"),
            ]
        )
        lowered = q.lower()
        if "ff14" in lowered or "ffxiv" in lowered:
            attempts.append(("ffxiv savage floor 4 guide latest", "general"))
        attempts.append((q, "news"))
    else:
        attempts.extend(
            [
                (q, "news"),
                (q, "general"),
                (f"{q} guide walkthrough strategy 解説 攻略", "general"),
            ]
        )

    deduped: list[tuple[str, Literal["general", "news"]]] = []
    seen: set[str] = set()
    for attempt_query, topic in attempts:
        key = f"{topic}::{attempt_query}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append((attempt_query, topic))
    return deduped


async def _tavily_search(
    client: httpx.AsyncClient,
    tavily_api_key: str,
    query: str,
    max_results: int,
    topic: Literal["general", "news"],
) -> list[FreshSource]:
    payload = {
        "api_key": tavily_api_key,
        "query": query,
        "search_depth": _fresh_search_depth(),
        "max_results": max_results,
        "topic": topic,
        "include_answer": False,
        "include_images": False,
        "include_raw_content": False,
    }
    response = await client.post("https://api.tavily.com/search", json=payload)
    response.raise_for_status()
    data = response.json()
    raw_results = data.get("results")
    if not isinstance(raw_results, list):
        return []

    sources: list[FreshSource] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        snippet = str(item.get("content") or "").strip()
        published = str(item.get("published_date") or "").strip() or None
        if not url or not snippet:
            continue
        sources.append(
            FreshSource(
                title=title or "Untitled",
                url=url,
                snippet=snippet,
                published_date=published,
            )
        )
        if len(sources) >= max_results:
            break
    return sources


async def _fetch_fresh_sources(query: str, max_results: int) -> list[FreshSource]:
    cached = _cached_fresh_sources(query)
    if cached is not None:
        return cached

    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        print("[magi] fresh_mode enabled but TAVILY_API_KEY is not set; fallback to normal prompt")
        return []

    primary_topic = _fresh_primary_topic()
    attempts = _fresh_query_attempts(query, primary_topic)
    seen_keys: set[str] = set()
    merged: list[FreshSource] = []

    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            for attempt_query, topic in attempts:
                found = await _tavily_search(client, tavily_api_key, attempt_query, max_results, topic)
                print(
                    f"[magi] fresh_mode search attempt topic={topic} "
                    f'query="{attempt_query[:60]}" results={len(found)}'
                )
                for item in found:
                    if item.url in seen_keys:
                        continue
                    seen_keys.add(item.url)
                    merged.append(item)
                    if len(merged) >= max_results:
                        _cache_fresh_sources(query, merged)
                        return merged
                if len(merged) >= max_results:
                    break
    except Exception as exc:  # noqa: BLE001
        print(f"[magi] fresh_mode search failed: {type(exc).__name__}: {exc}")
        return []

    _cache_fresh_sources(query, merged)
    return merged


def _inject_fresh_context(prompt: str, sources: list[FreshSource]) -> str:
    if not sources:
        return prompt
    now_utc = datetime.now(timezone.utc).isoformat()
    evidence_lines: list[str] = []
    for idx, source in enumerate(sources, start=1):
        published = source.published_date or "unknown"
        evidence_lines.append(
            f"[S{idx}] title={source.title}\n"
            f"url={source.url}\n"
            f"published={published}\n"
            f"snippet={source.snippet}"
        )
    evidence_text = "\n\n".join(evidence_lines)
    return (
        f"{prompt}\n\n"
        "[Fresh Web Evidence]\n"
        f"retrieved_at_utc={now_utc}\n"
        "Use this evidence as a primary source for time-sensitive facts.\n"
        "If a statement depends on freshness (today/latest/current), cite at least one source URL and published date.\n"
        "If evidence is insufficient, explicitly say what is uncertain.\n\n"
        f"{evidence_text}"
    )


async def _build_effective_prompt(prompt: str, fresh_mode: bool) -> str:
    if not fresh_mode:
        return prompt
    sources = await _fetch_fresh_sources(prompt, _fresh_max_results())
    if not sources:
        return prompt
    print(f"[magi] fresh_mode sources={len(sources)}")
    return _inject_fresh_context(prompt, sources)


def _history_enabled() -> bool:
    raw = os.getenv("HISTORY_CONTEXT_ENABLED", "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _history_similarity_threshold() -> float:
    raw = os.getenv("HISTORY_SIMILARITY_THRESHOLD", "0.78")
    try:
        value = float(raw)
    except ValueError:
        return 0.78
    return max(0.0, min(1.0, value))


def _history_candidate_limit() -> int:
    raw = os.getenv("HISTORY_SIMILAR_CANDIDATES", "120")
    try:
        return max(10, min(500, int(raw)))
    except ValueError:
        return 120


def _history_freshness_half_life_days(app_config: AppConfig | None = None) -> int:
    if app_config and app_config.history_context:
        return max(1, int(app_config.history_context.freshness_half_life_days))
    raw = os.getenv("HISTORY_FRESHNESS_HALF_LIFE_DAYS", "180")
    try:
        return max(1, int(raw))
    except ValueError:
        return 180


def _history_stale_weight(app_config: AppConfig | None = None) -> float:
    if app_config and app_config.history_context:
        return max(0.0, min(1.0, float(app_config.history_context.stale_weight)))
    raw = os.getenv("HISTORY_STALE_WEIGHT", "0.55")
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        return 0.55


def _history_superseded_weight(app_config: AppConfig | None = None) -> float:
    if app_config and app_config.history_context:
        return max(0.0, min(1.0, float(app_config.history_context.superseded_weight)))
    raw = os.getenv("HISTORY_SUPERSEDED_WEIGHT", "0.20")
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        return 0.20


def _history_max_references() -> int:
    raw = os.getenv("HISTORY_MAX_REFERENCES", "2")
    try:
        return max(1, min(5, int(raw)))
    except ValueError:
        return 2


def _thread_context_enabled() -> bool:
    raw = os.getenv("THREAD_CONTEXT_ENABLED", "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _thread_context_max_turns() -> int:
    raw = os.getenv("THREAD_CONTEXT_MAX_TURNS", "6")
    try:
        return max(1, min(20, int(raw)))
    except ValueError:
        return 6


def _normalize_thread_id(raw_thread_id: str | None) -> str | None:
    if raw_thread_id is None:
        return None
    value = raw_thread_id.strip()
    if not value:
        return None
    if len(value) > 120:
        raise HTTPException(status_code=400, detail="thread_id must be 120 characters or fewer")
    return value


def _normalize_similarity_text(text: str) -> str:
    lowered = text.strip().lower()
    return re.sub(r"\s+", " ", lowered)


def _history_validity_weight(validity_state: str | None, app_config: AppConfig | None = None) -> float:
    state = (validity_state or "active").strip().lower()
    if state == "superseded":
        return _history_superseded_weight(app_config)
    if state == "stale":
        return _history_stale_weight(app_config)
    return 1.0


def _history_freshness_weight(created_at: str | None, app_config: AppConfig | None = None) -> float:
    if not created_at:
        return 0.7
    try:
        created = datetime.fromisoformat(created_at)
    except ValueError:
        return 0.7
    now = datetime.now(timezone.utc)
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (now - created).total_seconds() / 86400.0)
    half_life_days = float(_history_freshness_half_life_days(app_config))
    return math.exp(-math.log(2) * (age_days / half_life_days))


def _weighted_history_score(
    similarity: float,
    created_at: str | None,
    validity_state: str | None,
    app_config: AppConfig | None = None,
) -> float:
    return similarity * _history_freshness_weight(created_at, app_config) * _history_validity_weight(
        validity_state, app_config
    )


def _tokenize_for_vector(text: str) -> list[str]:
    normalized = _normalize_similarity_text(text)
    if not normalized:
        return []

    tokens = [token for token in re.findall(r"\w+", normalized, flags=re.UNICODE) if len(token) > 1]
    compact = re.sub(r"\s+", "", normalized)
    # Add char 3-grams to support languages without explicit word boundaries.
    trigrams = [compact[i : i + 3] for i in range(max(0, len(compact) - 2))]
    return tokens + trigrams


def _term_frequency_vector(text: str) -> dict[str, float]:
    vector: dict[str, float] = defaultdict(float)
    for token in _tokenize_for_vector(text):
        vector[token] += 1.0
    return vector


def _cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    common = set(left.keys()) & set(right.keys())
    dot = sum(left[key] * right[key] for key in common)
    if dot <= 0:
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _load_history_candidates() -> list[sqlite3.Row]:
    candidates = _history_candidate_limit()
    with _get_db_connection() as conn:
        return conn.execute(
            """
            SELECT run_id, prompt, created_at, consensus_text, validity_state, superseded_by, superseded_at
            FROM runs
            WHERE consensus_status = 'OK'
              AND consensus_text IS NOT NULL
              AND consensus_text != ''
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (candidates,),
        ).fetchall()


def _rank_history_matches_lexical(
    prompt: str,
    rows: list[sqlite3.Row],
    threshold: float,
    limit: int,
    app_config: AppConfig | None = None,
) -> list[dict[str, str | float]]:
    query_vector = _term_frequency_vector(prompt)
    scored: list[dict[str, str | float]] = []
    for row in rows:
        previous_prompt = row["prompt"] or ""
        raw_similarity = _cosine_similarity(query_vector, _term_frequency_vector(previous_prompt))
        score = _weighted_history_score(raw_similarity, row["created_at"], row["validity_state"], app_config)
        if score < threshold:
            continue
        scored.append(
            {
                "run_id": row["run_id"],
                "prompt": previous_prompt,
                "created_at": row["created_at"] or "",
                "consensus_text": row["consensus_text"] or "",
                "validity_state": row["validity_state"] or "active",
                "superseded_by": row["superseded_by"] or "",
                "similarity": score,
                "raw_similarity": raw_similarity,
            }
        )

    scored.sort(
        key=lambda item: (
            float(item["similarity"]),
            str(item["created_at"]),
        ),
        reverse=True,
    )
    return scored[:limit]


def _history_embedding_config(
    app_config: AppConfig | None,
) -> tuple[str, str, int, int] | None:
    if not app_config or not app_config.history_context:
        return None
    cfg = app_config.history_context
    if cfg.strategy != "embedding":
        return None
    provider = (cfg.provider or "").strip()
    model = (cfg.model or "").strip()
    if not provider or not model:
        return None
    timeout_seconds = max(3, min(60, int(cfg.timeout_seconds)))
    batch_size = max(1, min(100, int(cfg.batch_size)))
    return provider, model, timeout_seconds, batch_size


def _extract_embedding_vectors(response: Any) -> list[list[float]]:
    data = getattr(response, "data", None)
    if data is None and isinstance(response, dict):
        data = response.get("data")
    if not isinstance(data, list):
        return []

    vectors: list[list[float]] = []
    for item in data:
        embedding = getattr(item, "embedding", None)
        if embedding is None and isinstance(item, dict):
            embedding = item.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            continue
        try:
            vectors.append([float(value) for value in embedding])
        except (TypeError, ValueError):
            continue
    return vectors


async def _embed_texts(
    provider: str,
    model: str,
    timeout_seconds: int,
    texts: list[str],
) -> list[list[float]]:
    if not texts:
        return []
    response = await asyncio.wait_for(
        aembedding(
            model=f"{provider}/{model}",
            input=texts,
        ),
        timeout=timeout_seconds,
    )
    vectors = _extract_embedding_vectors(response)
    if len(vectors) != len(texts):
        raise RuntimeError("embedding result size mismatch")
    return vectors


def _cosine_similarity_dense(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    if dot <= 0:
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


async def _rank_history_matches_embedding(
    prompt: str,
    rows: list[sqlite3.Row],
    threshold: float,
    limit: int,
    provider: str,
    model: str,
    timeout_seconds: int,
    batch_size: int,
    app_config: AppConfig | None = None,
) -> list[dict[str, str | float]]:
    query_vectors = await _embed_texts(provider, model, timeout_seconds, [prompt])
    if not query_vectors:
        return []
    query_vector = query_vectors[0]
    scored: list[dict[str, str | float]] = []

    for offset in range(0, len(rows), batch_size):
        batch_rows = rows[offset : offset + batch_size]
        batch_prompts = [str(row["prompt"] or "") for row in batch_rows]
        batch_vectors = await _embed_texts(provider, model, timeout_seconds, batch_prompts)

        for row, vector in zip(batch_rows, batch_vectors):
            raw_similarity = _cosine_similarity_dense(query_vector, vector)
            score = _weighted_history_score(raw_similarity, row["created_at"], row["validity_state"], app_config)
            if score < threshold:
                continue
            scored.append(
                {
                    "run_id": row["run_id"],
                    "prompt": row["prompt"] or "",
                    "created_at": row["created_at"] or "",
                    "consensus_text": row["consensus_text"] or "",
                    "validity_state": row["validity_state"] or "active",
                    "superseded_by": row["superseded_by"] or "",
                    "similarity": score,
                    "raw_similarity": raw_similarity,
                }
            )

    scored.sort(
        key=lambda item: (
            float(item["similarity"]),
            str(item["created_at"]),
        ),
        reverse=True,
    )
    return scored[:limit]


def _find_similar_history(prompt: str, limit: int) -> list[dict[str, str | float]]:
    rows = _load_history_candidates()
    return _rank_history_matches_lexical(prompt, rows, _history_similarity_threshold(), limit)


def _trim_for_prompt(text: str, max_chars: int = 700) -> str:
    value = text.strip()
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def _inject_history_context(base_prompt: str, matches: list[dict[str, str | float]]) -> str:
    if not matches:
        return base_prompt

    lines: list[str] = [
        "[Similar Past Runs]",
        "Use these only as reference. Re-check facts if they may be outdated.",
    ]
    for idx, item in enumerate(matches, start=1):
        similarity = float(item["similarity"])
        raw_similarity = float(item["raw_similarity"])
        validity_state = str(item.get("validity_state") or "active")
        superseded_by = str(item.get("superseded_by") or "").strip()
        validity_line = f"validity={validity_state}"
        if superseded_by:
            validity_line += f" superseded_by={superseded_by}"
        lines.append(
            f"[H{idx}] run_id={item['run_id']} created_at={item['created_at']} similarity={similarity:.2f}\n"
            f"raw_similarity={raw_similarity:.2f} {validity_line}\n"
            f"past_question={_trim_for_prompt(str(item['prompt']), 180)}\n"
            f"past_consensus={_trim_for_prompt(str(item['consensus_text']))}"
        )

    return f"{base_prompt}\n\n" + "\n\n".join(lines)


async def _build_prompt_with_history(
    original_prompt: str,
    base_prompt: str,
    app_config: AppConfig | None = None,
) -> str:
    if not _history_enabled():
        return base_prompt
    try:
        rows = _load_history_candidates()
    except Exception as exc:  # noqa: BLE001
        print(f"[magi] history_context lookup failed: {type(exc).__name__}: {exc}")
        return base_prompt
    threshold = _history_similarity_threshold()
    limit = _history_max_references()

    try:
        embedding_cfg = _history_embedding_config(app_config)
        if embedding_cfg:
            provider, model, timeout_seconds, batch_size = embedding_cfg
            matches = await _rank_history_matches_embedding(
                original_prompt,
                rows,
                threshold,
                limit,
                provider,
                model,
                timeout_seconds,
                batch_size,
                app_config,
            )
            print(f"[magi] history_context strategy=embedding model={provider}/{model} matches={len(matches)}")
        else:
            matches = _rank_history_matches_lexical(original_prompt, rows, threshold, limit, app_config)
            print(f"[magi] history_context strategy=lexical matches={len(matches)}")
    except Exception as exc:  # noqa: BLE001
        print(f"[magi] history_context embedding failed; fallback to lexical: {type(exc).__name__}: {exc}")
        try:
            matches = _rank_history_matches_lexical(original_prompt, rows, threshold, limit, app_config)
        except Exception as lexical_exc:  # noqa: BLE001
            print(f"[magi] history_context lookup failed: {type(lexical_exc).__name__}: {lexical_exc}")
            return base_prompt

    if not matches:
        return base_prompt
    print(f"[magi] history_context references={len(matches)}")
    return _inject_history_context(base_prompt, matches)


def _next_turn_index(thread_id: str) -> int:
    try:
        with _get_db_connection() as conn:
            row = conn.execute(
                "SELECT MAX(COALESCE(turn_index, 0)) AS max_turn FROM runs WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
    except sqlite3.OperationalError:
        return 1
    max_turn = int((row["max_turn"] if row and row["max_turn"] is not None else 0) or 0)
    return max_turn + 1


def _load_thread_history(thread_id: str, max_turns: int) -> list[sqlite3.Row]:
    try:
        with _get_db_connection() as conn:
            rows = conn.execute(
                """
                SELECT thread_id, turn_index, prompt, consensus_text, consensus_status, created_at
                FROM runs
                WHERE thread_id = ?
                ORDER BY turn_index DESC, created_at DESC
                LIMIT ?
                """,
                (thread_id, max_turns),
            ).fetchall()
    except sqlite3.OperationalError:
        return []
    return list(reversed(rows))


def _trim_thread_text(text: str, max_chars: int = 400) -> str:
    value = text.strip()
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def _extract_thread_assistant_text(consensus_text: str, consensus_status: str) -> str:
    if consensus_status != "OK":
        return "[no consensus answer]"

    text = consensus_text.strip()
    if not text:
        return "[no consensus answer]"

    marker = "Final answer:\n"
    if marker in text:
        start = text.find(marker) + len(marker)
        end_marker = "\n\nVote details:"
        end = text.find(end_marker, start)
        answer = text[start:end] if end >= 0 else text[start:]
        cleaned = answer.strip()
        if cleaned:
            return cleaned

    return text


def _inject_thread_context(base_prompt: str, thread_id: str, rows: list[sqlite3.Row]) -> str:
    if not rows:
        return base_prompt

    latest = rows[-1]
    latest_turn = int(latest["turn_index"] or 0)
    latest_user = _trim_thread_text(str(latest["prompt"] or ""), 260)
    latest_assistant = _trim_thread_text(
        _extract_thread_assistant_text(
            str(latest["consensus_text"] or ""),
            str(latest["consensus_status"] or ""),
        ),
        520,
    )

    lines = [
        "[High Priority Latest Turn]",
        "Treat this latest exchange as the strongest context for resolving the current question.",
        f"[T{latest_turn}] user={latest_user}\nassistant={latest_assistant}",
        "",
        "[Thread Conversation Context]",
        f"thread_id={thread_id}",
        "Interpret the current user question as a continuation of this thread unless the user clearly switches topic.",
        "When the current question is ambiguous, prioritize this thread context over generic interpretations.",
    ]
    for row in rows:
        turn_index = int(row["turn_index"] or 0)
        user_prompt = _trim_thread_text(str(row["prompt"] or ""), 220)
        consensus_status = str(row["consensus_status"] or "")
        assistant_text = _trim_thread_text(
            _extract_thread_assistant_text(str(row["consensus_text"] or ""), consensus_status),
            420,
        )
        lines.append(
            f"[T{turn_index}] user={user_prompt}\nassistant={assistant_text}"
        )

    return f"{base_prompt}\n\n" + "\n\n".join(lines)


def _build_prompt_with_thread_context(base_prompt: str, thread_id: str | None) -> str:
    if not _thread_context_enabled():
        return base_prompt
    if not thread_id:
        return base_prompt
    try:
        rows = _load_thread_history(thread_id, _thread_context_max_turns())
    except Exception as exc:  # noqa: BLE001
        print(f"[magi] thread_context lookup failed: {type(exc).__name__}: {exc}")
        return base_prompt
    if not rows:
        return base_prompt
    print(f"[magi] thread_context turns={len(rows)} thread_id={thread_id}")
    return _inject_thread_context(base_prompt, thread_id, rows)


def _db_path() -> Path:
    raw_path = os.getenv("MAGI_DB_PATH")
    if raw_path:
        return Path(raw_path)
    return Path(__file__).resolve().parents[1] / "data" / "magi.db"


def _get_db_connection() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _get_db_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                thread_id TEXT,
                turn_index INTEGER,
                profile TEXT NOT NULL,
                prompt TEXT NOT NULL,
                created_at TEXT NOT NULL,
                consensus_provider TEXT,
                consensus_model TEXT,
                consensus_text TEXT,
                consensus_status TEXT,
                consensus_latency_ms INTEGER,
                consensus_error_message TEXT,
                validity_state TEXT NOT NULL DEFAULT 'active',
                superseded_by TEXT,
                superseded_at TEXT
            );
            CREATE TABLE IF NOT EXISTS agent_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                agent TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                text TEXT NOT NULL,
                status TEXT NOT NULL,
                latency_ms INTEGER NOT NULL,
                error_message TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_agent_results_run_id ON agent_results(run_id);
            CREATE TABLE IF NOT EXISTS routing_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                request_id TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                router_input_json TEXT NOT NULL,
                router_output_json TEXT NOT NULL,
                execution_result_json TEXT NOT NULL,
                user_rating INTEGER NOT NULL DEFAULT 0,
                user_reason TEXT
            );
            CREATE TABLE IF NOT EXISTS routing_policy (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL UNIQUE,
                weights_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                stats_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_routing_events_thread_id ON routing_events(thread_id);
            CREATE INDEX IF NOT EXISTS idx_routing_events_request_id ON routing_events(request_id);
            """
        )
        _ensure_runs_columns(conn)
        _ensure_routing_columns(conn)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_validity_state ON runs(validity_state)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_thread_turn ON runs(thread_id, turn_index)")


def _ensure_runs_columns(conn: sqlite3.Connection) -> None:
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
    if "thread_id" not in existing:
        conn.execute("ALTER TABLE runs ADD COLUMN thread_id TEXT")
    if "turn_index" not in existing:
        conn.execute("ALTER TABLE runs ADD COLUMN turn_index INTEGER")
    if "validity_state" not in existing:
        conn.execute("ALTER TABLE runs ADD COLUMN validity_state TEXT NOT NULL DEFAULT 'active'")
    if "superseded_by" not in existing:
        conn.execute("ALTER TABLE runs ADD COLUMN superseded_by TEXT")
    if "superseded_at" not in existing:
        conn.execute("ALTER TABLE runs ADD COLUMN superseded_at TEXT")
    conn.execute(
        """
        UPDATE runs
        SET validity_state = COALESCE(NULLIF(TRIM(validity_state), ''), 'active')
        WHERE validity_state IS NULL OR TRIM(validity_state) = ''
        """
    )
    conn.execute(
        """
        UPDATE runs
        SET thread_id = COALESCE(NULLIF(TRIM(thread_id), ''), run_id)
        WHERE thread_id IS NULL OR TRIM(thread_id) = ''
        """
    )
    conn.execute(
        """
        UPDATE runs
        SET turn_index = 1
        WHERE turn_index IS NULL OR turn_index < 1
        """
    )


def _ensure_routing_columns(conn: sqlite3.Connection) -> None:
    event_columns = {row["name"] for row in conn.execute("PRAGMA table_info(routing_events)").fetchall()}
    if "user_rating" not in event_columns:
        conn.execute("ALTER TABLE routing_events ADD COLUMN user_rating INTEGER NOT NULL DEFAULT 0")
    if "user_reason" not in event_columns:
        conn.execute("ALTER TABLE routing_events ADD COLUMN user_reason TEXT")


def _json_dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _json_load_dict(payload: str | None) -> dict[str, Any]:
    if not payload:
        return {}
    try:
        loaded = json.loads(payload)
    except json.JSONDecodeError:
        return {}
    if not isinstance(loaded, dict):
        return {}
    return loaded


def _normalize_policy_weights(weights: dict[str, Any], available_profiles: set[str] | None = None) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key, value in weights.items():
        if available_profiles is not None and key not in available_profiles:
            continue
        try:
            normalized[key] = float(value)
        except (TypeError, ValueError):
            normalized[key] = 0.0
    if available_profiles:
        for profile in available_profiles:
            normalized.setdefault(profile, 0.0)
    return normalized


def _default_policy_stats() -> dict[str, float | int]:
    return {"n": 0, "avg_reward": 0.0}


def _routing_latency_threshold_ms(config: AppConfig | None = None) -> int:
    return max(1, int(_routing_learning_config(config).latency_threshold_ms))


def _routing_cost_threshold(config: AppConfig | None = None) -> float:
    return float(_routing_learning_config(config).cost_threshold)


def _routing_alpha(config: AppConfig | None = None) -> float:
    return float(_routing_learning_config(config).alpha)


def _routing_decay_lambda_per_day(config: AppConfig | None = None) -> float:
    return max(0.0, float(_routing_learning_config(config).decay_lambda_per_day))


def _routing_stats_ema_beta(config: AppConfig | None = None) -> float:
    return max(0.0, min(1.0, float(_routing_learning_config(config).stats_ema_beta)))


def _routing_clamp_range(config: AppConfig | None = None) -> tuple[float, float]:
    cfg = _routing_learning_config(config)
    lower = float(cfg.weight_min)
    upper = float(cfg.weight_max)
    if lower > upper:
        return upper, lower
    return lower, upper


def _clamp_weight(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _build_execution_result_snapshot(
    profile: str,
    results: list[AgentResult],
    consensus: ConsensusResult | None = None,
    failure_message: str | None = None,
) -> dict[str, Any]:
    models = [
        {
            "agent": item.agent,
            "provider": item.provider,
            "model": item.model,
            "status": item.status,
            "latency_ms": item.latency_ms,
            "error_message": item.error_message,
        }
        for item in results
    ]
    max_latency = max((item.latency_ms for item in results), default=0)
    token_estimate = max(0, sum(max(0, len(item.text)) // 4 for item in results))
    cost_estimate = round(token_estimate / 1000.0, 4)
    has_error = any(item.status == "ERROR" for item in results) or bool(failure_message)
    payload: dict[str, Any] = {
        "profile": profile,
        "models": models,
        "latency_ms": max_latency,
        "token_estimate": token_estimate,
        "cost_estimate": cost_estimate,
        "error": has_error,
        "error_message": failure_message,
    }
    if consensus:
        payload["consensus"] = {
            "provider": consensus.provider,
            "model": consensus.model,
            "status": consensus.status,
            "latency_ms": consensus.latency_ms,
            "error_message": consensus.error_message,
        }
    return payload


def _compute_routing_reward(
    user_rating: int,
    execution_result: dict[str, Any],
    config: AppConfig | None = None,
) -> float:
    reward = float(user_rating)
    if bool(execution_result.get("error")):
        reward -= 1.0
    latency_ms = int(execution_result.get("latency_ms") or 0)
    if latency_ms > _routing_latency_threshold_ms(config):
        reward -= 0.2
    profile = _safe_str(execution_result.get("profile")).lower()
    if profile != "local_only":
        cost_estimate = float(execution_result.get("cost_estimate") or 0.0)
        if cost_estimate > _routing_cost_threshold(config):
            reward -= 0.2
    return reward


def _next_policy_stats(stats: dict[str, Any], reward: float, config: AppConfig | None = None) -> dict[str, float | int]:
    prev_n = int(stats.get("n") or 0)
    prev_avg = float(stats.get("avg_reward") or 0.0)
    next_n = prev_n + 1
    ema_beta = _routing_stats_ema_beta(config)
    if ema_beta > 0.0:
        next_avg = (ema_beta * reward) + ((1.0 - ema_beta) * prev_avg)
    else:
        next_avg = ((prev_avg * prev_n) + reward) / next_n
    return {"n": next_n, "avg_reward": next_avg}


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    candidate = value.strip()
    if not candidate:
        return None
    try:
        dt = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _apply_weight_decay(
    weights: dict[str, float],
    updated_at: str | None,
    config: AppConfig | None = None,
) -> dict[str, float]:
    decay_lambda = _routing_decay_lambda_per_day(config)
    if decay_lambda <= 0.0 or not weights:
        return dict(weights)
    previous_time = _parse_iso_datetime(updated_at)
    if previous_time is None:
        return dict(weights)
    now = datetime.now(timezone.utc)
    elapsed_days = max(0.0, (now - previous_time).total_seconds() / 86400.0)
    if elapsed_days <= 0.0:
        return dict(weights)
    factor = math.exp(-decay_lambda * elapsed_days)
    return {name: float(value) * factor for name, value in weights.items()}


def _get_routing_policy(key: str) -> dict[str, Any]:
    _init_db()
    with _get_db_connection() as conn:
        row = conn.execute(
            "SELECT key, weights_json, stats_json, updated_at FROM routing_policy WHERE key = ?",
            (key,),
        ).fetchone()
    if row is None:
        return {"key": key, "weights": {}, "stats": _default_policy_stats(), "updated_at": None}
    return {
        "key": str(row["key"]),
        "weights": _normalize_policy_weights(_json_load_dict(row["weights_json"])),
        "stats": _json_load_dict(row["stats_json"]) or _default_policy_stats(),
        "updated_at": row["updated_at"],
    }


def _upsert_routing_policy(key: str, weights: dict[str, float], stats: dict[str, float | int]) -> None:
    _init_db()
    now = datetime.now(timezone.utc).isoformat()
    with _get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO routing_policy (key, weights_json, updated_at, stats_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                weights_json = excluded.weights_json,
                updated_at = excluded.updated_at,
                stats_json = excluded.stats_json
            """,
            (key, _json_dump(weights), now, _json_dump(stats)),
        )


def _save_routing_event(
    thread_id: str,
    request_id: str,
    router_input: dict[str, Any],
    router_output: dict[str, Any],
) -> None:
    _init_db()
    created_at = datetime.now(timezone.utc).isoformat()
    with _get_db_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO routing_events (
                thread_id, request_id, created_at, router_input_json, router_output_json,
                execution_result_json, user_rating, user_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                thread_id,
                request_id,
                created_at,
                _json_dump(router_input),
                _json_dump(router_output),
                _json_dump({}),
                0,
                None,
            ),
        )


def _apply_policy_update_for_event(request_id: str, config: AppConfig | None = None) -> str | None:
    _init_db()
    learning = _routing_learning_config(config)
    if not learning.enabled:
        return None
    with _get_db_connection() as conn:
        row = conn.execute(
            """
            SELECT request_id, router_output_json, execution_result_json, user_rating
            FROM routing_events
            WHERE request_id = ?
            """,
            (request_id,),
        ).fetchone()
        if row is None:
            return None

        router_output = _json_load_dict(row["router_output_json"])
        execution_result = _json_load_dict(row["execution_result_json"])
        user_rating = int(row["user_rating"] or 0)
        policy_key = _safe_str(router_output.get("policy_key")) or None
        chosen_profile = _safe_str(router_output.get("chosen_profile"))
        if not policy_key or not chosen_profile:
            return None

        policy_row = conn.execute(
            "SELECT weights_json, stats_json, updated_at FROM routing_policy WHERE key = ?",
            (policy_key,),
        ).fetchone()
        if policy_row is None:
            weights: dict[str, float] = {}
            stats: dict[str, Any] = _default_policy_stats()
            previous_updated_at: str | None = None
        else:
            weights = _normalize_policy_weights(_json_load_dict(policy_row["weights_json"]))
            stats = _json_load_dict(policy_row["stats_json"]) or _default_policy_stats()
            previous_updated_at = policy_row["updated_at"]

        weights = _apply_weight_decay(weights, previous_updated_at, config)

        reward = _compute_routing_reward(user_rating, execution_result, config)
        delta = _routing_alpha(config) * reward
        lower, upper = _routing_clamp_range(config)
        current_weight = float(weights.get(chosen_profile, 0.0))
        weights[chosen_profile] = _clamp_weight(current_weight + delta, lower, upper)
        next_stats = _next_policy_stats(stats, reward, config)
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            INSERT INTO routing_policy (key, weights_json, updated_at, stats_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                weights_json = excluded.weights_json,
                updated_at = excluded.updated_at,
                stats_json = excluded.stats_json
            """,
            (policy_key, _json_dump(weights), now, _json_dump(next_stats)),
        )
        return policy_key


def _update_routing_event_execution(request_id: str, execution_result: dict[str, Any], config: AppConfig | None = None) -> str | None:
    _init_db()
    with _get_db_connection() as conn:
        updated = conn.execute(
            "UPDATE routing_events SET execution_result_json = ? WHERE request_id = ?",
            (_json_dump(execution_result), request_id),
        ).rowcount
    if updated == 0:
        return None
    return _apply_policy_update_for_event(request_id, config)


def _update_routing_event_feedback(
    thread_id: str,
    request_id: str,
    rating: int,
    reason: str | None,
    config: AppConfig | None = None,
) -> str | None:
    _init_db()
    with _get_db_connection() as conn:
        updated = conn.execute(
            """
            UPDATE routing_events
            SET user_rating = ?, user_reason = ?
            WHERE thread_id = ? AND request_id = ?
            """,
            (int(rating), reason, thread_id, request_id),
        ).rowcount
    if updated == 0:
        return None
    return _apply_policy_update_for_event(request_id, config)


def _list_routing_events(thread_id: str | None, limit: int) -> RoutingEventsResponse:
    _init_db()
    with _get_db_connection() as conn:
        if thread_id:
            total_row = conn.execute(
                "SELECT COUNT(1) AS count FROM routing_events WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
            rows = conn.execute(
                """
                SELECT id, thread_id, request_id, created_at, router_input_json, router_output_json,
                       execution_result_json, user_rating, user_reason
                FROM routing_events
                WHERE thread_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (thread_id, limit),
            ).fetchall()
            total = int(total_row["count"] if total_row else 0)
        else:
            total_row = conn.execute("SELECT COUNT(1) AS count FROM routing_events").fetchone()
            rows = conn.execute(
                """
                SELECT id, thread_id, request_id, created_at, router_input_json, router_output_json,
                       execution_result_json, user_rating, user_reason
                FROM routing_events
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            total = int(total_row["count"] if total_row else 0)

    return RoutingEventsResponse(
        total=total,
        items=[
            RoutingEventItem(
                id=int(row["id"]),
                thread_id=str(row["thread_id"]),
                request_id=str(row["request_id"]),
                created_at=str(row["created_at"]),
                router_input=_json_load_dict(row["router_input_json"]),
                router_output=_json_load_dict(row["router_output_json"]),
                execution_result=_json_load_dict(row["execution_result_json"]),
                user_rating=int(row["user_rating"] or 0),
                user_reason=row["user_reason"],
            )
            for row in rows
        ],
    )


def _contains_any_term(text: str, terms: list[str]) -> bool:
    normalized = _normalize_similarity_text(text)
    return any(term.strip().lower() in normalized for term in terms if term.strip())


def _initial_validity_state(prompt: str, app_config: AppConfig | None) -> tuple[str, str | None]:
    if not app_config or not app_config.history_context:
        return "active", None
    for rule in app_config.history_context.deprecations:
        has_legacy = _contains_any_term(prompt, rule.legacy_terms)
        has_current = _contains_any_term(prompt, rule.current_terms)
        if has_legacy and not has_current:
            return "stale", rule.id
    return "active", None


def _apply_history_deprecations(
    conn: sqlite3.Connection,
    run_id: str,
    prompt: str,
    app_config: AppConfig | None,
) -> None:
    if not app_config or not app_config.history_context:
        return
    now = datetime.now(timezone.utc).isoformat()
    normalized = _normalize_similarity_text(prompt)

    for rule in app_config.history_context.deprecations:
        current_terms = [term.strip().lower() for term in rule.current_terms if term.strip()]
        legacy_terms = [term.strip().lower() for term in rule.legacy_terms if term.strip()]
        if not current_terms or not legacy_terms:
            continue
        if not any(term in normalized for term in current_terms):
            continue

        like_clauses = " OR ".join("LOWER(prompt) LIKE ?" for _ in legacy_terms)
        conn.execute(
            f"""
            UPDATE runs
            SET validity_state = 'superseded',
                superseded_by = ?,
                superseded_at = ?
            WHERE run_id != ?
              AND ({like_clauses})
              AND COALESCE(validity_state, 'active') != 'superseded'
            """,
            [rule.id, now, run_id, *[f"%{term}%" for term in legacy_terms]],
        )


def _save_run_history(
    run_id: str,
    profile: str,
    prompt: str,
    results: list[AgentResult],
    consensus: ConsensusResult,
    app_config: AppConfig | None = None,
    thread_id: str | None = None,
    turn_index: int | None = None,
) -> None:
    created_at = datetime.now(timezone.utc).isoformat()
    normalized_thread_id = thread_id or run_id
    normalized_turn_index = int(turn_index or 1)
    validity_state, superseded_hint = _initial_validity_state(prompt, app_config)
    with _get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO runs (
                run_id, thread_id, turn_index, profile, prompt, created_at,
                consensus_provider, consensus_model, consensus_text, consensus_status,
                consensus_latency_ms, consensus_error_message,
                validity_state, superseded_by, superseded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                normalized_thread_id,
                normalized_turn_index,
                profile,
                prompt,
                created_at,
                consensus.provider,
                consensus.model,
                consensus.text,
                consensus.status,
                consensus.latency_ms,
                consensus.error_message,
                validity_state,
                superseded_hint,
                None,
            ),
        )
        conn.executemany(
            """
            INSERT INTO agent_results (
                run_id, agent, provider, model, text, status, latency_ms, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    item.agent,
                    item.provider,
                    item.model,
                    item.text,
                    item.status,
                    item.latency_ms,
                    item.error_message,
                )
                for item in results
            ],
        )
        _apply_history_deprecations(conn, run_id, prompt, app_config)


def _update_run_consensus(run_id: str, consensus: ConsensusResult) -> bool:
    with _get_db_connection() as conn:
        result = conn.execute(
            """
            UPDATE runs
            SET consensus_provider = ?,
                consensus_model = ?,
                consensus_text = ?,
                consensus_status = ?,
                consensus_latency_ms = ?,
                consensus_error_message = ?
            WHERE run_id = ?
            """,
            (
                consensus.provider,
                consensus.model,
                consensus.text,
                consensus.status,
                consensus.latency_ms,
                consensus.error_message,
                run_id,
            ),
        )
        return int(result.rowcount or 0) > 0


def _get_run_thread_turn(run_id: str) -> tuple[str, int] | None:
    with _get_db_connection() as conn:
        row = conn.execute(
            "SELECT thread_id, turn_index FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    if row is None:
        return None
    return str(row["thread_id"] or run_id), int(row["turn_index"] or 1)


def _load_agent_results_map(run_ids: list[str]) -> dict[str, list[AgentResult]]:
    if not run_ids:
        return {}
    placeholders = ",".join("?" for _ in run_ids)
    query = (
        "SELECT run_id, agent, provider, model, text, status, latency_ms, error_message "
        f"FROM agent_results WHERE run_id IN ({placeholders}) ORDER BY id ASC"
    )
    by_run: dict[str, list[AgentResult]] = defaultdict(list)
    with _get_db_connection() as conn:
        rows = conn.execute(query, run_ids).fetchall()
    for row in rows:
        by_run[row["run_id"]].append(
            AgentResult(
                agent=row["agent"],
                provider=row["provider"],
                model=row["model"],
                text=row["text"],
                status=row["status"],
                latency_ms=row["latency_ms"],
                error_message=row["error_message"],
            )
        )
    return by_run


def _history_item_from_row(row: sqlite3.Row, results: list[AgentResult]) -> HistoryItem:
    consensus_status = row["consensus_status"]
    consensus: ConsensusResult | None = None
    if consensus_status:
        consensus = ConsensusResult(
            provider=row["consensus_provider"] or "-",
            model=row["consensus_model"] or "-",
            text=row["consensus_text"] or "",
            status=consensus_status,
            latency_ms=int(row["consensus_latency_ms"] or 0),
            error_message=row["consensus_error_message"],
        )

    return HistoryItem(
        run_id=row["run_id"],
        thread_id=row["thread_id"] or row["run_id"],
        turn_index=int(row["turn_index"] or 1),
        profile=row["profile"],
        prompt=row["prompt"],
        created_at=row["created_at"],
        results=results,
        consensus=consensus,
    )


def _list_history(limit: int, offset: int) -> HistoryListResponse:
    with _get_db_connection() as conn:
        total = conn.execute("SELECT COUNT(1) AS count FROM runs").fetchone()["count"]
        rows = conn.execute(
            """
            SELECT run_id, thread_id, turn_index, profile, prompt, created_at,
                   consensus_provider, consensus_model, consensus_text, consensus_status,
                   consensus_latency_ms, consensus_error_message
            FROM runs
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()

    run_ids = [row["run_id"] for row in rows]
    by_run = _load_agent_results_map(run_ids)
    items = [_history_item_from_row(row, by_run.get(row["run_id"], [])) for row in rows]
    return HistoryListResponse(total=total, items=items)


def _get_history_item(run_id: str) -> HistoryItem | None:
    with _get_db_connection() as conn:
        row = conn.execute(
            """
            SELECT run_id, thread_id, turn_index, profile, prompt, created_at,
                   consensus_provider, consensus_model, consensus_text, consensus_status,
                   consensus_latency_ms, consensus_error_message
            FROM runs
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()

    if row is None:
        return None

    by_run = _load_agent_results_map([run_id])
    return _history_item_from_row(row, by_run.get(run_id, []))


def _delete_thread_history(thread_id: str) -> int:
    normalized = thread_id.strip()
    if not normalized:
        return 0
    with _get_db_connection() as conn:
        run_rows = conn.execute("SELECT run_id FROM runs WHERE thread_id = ?", (normalized,)).fetchall()
        run_ids = [row["run_id"] for row in run_rows]
        if not run_ids:
            return 0
        placeholders = ",".join("?" for _ in run_ids)
        conn.execute(f"DELETE FROM agent_results WHERE run_id IN ({placeholders})", run_ids)
        result = conn.execute("DELETE FROM runs WHERE thread_id = ?", (normalized,))
        return int(result.rowcount or 0)


def _serialize_result_for_consensus(result: AgentResult) -> str:
    if result.status == "OK":
        body = result.text.strip()
    else:
        body = f"[ERROR] {result.error_message or 'unknown error'}"

    return (
        f"agent={result.agent}\n"
        f"provider={result.provider}\n"
        f"model={result.model}\n"
        f"status={result.status}\n"
        f"text={body}"
    )


def _build_single_model_consensus_prompt(prompt: str, results: list[AgentResult]) -> str:
    joined = "\n\n---\n\n".join(_serialize_result_for_consensus(item) for item in results)
    return (
        "You are MAGI Consensus.\n"
        "Given the user question and multiple model outputs, produce a concise final answer.\n"
        "You must weigh agreement and disagreement between models.\n"
        "Output format:\n"
        "- Conclusion: <one paragraph>\n"
        "- Evidence:\n"
        "  1) ...\n"
        "  2) ...\n"
        "- Confidence: <low|medium|high>\n\n"
        f"User question:\n{prompt}\n\n"
        f"Model outputs:\n{joined}\n"
    )


def _build_peer_vote_prompt(
    prompt: str,
    base_results: list[AgentResult],
    peer_answers: dict[str, str],
    agent_id: str,
    round_index: int,
    consensus_config: ConsensusConfig,
) -> str:
    base_section = "\n\n".join(_serialize_result_for_consensus(item) for item in base_results)
    peer_section = "\n".join(
        f"{key}: {value.strip() if value.strip() else '[NO ANSWER]'}" for key, value in sorted(peer_answers.items())
    )

    strict_rules = ""
    if consensus_config.debate_mode == "strict":
        min_criticisms = max(1, consensus_config.min_criticisms)
        strict_rules = (
            f"- criticisms must include at least {min_criticisms} concrete weaknesses from other agents.\n"
            "- each criticism should reference a specific weakness (logic gap, missing assumption, or evidence issue).\n"
            "- empty criticisms are invalid.\n"
        )

    return (
        f"You are Agent {agent_id} in MAGI deliberation round {round_index}.\n"
        "Task: read all answers, improve your own answer, and vote the best current answer.\n"
        "Return ONLY JSON with this schema:\n"
        '{"revised_answer":"...","preferred_agent":"A|B|C","reason":"...","confidence":0,"criticisms":["..."]}\n'
        "Rules:\n"
        "- revised_answer must be one concise final answer to the user question.\n"
        "- preferred_agent is the agent you think currently has the strongest final answer.\n"
        "- confidence is 0-100 integer.\n\n"
        f"{strict_rules}"
        f"User question:\n{prompt}\n\n"
        f"Initial model outputs:\n{base_section}\n\n"
        f"Current peer answers:\n{peer_section}\n"
    )


def _parse_deliberation_turn(
    text: str,
    agent_config: AgentConfig,
    latency_ms: int,
    consensus_config: ConsensusConfig,
) -> DeliberationTurn:
    stripped = text.strip()
    parsed: dict[str, Any] = {}

    if "{" in stripped and "}" in stripped:
        start = stripped.find("{")
        end = stripped.rfind("}")
        candidate = stripped[start : end + 1]
        try:
            parsed = json.loads(candidate)
        except Exception:  # noqa: BLE001
            parsed = {}

    revised_answer = str(parsed.get("revised_answer", "")).strip() if parsed else ""
    preferred_agent = parsed.get("preferred_agent") if parsed else None
    reason = str(parsed.get("reason", "")).strip() if parsed else ""
    confidence_raw = parsed.get("confidence") if parsed else None
    criticisms_raw = parsed.get("criticisms") if parsed else None

    confidence: int | None = None
    if isinstance(confidence_raw, (int, float)):
        confidence = max(0, min(100, int(confidence_raw)))

    if preferred_agent not in {"A", "B", "C"}:
        preferred_agent = None

    if not revised_answer:
        revised_answer = stripped

    criticisms: list[str] = []
    if isinstance(criticisms_raw, list):
        for item in criticisms_raw:
            if isinstance(item, str):
                value = item.strip()
                if value:
                    criticisms.append(value)

    if consensus_config.debate_mode == "strict":
        min_criticisms = max(1, consensus_config.min_criticisms)
        if len(criticisms) < min_criticisms:
            return DeliberationTurn(
                agent=agent_config.agent,
                provider=agent_config.provider,
                model=agent_config.model,
                status="ERROR",
                latency_ms=latency_ms,
                raw_text=stripped,
                revised_answer=revised_answer,
                error_message=f"strict debate requires at least {min_criticisms} criticisms",
            )

    return DeliberationTurn(
        agent=agent_config.agent,
        provider=agent_config.provider,
        model=agent_config.model,
        status="OK",
        latency_ms=latency_ms,
        raw_text=stripped,
        revised_answer=revised_answer,
        preferred_agent=preferred_agent,
        reason=reason or None,
        confidence=confidence,
        criticisms=criticisms,
    )


def _criticism_quality_score(criticisms: list[str]) -> int:
    # Favor specific critiques while capping impact to avoid overweighting verbosity.
    score = 0
    for item in criticisms:
        words = len(item.split())
        score += min(20, words)
    return min(60, score)


async def _call_model_text(
    full_model: str,
    prompt: str,
    timeout_seconds: int,
    max_tokens: int | None = None,
) -> tuple[str, int]:
    start = perf_counter()
    completion_args: dict[str, Any] = {
        "model": full_model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if max_tokens is not None:
        completion_args["max_tokens"] = max_tokens

    response = await asyncio.wait_for(
        acompletion(**completion_args),
        timeout=timeout_seconds,
    )
    text = _extract_text(response)
    latency_ms = int((perf_counter() - start) * 1000)
    return text, latency_ms


async def _run_single_agent(agent_config: AgentConfig, prompt: str, timeout_seconds: int) -> AgentResult:
    full_model = f"{agent_config.provider}/{agent_config.model}"
    start = perf_counter()
    print(f"[magi] agent={agent_config.agent} start model={full_model}")
    should_retry_on_timeout = agent_config.provider.strip().lower() in {"openai", "gemini"}
    max_attempts = 2 if should_retry_on_timeout else 1

    for attempt in range(1, max_attempts + 1):
        try:
            text, latency_ms = await _call_model_text(full_model, prompt, timeout_seconds)
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
            if attempt < max_attempts:
                print(f"[magi] agent={agent_config.agent} timeout attempt={attempt}; retrying once")
                continue
            print(f"[magi] agent={agent_config.agent} error=timeout")
            return AgentResult(
                agent=agent_config.agent,
                provider=agent_config.provider,
                model=agent_config.model,
                text="",
                status="ERROR",
                latency_ms=int((perf_counter() - start) * 1000),
                error_message="timeout",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[magi] agent={agent_config.agent} error={type(exc).__name__}: {exc}")
            return AgentResult(
                agent=agent_config.agent,
                provider=agent_config.provider,
                model=agent_config.model,
                text="",
                status="ERROR",
                latency_ms=int((perf_counter() - start) * 1000),
                error_message=_public_error_message(exc),
            )

    return AgentResult(
        agent=agent_config.agent,
        provider=agent_config.provider,
        model=agent_config.model,
        text="",
        status="ERROR",
        latency_ms=int((perf_counter() - start) * 1000),
        error_message="timeout",
    )


async def _run_deliberation_turn(
    agent_config: AgentConfig,
    prompt: str,
    base_results: list[AgentResult],
    peer_answers: dict[str, str],
    round_index: int,
    consensus_config: ConsensusConfig,
    timeout_seconds: int,
) -> DeliberationTurn:
    full_model = f"{agent_config.provider}/{agent_config.model}"
    start = perf_counter()
    print(f"[magi] deliberation round={round_index} agent={agent_config.agent} start")

    try:
        turn_prompt = _build_peer_vote_prompt(
            prompt,
            base_results,
            peer_answers,
            agent_config.agent,
            round_index,
            consensus_config,
        )
        text, latency_ms = await _call_model_text(full_model, turn_prompt, timeout_seconds)
        turn = _parse_deliberation_turn(text, agent_config, latency_ms, consensus_config)
        print(f"[magi] deliberation round={round_index} agent={agent_config.agent} success latency_ms={latency_ms}")
        return turn
    except asyncio.TimeoutError:
        print(f"[magi] deliberation round={round_index} agent={agent_config.agent} error=timeout")
        return DeliberationTurn(
            agent=agent_config.agent,
            provider=agent_config.provider,
            model=agent_config.model,
            status="ERROR",
            latency_ms=int((perf_counter() - start) * 1000),
            raw_text="",
            revised_answer="",
            error_message="timeout",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[magi] deliberation round={round_index} agent={agent_config.agent} error={type(exc).__name__}: {exc}")
        return DeliberationTurn(
            agent=agent_config.agent,
            provider=agent_config.provider,
            model=agent_config.model,
            status="ERROR",
            latency_ms=int((perf_counter() - start) * 1000),
            raw_text="",
            revised_answer="",
            error_message=_public_error_message(exc),
        )


async def _run_single_model_consensus(
    consensus_config: ConsensusConfig,
    prompt: str,
    results: list[AgentResult],
    timeout_seconds: int,
) -> ConsensusResult:
    ok_results = [item for item in results if item.status == "OK" and item.text.strip()]
    if len(ok_results) < consensus_config.min_ok_results:
        return ConsensusResult(
            provider=consensus_config.provider or "-",
            model=consensus_config.model or "-",
            text="",
            status="ERROR",
            latency_ms=0,
            error_message=f"consensus needs at least {consensus_config.min_ok_results} successful results",
        )

    if not consensus_config.provider or not consensus_config.model:
        return ConsensusResult(
            provider="-",
            model="-",
            text="",
            status="ERROR",
            latency_ms=0,
            error_message="single_model consensus requires provider/model",
        )

    full_model = f"{consensus_config.provider}/{consensus_config.model}"
    print(f"[magi] consensus(single_model) start model={full_model}")

    try:
        text, latency_ms = await _call_model_text(
            full_model,
            _build_single_model_consensus_prompt(prompt, results),
            timeout_seconds,
        )
        print(f"[magi] consensus(single_model) success latency_ms={latency_ms}")
        return ConsensusResult(
            provider=consensus_config.provider,
            model=consensus_config.model,
            text=text,
            status="OK",
            latency_ms=latency_ms,
        )
    except asyncio.TimeoutError:
        return ConsensusResult(
            provider=consensus_config.provider,
            model=consensus_config.model,
            text="",
            status="ERROR",
            latency_ms=int(timeout_seconds * 1000),
            error_message="timeout",
        )
    except Exception as exc:  # noqa: BLE001
        return ConsensusResult(
            provider=consensus_config.provider,
            model=consensus_config.model,
            text="",
            status="ERROR",
            latency_ms=0,
            error_message=_public_error_message(exc),
        )


async def _run_peer_vote_consensus(
    consensus_config: ConsensusConfig,
    prompt: str,
    results: list[AgentResult],
    agents: list[AgentConfig],
    timeout_seconds: int,
) -> ConsensusResult:
    start = perf_counter()
    ok_results = [item for item in results if item.status == "OK" and item.text.strip()]
    if len(ok_results) < consensus_config.min_ok_results:
        return ConsensusResult(
            provider="magi",
            model="peer_vote_v1",
            text="",
            status="ERROR",
            latency_ms=0,
            error_message=f"consensus needs at least {consensus_config.min_ok_results} successful results",
        )

    peer_answers = {item.agent: item.text for item in results if item.status == "OK" and item.text.strip()}
    rounds = max(1, consensus_config.rounds)
    last_turns: list[DeliberationTurn] = []

    for round_idx in range(1, rounds + 1):
        tasks = [
            _run_deliberation_turn(agent, prompt, results, peer_answers, round_idx, consensus_config, timeout_seconds)
            for agent in agents
        ]
        turns = await asyncio.gather(*tasks)
        last_turns = turns

        for turn in turns:
            if turn.status == "OK" and turn.revised_answer.strip():
                peer_answers[turn.agent] = turn.revised_answer.strip()

    valid_votes = [
        turn
        for turn in last_turns
        if turn.status == "OK" and turn.preferred_agent in {"A", "B", "C"}
    ]

    if len(valid_votes) < consensus_config.min_ok_results:
        return ConsensusResult(
            provider="magi",
            model="peer_vote_v1",
            text="",
            status="ERROR",
            latency_ms=int((perf_counter() - start) * 1000),
            error_message="insufficient valid deliberation votes",
        )

    vote_count: dict[str, int] = defaultdict(int)
    vote_conf: dict[str, int] = defaultdict(int)
    vote_crit: dict[str, int] = defaultdict(int)
    for turn in valid_votes:
        assert turn.preferred_agent is not None
        vote_count[turn.preferred_agent] += 1
        vote_conf[turn.preferred_agent] += turn.confidence if turn.confidence is not None else 50
        vote_crit[turn.preferred_agent] += _criticism_quality_score(turn.criticisms)

    ranked = sorted(
        vote_count.keys(),
        key=lambda key: (vote_count[key], vote_conf[key], vote_crit[key], key),
        reverse=True,
    )
    winner = ranked[0]

    winner_answer = peer_answers.get(winner, "")
    if not winner_answer:
        winner_turn = next((turn for turn in last_turns if turn.agent == winner and turn.revised_answer.strip()), None)
        winner_answer = winner_turn.revised_answer if winner_turn else ""

    vote_lines = []
    for turn in last_turns:
        if turn.status == "OK":
            vote_lines.append(
                f"- Agent {turn.agent} voted {turn.preferred_agent or '?'}"
                f" (confidence={turn.confidence if turn.confidence is not None else 'n/a'}): "
                f"{turn.reason or 'no reason'}"
            )
            if turn.criticisms:
                vote_lines.append("  criticisms: " + " | ".join(turn.criticisms))
        else:
            vote_lines.append(f"- Agent {turn.agent} failed in deliberation: {turn.error_message or 'unknown error'}")

    text = (
        f"Consensus winner: Agent {winner} "
        f"({vote_count[winner]}/{len(last_turns)} votes, total_confidence={vote_conf[winner]}, "
        f"critique_score={vote_crit[winner]}).\n\n"
        f"Final answer:\n{winner_answer}\n\n"
        "Vote details:\n"
        + "\n".join(vote_lines)
    )

    return ConsensusResult(
        provider="magi",
        model=f"peer_vote_r{rounds}",
        text=text,
        status="OK",
        latency_ms=int((perf_counter() - start) * 1000),
    )


async def _run_consensus(profile: ProfileConfig, prompt: str, results: list[AgentResult]) -> ConsensusResult:
    if profile.consensus.strategy == "single_model":
        return await _run_single_model_consensus(profile.consensus, prompt, results, profile.timeout_seconds)

    return await _run_peer_vote_consensus(
        profile.consensus,
        prompt,
        results,
        profile.agents,
        profile.timeout_seconds,
    )


def _build_pending_consensus(profile: ProfileConfig) -> ConsensusResult:
    if profile.consensus.strategy == "single_model":
        provider = profile.consensus.provider or "magi"
        model = profile.consensus.model or "single_model"
    else:
        provider = "magi"
        model = f"peer_vote_r{max(1, profile.consensus.rounds)}"
    return ConsensusResult(
        provider=provider,
        model=model,
        text="",
        status="LOADING",
        latency_ms=0,
        error_message=None,
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/magi/profiles", response_model=ProfilesResponse)
async def list_profiles() -> ProfilesResponse:
    try:
        config = load_config()
        profile_key, _ = _resolve_profile(config, None)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"invalid config: {exc}") from exc

    if config.profiles:
        profile_keys = sorted(config.profiles.keys())
        profile_agents = {key: config.profiles[key].agents for key in profile_keys}
        return ProfilesResponse(
            default_profile=config.default_profile or profile_key,
            profiles=profile_keys,
            profile_agents=profile_agents,
        )

    fallback_agents: list[AgentConfig] = []
    if config.agents:
        fallback_agents = config.agents
    return ProfilesResponse(
        default_profile=profile_key,
        profiles=[profile_key],
        profile_agents={profile_key: fallback_agents},
    )


@app.get("/api/magi/history", response_model=HistoryListResponse)
async def list_history(limit: int = 20, offset: int = 0) -> HistoryListResponse:
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be 0 or greater")
    try:
        return _list_history(limit=limit, offset=offset)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"failed to load history: {exc}") from exc


@app.get("/api/magi/history/{run_id}", response_model=HistoryItem)
async def get_history_item(run_id: str) -> HistoryItem:
    try:
        item = _get_history_item(run_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"failed to load history: {exc}") from exc
    if item is None:
        raise HTTPException(status_code=404, detail="run not found")
    return item


@app.delete("/api/magi/history/thread/{thread_id}")
async def delete_thread_history(thread_id: str) -> dict[str, int | str]:
    try:
        deleted = _delete_thread_history(thread_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"failed to delete thread history: {exc}") from exc
    if deleted == 0:
        raise HTTPException(status_code=404, detail="thread not found")
    return {"thread_id": thread_id, "deleted_runs": deleted}


@app.post("/api/magi/routing/feedback", response_model=RoutingFeedbackResponse)
async def submit_routing_feedback(payload: RoutingFeedbackRequest) -> RoutingFeedbackResponse:
    normalized_thread_id = _normalize_thread_id(payload.thread_id)
    if not normalized_thread_id:
        raise HTTPException(status_code=400, detail="thread_id is required")
    request_id = payload.request_id.strip()
    if not request_id:
        raise HTTPException(status_code=400, detail="request_id is required")
    reason = payload.reason.strip() if payload.reason else None
    try:
        config = load_config()
        policy_key = _update_routing_event_feedback(
            normalized_thread_id,
            request_id,
            int(payload.rating),
            reason,
            config,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"failed to persist routing feedback: {exc}") from exc
    if policy_key is None:
        raise HTTPException(status_code=404, detail="routing event not found")
    return RoutingFeedbackResponse(
        thread_id=normalized_thread_id,
        request_id=request_id,
        rating=int(payload.rating),
        policy_key=policy_key,
    )


@app.get("/api/magi/routing/policy", response_model=RoutingPolicyResponse)
async def get_routing_policy(key: str) -> RoutingPolicyResponse:
    normalized = key.strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="key is required")
    try:
        policy = _get_routing_policy(normalized)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"failed to load routing policy: {exc}") from exc
    return RoutingPolicyResponse(
        key=normalized,
        weights={name: float(value) for name, value in policy["weights"].items()},
        stats={
            "n": int(policy["stats"].get("n", 0)),
            "avg_reward": float(policy["stats"].get("avg_reward", 0.0)),
        },
        updated_at=policy.get("updated_at"),
    )


@app.get("/api/magi/routing/events", response_model=RoutingEventsResponse)
async def get_routing_events(thread_id: str | None = None, limit: int = 20) -> RoutingEventsResponse:
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100")
    normalized_thread_id = _normalize_thread_id(thread_id) if thread_id is not None else None
    try:
        return _list_routing_events(normalized_thread_id, limit)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"failed to load routing events: {exc}") from exc


@app.post("/api/magi/run", response_model=RunResponse)
async def run_magi(payload: RunRequest) -> RunResponse:
    prompt = _validate_prompt(payload.prompt)
    fresh_mode = _resolve_fresh_mode(prompt, payload.fresh_mode)
    thread_id = _normalize_thread_id(payload.thread_id) or str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    turn_index = _next_turn_index(thread_id)
    profile_name = payload.profile or ""
    profile: ProfileConfig | None = None
    router_input: dict[str, Any] = {}
    router_output: dict[str, Any] = {}
    routing_event_saved = False

    try:
        config = load_config()
        profile_name, profile, router_input, router_output = await _resolve_profile_with_router_trace(config, payload.profile, prompt)
        _save_routing_event(thread_id, run_id, router_input, router_output)
        routing_event_saved = True
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"invalid config: {exc}") from exc

    try:
        effective_prompt = await _build_effective_prompt(prompt, fresh_mode)
        if _should_apply_thread_context(profile_name):
            effective_prompt = _build_prompt_with_thread_context(effective_prompt, thread_id)
        else:
            print(f"[magi] thread_context skipped profile={profile_name}")
        if _should_apply_history_context(profile_name):
            effective_prompt = await _build_prompt_with_history(prompt, effective_prompt, config)
        else:
            print(f"[magi] history_context skipped profile={profile_name}")
        tasks = [_run_single_agent(agent, effective_prompt, profile.timeout_seconds) for agent in profile.agents]
        results = await asyncio.gather(*tasks)
        if profile_name == "local_only":
            results = [_postprocess_local_only_result(prompt, item) for item in results]
        consensus = _build_local_only_consensus(results) if profile_name == "local_only" else _build_pending_consensus(profile)
        try:
            _save_run_history(
                run_id,
                profile_name,
                prompt,
                results,
                consensus,
                config,
                thread_id=thread_id,
                turn_index=turn_index,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"failed to persist history: {exc}") from exc

        if routing_event_saved:
            execution_result = _build_execution_result_snapshot(profile_name, results, consensus=consensus)
            _update_routing_event_execution(run_id, execution_result, config)
    except HTTPException:
        if routing_event_saved:
            failure_result = _build_execution_result_snapshot(
                profile_name or "unknown",
                [],
                failure_message="request failed",
            )
            _update_routing_event_execution(run_id, failure_result, config)
        raise
    except Exception as exc:  # noqa: BLE001
        if routing_event_saved:
            failure_result = _build_execution_result_snapshot(
                profile_name or "unknown",
                [],
                failure_message=f"{type(exc).__name__}: {exc}",
            )
            _update_routing_event_execution(run_id, failure_result, config)
        raise

    return RunResponse(
        run_id=run_id,
        thread_id=thread_id,
        turn_index=turn_index,
        profile=profile_name,
        results=results,
        consensus=consensus,
        routing=RoutingDecisionInfo(
            profile=profile_name,
            reason=_build_routing_reason_display(
                profile_name,
                router_input,
                _safe_str(router_output.get("reason")) or None,
            ),
            intent=_safe_str(router_input.get("intent")) or None,
            complexity=_safe_str(router_input.get("complexity")) or None,
            safety=_safe_str(router_input.get("safety")) or None,
            execution_tier=_safe_str(router_input.get("execution_tier")) or None,
            policy_key=_safe_str(router_output.get("policy_key")) or None,
        ),
    )


@app.post("/api/magi/retry", response_model=RetryResponse)
async def retry_agent(payload: RetryRequest) -> RetryResponse:
    prompt = _validate_prompt(payload.prompt)
    fresh_mode = _resolve_fresh_mode(prompt, payload.fresh_mode)
    thread_id = _normalize_thread_id(payload.thread_id) or str(uuid.uuid4())

    try:
        config = load_config()
        profile_name, profile = _resolve_profile(config, payload.profile)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"invalid config: {exc}") from exc

    target_agent = next((agent for agent in profile.agents if agent.agent == payload.agent), None)
    if target_agent is None:
        raise HTTPException(status_code=400, detail=f"agent {payload.agent} is not configured")

    effective_prompt = await _build_effective_prompt(prompt, fresh_mode)
    if _should_apply_thread_context(profile_name):
        effective_prompt = _build_prompt_with_thread_context(effective_prompt, thread_id)
    else:
        print(f"[magi] thread_context skipped profile={profile_name}")
    result = await _run_single_agent(target_agent, effective_prompt, profile.timeout_seconds)
    if profile_name == "local_only":
        result = _postprocess_local_only_result(prompt, result)
    return RetryResponse(run_id=str(uuid.uuid4()), thread_id=thread_id, turn_index=0, profile=profile_name, result=result)


@app.post("/api/magi/consensus", response_model=ConsensusResponse)
async def recalc_consensus(payload: ConsensusRequest) -> ConsensusResponse:
    prompt = _validate_prompt(payload.prompt)
    fresh_mode = _resolve_fresh_mode(prompt, payload.fresh_mode)
    requested_run_id = payload.run_id.strip() if payload.run_id else None
    if requested_run_id == "":
        requested_run_id = None

    stored_thread_turn = _get_run_thread_turn(requested_run_id) if requested_run_id else None
    thread_id = (
        _normalize_thread_id(payload.thread_id)
        or (stored_thread_turn[0] if stored_thread_turn else None)
        or str(uuid.uuid4())
    )
    turn_index = stored_thread_turn[1] if stored_thread_turn else 0

    try:
        config = load_config()
        profile_name, profile = _resolve_profile(config, payload.profile)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"invalid config: {exc}") from exc

    effective_prompt = await _build_effective_prompt(prompt, fresh_mode)
    if _should_apply_thread_context(profile_name):
        effective_prompt = _build_prompt_with_thread_context(effective_prompt, thread_id)
    else:
        print(f"[magi] thread_context skipped profile={profile_name}")
    if profile_name == "local_only":
        consensus = _build_local_only_consensus(payload.results)
    else:
        consensus = await _run_consensus(profile, effective_prompt, payload.results)
    response_run_id = requested_run_id or str(uuid.uuid4())
    if requested_run_id and not _update_run_consensus(requested_run_id, consensus):
        raise HTTPException(status_code=404, detail="run not found")
    if requested_run_id:
        execution_result = _build_execution_result_snapshot(profile_name, payload.results, consensus=consensus)
        _update_routing_event_execution(requested_run_id, execution_result, config)

    return ConsensusResponse(
        run_id=response_run_id,
        thread_id=thread_id,
        turn_index=turn_index,
        profile=profile_name,
        consensus=consensus,
    )
