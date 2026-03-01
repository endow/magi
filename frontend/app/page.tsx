"use client";

import { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import ChamberVisualization from "./components/chamber-visualization";
import ChatTranscript from "./components/chat-transcript";
import FeedbackPanel from "./components/feedback-panel";
import PromptForm from "./components/prompt-form";
import RoutingInfoPanel from "./components/routing-info-panel";
import RunMetaBar from "./components/run-meta-bar";
import ThreadSidebar from "./components/thread-sidebar";

type AgentId = "A" | "B" | "C";
type AgentStatus = "OK" | "ERROR" | "LOADING";
type NodeState = "IDLE" | "BLINK" | "ON" | "ERROR" | "RELAY";
type ConfidenceMap = Record<AgentId, number | null>;

type AgentResult = {
  agent: AgentId;
  provider: string;
  model: string;
  text: string;
  status: AgentStatus;
  latency_ms: number;
  error_message?: string | null;
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
  total_tokens?: number | null;
  cost_estimate_usd?: number | null;
};

type RoutingInfo = {
  profile: string;
  reason?: string | null;
  intent?: string | null;
  complexity?: string | null;
  safety?: string | null;
  execution_tier?: string | null;
  policy_key?: string | null;
};

type ChatResponse = {
  run_id: string;
  thread_id: string;
  turn_index: number;
  profile: string;
  reply: string;
  results: AgentResult[];
  consensus: ConsensusResult;
  routing?: RoutingInfo | null;
};

type RunHistoryItem = {
  run_id: string;
  thread_id: string;
  turn_index: number;
  profile: string;
  prompt: string;
  results: AgentResult[];
  consensus: ConsensusResult | null;
  created_at: string;
};

type ConsensusResult = {
  provider: string;
  model: string;
  text: string;
  status: "OK" | "ERROR" | "LOADING";
  latency_ms: number;
  error_message?: string | null;
  error_code?: string | null;
};

type ProfilesResponse = {
  default_profile: string;
  profiles: string[];
  profile_agents: Record<string, Array<{ agent: AgentId; provider: string; model: string }>>;
};

type HistoryListResponse = {
  total: number;
  items: RunHistoryItem[];
};

type RoutingFeedbackResponse = {
  thread_id: string;
  request_id: string;
  rating: number;
  policy_key?: string | null;
};

type ThreadGroup = {
  thread_id: string;
  latest_at: string;
  turns: RunHistoryItem[];
};

type ThreadNameMap = Record<string, string>;
type ThreadCollapsedMap = Record<string, boolean>;
type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  run_id?: string;
  turn_index?: number;
  latency_ms?: number;
};

const RAW_API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const MAX_PROMPT_LENGTH = 4000;
const PHASE_VERSION = "v1.3";
const DEFAULT_OLLAMA_SYSTEM_PROMPT = "あなたは「MAGI」という名前のAIエージェントです。";

function resolveApiBaseUrl(): string {
  if (typeof window === "undefined") return RAW_API_BASE_URL;
  try {
    const configured = new URL(RAW_API_BASE_URL);
    const isLocalConfigured = configured.hostname === "localhost" || configured.hostname === "127.0.0.1";
    if (isLocalConfigured && window.location.hostname === "host.docker.internal") {
      configured.hostname = "host.docker.internal";
      return configured.toString().replace(/\/$/, "");
    }
  } catch {
    return RAW_API_BASE_URL;
  }
  return RAW_API_BASE_URL;
}
const loadingResults: AgentResult[] = [
  { agent: "A", provider: "-", model: "-", text: "Loading...", status: "LOADING", latency_ms: 0 },
  { agent: "B", provider: "-", model: "-", text: "Loading...", status: "LOADING", latency_ms: 0 },
  { agent: "C", provider: "-", model: "-", text: "Loading...", status: "LOADING", latency_ms: 0 }
];

function parseConfidenceMap(text: string | null | undefined): ConfidenceMap {
  const map: ConfidenceMap = { A: null, B: null, C: null };
  if (!text) return map;
  const pattern = /Agent ([ABC]).*?\(confidence=(\d+|n\/a)\)/g;
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(text)) !== null) {
    const agent = match[1] as AgentId;
    const raw = match[2];
    if (raw.toLowerCase() !== "n/a") {
      map[agent] = Math.max(0, Math.min(100, Number.parseInt(raw, 10)));
    }
  }
  return map;
}

function parseWinnerAgent(text: string | null | undefined): AgentId | null {
  if (!text) return null;
  const match = text.match(/Consensus winner:\s*Agent\s*([ABC])/i);
  if (!match) return null;
  const value = match[1].toUpperCase();
  if (value === "A" || value === "B" || value === "C") return value;
  return null;
}

function extractUnifiedReply(consensus: ConsensusResult | null | undefined, results: AgentResult[]): string {
  if (consensus?.text) {
    const marker = /\bFinal answer:\s*/i;
    const match = marker.exec(consensus.text);
    if (match) {
      const tail = consensus.text.slice(match.index + match[0].length);
      const voteMatch = /\n\s*Vote details:\s*/i.exec(tail);
      const body = voteMatch ? tail.slice(0, voteMatch.index) : tail;
      const cleaned = body.trim();
      if (cleaned) return cleaned;
    }
    const fallback = consensus.text.trim();
    if (fallback) return fallback;
  }
  const firstOk = results.find((item) => item.status === "OK" && item.text.trim());
  return firstOk?.text.trim() ?? "";
}

function resolveAssistantLatency(consensus: ConsensusResult | null | undefined, results: AgentResult[]): number {
  if (consensus && typeof consensus.latency_ms === "number" && consensus.latency_ms > 0) {
    return consensus.latency_ms;
  }
  return Math.max(0, ...results.map((item) => item.latency_ms || 0));
}

function formatElapsedMsToMinSec(elapsedMs: number): string {
  const totalSeconds = Math.max(0, Math.round(elapsedMs / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}m ${seconds}s`;
}

function buildConfiguredLoadingCards(
  profileAgents: ProfilesResponse["profile_agents"],
  selectedProfile: string
): AgentResult[] {
  if (!selectedProfile) return loadingResults;
  const configured = profileAgents[selectedProfile] ?? [];
  if (!configured.length) return loadingResults;
  const order: AgentId[] = ["A", "B", "C"];
  const byAgent = new Map(configured.map((item) => [item.agent, item]));
  const items = order
    .map((agent) => byAgent.get(agent))
    .filter((item): item is { agent: AgentId; provider: string; model: string } => Boolean(item))
    .map(
      (item): AgentResult => ({
        agent: item.agent,
        provider: item.provider,
        model: item.model,
        text: "Loading...",
        status: "LOADING",
        latency_ms: 0
      })
    );
  return items.length ? items : loadingResults;
}

export default function HomePage() {
  const apiBaseUrl = useMemo(() => resolveApiBaseUrl(), []);
  const [prompt, setPrompt] = useState("");
  const [selectedProfile, setSelectedProfile] = useState("");
  const [defaultProfile, setDefaultProfile] = useState("");
  const [resolvedProfile, setResolvedProfile] = useState("");
  const [availableProfiles, setAvailableProfiles] = useState<string[]>([]);
  const [profileAgents, setProfileAgents] = useState<ProfilesResponse["profile_agents"]>({});
  const [results, setResults] = useState<AgentResult[]>([]);
  const [consensus, setConsensus] = useState<ConsensusResult | null>(null);
  const [history, setHistory] = useState<RunHistoryItem[]>([]);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [runId, setRunId] = useState("");
  const [threadId, setThreadId] = useState("");
  const [turnIndex, setTurnIndex] = useState(0);
  const [routingInfo, setRoutingInfo] = useState<RoutingInfo | null>(null);
  const [feedbackRating, setFeedbackRating] = useState<-1 | 0 | 1 | null>(null);
  const [feedbackReason, setFeedbackReason] = useState("");
  const [feedbackSubmitting, setFeedbackSubmitting] = useState(false);
  const [feedbackMessage, setFeedbackMessage] = useState("");
  const [threadNames, setThreadNames] = useState<ThreadNameMap>({});
  const [collapsedThreads, setCollapsedThreads] = useState<ThreadCollapsedMap>({});
  const [editingThreadId, setEditingThreadId] = useState<string | null>(null);
  const [threadNameDraft, setThreadNameDraft] = useState("");
  const [confirmDeleteThreadId, setConfirmDeleteThreadId] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isConsensusLoading, setIsConsensusLoading] = useState(false);
  const [nodeStates, setNodeStates] = useState<Record<AgentId, NodeState>>({
    A: "IDLE",
    B: "IDLE",
    C: "IDLE"
  });
  const [localNodeState, setLocalNodeState] = useState<NodeState>("IDLE");
  const [showConclusion, setShowConclusion] = useState(false);
  const [conclusionElapsedMs, setConclusionElapsedMs] = useState<number | null>(null);
  const [loadingElapsedMs, setLoadingElapsedMs] = useState<number>(0);
  const [freshMode, setFreshMode] = useState(false);
  const [ollamaSystemPrompt, setOllamaSystemPrompt] = useState(DEFAULT_OLLAMA_SYSTEM_PROMPT);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settingsDraft, setSettingsDraft] = useState(DEFAULT_OLLAMA_SYSTEM_PROMPT);
  const [isDeletingAllThreads, setIsDeletingAllThreads] = useState(false);
  const isBusy = isLoading || isConsensusLoading;
  const isStrictDebate = selectedProfile === "performance" || selectedProfile === "ultra";
  const isUltra = selectedProfile === "ultra";
  const chamberRef = useRef<HTMLDivElement | null>(null);
  const localNodeRef = useRef<HTMLDivElement | null>(null);
  const peerGroupRef = useRef<HTMLDivElement | null>(null);
  const nodeRefs = useRef<Record<AgentId, HTMLDivElement | null>>({ A: null, B: null, C: null });
  const [linkPaths, setLinkPaths] = useState<string[]>([]);
  const [linkViewBox, setLinkViewBox] = useState("0 0 100 100");
  const nodeTimerRefs = useRef<number[]>([]);
  const consensusClockRef = useRef<number | null>(null);

  function markConsensusClockStart() {
    consensusClockRef.current = typeof performance !== "undefined" ? performance.now() : Date.now();
    setConclusionElapsedMs(null);
  }

  function markConsensusClockEnd() {
    if (consensusClockRef.current === null) return;
    const end = typeof performance !== "undefined" ? performance.now() : Date.now();
    setConclusionElapsedMs(Math.max(0, Math.round(end - consensusClockRef.current)));
    consensusClockRef.current = null;
  }

  async function fetchHistory() {
    try {
      const response = await fetch(`${apiBaseUrl}/api/magi/history?limit=20&offset=0`);
      if (!response.ok) return;
      const data = (await response.json()) as HistoryListResponse;
      setHistory(data.items);
    } catch {
      // ignore history load errors
    }
  }

  const cards = useMemo(() => {
    if (isLoading) return buildConfiguredLoadingCards(profileAgents, selectedProfile);
    if (!results.length) return [];
    const order: AgentId[] = ["A", "B", "C"];
    const byAgent = new Map(results.map((item) => [item.agent, item]));
    return order.map((agent) => byAgent.get(agent)).filter((item): item is AgentResult => Boolean(item));
  }, [isLoading, profileAgents, results, selectedProfile]);
  const displayNodes = useMemo(() => {
    if (cards.length) return cards;
    return buildConfiguredLoadingCards(profileAgents, selectedProfile);
  }, [cards, profileAgents, selectedProfile]);
  const localAgent = useMemo(() => (profileAgents.local_only ?? [])[0], [profileAgents]);
  const downstreamProfile = useMemo(() => {
    if (resolvedProfile && resolvedProfile !== "local_only") return resolvedProfile;
    if (selectedProfile && selectedProfile !== "local_only") return selectedProfile;
    return "";
  }, [resolvedProfile, selectedProfile]);
  const chamberNodes = useMemo(() => {
    if (!downstreamProfile) return buildConfiguredLoadingCards(profileAgents, "");
    const base = buildConfiguredLoadingCards(profileAgents, downstreamProfile);
    if (isLoading) return base;
    if (resolvedProfile === "local_only") return base;
    if (!results.length) return base;
    const byAgent = new Map(results.map((item) => [item.agent, item]));
    return base.map((node) => byAgent.get(node.agent) ?? node);
  }, [downstreamProfile, isLoading, profileAgents, resolvedProfile, results]);
  const localRouteHint = useMemo(() => {
    if (isLoading && !resolvedProfile) {
      if (loadingElapsedMs < 7000) return "routing + context prep";
      return "waiting provider responses";
    }
    if (isLoading && resolvedProfile && resolvedProfile !== "local_only") return `dispatching to ${resolvedProfile}`;
    if (isConsensusLoading) return "consensus in progress";
    if (localNodeState === "BLINK") return "classifying";
    if (resolvedProfile === "local_only") return "handled local_only";
    if (resolvedProfile) return `routed to ${resolvedProfile}`;
    return "pre-routing";
  }, [isConsensusLoading, isLoading, loadingElapsedMs, localNodeState, resolvedProfile]);
  const localOnlyHandled = useMemo(
    () => resolvedProfile === "local_only" && (results.length > 0 || consensus?.status === "OK"),
    [consensus?.status, resolvedProfile, results.length]
  );
  const confidenceMap = useMemo(
    () => parseConfidenceMap(consensus?.status === "OK" ? consensus.text : ""),
    [consensus]
  );
  const winnerAgent = useMemo(
    () => parseWinnerAgent(consensus?.status === "OK" ? consensus.text : ""),
    [consensus]
  );
  const totalTokens = useMemo(() => {
    if (!results.length) return null;
    const known = results
      .map((item) => (typeof item.total_tokens === "number" ? item.total_tokens : null))
      .filter((value): value is number => value !== null);
    if (!known.length) return null;
    return known.reduce((sum, value) => sum + value, 0);
  }, [results]);
  const totalCostUsd = useMemo(() => {
    if (!results.length) return null;
    const known = results
      .map((item) => (typeof item.cost_estimate_usd === "number" ? item.cost_estimate_usd : null))
      .filter((value): value is number => value !== null);
    if (!known.length) return null;
    return known.reduce((sum, value) => sum + value, 0);
  }, [results]);
  const showDiscussionBadge = useMemo(
    () =>
      !showConclusion &&
      resolvedProfile !== "" &&
      resolvedProfile !== "local_only" &&
      (isConsensusLoading || consensus?.status === "LOADING"),
    [consensus?.status, isConsensusLoading, resolvedProfile, showConclusion]
  );
  const showRoutingBadge = useMemo(
    () => !showConclusion && isLoading && loadingElapsedMs < 7000,
    [isLoading, loadingElapsedMs, showConclusion]
  );
  const showExecutingBadge = useMemo(
    () =>
      !showConclusion &&
      isLoading &&
      loadingElapsedMs >= 7000 &&
      Boolean(resolvedProfile),
    [isLoading, loadingElapsedMs, resolvedProfile, showConclusion]
  );
  const threadGroups = useMemo(() => {
    const byThread = new Map<string, RunHistoryItem[]>();
    for (const item of history) {
      const key = item.thread_id || item.run_id;
      const list = byThread.get(key) ?? [];
      list.push(item);
      byThread.set(key, list);
    }

    const groups: ThreadGroup[] = Array.from(byThread.entries()).map(([key, items]) => {
      const sortedTurns = [...items].sort((a, b) => {
        if (b.turn_index !== a.turn_index) return b.turn_index - a.turn_index;
        return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
      });
      return {
        thread_id: key,
        latest_at: sortedTurns[0]?.created_at ?? "",
        turns: sortedTurns
      };
    });

    return groups.sort((a, b) => new Date(b.latest_at).getTime() - new Date(a.latest_at).getTime());
  }, [history]);
  const chamberActive = localNodeState === "RELAY" || Object.values(nodeStates).some((state) => state === "BLINK");
  const chamberMode = useMemo(() => {
    if (showConclusion) return "conclusion";
    if (showDiscussionBadge || chamberActive) return "discussion";
    return "idle";
  }, [chamberActive, showConclusion, showDiscussionBadge]);

  function clearNodeTimers() {
    nodeTimerRefs.current.forEach((id) => window.clearTimeout(id));
    nodeTimerRefs.current = [];
  }

  function nodeStatesFromResults(items: AgentResult[]): Record<AgentId, NodeState> {
    const base: Record<AgentId, NodeState> = { A: "IDLE", B: "IDLE", C: "IDLE" };
    for (const item of items) {
      base[item.agent] = item.status === "OK" ? "ON" : item.status === "LOADING" ? "BLINK" : "ERROR";
    }
    return base;
  }

  function runNodeTransition(
    profile: string,
    items: AgentResult[],
    consensusStatus: "OK" | "ERROR" | "LOADING"
  ) {
    clearNodeTimers();
    setShowConclusion(false);
    setLocalNodeState("BLINK");
    setNodeStates({ A: "IDLE", B: "IDLE", C: "IDLE" });

    const localTimer = window.setTimeout(() => {
      if (profile === "local_only") {
        setLocalNodeState("ON");
        setShowConclusion(false);
        return;
      }
      setLocalNodeState("RELAY");
      setNodeStates({ A: "BLINK", B: "BLINK", C: "BLINK" });
    }, 240);
    nodeTimerRefs.current.push(localTimer);

    if (profile === "local_only") return;

    const maxLatency = Math.max(...items.map((item) => Math.max(1, item.latency_ms)), 1);
    const maxDelay = 1800;
    let doneAt = 240;

    for (const item of items) {
      const delay = Math.max(260, Math.round((item.latency_ms / maxLatency) * maxDelay));
      doneAt = Math.max(doneAt, delay);
      const timer = window.setTimeout(() => {
        setNodeStates((prev) => ({
          ...prev,
          [item.agent]: item.status === "OK" ? "ON" : "ERROR"
        }));
      }, delay);
      nodeTimerRefs.current.push(timer);
    }

    const endTimer = window.setTimeout(() => {
      setLocalNodeState(consensusStatus === "LOADING" ? "RELAY" : "ON");
      setShowConclusion(consensusStatus === "OK" && profile !== "local_only");
    }, doneAt + 200);
    nodeTimerRefs.current.push(endTimer);
  }

  useEffect(() => {
    async function loadProfiles() {
      try {
        const response = await fetch(`${apiBaseUrl}/api/magi/profiles`);
        if (!response.ok) return;
        const data = (await response.json()) as ProfilesResponse;
        if (!data.profiles.length) return;
        setAvailableProfiles(data.profiles);
        setDefaultProfile(data.default_profile || "");
        setSelectedProfile("");
        setProfileAgents(data.profile_agents ?? {});
      } catch {
        // ignore and fallback to default
      }
    }

    void loadProfiles();
  }, []);

  useEffect(() => {
    void fetchHistory();
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const rawNames = window.localStorage.getItem("magi_thread_names");
      const rawCollapsed = window.localStorage.getItem("magi_thread_collapsed");
      const rawSystemPrompt = window.localStorage.getItem("magi_ollama_system_prompt");
      if (rawNames) setThreadNames(JSON.parse(rawNames) as ThreadNameMap);
      if (rawCollapsed) setCollapsedThreads(JSON.parse(rawCollapsed) as ThreadCollapsedMap);
      if (rawSystemPrompt && rawSystemPrompt.trim()) {
        setOllamaSystemPrompt(rawSystemPrompt);
        setSettingsDraft(rawSystemPrompt);
      }
    } catch {
      // ignore storage parse error
    }
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem("magi_thread_names", JSON.stringify(threadNames));
  }, [threadNames]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem("magi_thread_collapsed", JSON.stringify(collapsedThreads));
  }, [collapsedThreads]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem("magi_ollama_system_prompt", ollamaSystemPrompt);
  }, [ollamaSystemPrompt]);

  useEffect(() => {
    return () => {
      clearNodeTimers();
    };
  }, []);

  useEffect(() => {
    if (!isLoading) {
      setLoadingElapsedMs(0);
      return;
    }
    const startedAt = Date.now();
    const id = window.setInterval(() => {
      setLoadingElapsedMs(Date.now() - startedAt);
    }, 250);
    return () => {
      window.clearInterval(id);
    };
  }, [isLoading]);

  useEffect(() => {
    function updateLinks() {
      const chamber = chamberRef.current;
      const nodeA = nodeRefs.current.A;
      const nodeB = nodeRefs.current.B;
      const nodeC = nodeRefs.current.C;
      if (!chamber || !nodeA || !nodeB || !nodeC) return;

      const chamberRect = chamber.getBoundingClientRect();
      if (window.innerWidth < 768) {
        setLinkPaths([]);
        return;
      }

      function centerOf(el: HTMLDivElement) {
        const rect = el.getBoundingClientRect();
        return {
          rect,
          x: rect.left - chamberRect.left + rect.width / 2,
          y: rect.top - chamberRect.top + rect.height / 2
        };
      }

      const a = centerOf(nodeA);
      const b = centerOf(nodeB);
      const c = centerOf(nodeC);

      const aTop = { x: a.x, y: a.y - a.rect.height / 2 };
      const cTop = { x: c.x, y: c.y - c.rect.height / 2 };
      const bLeft = { x: b.x - b.rect.width / 2, y: b.y };
      const bRight = { x: b.x + b.rect.width / 2, y: b.y };
      const aRight = { x: a.x + a.rect.width / 2, y: a.y };
      const cLeft = { x: c.x - c.rect.width / 2, y: c.y };

      setLinkViewBox(`0 0 ${Math.max(1, chamberRect.width)} ${Math.max(1, chamberRect.height)}`);
      setLinkPaths([
        `M ${bLeft.x} ${bLeft.y} L ${aTop.x} ${aTop.y}`,
        `M ${bRight.x} ${bRight.y} L ${cTop.x} ${cTop.y}`,
        `M ${aRight.x} ${aRight.y} L ${cLeft.x} ${cLeft.y}`
      ]);
    }

    const id = window.requestAnimationFrame(updateLinks);
    window.addEventListener("resize", updateLinks);
    return () => {
      window.cancelAnimationFrame(id);
      window.removeEventListener("resize", updateLinks);
    };
  }, [chamberNodes, isLoading, localNodeState, resolvedProfile]);

  function validatePrompt(input: string): string | null {
    const trimmed = input.trim();
    if (!trimmed) return "prompt must not be empty";
    if (trimmed.length > MAX_PROMPT_LENGTH) return `prompt must be ${MAX_PROMPT_LENGTH} characters or fewer`;
    return null;
  }

  function openSettingsDialog() {
    setSettingsDraft(ollamaSystemPrompt);
    setSettingsOpen(true);
  }

  function closeSettingsDialog() {
    setSettingsOpen(false);
  }

  function saveSettingsDialog() {
    const next = settingsDraft.trim();
    setOllamaSystemPrompt(next || DEFAULT_OLLAMA_SYSTEM_PROMPT);
    setSettingsOpen(false);
  }

  async function deleteAllThreadsFromSettings() {
    if (isBusy || isDeletingAllThreads) return;
    const confirmed =
      typeof window !== "undefined"
        ? window.confirm("Delete all thread history permanently? This action cannot be undone.")
        : false;
    if (!confirmed) return;

    setIsDeletingAllThreads(true);
    try {
      const response = await fetch(`${apiBaseUrl}/api/magi/history`, { method: "DELETE" });
      if (!response.ok) {
        const data = (await response.json()) as { detail?: string };
        setError(data.detail ?? "failed to delete all thread history");
        return;
      }
      setHistory([]);
      setThreadNames({});
      setCollapsedThreads({});
      setEditingThreadId(null);
      setThreadNameDraft("");
      setConfirmDeleteThreadId(null);
      startNewChat();
      setSettingsOpen(false);
      setError("");
    } catch {
      setError("backend connection failed");
    } finally {
      setIsDeletingAllThreads(false);
    }
  }

  async function requestChat(trimmedPrompt: string): Promise<ChatResponse | null> {
    try {
      const response = await fetch(`${apiBaseUrl}/api/magi/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          prompt: trimmedPrompt,
          profile: selectedProfile || undefined,
          fresh_mode: freshMode,
          thread_id: threadId || undefined,
          ollama_system_prompt: ollamaSystemPrompt
        })
      });

      const data = (await response.json()) as ChatResponse | { detail?: string };
      if (!response.ok) {
        setError((data as { detail?: string }).detail ?? "backend request failed");
        return null;
      }
      return data as ChatResponse;
    } catch {
      setError("backend connection failed");
    }
    return null;
  }

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (isBusy) return;
    const trimmed = prompt.trim();
    const validationError = validatePrompt(trimmed);

    if (validationError) {
      setError(validationError);
      return;
    }

    setError("");
    setRunId("");
    setRoutingInfo(null);
    setFeedbackRating(null);
    setFeedbackReason("");
    setFeedbackMessage("");
    setIsLoading(true);
    setIsConsensusLoading(false);
    clearNodeTimers();
    setShowConclusion(false);
    setResolvedProfile("");
    setLocalNodeState("BLINK");
    setNodeStates({ A: "IDLE", B: "IDLE", C: "IDLE" });
    setResults([]);
    setConsensus({
      provider: "-",
      model: "-",
      text: "Loading...",
      status: "LOADING",
      latency_ms: 0
    });
    setChatMessages((prev) => [
      ...prev,
      {
        id: `u-${Date.now()}`,
        role: "user",
        text: trimmed
      }
    ]);
    markConsensusClockStart();

    try {
      const chat = await requestChat(trimmed);
      if (!chat) {
        setLocalNodeState("IDLE");
        setNodeStates({ A: "IDLE", B: "IDLE", C: "IDLE" });
        return;
      }
      setResolvedProfile(chat.profile);
      setRunId(chat.run_id);
      setThreadId(chat.thread_id || threadId || chat.run_id);
      setTurnIndex(chat.turn_index || turnIndex);
      setRoutingInfo(chat.routing ?? null);
      setResults(chat.results);
      setConsensus(chat.consensus);
      runNodeTransition(chat.profile, chat.results, chat.consensus.status);
      setChatMessages((prev) => [
        ...prev,
        {
          id: `a-${chat.run_id}`,
          role: "assistant",
          text: chat.reply || extractUnifiedReply(chat.consensus, chat.results),
          run_id: chat.run_id,
          turn_index: chat.turn_index,
          latency_ms: resolveAssistantLatency(chat.consensus, chat.results)
        }
      ]);
      markConsensusClockEnd();
      await fetchHistory();
    } finally {
      setIsLoading(false);
    }
  }

  async function copyRunId() {
    if (!runId) return;
    try {
      await navigator.clipboard.writeText(runId);
    } catch {
      setError("failed to copy run_id");
    }
  }

  async function submitRoutingFeedback() {
    if (!runId || !threadId || feedbackRating === null || feedbackSubmitting) return;
    setFeedbackSubmitting(true);
    setFeedbackMessage("");
    try {
      const response = await fetch(`${apiBaseUrl}/api/magi/routing/feedback`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          thread_id: threadId,
          request_id: runId,
          rating: feedbackRating,
          reason: feedbackReason.trim() || undefined
        })
      });
      const data = (await response.json()) as RoutingFeedbackResponse | { detail?: string };
      if (!response.ok) {
        setFeedbackMessage((data as { detail?: string }).detail ?? "failed to save feedback");
        return;
      }
      const policyKey = (data as RoutingFeedbackResponse).policy_key || routingInfo?.policy_key;
      setFeedbackMessage(policyKey ? `saved (policy: ${policyKey})` : "saved");
    } catch {
      setFeedbackMessage("failed to save feedback");
    } finally {
      setFeedbackSubmitting(false);
    }
  }

  function onTextareaKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key !== "Enter" || event.shiftKey || event.nativeEvent.isComposing) return;
    event.preventDefault();
    event.currentTarget.form?.requestSubmit();
  }

  function restoreHistory(item: RunHistoryItem) {
    if (isBusy) return;
    setError("");
    setPrompt(item.prompt);
    setRunId(item.run_id);
    setThreadId(item.thread_id || item.run_id);
    setTurnIndex(item.turn_index || 0);
    setResolvedProfile(item.profile);
    setResults(item.results);
    setConsensus(item.consensus);
    clearNodeTimers();
    const hasLoading =
      item.consensus?.status === "LOADING" || item.results.some((result) => result.status === "LOADING");
    if (item.profile === "local_only") {
      setLocalNodeState(hasLoading ? "BLINK" : "ON");
    } else {
      setLocalNodeState(hasLoading ? "RELAY" : "ON");
    }
    setNodeStates(nodeStatesFromResults(item.results));
    setShowConclusion(item.profile !== "local_only" && item.consensus?.status === "OK");
    setConclusionElapsedMs(item.consensus ? item.consensus.latency_ms : null);
    setRoutingInfo(null);
    setFeedbackRating(null);
    setFeedbackReason("");
    setFeedbackMessage("");
    consensusClockRef.current = null;
    const turns = history
      .filter((row) => (row.thread_id || row.run_id) === (item.thread_id || item.run_id))
      .sort((a, b) => a.turn_index - b.turn_index);
    const nextMessages: ChatMessage[] = [];
    for (const turn of turns) {
      nextMessages.push({
        id: `u-${turn.run_id}`,
        role: "user",
        text: turn.prompt,
        run_id: turn.run_id,
        turn_index: turn.turn_index
      });
      nextMessages.push({
        id: `a-${turn.run_id}`,
        role: "assistant",
        text: extractUnifiedReply(turn.consensus, turn.results),
        run_id: turn.run_id,
        turn_index: turn.turn_index,
        latency_ms: resolveAssistantLatency(turn.consensus, turn.results)
      });
    }
    setChatMessages(nextMessages);
  }

  function startNewChat() {
    if (isBusy) return;
    clearNodeTimers();
    setError("");
    setRunId("");
    setThreadId("");
    setTurnIndex(0);
    setPrompt("");
    setResolvedProfile("");
    setResults([]);
    setConsensus(null);
    setChatMessages([]);
    setLocalNodeState("IDLE");
    setNodeStates({ A: "IDLE", B: "IDLE", C: "IDLE" });
    setShowConclusion(false);
    setConclusionElapsedMs(null);
    setRoutingInfo(null);
    setFeedbackRating(null);
    setFeedbackReason("");
    setFeedbackMessage("");
    consensusClockRef.current = null;
  }

  function threadLabel(threadKey: string): string {
    const custom = (threadNames[threadKey] ?? "").trim();
    if (custom) return custom;
    return `Thread ${threadKey.slice(0, 8)}`;
  }

  function beginRenameThread(threadKey: string) {
    setEditingThreadId(threadKey);
    setThreadNameDraft(threadNames[threadKey] ?? "");
  }

  function cancelRenameThread() {
    setEditingThreadId(null);
    setThreadNameDraft("");
  }

  function saveRenameThread(threadKey: string) {
    const next = threadNameDraft.trim();
    setThreadNames((prev) => {
      const copy = { ...prev };
      if (!next) {
        delete copy[threadKey];
      } else {
        copy[threadKey] = next;
      }
      return copy;
    });
    setEditingThreadId(null);
    setThreadNameDraft("");
  }

  function toggleThreadCollapse(threadKey: string) {
    setCollapsedThreads((prev) => ({ ...prev, [threadKey]: !prev[threadKey] }));
  }

  async function deleteThread(threadKey: string) {
    if (isBusy) return;
    try {
      const response = await fetch(`${apiBaseUrl}/api/magi/history/thread/${encodeURIComponent(threadKey)}`, {
        method: "DELETE"
      });
      if (!response.ok) {
        const data = (await response.json()) as { detail?: string };
        setError(data.detail ?? "failed to delete thread");
        return;
      }

      setHistory((prev) => prev.filter((item) => (item.thread_id || item.run_id) !== threadKey));
      setThreadNames((prev) => {
        const copy = { ...prev };
        delete copy[threadKey];
        return copy;
      });
      setCollapsedThreads((prev) => {
        const copy = { ...prev };
        delete copy[threadKey];
        return copy;
      });
      if (threadKey === threadId) {
        startNewChat();
      }
      setConfirmDeleteThreadId(null);
      setError("");
    } catch {
      setError("backend connection failed");
    }
  }

  return (
    <main className="magi-grid magi-scan mx-auto min-h-screen w-full max-w-7xl px-4 py-8 md:px-8">
      <div className="grid items-start grid-cols-1 gap-6 md:grid-cols-[280px_1fr]">
        <ThreadSidebar
          isBusy={isBusy}
          threadGroups={threadGroups}
          threadId={threadId}
          editingThreadId={editingThreadId}
          threadNameDraft={threadNameDraft}
          collapsedThreads={collapsedThreads}
          confirmDeleteThreadId={confirmDeleteThreadId}
          threadLabel={threadLabel}
          onStartNewChat={startNewChat}
          onThreadNameDraftChange={setThreadNameDraft}
          onSaveRenameThread={saveRenameThread}
          onCancelRenameThread={cancelRenameThread}
          onToggleThreadCollapse={toggleThreadCollapse}
          onBeginRenameThread={beginRenameThread}
          onToggleDeleteConfirm={(id) =>
            setConfirmDeleteThreadId((prev) => (prev === id ? null : id))
          }
          onDeleteThread={(id) => void deleteThread(id)}
          onRestoreHistory={restoreHistory}
        />

        <div>
      <section className="panel p-4 md:p-6">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <h1 className="text-xl font-semibold tracking-[0.2em] text-terminal-accent md:text-2xl">MAGI</h1>
            <span className="rounded border border-terminal-border px-2 py-0.5 text-[11px] text-terminal-dim">
              phase {PHASE_VERSION}
            </span>
          </div>
          <button
            type="button"
            onClick={openSettingsDialog}
            className="inline-flex h-9 w-9 items-center justify-center rounded-md border border-terminal-border bg-[#02060b] text-terminal-dim hover:text-terminal-accent"
            aria-label="Open settings"
            title="Settings"
          >
            <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.9">
              <circle cx="12" cy="12" r="3.2" />
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M12 3.5v2.2m0 12.6v2.2m8.5-8.5h-2.2M5.7 12H3.5m14.36-6.36-1.55 1.55M7.69 16.31l-1.55 1.55m0-12.22 1.55 1.55m8.67 8.67 1.55 1.55"
              />
              <circle cx="12" cy="12" r="7.1" />
            </svg>
          </button>
        </div>
        <p className="mt-2 text-sm text-terminal-dim">Command chamber: local pre-router, then three models, then one consensus core.</p>

        <ChamberVisualization
          setChamberRef={(el) => {
            chamberRef.current = el;
          }}
          setPeerGroupRef={(el) => {
            peerGroupRef.current = el;
          }}
          setLocalNodeRef={(el) => {
            localNodeRef.current = el;
          }}
          linkPaths={linkPaths}
          linkViewBox={linkViewBox}
          chamberActive={chamberActive}
          chamberMode={chamberMode}
          localOnlyHandled={localOnlyHandled}
          localNodeState={localNodeState}
          showConclusion={showConclusion}
          localAgent={localAgent}
          localRouteHint={localRouteHint}
          chamberNodes={chamberNodes}
          showExecutingBadge={showExecutingBadge}
          resolvedProfile={resolvedProfile}
          nodeStates={nodeStates}
          winnerAgent={winnerAgent}
          confidenceMap={confidenceMap}
          showDiscussionBadge={showDiscussionBadge}
          showRoutingBadge={showRoutingBadge}
          conclusionElapsedMs={conclusionElapsedMs}
          setNodeRef={(agent, el) => {
            nodeRefs.current[agent] = el;
          }}
          formatElapsedMsToMinSec={formatElapsedMsToMinSec}
        />

        <PromptForm
          prompt={prompt}
          maxPromptLength={MAX_PROMPT_LENGTH}
          isBusy={isBusy}
          isLoading={isLoading}
          isConsensusLoading={isConsensusLoading}
          selectedProfile={selectedProfile}
          availableProfiles={availableProfiles}
          freshMode={freshMode}
          isStrictDebate={isStrictDebate}
          isUltra={isUltra}
          onSubmit={onSubmit}
          onPromptChange={setPrompt}
          onPromptKeyDown={onTextareaKeyDown}
          onProfileChange={setSelectedProfile}
          onFreshModeChange={setFreshMode}
        />

        {error ? <p className="mt-3 text-sm status-error">{error}</p> : null}

        <details className="mt-4 rounded-md border border-terminal-border bg-[#02060b] px-3 py-2">
          <summary className="cursor-pointer select-none text-sm text-terminal-dim">
            run details / routing / feedback
          </summary>
          <div className="mt-3 space-y-3">
            <RunMetaBar
              runId={runId}
              threadId={threadId}
              turnIndex={turnIndex}
              selectedProfile={selectedProfile}
              freshMode={freshMode}
              totalTokens={totalTokens}
              totalCostUsd={totalCostUsd}
              onCopyRunId={copyRunId}
            />
            <RoutingInfoPanel routingInfo={routingInfo} />
            <FeedbackPanel
              runId={runId}
              threadId={threadId}
              isBusy={isBusy}
              feedbackSubmitting={feedbackSubmitting}
              feedbackRating={feedbackRating}
              feedbackReason={feedbackReason}
              feedbackMessage={feedbackMessage}
              onSelectGood={() => setFeedbackRating(1)}
              onSelectBad={() => setFeedbackRating(-1)}
              onReasonChange={setFeedbackReason}
              onSubmit={() => void submitRoutingFeedback()}
            />
            <div className="rounded border border-terminal-border bg-[#050a10] px-3 py-3 text-xs text-terminal-dim">
              <p className="font-semibold text-terminal-text">Model execution status</p>
              {cards.length ? (
                <div className="mt-2 space-y-2">
                  {cards.map((item) => (
                    <div
                      key={`status-${item.agent}`}
                      className="rounded border border-terminal-border bg-[#02060b] px-2 py-2"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="font-semibold text-terminal-text">
                          Agent {item.agent}: {item.provider}/{item.model}
                        </span>
                        <span
                          className={
                            item.status === "OK"
                              ? "status-ok"
                              : item.status === "ERROR"
                                ? "status-error"
                                : "status-loading"
                          }
                        >
                          {item.status}
                        </span>
                      </div>
                      <p className="mt-1">latency_ms: {item.latency_ms}</p>
                      {item.error_message ? <p className="mt-1 status-error">error: {item.error_message}</p> : null}
                    </div>
                  ))}
                  {consensus?.status === "ERROR" ? (
                    <div className="rounded border border-terminal-border bg-[#02060b] px-2 py-2">
                      <p className="font-semibold text-terminal-text">Consensus</p>
                      {consensus.error_code ? <p className="mt-1">error_code: {consensus.error_code}</p> : null}
                      {consensus.error_message ? <p className="mt-1 status-error">error: {consensus.error_message}</p> : null}
                    </div>
                  ) : null}
                </div>
              ) : (
                <p className="mt-2 text-[11px]">No run yet.</p>
              )}
            </div>
          </div>
        </details>

      </section>

      <ChatTranscript messages={chatMessages} />
        </div>
      </div>

      {settingsOpen ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
          <div className="w-full max-w-2xl rounded-md border border-terminal-border bg-[#02060b] p-4">
            <div className="flex items-center justify-between">
              <h2 className="text-base font-semibold text-terminal-accent">Settings</h2>
              <button
                type="button"
                onClick={closeSettingsDialog}
                className="rounded border border-terminal-border px-2 py-1 text-xs text-terminal-dim hover:text-terminal-accent"
              >
                Close
              </button>
            </div>
            <label className="mt-4 block text-sm text-terminal-dim">
              Ollama system prompt
              <textarea
                className="mt-2 h-36 w-full resize-y rounded-md border border-terminal-border bg-[#01040a] p-3 text-sm text-terminal-text outline-none ring-terminal-accent focus:ring"
                value={settingsDraft}
                onChange={(event) => setSettingsDraft(event.target.value)}
                placeholder={DEFAULT_OLLAMA_SYSTEM_PROMPT}
              />
            </label>
            <p className="mt-2 text-xs text-terminal-dim">
              This prompt is sent to Ollama agents for local draft/local_only execution.
            </p>
            <div className="mt-4 flex items-center gap-2">
              <button
                type="button"
                onClick={saveSettingsDialog}
                className="rounded border border-terminal-accent bg-[#0d1d2a] px-3 py-1.5 text-sm text-terminal-accent"
              >
                Save
              </button>
              <button
                type="button"
                onClick={() => setSettingsDraft(DEFAULT_OLLAMA_SYSTEM_PROMPT)}
                className="rounded border border-terminal-border px-3 py-1.5 text-sm text-terminal-dim"
              >
                Reset Default
              </button>
            </div>
            <div className="mt-5 border-t border-terminal-border pt-4">
              <p className="text-xs text-terminal-dim">Danger Zone</p>
              <button
                type="button"
                onClick={() => void deleteAllThreadsFromSettings()}
                disabled={isBusy || isDeletingAllThreads}
                className="mt-2 rounded border border-terminal-err bg-[#160a0a] px-3 py-1.5 text-sm text-terminal-err disabled:cursor-not-allowed disabled:opacity-50"
              >
                {isDeletingAllThreads ? "Deleting..." : "Delete All Threads"}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </main>
  );
}

