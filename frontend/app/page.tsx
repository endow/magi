"use client";

import { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import AgentResultsGrid from "./components/agent-results-grid";
import ChamberVisualization from "./components/chamber-visualization";
import ConsensusPanel from "./components/consensus-panel";
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

type RunResponse = {
  run_id: string;
  thread_id: string;
  turn_index: number;
  profile: string;
  results: AgentResult[];
  consensus: ConsensusResult;
  routing?: RoutingInfo | null;
};

type RetryResponse = {
  run_id: string;
  thread_id: string;
  turn_index: number;
  profile: string;
  result: AgentResult;
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
};

type ConsensusResponse = {
  run_id: string;
  thread_id: string;
  turn_index: number;
  profile: string;
  consensus: ConsensusResult;
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

const RAW_API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const MAX_PROMPT_LENGTH = 4000;
const PHASE_VERSION = "v1.1";

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

function splitConsensusText(text: string | null | undefined): { main: string; voteDetails: string | null } {
  if (!text) return { main: "", voteDetails: null };
  const marker = "\n\nVote details:";
  const markerIndex = text.indexOf(marker);
  if (markerIndex < 0) return { main: text, voteDetails: null };
  return {
    main: text.slice(0, markerIndex),
    voteDetails: text.slice(markerIndex + marker.length).trim()
  };
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
  const [lastRunPrompt, setLastRunPrompt] = useState("");
  const [selectedProfile, setSelectedProfile] = useState("");
  const [defaultProfile, setDefaultProfile] = useState("");
  const [resolvedProfile, setResolvedProfile] = useState("");
  const [availableProfiles, setAvailableProfiles] = useState<string[]>([]);
  const [profileAgents, setProfileAgents] = useState<ProfilesResponse["profile_agents"]>({});
  const [results, setResults] = useState<AgentResult[]>([]);
  const [consensus, setConsensus] = useState<ConsensusResult | null>(null);
  const [history, setHistory] = useState<RunHistoryItem[]>([]);
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
  const parsedConsensus = useMemo(() => splitConsensusText(consensus?.text), [consensus?.text]);
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
    () => !showConclusion && isLoading && loadingElapsedMs >= 7000,
    [isLoading, loadingElapsedMs, showConclusion]
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
      if (rawNames) setThreadNames(JSON.parse(rawNames) as ThreadNameMap);
      if (rawCollapsed) setCollapsedThreads(JSON.parse(rawCollapsed) as ThreadCollapsedMap);
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

  async function requestRun(trimmedPrompt: string): Promise<RunResponse | null> {
    try {
      const response = await fetch(`${apiBaseUrl}/api/magi/run`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          prompt: trimmedPrompt,
          profile: selectedProfile || undefined,
          fresh_mode: freshMode,
          thread_id: threadId || undefined
        })
      });

      const data = (await response.json()) as RunResponse | { detail?: string };

      if (!response.ok) {
        setError((data as { detail?: string }).detail ?? "backend request failed");
        return null;
      }

      return data as RunResponse;
    } catch {
      setError("backend connection failed");
    }
    return null;
  }

  async function requestRetry(trimmedPrompt: string, agent: AgentId): Promise<RetryResponse | null> {
    try {
      const response = await fetch(`${apiBaseUrl}/api/magi/retry`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          prompt: trimmedPrompt,
          agent,
          profile: selectedProfile || undefined,
          fresh_mode: freshMode,
          thread_id: threadId || undefined
        })
      });

      const data = (await response.json()) as RetryResponse | { detail?: string };

      if (!response.ok) {
        setError((data as { detail?: string }).detail ?? "backend request failed");
        return null;
      }

      return data as RetryResponse;
    } catch {
      setError("backend connection failed");
    }
    return null;
  }

  async function requestConsensus(
    trimmedPrompt: string,
    latestResults: AgentResult[],
    options?: { runId?: string; threadId?: string; profile?: string; freshMode?: boolean }
  ): Promise<ConsensusResponse | null> {
    try {
      const response = await fetch(`${apiBaseUrl}/api/magi/consensus`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          prompt: trimmedPrompt,
          results: latestResults,
          profile: options?.profile ?? (selectedProfile || undefined),
          fresh_mode: options?.freshMode ?? freshMode,
          thread_id: options?.threadId ?? (threadId || undefined),
          run_id: options?.runId
        })
      });

      const data = (await response.json()) as ConsensusResponse | { detail?: string };
      if (!response.ok) {
        setError((data as { detail?: string }).detail ?? "backend request failed");
        return null;
      }

      return data as ConsensusResponse;
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
    setLastRunPrompt(trimmed);
    markConsensusClockStart();

    try {
      const success = await requestRun(trimmed);
      if (!success) {
        setLocalNodeState("IDLE");
        setNodeStates({ A: "IDLE", B: "IDLE", C: "IDLE" });
        return;
      }
      setResolvedProfile(success.profile);
      setRunId(success.run_id);
      setThreadId(success.thread_id || threadId || success.run_id);
      setTurnIndex(success.turn_index || turnIndex);
      setRoutingInfo(success.routing ?? null);
      setResults(success.results);
      setConsensus(success.consensus);
      runNodeTransition(success.profile, success.results, success.consensus.status);
      await fetchHistory();
      setIsLoading(false);

      if (success.consensus.status !== "LOADING") {
        markConsensusClockEnd();
        return;
      }

      setIsConsensusLoading(true);
      try {
        const finalized = await requestConsensus(trimmed, success.results, {
          runId: success.run_id,
          threadId: success.thread_id,
          profile: success.profile,
          freshMode
        });
        if (!finalized || finalized.run_id !== success.run_id) {
          return;
        }
        setThreadId(finalized.thread_id || success.thread_id);
        if ((finalized.turn_index ?? 0) > 0) setTurnIndex(finalized.turn_index);
        setResolvedProfile(finalized.profile);
        setConsensus(finalized.consensus);
        runNodeTransition(finalized.profile, success.results, finalized.consensus.status);
        markConsensusClockEnd();
        await fetchHistory();
      } finally {
        setIsConsensusLoading(false);
      }
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

  async function copyResultText(result: AgentResult) {
    if (!result.text) return;
    try {
      await navigator.clipboard.writeText(result.text);
    } catch {
      setError(`failed to copy Agent ${result.agent} text`);
    }
  }

  async function retryAgent(agent: AgentId) {
    if (!lastRunPrompt || isBusy) return;
    setError("");
    setIsLoading(true);
    clearNodeTimers();
    setShowConclusion(false);
    setRoutingInfo(null);
    setFeedbackRating(null);
    setFeedbackReason("");
    setFeedbackMessage("");
    markConsensusClockStart();
    setLocalNodeState("RELAY");
    setNodeStates((prev) => ({ ...prev, [agent]: "BLINK" }));

    try {
      const retried = await requestRetry(lastRunPrompt, agent);
      if (!retried) {
        setNodeStates(nodeStatesFromResults(results));
        return;
      }

      setRunId(retried.run_id);
      setResolvedProfile(retried.profile);
      setThreadId(retried.thread_id || threadId || retried.run_id);
      if ((retried.turn_index ?? 0) > 0) setTurnIndex(retried.turn_index);
      setConsensus({
        provider: consensus?.provider ?? "-",
        model: consensus?.model ?? "-",
        text: "Loading...",
        status: "LOADING",
        latency_ms: 0
      });
      const updatedResults = (() => {
        const baseResults = results.length ? results : [];
        const byAgent = new Map(baseResults.map((item) => [item.agent, item]));
        byAgent.set(agent, retried.result);
        const order: AgentId[] = ["A", "B", "C"];
        return order.map((id) => byAgent.get(id)).filter((item): item is AgentResult => Boolean(item));
      })();

      setResults(updatedResults);
      const recalculated = await requestConsensus(lastRunPrompt, updatedResults);
      if (recalculated) {
        setResolvedProfile(recalculated.profile);
        setThreadId(recalculated.thread_id || threadId || recalculated.run_id);
        if ((recalculated.turn_index ?? 0) > 0) setTurnIndex(recalculated.turn_index);
        setConsensus(recalculated.consensus);
        runNodeTransition(recalculated.profile, updatedResults, recalculated.consensus.status);
        markConsensusClockEnd();
      } else {
        setNodeStates(nodeStatesFromResults(updatedResults));
      }
    } finally {
      setIsLoading(false);
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
    setLastRunPrompt(item.prompt);
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
  }

  function startNewChat() {
    if (isBusy) return;
    clearNodeTimers();
    setError("");
    setRunId("");
    setThreadId("");
    setTurnIndex(0);
    setPrompt("");
    setLastRunPrompt("");
    setResolvedProfile("");
    setResults([]);
    setConsensus(null);
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
        <div className="flex items-center gap-2">
          <h1 className="text-xl font-semibold tracking-[0.2em] text-terminal-accent md:text-2xl">MAGI</h1>
          <span className="rounded border border-terminal-border px-2 py-0.5 text-[11px] text-terminal-dim">
            phase {PHASE_VERSION}
          </span>
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

        <RunMetaBar
          runId={runId}
          threadId={threadId}
          turnIndex={turnIndex}
          selectedProfile={selectedProfile}
          freshMode={freshMode}
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

      </section>

      {consensus ? (
        <ConsensusPanel
          consensus={consensus}
          parsedConsensus={parsedConsensus}
          isStrictDebate={isStrictDebate}
          isUltra={isUltra}
        />
      ) : null}

      <AgentResultsGrid
        cards={cards}
        isBusy={isBusy}
        onRetry={(agent) => void retryAgent(agent)}
        onCopy={copyResultText}
      />
        </div>
      </div>
    </main>
  );
}

