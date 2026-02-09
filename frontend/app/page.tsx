"use client";

import { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";

type AgentId = "A" | "B" | "C";
type AgentStatus = "OK" | "ERROR" | "LOADING";
type NodeState = "IDLE" | "BLINK" | "ON" | "ERROR";
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

type RunResponse = {
  run_id: string;
  profile: string;
  results: AgentResult[];
  consensus: ConsensusResult;
};

type RetryResponse = {
  run_id: string;
  profile: string;
  result: AgentResult;
};

type RunHistoryItem = {
  run_id: string;
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
  profile: string;
  consensus: ConsensusResult;
};

type ProfilesResponse = {
  default_profile: string;
  profiles: string[];
};

type HistoryListResponse = {
  total: number;
  items: RunHistoryItem[];
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const MAX_PROMPT_LENGTH = 4000;
const PHASE_VERSION = "v0.4";

const loadingResults: AgentResult[] = [
  { agent: "A", provider: "-", model: "-", text: "Loading...", status: "LOADING", latency_ms: 0 },
  { agent: "B", provider: "-", model: "-", text: "Loading...", status: "LOADING", latency_ms: 0 },
  { agent: "C", provider: "-", model: "-", text: "Loading...", status: "LOADING", latency_ms: 0 }
];

function statusClass(status: AgentStatus): string {
  if (status === "OK") return "status-ok";
  if (status === "ERROR") return "status-error";
  return "status-loading";
}

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

export default function HomePage() {
  const [prompt, setPrompt] = useState("");
  const [lastRunPrompt, setLastRunPrompt] = useState("");
  const [selectedProfile, setSelectedProfile] = useState("default");
  const [availableProfiles, setAvailableProfiles] = useState<string[]>(["default"]);
  const [results, setResults] = useState<AgentResult[]>([]);
  const [consensus, setConsensus] = useState<ConsensusResult | null>(null);
  const [history, setHistory] = useState<RunHistoryItem[]>([]);
  const [runId, setRunId] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [nodeStates, setNodeStates] = useState<Record<AgentId, NodeState>>({
    A: "IDLE",
    B: "IDLE",
    C: "IDLE"
  });
  const [showConclusion, setShowConclusion] = useState(false);
  const isStrictDebate = selectedProfile === "performance";
  const chamberRef = useRef<HTMLDivElement | null>(null);
  const nodeRefs = useRef<Record<AgentId, HTMLDivElement | null>>({ A: null, B: null, C: null });
  const [linkPaths, setLinkPaths] = useState<string[]>([]);
  const [linkViewBox, setLinkViewBox] = useState("0 0 100 100");
  const nodeTimerRefs = useRef<number[]>([]);

  async function fetchHistory() {
    try {
      const response = await fetch(`${API_BASE_URL}/api/magi/history?limit=20&offset=0`);
      if (!response.ok) return;
      const data = (await response.json()) as HistoryListResponse;
      setHistory(data.items);
    } catch {
      // ignore history load errors
    }
  }

  const cards = useMemo(() => {
    if (isLoading) return loadingResults;
    if (!results.length) return [];
    const order: AgentId[] = ["A", "B", "C"];
    const byAgent = new Map(results.map((item) => [item.agent, item]));
    return order.map((agent) => byAgent.get(agent)).filter((item): item is AgentResult => Boolean(item));
  }, [isLoading, results]);
  const displayNodes = cards.length ? cards : loadingResults;
  const confidenceMap = useMemo(
    () => parseConfidenceMap(consensus?.status === "OK" ? consensus.text : ""),
    [consensus]
  );
  const winnerAgent = useMemo(
    () => parseWinnerAgent(consensus?.status === "OK" ? consensus.text : ""),
    [consensus]
  );
  const nodePositionClass: Record<AgentId, string> = {
    A: "magi-node-a",
    B: "magi-node-b",
    C: "magi-node-c"
  };
  const chamberActive = Object.values(nodeStates).some((state) => state === "BLINK");

  function clearNodeTimers() {
    nodeTimerRefs.current.forEach((id) => window.clearTimeout(id));
    nodeTimerRefs.current = [];
  }

  function nodeStatesFromResults(items: AgentResult[]): Record<AgentId, NodeState> {
    const base: Record<AgentId, NodeState> = { A: "IDLE", B: "IDLE", C: "IDLE" };
    for (const item of items) {
      base[item.agent] = item.status === "OK" ? "ON" : "ERROR";
    }
    return base;
  }

  function runNodeTransition(items: AgentResult[], consensusStatus: "OK" | "ERROR" | "LOADING") {
    clearNodeTimers();
    setShowConclusion(false);
    setNodeStates({ A: "BLINK", B: "BLINK", C: "BLINK" });

    const maxLatency = Math.max(...items.map((item) => Math.max(1, item.latency_ms)), 1);
    const maxDelay = 1800;
    let doneAt = 0;

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
      setShowConclusion(consensusStatus === "OK");
    }, doneAt + 200);
    nodeTimerRefs.current.push(endTimer);
  }

  useEffect(() => {
    async function loadProfiles() {
      try {
        const response = await fetch(`${API_BASE_URL}/api/magi/profiles`);
        if (!response.ok) return;
        const data = (await response.json()) as ProfilesResponse;
        if (!data.profiles.length) return;
        setAvailableProfiles(data.profiles);
        setSelectedProfile(data.default_profile);
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
    return () => {
      clearNodeTimers();
    };
  }, []);

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

      function edgePoint(
        from: { x: number; y: number; rect: DOMRect },
        to: { x: number; y: number; rect: DOMRect }
      ): { x: number; y: number } {
        const dx = to.x - from.x;
        const dy = to.y - from.y;
        const absDx = Math.abs(dx) || 0.0001;
        const absDy = Math.abs(dy) || 0.0001;
        const halfW = from.rect.width / 2;
        const halfH = from.rect.height / 2;
        const scale = Math.min(halfW / absDx, halfH / absDy);
        return {
          x: from.x + dx * scale,
          y: from.y + dy * scale
        };
      }

      const ab1 = edgePoint(a, b);
      const ab2 = edgePoint(b, a);
      const bc1 = edgePoint(b, c);
      const bc2 = edgePoint(c, b);
      const ac1 = edgePoint(a, c);
      const ac2 = edgePoint(c, a);

      setLinkViewBox(`0 0 ${Math.max(1, chamberRect.width)} ${Math.max(1, chamberRect.height)}`);
      setLinkPaths([
        `M ${ab1.x} ${ab1.y} L ${ab2.x} ${ab2.y}`,
        `M ${bc1.x} ${bc1.y} L ${bc2.x} ${bc2.y}`,
        `M ${ac1.x} ${ac1.y} L ${ac2.x} ${ac2.y}`
      ]);
    }

    const id = window.requestAnimationFrame(updateLinks);
    window.addEventListener("resize", updateLinks);
    return () => {
      window.cancelAnimationFrame(id);
      window.removeEventListener("resize", updateLinks);
    };
  }, [cards, isLoading]);

  function validatePrompt(input: string): string | null {
    const trimmed = input.trim();
    if (!trimmed) return "prompt must not be empty";
    if (trimmed.length > MAX_PROMPT_LENGTH) return `prompt must be ${MAX_PROMPT_LENGTH} characters or fewer`;
    return null;
  }

  async function requestRun(trimmedPrompt: string): Promise<RunResponse | null> {
    try {
      const response = await fetch(`${API_BASE_URL}/api/magi/run`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ prompt: trimmedPrompt, profile: selectedProfile })
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
      const response = await fetch(`${API_BASE_URL}/api/magi/retry`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ prompt: trimmedPrompt, agent, profile: selectedProfile })
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

  async function requestConsensus(trimmedPrompt: string, latestResults: AgentResult[]): Promise<ConsensusResponse | null> {
    try {
      const response = await fetch(`${API_BASE_URL}/api/magi/consensus`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ prompt: trimmedPrompt, results: latestResults, profile: selectedProfile })
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
    const trimmed = prompt.trim();
    const validationError = validatePrompt(trimmed);

    if (validationError) {
      setError(validationError);
      return;
    }

    setError("");
    setRunId("");
    setIsLoading(true);
    clearNodeTimers();
    setShowConclusion(false);
    setNodeStates({ A: "BLINK", B: "BLINK", C: "BLINK" });
    setResults([]);
    setConsensus({
      provider: "-",
      model: "-",
      text: "Loading...",
      status: "LOADING",
      latency_ms: 0
    });
    setLastRunPrompt(trimmed);

    try {
      const success = await requestRun(trimmed);
      if (!success) {
        setNodeStates({ A: "IDLE", B: "IDLE", C: "IDLE" });
        return;
      }
      setRunId(success.run_id);
      setSelectedProfile(success.profile);
      setResults(success.results);
      setConsensus(success.consensus);
      runNodeTransition(success.results, success.consensus.status);
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

  async function copyResultText(result: AgentResult) {
    if (!result.text) return;
    try {
      await navigator.clipboard.writeText(result.text);
    } catch {
      setError(`failed to copy Agent ${result.agent} text`);
    }
  }

  async function retryAgent(agent: AgentId) {
    if (!lastRunPrompt || isLoading) return;
    setError("");
    setIsLoading(true);
    clearNodeTimers();
    setShowConclusion(false);
    setNodeStates((prev) => ({ ...prev, [agent]: "BLINK" }));

    try {
      const retried = await requestRetry(lastRunPrompt, agent);
      if (!retried) {
        setNodeStates(nodeStatesFromResults(results));
        return;
      }

      setRunId(retried.run_id);
      setSelectedProfile(retried.profile);
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
        setSelectedProfile(recalculated.profile);
        setConsensus(recalculated.consensus);
        runNodeTransition(updatedResults, recalculated.consensus.status);
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
    if (isLoading) return;
    setError("");
    setPrompt(item.prompt);
    setLastRunPrompt(item.prompt);
    setRunId(item.run_id);
    setSelectedProfile(item.profile);
    setResults(item.results);
    setConsensus(item.consensus);
    clearNodeTimers();
    setNodeStates(nodeStatesFromResults(item.results));
    setShowConclusion(item.consensus?.status === "OK");
  }

  function startNewChat() {
    if (isLoading) return;
    clearNodeTimers();
    setError("");
    setRunId("");
    setPrompt("");
    setLastRunPrompt("");
    setResults([]);
    setConsensus(null);
    setNodeStates({ A: "IDLE", B: "IDLE", C: "IDLE" });
    setShowConclusion(false);
  }

  return (
    <main className="magi-grid magi-scan mx-auto min-h-screen w-full max-w-7xl px-4 py-8 md:px-8">
      <div className="grid items-start grid-cols-1 gap-6 md:grid-cols-[280px_1fr]">
        <aside className="panel p-3 md:max-h-[calc(100vh-3rem)] md:overflow-auto">
          <button
            type="button"
            onClick={startNewChat}
            disabled={isLoading}
            className="w-full rounded border border-terminal-accent bg-[#1f120b] px-3 py-2 text-xs font-semibold text-terminal-accent disabled:cursor-not-allowed disabled:opacity-50"
          >
            Start New Chat
          </button>
          <p className="mt-3 text-xs font-semibold text-terminal-dim">Run history</p>
          <div className="mt-2 space-y-2">
            {history.length ? (
              history.map((item) => {
                const statusSummary = item.results
                  .map((result) => `${result.agent}:${result.status}`)
                  .join(" ");
                return (
                  <button
                    key={item.run_id}
                    type="button"
                    onClick={() => restoreHistory(item)}
                    className="w-full rounded border border-terminal-border px-2 py-2 text-left text-xs text-terminal-dim hover:border-terminal-accent hover:text-terminal-text"
                  >
                    <p>{new Date(item.created_at).toLocaleString()}</p>
                    <p>mode: {item.profile}</p>
                    <p>status: {statusSummary}</p>
                    <p className="truncate">prompt: {item.prompt}</p>
                  </button>
                );
              })
            ) : (
              <p className="text-xs text-terminal-dim">No history yet.</p>
            )}
          </div>
        </aside>

        <div>
      <section className="panel p-4 md:p-6">
        <div className="flex items-center gap-2">
          <h1 className="text-xl font-semibold tracking-[0.2em] text-terminal-accent md:text-2xl">MAGI</h1>
          <span className="rounded border border-terminal-border px-2 py-0.5 text-[11px] text-terminal-dim">
            phase {PHASE_VERSION}
          </span>
        </div>
        <p className="mt-2 text-sm text-terminal-dim">Command chamber: one prompt, three models, one consensus core.</p>

        <div className="magi-wire mt-4 rounded-md p-3">
          <div ref={chamberRef} className="magi-chamber grid grid-cols-1 gap-3 md:block">
            <svg
              className={`magi-links ${chamberActive ? "is-active" : ""}`}
              viewBox={linkViewBox}
              preserveAspectRatio="xMidYMid meet"
              aria-hidden="true"
            >
              {linkPaths.map((path, index) => (
                <path key={`link-${index}`} className="magi-link-path" d={path} />
              ))}
            </svg>
            {displayNodes.map((node) => (
              <div
                key={`node-${node.agent}`}
                ref={(el) => {
                  nodeRefs.current[node.agent] = el;
                }}
                className={`magi-node-wrap ${nodePositionClass[node.agent]}`}
              >
                <div
                  className={`magi-node p-4 magi-node-${nodeStates[node.agent].toLowerCase()} ${
                    winnerAgent && nodeStates[node.agent] === "ON" && node.agent !== winnerAgent
                      ? "magi-node-rejected"
                      : ""
                  }`}
                >
                  <p className="magi-node-label">NODE-{node.agent}</p>
                  <p className="mt-2 truncate text-sm font-semibold">
                    {node.provider === "-" ? `AGENT ${node.agent}` : `${node.provider}/${node.model}`}
                  </p>
                  <div className="magi-node-status-slot mt-2 w-full px-6">
                    {confidenceMap[node.agent] !== null ? (
                      <div className="magi-confidence-track">
                        <div className="magi-confidence-fill" style={{ width: `${confidenceMap[node.agent]}%` }} />
                      </div>
                    ) : null}
                    {confidenceMap[node.agent] !== null ? (
                      <p className="mt-1 text-[11px] font-semibold">confidence {confidenceMap[node.agent]}</p>
                    ) : (
                      <p className="mt-1 text-[11px] font-semibold opacity-0">confidence --</p>
                    )}
                    {nodeStates[node.agent] === "BLINK" ? (
                      <div className="magi-node-progress" />
                    ) : null}
                  </div>
                </div>
              </div>
            ))}
            {showConclusion ? <div className="magi-conclusion">Conclusion</div> : null}
          </div>
        </div>

        <form className="mt-4 space-y-3" onSubmit={onSubmit}>
          <textarea
            className="h-40 w-full resize-y rounded-md border border-terminal-border bg-[#02060b] p-3 text-sm outline-none ring-terminal-accent focus:ring"
            placeholder="Type your prompt..."
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            onKeyDown={onTextareaKeyDown}
          />
          <div className="flex items-center gap-3">
            <button
              type="submit"
              disabled={isLoading}
              className="rounded-md border border-terminal-accent bg-[#0d1d2a] px-4 py-2 text-sm text-terminal-accent disabled:cursor-not-allowed disabled:opacity-50"
            >
              {isLoading ? "Running..." : "Run MAGI"}
            </button>
            <span className="text-xs text-terminal-dim">
              {prompt.length}/{MAX_PROMPT_LENGTH} chars
            </span>
            <label className="text-xs text-terminal-dim">
              mode:
              <select
                className="ml-2 rounded border border-terminal-border bg-[#02060b] px-2 py-1 text-xs"
                value={selectedProfile}
                onChange={(event) => setSelectedProfile(event.target.value)}
                disabled={isLoading}
              >
                {availableProfiles.map((profile) => (
                  <option key={profile} value={profile}>
                    {profile}
                  </option>
                ))}
              </select>
            </label>
            {isStrictDebate ? (
              <span className="rounded border border-terminal-accent px-2 py-1 text-[11px] text-terminal-accent">
                strict debate
              </span>
            ) : null}
          </div>
          <p className="text-xs text-terminal-dim">Enter: submit / Shift+Enter: newline</p>
        </form>

        {error ? <p className="mt-3 text-sm status-error">{error}</p> : null}

        <div className="mt-4 flex items-center gap-2 text-xs text-terminal-dim">
          <span>run_id: {runId || "-"}</span>
          <span>mode: {selectedProfile}</span>
          <button
            type="button"
            onClick={copyRunId}
            disabled={!runId}
            className="rounded border border-terminal-border px-2 py-1 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Copy
          </button>
        </div>

      </section>

      {consensus ? (
        <section className="panel mt-6 p-4">
          <div className="flex items-center justify-between border-b border-terminal-border pb-2 text-sm">
            <div className="flex items-center gap-2">
              <span className="font-semibold tracking-wide text-terminal-accent">Consensus Core</span>
              {isStrictDebate ? (
                <span className="rounded border border-terminal-accent px-2 py-0.5 text-[11px] text-terminal-accent">
                  strict debate
                </span>
              ) : null}
            </div>
            <span className={statusClass(consensus.status)}>{consensus.status}</span>
          </div>
          <div className="mt-3 space-y-2 text-xs">
            <p className="text-terminal-dim">model: {consensus.provider}/{consensus.model}</p>
            <p className="text-terminal-dim">latency_ms: {consensus.latency_ms}</p>
            {consensus.error_message ? <p className="status-error">error: {consensus.error_message}</p> : null}
            <pre className="mt-2 whitespace-pre-wrap break-words rounded-md bg-[#02060b] p-3 text-sm leading-6">
              {consensus.text}
            </pre>
          </div>
        </section>
      ) : null}

      {cards.length ? (
        <section className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-3">
          {cards.map((result) => (
            <article key={result.agent} className="panel p-4">
              <div className="flex items-center justify-between border-b border-terminal-border pb-2 text-sm">
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-terminal-accent">{result.provider}/{result.model}</span>
                  <span className="text-[11px] text-terminal-dim">Agent {result.agent}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className={statusClass(result.status)}>{result.status}</span>
                  <button
                    type="button"
                    onClick={() => copyResultText(result)}
                    disabled={!result.text || isLoading}
                    className="rounded border border-terminal-border px-2 py-1 text-[11px] disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    Copy
                  </button>
                  {result.status === "ERROR" ? (
                    <button
                      type="button"
                      onClick={() => retryAgent(result.agent)}
                      disabled={isLoading}
                      className="rounded border border-terminal-err px-2 py-1 text-[11px] text-terminal-err disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      Retry
                    </button>
                  ) : null}
                </div>
              </div>

              <div className="mt-3 space-y-2 text-xs">
                <p className="text-terminal-dim">latency_ms: {result.latency_ms}</p>
                {result.error_message ? <p className="status-error">error: {result.error_message}</p> : null}
                <pre className="mt-2 whitespace-pre-wrap break-words rounded-md bg-[#02060b] p-3 text-sm leading-6">
                  {result.text}
                </pre>
              </div>
            </article>
          ))}
        </section>
      ) : null}
        </div>
      </div>
    </main>
  );
}
