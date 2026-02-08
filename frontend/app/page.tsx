"use client";

import { FormEvent, KeyboardEvent, useEffect, useMemo, useState } from "react";

type AgentId = "A" | "B" | "C";
type AgentStatus = "OK" | "ERROR" | "LOADING";

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
      if (!success) return;
      setRunId(success.run_id);
      setSelectedProfile(success.profile);
      setResults(success.results);
      setConsensus(success.consensus);
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

    try {
      const retried = await requestRetry(lastRunPrompt, agent);
      if (!retried) return;

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
  }

  return (
    <main className="mx-auto min-h-screen w-full max-w-7xl px-4 py-8 md:px-8">
      <section className="panel p-4 md:p-6">
        <div className="flex items-center gap-2">
          <h1 className="text-xl font-semibold tracking-wide text-terminal-accent md:text-2xl">MAGI</h1>
          <span className="rounded border border-terminal-border px-2 py-0.5 text-[11px] text-terminal-dim">
            phase {PHASE_VERSION}
          </span>
        </div>
        <p className="mt-2 text-sm text-terminal-dim">Single prompt. 3 parallel models. Side-by-side outputs.</p>

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
              profile:
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
          </div>
          <p className="text-xs text-terminal-dim">Enter: submit / Shift+Enter: newline</p>
        </form>

        {error ? <p className="mt-3 text-sm status-error">{error}</p> : null}

        <div className="mt-4 flex items-center gap-2 text-xs text-terminal-dim">
          <span>run_id: {runId || "-"}</span>
          <span>profile: {selectedProfile}</span>
          <button
            type="button"
            onClick={copyRunId}
            disabled={!runId}
            className="rounded border border-terminal-border px-2 py-1 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Copy
          </button>
        </div>

        {history.length ? (
          <div className="mt-4 rounded-md border border-terminal-border bg-[#02060b] p-3">
            <p className="text-xs font-semibold text-terminal-dim">Run history (persisted)</p>
            <div className="mt-2 space-y-2">
              {history.map((item) => {
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
                    <p>run_id: {item.run_id}</p>
                    <p>profile: {item.profile}</p>
                    <p>time: {new Date(item.created_at).toLocaleTimeString()}</p>
                    <p>status: {statusSummary}</p>
                    <p>consensus: {item.consensus?.status ?? "-"}</p>
                    <p>prompt: {item.prompt.slice(0, 80)}</p>
                  </button>
                );
              })}
            </div>
          </div>
        ) : null}
      </section>

      {consensus ? (
        <section className="panel mt-6 p-4">
          <div className="flex items-center justify-between border-b border-terminal-border pb-2 text-sm">
            <span className="font-semibold">Consensus</span>
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
                <span className="font-semibold">Agent {result.agent}</span>
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
                <p className="text-terminal-dim">model: {result.provider}/{result.model}</p>
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
    </main>
  );
}
