"use client";

import { FormEvent, KeyboardEvent, useMemo, useState } from "react";

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
  results: AgentResult[];
};

type RunHistoryItem = {
  run_id: string;
  prompt: string;
  results: AgentResult[];
  created_at: string;
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const MAX_PROMPT_LENGTH = 4000;

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
  const [results, setResults] = useState<AgentResult[]>([]);
  const [history, setHistory] = useState<RunHistoryItem[]>([]);
  const [runId, setRunId] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const cards = useMemo(() => {
    if (isLoading) return loadingResults;
    if (!results.length) return [];
    const order: AgentId[] = ["A", "B", "C"];
    const byAgent = new Map(results.map((item) => [item.agent, item]));
    return order.map((agent) => byAgent.get(agent)).filter((item): item is AgentResult => Boolean(item));
  }, [isLoading, results]);

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
        body: JSON.stringify({ prompt: trimmedPrompt })
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
    setLastRunPrompt(trimmed);

    try {
      const success = await requestRun(trimmed);
      if (!success) return;
      setRunId(success.run_id);
      setResults(success.results);
      setHistory((current) => [
        {
          run_id: success.run_id,
          prompt: trimmed,
          results: success.results,
          created_at: new Date().toISOString()
        },
        ...current
      ].slice(0, 20));
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
      const rerun = await requestRun(lastRunPrompt);
      if (!rerun) return;

      const retried = rerun.results.find((item) => item.agent === agent);
      if (!retried) return;

      setRunId(rerun.run_id);
      setResults((current) => {
        const byAgent = new Map(current.map((item) => [item.agent, item]));
        byAgent.set(agent, retried);
        const order: AgentId[] = ["A", "B", "C"];
        return order.map((id) => byAgent.get(id)).filter((item): item is AgentResult => Boolean(item));
      });
      setHistory((current) => [
        {
          run_id: rerun.run_id,
          prompt: lastRunPrompt,
          results: rerun.results,
          created_at: new Date().toISOString()
        },
        ...current
      ].slice(0, 20));
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
    setResults(item.results);
  }

  return (
    <main className="mx-auto min-h-screen w-full max-w-7xl px-4 py-8 md:px-8">
      <section className="panel p-4 md:p-6">
        <h1 className="text-xl font-semibold tracking-wide text-terminal-accent md:text-2xl">MAGI v0</h1>
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
          </div>
          <p className="text-xs text-terminal-dim">Enter: submit / Shift+Enter: newline</p>
        </form>

        {error ? <p className="mt-3 text-sm status-error">{error}</p> : null}

        <div className="mt-4 flex items-center gap-2 text-xs text-terminal-dim">
          <span>run_id: {runId || "-"}</span>
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
            <p className="text-xs font-semibold text-terminal-dim">Session history (memory only)</p>
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
                    <p>time: {new Date(item.created_at).toLocaleTimeString()}</p>
                    <p>status: {statusSummary}</p>
                    <p>prompt: {item.prompt.slice(0, 80)}</p>
                  </button>
                );
              })}
            </div>
          </div>
        ) : null}
      </section>

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
