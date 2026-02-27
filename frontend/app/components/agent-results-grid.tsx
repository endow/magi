import React from "react";

type AgentId = "A" | "B" | "C";
type AgentStatus = "OK" | "ERROR" | "LOADING";

type AgentResultView = {
  agent: AgentId;
  provider: string;
  model: string;
  text: string;
  status: AgentStatus;
  latency_ms: number;
  error_message?: string | null;
};

type AgentResultsGridProps = {
  cards: AgentResultView[];
  isBusy: boolean;
  onRetry: (agent: AgentId) => void;
  onCopy: (result: AgentResultView) => void;
};

function statusClass(status: AgentStatus): string {
  if (status === "OK") return "status-ok";
  if (status === "ERROR") return "status-error";
  return "status-loading";
}

function providerDisplayName(provider: string): string {
  const normalized = provider.trim().toLowerCase();
  if (normalized === "openai") return "OpenAI";
  if (normalized === "anthropic") return "Claude";
  if (normalized === "gemini" || normalized === "google") return "Gemini";
  if (!normalized || normalized === "-") return "-";
  return provider;
}

export default function AgentResultsGrid({ cards, isBusy, onRetry, onCopy }: AgentResultsGridProps) {
  if (!cards.length) return null;

  return (
    <section className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-3">
      {cards.map((result) => (
        <article key={result.agent} className="panel p-4">
          <div className="flex items-center justify-between border-b border-terminal-border pb-2 text-sm">
            <div className="flex items-center gap-2">
              <span className="font-semibold text-terminal-accent">
                {providerDisplayName(result.provider)}/{result.model}
              </span>
              <span className="text-[11px] text-terminal-dim">Agent {result.agent}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className={statusClass(result.status)}>{result.status}</span>
              {result.status === "ERROR" ? (
                <button
                  type="button"
                  onClick={() => onRetry(result.agent)}
                  disabled={isBusy}
                  className="rounded border border-terminal-err px-2 py-1 text-[11px] text-terminal-err disabled:cursor-not-allowed disabled:opacity-50"
                >
                  Retry
                </button>
              ) : (
                <button
                  type="button"
                  onClick={() => onCopy(result)}
                  disabled={!result.text || isBusy}
                  className="rounded border border-terminal-border px-2 py-1 text-[11px] disabled:cursor-not-allowed disabled:opacity-50"
                >
                  Copy
                </button>
              )}
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
  );
}
