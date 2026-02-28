import React from "react";

type ConsensusStatus = "OK" | "ERROR" | "LOADING";

type ConsensusResultView = {
  provider: string;
  model: string;
  text: string;
  status: ConsensusStatus;
  latency_ms: number;
  error_message?: string | null;
  error_code?: string | null;
};

type ParsedConsensus = {
  main: string;
  voteDetails: string | null;
};

type ConsensusPanelProps = {
  consensus: ConsensusResultView;
  parsedConsensus: ParsedConsensus;
  isStrictDebate: boolean;
  isUltra: boolean;
};

function statusClass(status: ConsensusStatus): string {
  if (status === "OK") return "status-ok";
  if (status === "ERROR") return "status-error";
  return "status-loading";
}

export default function ConsensusPanel({
  consensus,
  parsedConsensus,
  isStrictDebate,
  isUltra
}: ConsensusPanelProps) {
  return (
    <section className="panel mt-6 p-4">
      <div className="flex items-center justify-between border-b border-terminal-border pb-2 text-sm">
        <div className="flex items-center gap-2">
          <span className="font-semibold tracking-wide text-terminal-accent">Consensus Core</span>
          {isStrictDebate ? (
            <span className="rounded border border-terminal-accent px-2 py-0.5 text-[11px] text-terminal-accent">
              strict debate
            </span>
          ) : null}
          {isUltra ? (
            <span className="rounded border border-terminal-err px-2 py-0.5 text-[11px] text-terminal-err">
              high cost
            </span>
          ) : null}
        </div>
        <span className={statusClass(consensus.status)}>{consensus.status}</span>
      </div>
      <div className="mt-3 space-y-2 text-xs">
        <p className="text-terminal-dim">model: {consensus.provider}/{consensus.model}</p>
        <p className="text-terminal-dim">latency_ms: {consensus.latency_ms}</p>
        {consensus.error_code ? <p className="text-terminal-dim">error_code: {consensus.error_code}</p> : null}
        {consensus.error_message ? <p className="status-error">error: {consensus.error_message}</p> : null}
        <pre className="mt-2 whitespace-pre-wrap break-words rounded-md bg-[#02060b] p-3 text-sm leading-6">
          {parsedConsensus.main}
        </pre>
        {parsedConsensus.voteDetails ? (
          <details className="rounded-md border border-terminal-border bg-[#02060b] p-3">
            <summary className="cursor-pointer text-sm font-semibold text-terminal-dim">Vote details</summary>
            <pre className="mt-2 whitespace-pre-wrap break-words text-sm leading-6">{parsedConsensus.voteDetails}</pre>
          </details>
        ) : null}
      </div>
    </section>
  );
}
