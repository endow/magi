import React from "react";

type RunMetaBarProps = {
  runId: string;
  threadId: string;
  turnIndex: number;
  selectedProfile: string;
  freshMode: boolean;
  totalTokens: number | null;
  totalCostUsd: number | null;
  onCopyRunId: () => void;
};

export default function RunMetaBar({
  runId,
  threadId,
  turnIndex,
  selectedProfile,
  freshMode,
  totalTokens,
  totalCostUsd,
  onCopyRunId
}: RunMetaBarProps) {
  return (
    <div className="mt-4 flex items-center gap-2 text-xs text-terminal-dim">
      <span>run_id: {runId || "-"}</span>
      <span>thread_id: {threadId || "-"}</span>
      <span>turn: {turnIndex > 0 ? turnIndex : "-"}</span>
      <span>mode: {selectedProfile || "auto"}</span>
      <span>fresh: {freshMode ? "on" : "off"}</span>
      <span>tokens: {typeof totalTokens === "number" ? totalTokens : "-"}</span>
      <span>cost_usd_est: {typeof totalCostUsd === "number" ? totalCostUsd.toFixed(6) : "-"}</span>
      <button
        type="button"
        onClick={onCopyRunId}
        disabled={!runId}
        className="rounded border border-terminal-border px-2 py-1 disabled:cursor-not-allowed disabled:opacity-50"
      >
        Copy
      </button>
    </div>
  );
}
