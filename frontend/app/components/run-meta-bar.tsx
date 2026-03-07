import React from "react";

type RunMetaBarProps = {
  runId: string;
  threadId: string;
  turnIndex: number;
  selectedProfile: string;
  resolvedProfile: string;
  freshMode: "auto" | "on" | "off";
  totalTokens: number | null;
  totalCostUsd: number | null;
  onCopyRunId: () => void;
};

export default function RunMetaBar({
  runId,
  threadId,
  turnIndex,
  selectedProfile,
  resolvedProfile,
  freshMode,
  totalTokens,
  totalCostUsd,
  onCopyRunId
}: RunMetaBarProps) {
  const modeLabel = selectedProfile || (resolvedProfile ? `auto -> ${resolvedProfile}` : "auto");
  const freshLabel = freshMode === "auto" ? "auto" : freshMode === "on" ? "on" : "off";
  return (
    <div className="mt-4 flex items-center gap-2 text-xs text-terminal-dim">
      <span>run_id: {runId || "-"}</span>
      <span>thread_id: {threadId || "-"}</span>
      <span>turn: {turnIndex > 0 ? turnIndex : "-"}</span>
      <span>profile: {modeLabel}</span>
      <span>fresh: {freshLabel}</span>
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
