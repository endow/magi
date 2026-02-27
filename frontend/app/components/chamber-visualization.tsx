import React from "react";

type AgentId = "A" | "B" | "C";
type NodeState = "IDLE" | "BLINK" | "ON" | "ERROR" | "RELAY";
type ConfidenceMap = Record<AgentId, number | null>;

type ChamberNode = {
  agent: AgentId;
  provider: string;
  model: string;
};

type ChamberVisualizationProps = {
  setChamberRef: (el: HTMLDivElement | null) => void;
  setPeerGroupRef: (el: HTMLDivElement | null) => void;
  setLocalNodeRef: (el: HTMLDivElement | null) => void;
  linkPaths: string[];
  linkViewBox: string;
  chamberActive: boolean;
  chamberMode: "idle" | "discussion" | "conclusion";
  localOnlyHandled: boolean;
  localNodeState: NodeState;
  showConclusion: boolean;
  localAgent: { provider: string; model: string } | undefined;
  localRouteHint: string;
  chamberNodes: ChamberNode[];
  showExecutingBadge: boolean;
  resolvedProfile: string;
  nodeStates: Record<AgentId, NodeState>;
  winnerAgent: AgentId | null;
  confidenceMap: ConfidenceMap;
  showDiscussionBadge: boolean;
  showRoutingBadge: boolean;
  conclusionElapsedMs: number | null;
  setNodeRef: (agent: AgentId, el: HTMLDivElement | null) => void;
  formatElapsedMsToMinSec: (elapsedMs: number) => string;
};

function providerDisplayName(provider: string): string {
  const normalized = provider.trim().toLowerCase();
  if (normalized === "openai") return "OpenAI";
  if (normalized === "anthropic") return "Claude";
  if (normalized === "gemini" || normalized === "google") return "Gemini";
  if (!normalized || normalized === "-") return "-";
  return provider;
}

function buildLlmLabel(provider: string, model: string, fallbackAgent: AgentId): string {
  if (provider === "-" || model === "-") return `LLM-${fallbackAgent}`;
  return providerDisplayName(provider);
}

export default function ChamberVisualization({
  setChamberRef,
  setPeerGroupRef,
  setLocalNodeRef,
  linkPaths,
  linkViewBox,
  chamberActive,
  chamberMode,
  localOnlyHandled,
  localNodeState,
  showConclusion,
  localAgent,
  localRouteHint,
  chamberNodes,
  showExecutingBadge,
  resolvedProfile,
  nodeStates,
  winnerAgent,
  confidenceMap,
  showDiscussionBadge,
  showRoutingBadge,
  conclusionElapsedMs,
  setNodeRef,
  formatElapsedMsToMinSec
}: ChamberVisualizationProps) {
  const nodePositionClass: Record<AgentId, string> = {
    A: "magi-node-a",
    B: "magi-node-b",
    C: "magi-node-c"
  };

  return (
    <div className="magi-wire mt-4 rounded-md p-3">
      <div ref={setChamberRef} className={`magi-chamber magi-chamber-${chamberMode} grid grid-cols-1 gap-3 md:block`}>
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
        <div
          ref={setPeerGroupRef}
          className={`magi-peer-group ${localOnlyHandled ? "magi-peer-group-skipped" : ""}`}
          aria-hidden="true"
        />
        <div ref={setLocalNodeRef} className="magi-node-wrap magi-node-local">
          <div className={`magi-node p-4 magi-node-${localNodeState.toLowerCase()} ${showConclusion ? "magi-node-conclusion" : ""}`}>
            <p className="magi-node-label">Local LLM</p>
            <p className="magi-node-model mt-2 text-sm font-semibold">
              {localAgent ? `${localAgent.provider}/${localAgent.model}` : "ollama/local"}
            </p>
            <div className="magi-node-status-slot mt-2 w-full px-6">
              <p className="mt-1 text-[11px] font-semibold">{localRouteHint}</p>
              {localNodeState === "BLINK" || localNodeState === "RELAY" ? <div className="magi-node-progress" /> : null}
            </div>
          </div>
        </div>
        {chamberNodes.map((node) => {
          const displayNodeState: NodeState =
            showExecutingBadge && !resolvedProfile && !localOnlyHandled ? "BLINK" : nodeStates[node.agent];
          return (
            <div
              key={`node-${node.agent}`}
              ref={(el) => setNodeRef(node.agent, el)}
              className={`magi-node-wrap ${nodePositionClass[node.agent]}`}
            >
              <div
                className={`magi-node p-4 magi-node-${displayNodeState.toLowerCase()} ${
                  winnerAgent && displayNodeState === "ON" && node.agent !== winnerAgent
                    ? "magi-node-rejected"
                    : ""
                } ${localOnlyHandled ? "magi-node-skipped" : ""} ${showConclusion ? "magi-node-conclusion" : ""}`}
              >
                <p className="magi-node-label">{buildLlmLabel(node.provider, node.model, node.agent)}</p>
                <p className="magi-node-model mt-2 text-sm font-semibold">
                  {node.provider === "-" ? `AGENT ${node.agent}` : `${node.provider}/${node.model}`}
                </p>
                <div className="magi-node-status-slot mt-2 w-full px-6">
                  {localOnlyHandled ? (
                    <p className="mt-1 text-[11px] font-semibold">skipped (not routed)</p>
                  ) : confidenceMap[node.agent] !== null ? (
                    <div className="magi-confidence-track">
                      <div className="magi-confidence-fill" style={{ width: `${confidenceMap[node.agent]}%` }} />
                    </div>
                  ) : null}
                  {!localOnlyHandled && confidenceMap[node.agent] !== null ? (
                    <p className="mt-1 text-[11px] font-semibold">confidence {confidenceMap[node.agent]}</p>
                  ) : (
                    <p className="mt-1 text-[11px] font-semibold opacity-0">confidence --</p>
                  )}
                  {!localOnlyHandled && displayNodeState === "BLINK" ? <div className="magi-node-progress" /> : null}
                </div>
              </div>
            </div>
          );
        })}
        {showConclusion || showDiscussionBadge || showRoutingBadge || showExecutingBadge ? (
          <>
            <div
              className={
                showConclusion
                  ? "magi-conclusion-badge"
                  : showExecutingBadge
                    ? "magi-executing-badge"
                    : showRoutingBadge
                      ? "magi-routing-badge"
                      : "magi-discussion-badge"
              }
            >
              {showConclusion ? "Conclusion" : showRoutingBadge ? "Routing / Prep" : showExecutingBadge ? "Executing" : "Discussion"}
            </div>
            {showConclusion && conclusionElapsedMs !== null ? (
              <div className="magi-conclusion-time">elapsed {formatElapsedMsToMinSec(conclusionElapsedMs)}</div>
            ) : null}
          </>
        ) : null}
      </div>
    </div>
  );
}
