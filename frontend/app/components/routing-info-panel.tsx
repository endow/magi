import React from "react";

type RoutingInfo = {
  profile: string;
  reason?: string | null;
  intent?: string | null;
  complexity?: string | null;
  execution_tier?: string | null;
};

type RoutingInfoPanelProps = {
  routingInfo: RoutingInfo | null;
};

export default function RoutingInfoPanel({ routingInfo }: RoutingInfoPanelProps) {
  if (!routingInfo) return null;

  return (
    <div className="mt-2 rounded border border-terminal-border bg-[#050a10] px-3 py-2 text-xs text-terminal-dim">
      <p>
        routing: <span className="text-terminal-text">{routingInfo.profile}</span>
        {routingInfo.intent ? ` | intent=${routingInfo.intent}` : ""}
        {routingInfo.complexity ? ` | complexity=${routingInfo.complexity}` : ""}
        {routingInfo.execution_tier ? ` | tier=${routingInfo.execution_tier}` : ""}
      </p>
      <p className="mt-1">
        reason: <span className="text-terminal-text">{routingInfo.reason || "-"}</span>
      </p>
    </div>
  );
}
