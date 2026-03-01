import React from "react";

type FeedbackPanelProps = {
  runId: string;
  threadId: string;
  isBusy: boolean;
  feedbackSubmitting: boolean;
  feedbackRating: -1 | 0 | 1 | null;
  feedbackReason: string;
  feedbackMessage: string;
  onSelectGood: () => void;
  onSelectBad: () => void;
  onReasonChange: (value: string) => void;
  onSubmit: () => void;
};

export default function FeedbackPanel({
  runId,
  threadId,
  isBusy,
  feedbackSubmitting,
  feedbackRating,
  feedbackReason,
  feedbackMessage,
  onSelectGood,
  onSelectBad,
  onReasonChange,
  onSubmit
}: FeedbackPanelProps) {
  const ready = Boolean(runId && threadId);

  return (
    <div className="mt-3 rounded border border-terminal-border bg-[#050a10] px-3 py-3 text-xs text-terminal-dim">
      <p className="font-semibold text-terminal-text">Rate this answer</p>
      {!ready ? <p className="mt-1 text-[11px] text-terminal-dim">Run a prompt to enable feedback.</p> : null}
      <div className="mt-2 flex items-center gap-2">
        <button
          type="button"
          onClick={onSelectGood}
          disabled={!ready || isBusy || feedbackSubmitting}
          className={`rounded border px-2 py-1 text-[11px] disabled:cursor-not-allowed disabled:opacity-50 ${
            feedbackRating === 1 ? "border-terminal-ok text-terminal-ok" : "border-terminal-border text-terminal-dim"
          }`}
        >
          Good
        </button>
        <button
          type="button"
          onClick={onSelectBad}
          disabled={!ready || isBusy || feedbackSubmitting}
          className={`rounded border px-2 py-1 text-[11px] disabled:cursor-not-allowed disabled:opacity-50 ${
            feedbackRating === -1 ? "border-terminal-err text-terminal-err" : "border-terminal-border text-terminal-dim"
          }`}
        >
          Bad
        </button>
      </div>
      <textarea
        value={feedbackReason}
        onChange={(event) => onReasonChange(event.target.value)}
        disabled={!ready || isBusy || feedbackSubmitting}
        placeholder="reason (optional)"
        className="mt-2 h-20 w-full resize-y rounded border border-terminal-border bg-[#02060b] px-2 py-1 text-xs text-terminal-text outline-none disabled:cursor-not-allowed disabled:opacity-50"
      />
      <div className="mt-2 flex items-center gap-2">
        <button
          type="button"
          onClick={onSubmit}
          disabled={!ready || isBusy || feedbackSubmitting || feedbackRating === null}
          className="rounded border border-terminal-accent px-2 py-1 text-[11px] text-terminal-accent disabled:cursor-not-allowed disabled:opacity-50"
        >
          {feedbackSubmitting ? "Saving..." : "Send Feedback"}
        </button>
        {feedbackMessage ? <span>{feedbackMessage}</span> : null}
      </div>
    </div>
  );
}
