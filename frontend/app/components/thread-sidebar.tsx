import React from "react";

type AgentStatus = "OK" | "ERROR" | "LOADING";

type RunHistoryItemView = {
  run_id: string;
  thread_id: string;
  turn_index: number;
  profile: string;
  prompt: string;
  consensus: {
    provider: string;
    model: string;
    text: string;
    status: AgentStatus;
    latency_ms: number;
    error_message?: string | null;
  } | null;
  results: Array<{
    agent: "A" | "B" | "C";
    provider: string;
    model: string;
    text: string;
    status: AgentStatus;
    latency_ms: number;
    error_message?: string | null;
  }>;
  created_at: string;
};

type ThreadGroupView = {
  thread_id: string;
  latest_at: string;
  turns: RunHistoryItemView[];
};

type ThreadSidebarProps = {
  isBusy: boolean;
  threadGroups: ThreadGroupView[];
  threadId: string;
  editingThreadId: string | null;
  threadNameDraft: string;
  collapsedThreads: Record<string, boolean>;
  confirmDeleteThreadId: string | null;
  threadLabel: (threadId: string) => string;
  onStartNewChat: () => void;
  onThreadNameDraftChange: (value: string) => void;
  onSaveRenameThread: (threadId: string) => void;
  onCancelRenameThread: () => void;
  onToggleThreadCollapse: (threadId: string) => void;
  onBeginRenameThread: (threadId: string) => void;
  onToggleDeleteConfirm: (threadId: string) => void;
  onDeleteThread: (threadId: string) => void;
  onRestoreHistory: (item: RunHistoryItemView) => void;
};

export default function ThreadSidebar({
  isBusy,
  threadGroups,
  threadId,
  editingThreadId,
  threadNameDraft,
  collapsedThreads,
  confirmDeleteThreadId,
  threadLabel,
  onStartNewChat,
  onThreadNameDraftChange,
  onSaveRenameThread,
  onCancelRenameThread,
  onToggleThreadCollapse,
  onBeginRenameThread,
  onToggleDeleteConfirm,
  onDeleteThread,
  onRestoreHistory
}: ThreadSidebarProps) {
  return (
    <aside className="panel p-3 md:max-h-[calc(100vh-3rem)] md:overflow-auto">
      <button
        type="button"
        onClick={onStartNewChat}
        disabled={isBusy}
        className="w-full rounded border border-terminal-accent bg-[#1f120b] px-3 py-2 text-xs font-semibold text-terminal-accent disabled:cursor-not-allowed disabled:opacity-50"
      >
        Start New Chat
      </button>
      <p className="mt-3 text-xs font-semibold text-terminal-dim">Threads</p>
      <div className="mt-2 space-y-2">
        {threadGroups.length ? (
          threadGroups.map((group) => (
            <div
              key={group.thread_id}
              className={`rounded border px-2 py-2 transition-colors ${
                group.thread_id === threadId
                  ? "border-terminal-accent bg-[#091015]"
                  : "border-terminal-border bg-[#060a0f]"
              }`}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0 flex-1">
                  {editingThreadId === group.thread_id ? (
                    <div className="flex items-center gap-1">
                      <input
                        value={threadNameDraft}
                        onChange={(event) => onThreadNameDraftChange(event.target.value)}
                        onKeyDown={(event) => {
                          if (event.key === "Enter") onSaveRenameThread(group.thread_id);
                          if (event.key === "Escape") onCancelRenameThread();
                        }}
                        className="w-full rounded border border-terminal-accent bg-[#02060b] px-2 py-1 text-[11px] text-terminal-text outline-none"
                        maxLength={64}
                        autoFocus
                      />
                      <button
                        type="button"
                        onClick={() => onSaveRenameThread(group.thread_id)}
                        className="rounded border border-terminal-accent px-1.5 py-0.5 text-[10px] text-terminal-accent"
                      >
                        Save
                      </button>
                      <button
                        type="button"
                        onClick={onCancelRenameThread}
                        className="rounded border border-terminal-border px-1.5 py-0.5 text-[10px] text-terminal-dim"
                      >
                        Cancel
                      </button>
                    </div>
                  ) : (
                    <p className="truncate text-[11px] font-semibold text-terminal-text">{threadLabel(group.thread_id)}</p>
                  )}
                  <p className="text-[11px] text-terminal-dim">{group.turns.length} turns</p>
                </div>
                <div className="flex items-center gap-1 pl-1">
                  <button
                    type="button"
                    onClick={() => onToggleThreadCollapse(group.thread_id)}
                    title={collapsedThreads[group.thread_id] ? "Expand thread" : "Fold thread"}
                    className="rounded border border-terminal-border px-1.5 py-0.5 text-[10px] text-terminal-dim hover:border-terminal-accent hover:text-terminal-text"
                  >
                    {collapsedThreads[group.thread_id] ? ">" : "v"}
                  </button>
                  <button
                    type="button"
                    onClick={() => onBeginRenameThread(group.thread_id)}
                    title="Rename thread"
                    className="rounded border border-terminal-border px-1.5 py-0.5 text-[10px] text-terminal-dim hover:border-terminal-accent hover:text-terminal-text"
                  >
                    Edit
                  </button>
                  <button
                    type="button"
                    onClick={() => onToggleDeleteConfirm(group.thread_id)}
                    title="Delete thread"
                    className="rounded border border-terminal-err px-1.5 py-0.5 text-[10px] text-terminal-err hover:opacity-90"
                  >
                    Del
                  </button>
                </div>
              </div>

              <p className="mt-1 text-[11px] text-terminal-dim">id: {group.thread_id}</p>
              <p className="text-[11px] text-terminal-dim">
                updated: {group.latest_at ? new Date(group.latest_at).toLocaleString() : "-"}
              </p>

              {confirmDeleteThreadId === group.thread_id ? (
                <div className="mt-2 rounded border border-terminal-err bg-[#160a0a] p-2 text-[11px]">
                  <p className="text-terminal-err">Delete this thread permanently?</p>
                  <div className="mt-1 flex items-center gap-1">
                    <button
                      type="button"
                      onClick={() => onDeleteThread(group.thread_id)}
                      className="rounded border border-terminal-err px-2 py-0.5 text-[10px] text-terminal-err"
                    >
                      Confirm delete
                    </button>
                    <button
                      type="button"
                      onClick={() => onToggleDeleteConfirm(group.thread_id)}
                      className="rounded border border-terminal-border px-2 py-0.5 text-[10px] text-terminal-dim"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              ) : null}

              <div className={`mt-2 space-y-1 ${collapsedThreads[group.thread_id] ? "hidden" : ""}`}>
                {group.turns.map((item) => {
                  const statusSummary = item.results.map((result) => `${result.agent}:${result.status}`).join(" ");
                  return (
                    <button
                      key={item.run_id}
                      type="button"
                      onClick={() => onRestoreHistory(item)}
                      className="w-full rounded border border-terminal-border bg-[#02060b] px-2 py-2 text-left text-xs text-terminal-dim transition-colors hover:border-terminal-accent hover:text-terminal-text"
                    >
                      <p>{new Date(item.created_at).toLocaleString()}</p>
                      <p>turn: {item.turn_index}</p>
                      <p>mode: {item.profile}</p>
                      <p>status: {statusSummary}</p>
                      <p className="truncate">prompt: {item.prompt}</p>
                    </button>
                  );
                })}
              </div>
            </div>
          ))
        ) : (
          <p className="text-xs text-terminal-dim">No history yet.</p>
        )}
      </div>
    </aside>
  );
}
