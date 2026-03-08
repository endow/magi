"use client";

import { useState } from "react";

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  run_id?: string;
  turn_index?: number;
  latency_ms?: number;
};

type ChatTranscriptProps = {
  messages: ChatMessage[];
};

function CopyIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2.3">
      <rect x="8" y="8" width="11" height="11" rx="1.2" />
      <rect x="5" y="5" width="11" height="11" rx="1.2" fill="#0a0f17" />
    </svg>
  );
}

export default function ChatTranscript({ messages }: ChatTranscriptProps) {
  const [copiedId, setCopiedId] = useState<string | null>(null);

  async function copyMessage(message: ChatMessage) {
    try {
      await navigator.clipboard.writeText(message.text);
      setCopiedId(message.id);
      window.setTimeout(() => {
        setCopiedId((current) => (current === message.id ? null : current));
      }, 1200);
    } catch {
      // noop
    }
  }

  if (!messages.length) {
    return (
      <section className="panel mt-6 p-4">
        <div className="border-b border-terminal-border pb-2 text-sm font-semibold tracking-wide text-terminal-accent">
          Dialogue
        </div>
        <p className="mt-3 text-sm text-terminal-dim">No dialogue yet. Send a prompt in chat mode.</p>
      </section>
    );
  }

  return (
    <section className="panel mt-6 p-4">
      <div className="border-b border-terminal-border pb-2 text-sm font-semibold tracking-wide text-terminal-accent">
        Dialogue
      </div>
      <div className="mt-3 space-y-3">
        {messages.map((message) => (
          <article
            key={message.id}
            className={`rounded-md border p-3 text-sm leading-6 ${
              message.role === "assistant"
                ? "border-terminal-accent bg-[#06121c] text-terminal-text"
                : "border-terminal-border bg-[#02060b] text-terminal-text"
            }`}
          >
            <div className="mb-1 flex items-center justify-between text-[11px] text-terminal-dim">
              <span>{message.role === "assistant" ? "assistant" : "user"}</span>
              <span className="flex items-center gap-2">
                {message.latency_ms && message.role === "assistant" ? (
                  <span>latency_ms: {message.latency_ms}</span>
                ) : null}
                {message.run_id ? <span>run_id: {message.run_id}</span> : null}
                <button
                  type="button"
                  onClick={() => void copyMessage(message)}
                  className={`inline-flex h-9 w-9 items-center justify-center rounded-sm border transition ${
                    copiedId === message.id
                      ? "border-terminal-accent bg-[#0f1d29] text-terminal-accent"
                      : "border-terminal-border bg-[#0a0f17] text-terminal-text hover:bg-[#162033]"
                  }`}
                  aria-label={copiedId === message.id ? "Copied" : "Copy message"}
                  title={copiedId === message.id ? "Copied" : "Copy"}
                >
                  <CopyIcon />
                </button>
              </span>
            </div>
            <pre className="whitespace-pre-wrap break-words">{message.text}</pre>
          </article>
        ))}
      </div>
    </section>
  );
}
