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

export default function ChatTranscript({ messages }: ChatTranscriptProps) {
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
              <span className="flex items-center gap-3">
                {message.latency_ms && message.role === "assistant" ? (
                  <span>latency_ms: {message.latency_ms}</span>
                ) : null}
                {message.run_id ? <span>run_id: {message.run_id}</span> : null}
              </span>
            </div>
            <pre className="whitespace-pre-wrap break-words">{message.text}</pre>
          </article>
        ))}
      </div>
    </section>
  );
}
