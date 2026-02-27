import { FormEvent, KeyboardEvent } from "react";

type PromptFormProps = {
  prompt: string;
  maxPromptLength: number;
  isBusy: boolean;
  isLoading: boolean;
  isConsensusLoading: boolean;
  selectedProfile: string;
  availableProfiles: string[];
  freshMode: boolean;
  isStrictDebate: boolean;
  isUltra: boolean;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  onPromptChange: (value: string) => void;
  onPromptKeyDown: (event: KeyboardEvent<HTMLTextAreaElement>) => void;
  onProfileChange: (value: string) => void;
  onFreshModeChange: (value: boolean) => void;
};

export default function PromptForm({
  prompt,
  maxPromptLength,
  isBusy,
  isLoading,
  isConsensusLoading,
  selectedProfile,
  availableProfiles,
  freshMode,
  isStrictDebate,
  isUltra,
  onSubmit,
  onPromptChange,
  onPromptKeyDown,
  onProfileChange,
  onFreshModeChange
}: PromptFormProps) {
  return (
    <form className="mt-4 space-y-3" onSubmit={onSubmit}>
      <textarea
        className="h-40 w-full resize-y rounded-md border border-terminal-border bg-[#02060b] p-3 text-sm outline-none ring-terminal-accent focus:ring"
        placeholder="Type your prompt..."
        value={prompt}
        onChange={(event) => onPromptChange(event.target.value)}
        onKeyDown={onPromptKeyDown}
        disabled={isBusy}
      />
      <div className="flex items-center gap-3">
        <button
          type="submit"
          disabled={isBusy}
          className="rounded-md border border-terminal-accent bg-[#0d1d2a] px-4 py-2 text-sm text-terminal-accent disabled:cursor-not-allowed disabled:opacity-50"
        >
          {isLoading ? "Running..." : isConsensusLoading ? "Finalizing..." : "Run MAGI"}
        </button>
        <span className="text-xs text-terminal-dim">
          {prompt.length}/{maxPromptLength} chars
        </span>
        <label className="text-xs text-terminal-dim">
          mode:
          <select
            className="ml-2 rounded border border-terminal-border bg-[#02060b] px-2 py-1 text-xs"
            value={selectedProfile}
            onChange={(event) => onProfileChange(event.target.value)}
            disabled={isBusy}
          >
            <option value="">auto (unset)</option>
            {availableProfiles.map((profile) => (
              <option key={profile} value={profile}>
                {profile}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2 text-xs text-terminal-dim">
          <input
            type="checkbox"
            checked={freshMode}
            onChange={(event) => onFreshModeChange(event.target.checked)}
            disabled={isBusy}
            className="h-3.5 w-3.5 accent-terminal-accent"
          />
          fresh mode
        </label>
        {isStrictDebate ? (
          <span className="rounded border border-terminal-accent px-2 py-1 text-[11px] text-terminal-accent">
            strict debate
          </span>
        ) : null}
        {isUltra ? (
          <span className="rounded border border-terminal-err px-2 py-1 text-[11px] text-terminal-err">
            high cost
          </span>
        ) : null}
      </div>
      <p className="text-xs text-terminal-dim">Enter: submit / Shift+Enter: newline</p>
    </form>
  );
}
