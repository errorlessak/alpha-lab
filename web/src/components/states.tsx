/**
 * Loading, empty and error states (§13.5).
 *
 * Three rules, all from the quality bar:
 *
 * - **Skeletons matched to the final layout.** No spinners, no layout shift. A
 *   spinner says "wait"; a skeleton says "here is the shape of what is coming",
 *   and the second one is information.
 * - **Errors name the cause and the fix.** The API already returns RFC 9457
 *   problem documents whose `detail` does exactly that, so the UI renders that
 *   text rather than inventing something friendlier and less useful. "Something
 *   went wrong" appears nowhere in this codebase.
 * - **An empty screen is an invitation to act**, with the exact command.
 */

import { useState } from 'react'

import { ApiError } from '../lib/api'

/** A block skeleton sized to what will replace it. */
export function Skeleton({ height, className = '' }: { height: number; className?: string }) {
  return (
    <div
      aria-hidden
      className={`animate-pulse bg-[color:var(--color-paper-sunk)] ${className}`}
      style={{ height }}
    />
  )
}

export function SkeletonLines({ rows = 4 }: { rows?: number }) {
  return (
    <div className="space-y-2" aria-hidden>
      {Array.from({ length: rows }, (_, index) => (
        <Skeleton key={index} height={14} className={index % 3 === 2 ? 'w-2/3' : 'w-full'} />
      ))}
    </div>
  )
}

/**
 * An error, in the service's own words.
 *
 * The heading is the problem's `title`, the body is its `detail`. Both come
 * from the server, which knows which phase failed and what to do about it.
 */
export function ErrorState({ error, context }: { error: unknown; context?: string }) {
  const api = error instanceof ApiError ? error : null
  const title = api?.title ?? 'Request failed'
  const detail =
    api?.problem?.detail ??
    (error instanceof Error ? error.message : 'The service did not respond.')

  return (
    <div
      role="alert"
      className="border-l-2 border-[color:var(--color-oxblood)] bg-[color:var(--color-paper-sunk)] p-4"
    >
      <p className="font-display text-sm font-semibold text-[color:var(--color-ink)]">{title}</p>
      <p className="mt-1 text-xs leading-relaxed text-[color:var(--color-ink)]">{detail}</p>
      {context ? (
        <p className="tabular mt-2 text-2xs text-[color:var(--color-ink-muted)]">{context}</p>
      ) : null}
    </div>
  )
}

/**
 * A copyable command.
 *
 * §13.5 asks the empty state on `/` to be an invitation carrying the exact CLI
 * command. Copy confirms in place rather than through a toast, because a toast
 * is a second thing to look at for a one-word acknowledgement.
 */
export function CommandBlock({ command }: { command: string }) {
  const [copied, setCopied] = useState(false)

  return (
    <div className="flex items-stretch gap-2">
      <code className="tabular flex-1 overflow-x-auto whitespace-pre border border-[color:var(--color-rule)] bg-[color:var(--color-paper-sunk)] px-3 py-2 text-xs text-[color:var(--color-ink)]">
        {command}
      </code>
      <button
        type="button"
        className="shrink-0 border border-[color:var(--color-rule)] px-3 text-2xs text-[color:var(--color-ink-muted)] hover:border-[color:var(--color-ink)] hover:text-[color:var(--color-ink)]"
        onClick={() => {
          void navigator.clipboard?.writeText(command).then(
            () => {
              setCopied(true)
              window.setTimeout(() => setCopied(false), 2000)
            },
            () => setCopied(false),
          )
        }}
      >
        {copied ? 'Copied' : 'Copy'}
      </button>
    </div>
  )
}

/** An empty state: what is absent, and the command that fixes it. */
export function EmptyState({
  headline,
  body,
  command,
  aside,
}: {
  headline: string
  body: string
  command?: string
  aside?: string
}) {
  return (
    <div className="border border-dashed border-[color:var(--color-rule)] p-8">
      <p className="font-display text-base font-semibold text-[color:var(--color-ink)]">
        {headline}
      </p>
      <p className="mt-2 max-w-prose text-sm text-[color:var(--color-ink-muted)]">{body}</p>
      {command ? (
        <div className="mt-4 max-w-xl">
          <CommandBlock command={command} />
        </div>
      ) : null}
      {aside ? (
        <p className="mt-3 max-w-prose text-xs text-[color:var(--color-ink-muted)]">{aside}</p>
      ) : null}
    </div>
  )
}
