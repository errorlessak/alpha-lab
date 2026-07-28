/**
 * `/jobs/$id` — the live console.
 *
 * The API streams `log`, `progress` and one terminal `status` event over SSE,
 * resumable from `Last-Event-ID`. `EventSource` handles the reconnect and the
 * resume header itself, which is the reason the server emits ids at all.
 *
 * The console follows the tail unless the reader has scrolled up. Yanking
 * someone back to the bottom while they are reading the line that explains the
 * failure is the worst thing this screen could do.
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link, createFileRoute } from '@tanstack/react-router'
import { useEffect, useRef, useState } from 'react'

import { ErrorState, SkeletonLines } from '../components/states'
import { api } from '../lib/api'
import { fmtDateTime, fmtDuration } from '../lib/format'

export const Route = createFileRoute('/jobs/$jobId')({
  component: JobConsole,
})

interface LogLine {
  seq: number
  ts: string
  level: string
  phase: string
  message: string
}

const TERMINAL = new Set(['completed', 'failed', 'cancelled'])

function JobConsole() {
  const { jobId } = Route.useParams()
  const queryClient = useQueryClient()
  const [lines, setLines] = useState<LogLine[]>([])
  const [streamClosed, setStreamClosed] = useState(false)

  const job = useQuery({
    queryKey: ['job', jobId],
    queryFn: () => api.getJob(jobId),
    staleTime: 0,
    refetchInterval: (query) =>
      query.state.data && TERMINAL.has(query.state.data.status) ? false : 2000,
  })

  const cancel = useMutation({
    mutationFn: () => api.cancelJob(jobId),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: ['job', jobId] }),
  })

  useEffect(() => {
    const source = new EventSource(api.jobEventsUrl(jobId))

    source.addEventListener('log', (event) => {
      const line = JSON.parse((event as MessageEvent).data) as LogLine
      setLines((previous) =>
        previous.some((existing) => existing.seq === line.seq) ? previous : [...previous, line],
      )
    })

    source.addEventListener('progress', () => {
      void queryClient.invalidateQueries({ queryKey: ['job', jobId] })
    })

    source.addEventListener('status', () => {
      void queryClient.invalidateQueries({ queryKey: ['job', jobId] })
      void queryClient.invalidateQueries({ queryKey: ['runs'] })
      setStreamClosed(true)
      source.close()
    })

    source.onerror = () => {
      // EventSource reconnects on its own and replays Last-Event-ID, so a
      // dropped connection is not an error worth showing. A closed one is.
      if (source.readyState === EventSource.CLOSED) setStreamClosed(true)
    }

    return () => source.close()
  }, [jobId, queryClient])

  if (job.isError) {
    return <ErrorState error={job.error} context={`GET /api/v1/jobs/${jobId}`} />
  }

  const record = job.data
  const isTerminal = record ? TERMINAL.has(record.status) : false

  return (
    <div className="max-w-[1000px]">
      <p className="tabular text-2xs text-[color:var(--color-ink-muted)]">
        <Link to="/jobs" className="underline underline-offset-2">
          jobs
        </Link>{' '}
        / {jobId.slice(0, 8)}
      </p>

      <div className="mt-1 flex flex-wrap items-baseline justify-between gap-3">
        <h1
          className="font-display text-lg font-semibold text-[color:var(--color-ink)]"
          style={{ fontStretch: '88%' }}
        >
          {record ? `${record.type} · ${record.status}` : 'job'}
        </h1>
        {record && !isTerminal ? (
          <button
            type="button"
            onClick={() => cancel.mutate()}
            disabled={cancel.isPending || record.cancel_requested}
            className="border border-[color:var(--color-rule)] px-3 py-1 text-2xs hover:border-[color:var(--color-ink)] disabled:opacity-50"
          >
            {record.cancel_requested ? 'Cancelling at the next phase' : 'Cancel'}
          </button>
        ) : null}
      </div>

      {record ? (
        <>
          <p className="tabular mt-2 text-2xs text-[color:var(--color-ink-muted)]">
            {record.config_ref} · submitted {fmtDateTime(record.created_at)} ·{' '}
            {fmtDuration(record.duration_s)}
            {record.run_id ? (
              <>
                {' · '}
                <Link
                  to="/runs/$runId"
                  params={{ runId: record.run_id }}
                  className="text-[color:var(--color-link)] underline underline-offset-2"
                >
                  open the run
                </Link>
              </>
            ) : null}
          </p>

          {record.progress.total > 0 ? (
            <div className="mt-3 max-w-md">
              <div className="tabular flex justify-between text-2xs text-[color:var(--color-ink-muted)]">
                <span>
                  {record.progress.phase} — {record.progress.message}
                </span>
                <span>
                  {record.progress.step} / {record.progress.total}
                </span>
              </div>
              <div className="mt-1 h-1.5 w-full bg-[color:var(--color-paper-sunk)]">
                <div
                  className="h-full bg-[color:var(--color-prussian)]"
                  style={{ width: `${(record.progress.fraction ?? 0) * 100}%` }}
                />
              </div>
            </div>
          ) : null}

          {record.error ? (
            <div className="mt-4">
              <ErrorState
                error={new Error(record.error)}
                context="The run directory is finalised — it will not claim to be complete."
              />
            </div>
          ) : null}
        </>
      ) : (
        <SkeletonLines rows={3} />
      )}

      <Console lines={lines} closed={streamClosed} />
    </div>
  )
}

function Console({ lines, closed }: { lines: LogLine[]; closed: boolean }) {
  const box = useRef<HTMLDivElement>(null)
  const [follow, setFollow] = useState(true)

  useEffect(() => {
    if (follow && box.current) box.current.scrollTop = box.current.scrollHeight
  }, [lines, follow])

  return (
    <section className="mt-6" aria-label="Job log">
      <div className="flex items-baseline justify-between">
        <h2 className="text-xs font-medium">Log</h2>
        <label className="flex items-center gap-2 text-2xs text-[color:var(--color-ink-muted)]">
          <input
            type="checkbox"
            checked={follow}
            onChange={(event) => setFollow(event.target.checked)}
          />
          follow the tail
        </label>
      </div>

      <div
        ref={box}
        tabIndex={0}
        className="tabular mt-2 max-h-[460px] overflow-auto border border-[color:var(--color-rule)] bg-[color:var(--color-paper-sunk)] p-3 text-2xs leading-relaxed"
        onScroll={(event) => {
          const element = event.currentTarget
          const atBottom = element.scrollHeight - element.scrollTop - element.clientHeight < 24
          if (!atBottom && follow) setFollow(false)
        }}
      >
        {lines.length === 0 ? (
          <p className="text-[color:var(--color-ink-muted)]">
            {closed ? 'This job produced no log lines.' : 'Waiting for the first line…'}
          </p>
        ) : (
          lines.map((line) => (
            <p key={line.seq} className="whitespace-pre-wrap">
              <span className="text-[color:var(--color-ink-muted)]">
                {line.ts.slice(11, 19)} {line.phase.padEnd(9)}
              </span>
              <span
                style={{
                  color:
                    line.level === 'ERROR'
                      ? 'var(--color-oxblood)'
                      : line.level === 'WARNING'
                        ? 'var(--color-sulphur)'
                        : 'var(--color-ink)',
                }}
              >
                {line.message}
              </span>
            </p>
          ))
        )}
      </div>

      {closed ? (
        <p className="tabular mt-2 text-2xs text-[color:var(--color-ink-muted)]">
          Stream closed. The complete log is in the run directory as logs.jsonl — this console shows
          the live tail.
        </p>
      ) : null}
    </section>
  )
}
