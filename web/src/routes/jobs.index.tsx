/** `/jobs` — what has been asked of the runner. */

import { useQuery } from '@tanstack/react-query'
import { Link, createFileRoute } from '@tanstack/react-router'

import { EmptyState, ErrorState, SkeletonLines } from '../components/states'
import { api } from '../lib/api'
import { fmtDateTime, fmtDuration } from '../lib/format'

export const Route = createFileRoute('/jobs/')({
  component: JobIndex,
})

function JobIndex() {
  const jobs = useQuery({
    queryKey: ['jobs'],
    queryFn: () => api.listJobs({ limit: 100 }),
    // Jobs change by design, unlike runs. This is the one query that refetches.
    staleTime: 0,
    refetchInterval: 3000,
  })

  const rows = jobs.data?.jobs ?? []

  return (
    <div className="max-w-[1000px]">
      <h1
        className="font-display text-xl font-semibold text-[color:var(--color-ink)]"
        style={{ fontStretch: '88%' }}
      >
        Jobs
      </h1>

      {jobs.isPending ? <SkeletonLines rows={5} /> : null}
      {jobs.isError ? <ErrorState error={jobs.error} context="GET /api/v1/jobs" /> : null}

      {jobs.isSuccess && rows.length === 0 ? (
        <div className="mt-4">
          <EmptyState
            headline="Nothing has been submitted."
            body="Jobs appear here when a run is started through the API. From a shell, the CLI does the same work without a job record:"
            command="uv run alpha run -c configs/experiments/smoke.yaml"
          />
        </div>
      ) : null}

      {rows.length > 0 ? (
        <table className="mt-4 w-full border-collapse text-xs">
          <caption className="sr-only">Submitted jobs, newest first</caption>
          <thead>
            <tr className="border-b border-[color:var(--color-ink)] text-left">
              <th scope="col" className="py-2 pr-3 font-medium">
                job
              </th>
              <th scope="col" className="py-2 pr-3 font-medium">
                type
              </th>
              <th scope="col" className="py-2 pr-3 font-medium">
                status
              </th>
              <th scope="col" className="py-2 pr-3 font-medium">
                progress
              </th>
              <th scope="col" className="py-2 pr-3 font-medium">
                submitted
              </th>
              <th scope="col" className="py-2 pr-3 text-right font-medium">
                took
              </th>
              <th scope="col" className="py-2 font-medium">
                run
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((job) => (
              <tr key={job.job_id} className="border-b border-[color:var(--color-rule)]">
                <td className="tabular py-1.5 pr-3">
                  <Link
                    to="/jobs/$jobId"
                    params={{ jobId: job.job_id }}
                    className="text-[color:var(--color-link)] underline underline-offset-2"
                  >
                    {job.job_id.slice(0, 8)}
                  </Link>
                </td>
                <td className="py-1.5 pr-3">{job.type}</td>
                <td className="py-1.5 pr-3">{job.status}</td>
                <td className="tabular py-1.5 pr-3 text-[color:var(--color-ink-muted)]">
                  {job.progress.phase || '—'}
                  {typeof job.progress.fraction === 'number'
                    ? ` ${(job.progress.fraction * 100).toFixed(0)}%`
                    : ''}
                </td>
                <td className="tabular py-1.5 pr-3 text-[color:var(--color-ink-muted)]">
                  {fmtDateTime(job.created_at)}
                </td>
                <td className="tabular py-1.5 pr-3 text-right">{fmtDuration(job.duration_s)}</td>
                <td className="tabular py-1.5">
                  {job.run_id ? (
                    <Link
                      to="/runs/$runId"
                      params={{ runId: job.run_id }}
                      className="text-[color:var(--color-link)] underline underline-offset-2"
                    >
                      open
                    </Link>
                  ) : (
                    '—'
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
    </div>
  )
}
