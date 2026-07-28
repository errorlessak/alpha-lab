/**
 * The run layout: header, stage rail, and the Fold Strip.
 *
 * The Fold Strip lives here rather than on the overview because §13.2 makes it
 * the global fold selector. It stays on screen across every sub-route, so the
 * validation scheme is never out of sight while a number is being read.
 *
 * The stage rail is the sub-route navigation, in **pipeline order**, numbered.
 * The `frontend-design` skill warns that numbered markers are decoration unless
 * the content really is a sequence — here it is, and the numbering is
 * load-bearing: a run that stopped at stage three makes stages four onward
 * meaningless, and the rail says how far it got.
 */

import { useQuery } from '@tanstack/react-query'
import { Link, Outlet, createFileRoute } from '@tanstack/react-router'
import { useEffect, useState } from 'react'

import { FoldStrip } from '../components/FoldStrip'
import { ErrorState, Skeleton } from '../components/states'
import { api } from '../lib/api'
import { fmtDate, shortId } from '../lib/format'
import { RunContext } from '../lib/run-context'

export const Route = createFileRoute('/runs/$runId')({
  component: RunLayout,
})

const STAGES = [
  { to: '/runs/$runId/data', label: 'data' },
  { to: '/runs/$runId/features', label: 'features' },
  { to: '/runs/$runId/folds', label: 'folds' },
  { to: '/runs/$runId/models', label: 'models' },
  { to: '/runs/$runId/backtest', label: 'backtest' },
  { to: '/runs/$runId/explain', label: 'explain' },
] as const

function RunLayout() {
  const { runId } = Route.useParams()
  const [selectedFold, setSelectedFold] = useState<number | null>(null)
  const reduceMotion = usePrefersReducedMotion()

  const run = useQuery({ queryKey: ['run', runId], queryFn: () => api.getRun(runId) })
  const folds = useQuery({
    queryKey: ['folds', runId],
    queryFn: () => api.getFolds(runId),
    retry: false,
  })
  const price = useQuery({
    queryKey: ['series', runId, 'equity', 400],
    queryFn: () => api.getSeries(runId, 'equity', 400),
    retry: false,
  })

  if (run.isError) {
    return (
      <div className="max-w-2xl">
        <ErrorState error={run.error} context={`GET /api/v1/runs/${runId}`} />
        <p className="mt-4 text-xs">
          <Link to="/" className="underline underline-offset-4">
            Back to the run index
          </Link>
        </p>
      </div>
    )
  }

  const sheet = run.data

  return (
    <RunContext.Provider value={{ runId, tearsheet: sheet, selectedFold }}>
      <header className="mb-5">
        <p className="tabular text-2xs text-[color:var(--color-ink-muted)]">
          <Link to="/" className="underline underline-offset-2">
            runs
          </Link>{' '}
          / {shortId(runId)}
        </p>
        <h1
          className="font-display mt-1 text-lg font-semibold text-[color:var(--color-ink)]"
          style={{ fontStretch: '88%' }}
        >
          {sheet ? sheet.run_id : <Skeleton height={26} className="w-96" />}
        </h1>
        {sheet ? (
          <p className="tabular mt-2 text-2xs text-[color:var(--color-ink-muted)]">
            {sheet.dataset.symbols.join(', ')} · {sheet.dataset.interval} · {sheet.dataset.calendar}{' '}
            · {fmtDate(sheet.dataset.span[0])} → {fmtDate(sheet.dataset.span[1])} ·{' '}
            {sheet.dataset.n_bars.toLocaleString()} bars
            <br />
            config {sheet.manifest.config_hash.slice(0, 8)} · data{' '}
            {sheet.manifest.data_hash?.slice(0, 8) ?? 'unrecorded'} ·{' '}
            {sheet.manifest.code_version ?? 'unrecorded'} · seed {sheet.manifest.seed} ·{' '}
            {sheet.dataset.periods_per_year.toFixed(2)} bars/year
          </p>
        ) : null}
      </header>

      <nav aria-label="Pipeline stages" className="mb-5 border-y border-[color:var(--color-rule)]">
        <ol className="flex flex-wrap items-center gap-x-1 gap-y-1 py-2 text-xs">
          <li>
            <StageLink to="/runs/$runId" runId={runId} index={0} label="overview" exact />
          </li>
          {STAGES.map((stage, index) => (
            <li key={stage.to} className="flex items-center">
              <span aria-hidden className="px-1 text-[color:var(--color-rule)]">
                ·
              </span>
              <StageLink to={stage.to} runId={runId} index={index + 1} label={stage.label} />
            </li>
          ))}
        </ol>
      </nav>

      <section
        aria-label="Walk-forward validation"
        className="sticky top-0 z-10 mb-6 border-b border-[color:var(--color-rule)] bg-[color:var(--color-paper)] pb-4 pt-2"
      >
        {folds.isPending ? (
          <Skeleton height={96} />
        ) : folds.isError ? (
          <p className="text-xs text-[color:var(--color-ink-muted)]">
            This run has no folds — it stopped before cross-validation, so nothing below it was
            validated.
          </p>
        ) : (
          <>
            <FoldStrip
              folds={folds.data}
              price={price.data}
              selected={selectedFold}
              onSelect={setSelectedFold}
              reduceMotion={reduceMotion}
            />
            <p className="tabular mt-3 text-2xs text-[color:var(--color-ink-muted)]">
              {folds.data?.scheme ?? 'unknown scheme'} · {folds.data?.folds.length ?? 0} folds ·{' '}
              {sumField(folds.data?.folds, 'purged').toLocaleString()} purged ·{' '}
              {sumField(folds.data?.folds, 'embargoed').toLocaleString()} embargoed ·{' '}
              {((folds.data?.embargo_pct ?? 0) * 100).toFixed(1)}% embargo
            </p>
          </>
        )}
      </section>

      <Outlet />
    </RunContext.Provider>
  )
}

function StageLink({
  to,
  runId,
  index,
  label,
  exact = false,
}: {
  to: string
  runId: string
  index: number
  label: string
  exact?: boolean
}) {
  return (
    <Link
      to={to}
      params={{ runId }}
      activeOptions={{ exact }}
      className="tabular inline-flex items-center gap-1.5 px-2 py-1 text-[color:var(--color-ink-muted)] hover:text-[color:var(--color-ink)]"
      activeProps={{
        className: 'text-[color:var(--color-ink)] border-b-2 border-[color:var(--color-prussian)]',
      }}
    >
      <span className="text-2xs text-[color:var(--color-rule)]">
        {String(index).padStart(2, '0')}
      </span>
      {label}
    </Link>
  )
}

function sumField(folds: Array<Record<string, unknown>> | undefined, field: string): number {
  if (!folds) return 0
  return folds.reduce((total, fold) => {
    const value = fold[field]
    return total + (typeof value === 'number' ? value : 0)
  }, 0)
}

/** Honours the operating system's reduced-motion setting, live. */
export function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(
    () => window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false,
  )

  useEffect(() => {
    const query = window.matchMedia?.('(prefers-reduced-motion: reduce)')
    if (!query) return
    const listener = (event: MediaQueryListEvent) => setReduced(event.matches)
    query.addEventListener('change', listener)
    return () => query.removeEventListener('change', listener)
  }, [])

  return reduced
}
