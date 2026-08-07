/** `/runs/$id/backtest` — the trades, and the cost curve that judges them. */

import { useQuery } from '@tanstack/react-query'
import { createFileRoute } from '@tanstack/react-router'
import { useVirtualizer } from '@tanstack/react-virtual'
import { useRef } from 'react'

import { Plate } from '../components/Plate'
import { SeriesChart } from '../components/SeriesChart'
import { stampFrom } from '../components/ProvenanceStamp'
import { ErrorState, Skeleton, SkeletonLines } from '../components/states'
import { api } from '../lib/api'
import { fmtBps, fmtDate, fmtNum, fmtPct } from '../lib/format'
import { useRun } from '../lib/run-context'

export const Route = createFileRoute('/runs/$runId/backtest')({
  component: BacktestStage,
})

const ROW_HEIGHT = 28

function BacktestStage() {
  const { runId, tearsheet } = useRun()

  const rolling = useQuery({
    queryKey: ['series', runId, 'rolling_sharpe', 1200],
    queryFn: () => api.getSeries(runId, 'rolling_sharpe', 1200),
    retry: false,
  })
  const trades = useQuery({
    queryKey: ['trades', runId],
    queryFn: () => api.getTrades(runId, { limit: 1000 }),
    retry: false,
  })

  if (!tearsheet) return <SkeletonLines rows={6} />

  const curve = tearsheet.backtest?.cost_sensitivity ?? []
  const stamp = stampFrom(tearsheet)

  return (
    <div data-plates className="max-w-[1200px]">
      <Plate
        id="costs"
        title="Cost sensitivity"
        meta={`break-even ${fmtBps(tearsheet.backtest?.break_even_bps)} against ${fmtBps(tearsheet.backtest?.round_trip_bps)} assumed`}
        stamp={stamp}
      >
        <p className="mb-4 max-w-prose text-xs text-[color:var(--color-ink-muted)]">
          The same signal and the same sizing, re-run at each cost level. A strategy whose Sharpe
          collapses between 0 and 2 basis points is not a strategy — it is an artifact of assuming
          free execution, and the shape of this curve says so at a glance.
        </p>

        {curve.length > 0 ? (
          <table className="w-full max-w-2xl border-collapse text-xs">
            <caption className="sr-only">Sharpe against round-trip cost</caption>
            <thead>
              <tr className="border-b border-[color:var(--color-ink)] text-left">
                <th scope="col" className="py-2 pr-3 text-right font-medium">
                  round trip
                </th>
                <th scope="col" className="py-2 pr-3 text-right font-medium">
                  Sharpe
                </th>
                <th scope="col" className="py-2 pr-3 text-right font-medium">
                  total return
                </th>
                <th scope="col" className="py-2 pr-3 text-right font-medium">
                  trades
                </th>
                <th scope="col" className="w-1/3 py-2 font-medium">
                  <span className="sr-only">Sharpe, drawn to scale</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {curve.map((point) => (
                <tr key={point['bps']} className="border-b border-[color:var(--color-rule)]">
                  <td className="tabular py-1.5 pr-3 text-right">{fmtBps(point['bps'])}</td>
                  <td className="tabular py-1.5 pr-3 text-right">{fmtNum(point['sharpe'], 3)}</td>
                  <td className="tabular py-1.5 pr-3 text-right">
                    {fmtPct(point['total_return'], 2)}
                  </td>
                  <td className="tabular py-1.5 pr-3 text-right">{fmtNum(point['n_trades'], 0)}</td>
                  <td className="py-1.5">
                    <Bar value={point['sharpe'] ?? 0} max={maxAbs(curve)} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="text-sm text-[color:var(--color-ink-muted)]">This run has no cost sweep.</p>
        )}
      </Plate>

      <Plate id="rolling" title="Rolling Sharpe" meta="63-bar window" stamp={stamp}>
        {rolling.isPending ? <Skeleton height={200} /> : null}
        {rolling.isError ? (
          <ErrorState
            error={rolling.error}
            context={`GET /api/v1/runs/${runId}/series/rolling_sharpe`}
          />
        ) : null}
        {rolling.data ? (
          <SeriesChart
            series={rolling.data}
            label="Rolling Sharpe"
            stroke="var(--color-prussian)"
            height={200}
            zeroLine
          />
        ) : null}
      </Plate>

      <Plate
        id="trades"
        title="Trades"
        meta={trades.data ? `${trades.data.total.toLocaleString()} realising events` : undefined}
        stamp={stamp}
      >
        <p className="mb-3 max-w-prose text-2xs text-[color:var(--color-ink-muted)]">
          A row is one realising event: a fill that reduced the position, with its own entry price
          and P&amp;L. A same-direction rebalance is not a round trip and does not appear here.
        </p>

        {trades.isPending ? <SkeletonLines rows={8} /> : null}
        {trades.isError ? (
          <ErrorState error={trades.error} context={`GET /api/v1/runs/${runId}/trades`} />
        ) : null}
        {trades.data ? <TradeTable rows={trades.data.rows} /> : null}
      </Plate>
    </div>
  )
}

function TradeTable({ rows }: { rows: Array<Record<string, unknown>> }) {
  const parent = useRef<HTMLDivElement>(null)
  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => parent.current,
    estimateSize: () => ROW_HEIGHT,
    overscan: 12,
  })

  if (rows.length === 0) {
    return (
      <p className="text-sm text-[color:var(--color-ink-muted)]">
        This run closed no trades. The position was never reduced.
      </p>
    )
  }

  return (
    <>
      <div className="tabular grid grid-cols-7 gap-2 border-b border-[color:var(--color-ink)] pb-1 text-2xs font-medium">
        <span>entry</span>
        <span>exit</span>
        <span className="text-right">side</span>
        <span className="text-right">qty</span>
        <span className="text-right">entry px</span>
        <span className="text-right">exit px</span>
        <span className="text-right">P&amp;L</span>
      </div>
      <div ref={parent} className="max-h-[420px] overflow-auto" tabIndex={0}>
        <div style={{ height: virtualizer.getTotalSize(), position: 'relative' }}>
          {virtualizer.getVirtualItems().map((item) => {
            const row = rows[item.index]
            if (!row) return null
            const side = Number(row['side'] ?? 0)
            const pnl = Number(row['pnl'] ?? 0)
            return (
              <div
                key={item.key}
                className="tabular absolute inset-x-0 grid grid-cols-7 gap-2 border-b border-[color:var(--color-rule)] py-1 text-2xs"
                style={{ height: ROW_HEIGHT, transform: `translateY(${item.start}px)` }}
              >
                <span>{fmtDate(String(row['entry_ts'] ?? ''))}</span>
                <span>{fmtDate(String(row['exit_ts'] ?? ''))}</span>
                {/* ▲/▼ rather than colour alone (§13.2) */}
                <span className="text-right">{side >= 0 ? '▲ long' : '▼ short'}</span>
                <span className="text-right">{fmtNum(Number(row['quantity'] ?? 0), 0)}</span>
                <span className="text-right">{fmtNum(Number(row['entry_price'] ?? 0), 2)}</span>
                <span className="text-right">{fmtNum(Number(row['exit_price'] ?? 0), 2)}</span>
                <span
                  className="text-right"
                  style={{ color: pnl < 0 ? 'var(--color-danger-text)' : 'var(--color-sage)' }}
                >
                  {fmtNum(pnl, 2)}
                </span>
              </div>
            )
          })}
        </div>
      </div>
    </>
  )
}

function Bar({ value, max }: { value: number; max: number }) {
  const width = max > 0 ? Math.min((Math.abs(value) / max) * 100, 100) : 0
  return (
    <span
      aria-hidden
      className="block h-2"
      style={{
        width: `${width}%`,
        background: value < 0 ? 'var(--color-oxblood)' : 'var(--color-prussian)',
      }}
    />
  )
}

function maxAbs(curve: Array<Record<string, number>>): number {
  return curve.reduce((peak, point) => Math.max(peak, Math.abs(point['sharpe'] ?? 0)), 0)
}
