/**
 * A line or area chart over a downsampled series.
 *
 * uPlot, on canvas. §13.5 requires a 100k-point equity chart to pan at 60fps,
 * and the way that is achieved is not a faster renderer: **the server
 * downsamples with LTTB to the requested pixel width**, so the browser never
 * receives 100k points in the first place. This component asks for the width it
 * is actually drawing at.
 *
 * Styled from the tokens rather than uPlot's defaults, which are recognisable
 * on sight and belong to a different product.
 */

import { useEffect, useMemo, useRef } from 'react'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'

import type { SeriesResponse } from '../lib/api'

interface SeriesChartProps {
  series: SeriesResponse | undefined
  /** Which numeric column to draw. Defaults to the first. */
  column?: string
  height?: number
  /** Fill under the line — used for the underwater plot. */
  fill?: string
  stroke?: string
  label: string
  /** Draw a zero baseline, for drawdown and returns. */
  zeroLine?: boolean
}

export function SeriesChart({
  series,
  column,
  height = 220,
  fill,
  stroke = 'var(--color-prussian)',
  label,
  zeroLine = false,
}: SeriesChartProps) {
  const host = useRef<HTMLDivElement>(null)
  const plot = useRef<uPlot | null>(null)

  const data = useMemo(() => toData(series, column), [series, column])

  useEffect(() => {
    const element = host.current
    if (!element || !data) return

    const options: uPlot.Options = {
      width: element.clientWidth || 600,
      height,
      padding: [8, 4, 0, 0],
      legend: { show: false },
      cursor: { drag: { x: true, y: false }, points: { size: 6 } },
      axes: [
        {
          stroke: 'var(--color-ink-muted)',
          grid: { stroke: 'var(--color-rule)', width: 1 },
          ticks: { stroke: 'var(--color-rule)', width: 1 },
          font: '11px "IBM Plex Mono", monospace',
        },
        {
          stroke: 'var(--color-ink-muted)',
          grid: { stroke: 'var(--color-rule)', width: 1 },
          ticks: { stroke: 'var(--color-rule)', width: 1 },
          font: '11px "IBM Plex Mono", monospace',
          size: 56,
        },
      ],
      series: [
        {},
        {
          label,
          stroke,
          width: 1.5,
          ...(fill ? { fill } : {}),
        },
      ],
      hooks: zeroLine
        ? {
            draw: [
              (u) => {
                const y = u.valToPos(0, 'y', true)
                const { ctx } = u
                ctx.save()
                ctx.strokeStyle = 'var(--color-ink-muted)'
                ctx.setLineDash([3, 3])
                ctx.beginPath()
                ctx.moveTo(u.bbox.left, y)
                ctx.lineTo(u.bbox.left + u.bbox.width, y)
                ctx.stroke()
                ctx.restore()
              },
            ],
          }
        : {},
    }

    plot.current = new uPlot(options, data, element)

    const observer = new ResizeObserver(() => {
      plot.current?.setSize({ width: element.clientWidth, height })
    })
    observer.observe(element)

    return () => {
      observer.disconnect()
      plot.current?.destroy()
      plot.current = null
    }
  }, [data, height, stroke, fill, label, zeroLine])

  if (!series) return <div style={{ height }} aria-hidden />

  if (!data) {
    return (
      <p className="text-xs text-[color:var(--color-ink-muted)]">
        {label} has no numeric column to draw.
      </p>
    )
  }

  return (
    <figure className="m-0">
      <div ref={host} role="img" aria-label={`${label}, ${series.n_points} points`} />
      <figcaption className="tabular mt-1 text-2xs text-[color:var(--color-ink-muted)]">
        {series.downsampled
          ? `${series.n_points.toLocaleString()} of ${series.n_source_points.toLocaleString()} points, downsampled server-side`
          : `${series.n_points.toLocaleString()} points`}
      </figcaption>
    </figure>
  )
}

function toData(series: SeriesResponse | undefined, column?: string): uPlot.AlignedData | null {
  if (!series) return null
  const name = column && column in series.values ? column : Object.keys(series.values)[0]
  if (!name) return null
  const values = series.values[name]
  if (!values?.length) return null

  const x = series.index.map((iso) => {
    const parsed = Date.parse(iso)
    return Number.isFinite(parsed) ? parsed / 1000 : 0
  })
  return [x, values] as unknown as uPlot.AlignedData
}
