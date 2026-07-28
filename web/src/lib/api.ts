/**
 * The API client.
 *
 * Types come from `api/generated/schema.d.ts`, which is generated from the
 * service's own OpenAPI document by `just client`. Nothing here restates a
 * shape by hand: a hand-written interface is a second source of truth that
 * drifts silently, and the whole point of generating the client is that it
 * cannot.
 *
 * Errors arrive as RFC 9457 problem documents whose `detail` already names the
 * cause and the fix — "Run a3f91c has no tearsheet.json. Its manifest says
 * failed". §13.5 asks the UI to say exactly that, so `ApiError` carries the
 * problem through unchanged and the UI renders it rather than inventing
 * something friendlier and less useful.
 */

import type { components, paths } from '../api/generated/schema'

export type Tearsheet = components['schemas']['Tearsheet']
export type LeaderboardRow = components['schemas']['LeaderboardRow']
export type RunSummaryRow = components['schemas']['RunSummaryRow']
export type RunPage = components['schemas']['RunPage']
export type SeriesResponse = components['schemas']['SeriesResponse']
export type TablePage = components['schemas']['TablePage']
export type FoldsResponse = components['schemas']['FoldsResponse']
export type ConfigResponse = components['schemas']['ConfigResponse']
export type LeaderboardResponse = components['schemas']['LeaderboardResponse']
export type CompareResponse = components['schemas']['CompareResponse']
export type ShapSummaryResponse = components['schemas']['ShapSummaryResponse']
export type DatasetList = components['schemas']['DatasetList']
export type DatasetDetail = components['schemas']['DatasetDetail']
export type Health = components['schemas']['Health']
export type JobOut = components['schemas']['JobOut']
export type JobList = components['schemas']['JobList']
export type JobSubmission = components['schemas']['JobSubmission']
export type ProblemDocument = components['schemas']['ProblemDocument']

export type SeriesName =
  paths['/api/v1/runs/{run_id}/series/{name}']['get']['parameters']['path']['name']

const BASE = '/api/v1'

/** An API failure carrying the server's own problem document. */
export class ApiError extends Error {
  readonly status: number
  readonly problem: ProblemDocument | null

  constructor(status: number, problem: ProblemDocument | null, fallback: string) {
    super(problem?.detail ?? fallback)
    this.name = 'ApiError'
    this.status = status
    this.problem = problem
  }

  /** The short summary, for a heading above the detail. */
  get title(): string {
    return this.problem?.title ?? 'Request failed'
  }

  /** True when the resource is simply absent rather than broken. */
  get isMissing(): boolean {
    return this.status === 404
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { accept: 'application/json', ...init?.headers },
  })

  if (!response.ok) {
    let problem: ProblemDocument | null = null
    try {
      problem = (await response.json()) as ProblemDocument
    } catch {
      // A non-JSON error body means something upstream of the application
      // answered — a proxy, usually. There is no problem document to show.
      problem = null
    }
    throw new ApiError(response.status, problem, `${response.status} ${response.statusText}`)
  }

  return (await response.json()) as T
}

export const api = {
  health: () => request<Health>('/health'),

  listRuns: (params: { symbol?: string; status?: string; limit?: number; cursor?: string } = {}) =>
    request<RunPage>(`/runs${query(params)}`),

  getRun: (runId: string) => request<Tearsheet>(`/runs/${encodeURIComponent(runId)}`),

  getConfig: (runId: string) =>
    request<ConfigResponse>(`/runs/${encodeURIComponent(runId)}/config`),

  getFolds: (runId: string) => request<FoldsResponse>(`/runs/${encodeURIComponent(runId)}/folds`),

  getLeaderboard: (runId: string) =>
    request<LeaderboardResponse>(`/runs/${encodeURIComponent(runId)}/leaderboard`),

  getSeries: (runId: string, name: SeriesName, width: number) =>
    request<SeriesResponse>(`/runs/${encodeURIComponent(runId)}/series/${name}${query({ width })}`),

  getTrades: (runId: string, params: { limit?: number; cursor?: string } = {}) =>
    request<TablePage>(`/runs/${encodeURIComponent(runId)}/trades${query(params)}`),

  getPredictions: (
    runId: string,
    params: { fold?: number; limit?: number; cursor?: string } = {},
  ) => request<TablePage>(`/runs/${encodeURIComponent(runId)}/predictions${query(params)}`),

  getShapSummary: (runId: string) =>
    request<ShapSummaryResponse>(`/runs/${encodeURIComponent(runId)}/shap/summary`),

  getResearch: (runId: string) =>
    request<{ run_id: string; markdown: string }>(`/runs/${encodeURIComponent(runId)}/research`),

  compare: (a: string, b: string) => request<CompareResponse>(`/runs/compare${query({ a, b })}`),

  listDatasets: () => request<DatasetList>('/datasets'),

  getDataset: (datasetId: string) =>
    request<DatasetDetail>(`/datasets/${encodeURIComponent(datasetId)}`),

  listJobs: (params: { status?: string; limit?: number } = {}) =>
    request<JobList>(`/jobs${query(params)}`),

  getJob: (jobId: string) => request<JobOut>(`/jobs/${encodeURIComponent(jobId)}`),

  createJob: (submission: JobSubmission) =>
    request<JobOut>('/jobs', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(submission),
    }),

  cancelJob: (jobId: string) =>
    request<JobOut>(`/jobs/${encodeURIComponent(jobId)}/cancel`, { method: 'POST' }),

  /** The URL of a job's SSE stream, for `EventSource`. */
  jobEventsUrl: (jobId: string) => `${BASE}/jobs/${encodeURIComponent(jobId)}/events`,

  /** The URL of an artifact download, for a plain anchor. */
  exportUrl: (runId: string, artifact: string) =>
    `${BASE}/runs/${encodeURIComponent(runId)}/export/${encodeURIComponent(artifact)}`,
}

function query(params: Record<string, string | number | undefined>): string {
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== '') search.set(key, String(value))
  }
  const rendered = search.toString()
  return rendered ? `?${rendered}` : ''
}
