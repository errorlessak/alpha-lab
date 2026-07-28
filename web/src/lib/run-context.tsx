/**
 * The run currently on screen, and which fold is filtering it.
 *
 * In its own module rather than in `routes/runs.$runId.tsx` for a specific
 * reason: the router's automatic code splitting divides each route file into
 * separate chunks, and a `createContext` call living in a route file can end up
 * duplicated across them. The child route then reads a *different* context
 * instance from the one the layout provides, and every consumer sees the
 * default value — which looks exactly like a page that never finished loading.
 */

import { createContext, useContext } from 'react'

import type { Tearsheet } from './api'

export interface RunState {
  runId: string
  tearsheet: Tearsheet | undefined
  selectedFold: number | null
}

export const RunContext = createContext<RunState>({
  runId: '',
  tearsheet: undefined,
  selectedFold: null,
})

/** The run on screen. Undefined `tearsheet` means it has not arrived yet. */
export function useRun(): RunState {
  return useContext(RunContext)
}
