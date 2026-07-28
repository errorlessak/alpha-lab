/**
 * The application shell.
 *
 * A masthead, the content column, and nothing else chrome-like. There is no
 * left sidebar: §13.4's routes are the pipeline stages of one run, and they
 * belong to the run rather than to the app. A persistent sidebar would encode
 * a habit rather than a fact.
 */

import type { QueryClient } from '@tanstack/react-query'
import { Link, Outlet, createRootRouteWithContext } from '@tanstack/react-router'

export interface RouterContext {
  queryClient: QueryClient
}

export const Route = createRootRouteWithContext<RouterContext>()({
  component: Shell,
  notFoundComponent: NotFound,
})

function Shell() {
  return (
    <div className="min-h-dvh bg-[color:var(--color-paper)]">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:border focus:border-[color:var(--color-ink)] focus:bg-[color:var(--color-paper)] focus:px-3 focus:py-2 focus:text-sm"
      >
        Skip to content
      </a>

      <header className="border-b border-[color:var(--color-rule)]">
        <div className="mx-auto flex max-w-[1600px] flex-wrap items-baseline justify-between gap-4 px-6 py-4">
          <Link
            to="/"
            className="font-display text-base font-semibold tracking-tight text-[color:var(--color-ink)]"
            style={{ fontStretch: '85%' }}
          >
            ALPHA LAB
          </Link>
          <nav aria-label="Primary" className="flex gap-5 text-xs">
            <TopLink to="/">runs</TopLink>
            <TopLink to="/compare">compare</TopLink>
            <TopLink to="/jobs">jobs</TopLink>
          </nav>
        </div>
      </header>

      <main id="main" className="mx-auto max-w-[1600px] px-6 py-8">
        <Outlet />
      </main>

      <footer className="mx-auto max-w-[1600px] px-6 pb-10 pt-4">
        <p className="tabular border-t border-[color:var(--color-rule)] pt-3 text-2xs text-[color:var(--color-ink-muted)]">
          Research output. Not investment advice. Backtested performance is not indicative of
          anything.
        </p>
      </footer>
    </div>
  )
}

function TopLink({ to, children }: { to: string; children: string }) {
  return (
    <Link
      to={to}
      className="text-[color:var(--color-ink-muted)] hover:text-[color:var(--color-ink)]"
      activeProps={{
        className: 'text-[color:var(--color-ink)] underline underline-offset-4',
      }}
      activeOptions={{ exact: to === '/' }}
    >
      {children}
    </Link>
  )
}

function NotFound() {
  return (
    <div className="border border-dashed border-[color:var(--color-rule)] p-8">
      <p className="font-display text-base font-semibold">No such page</p>
      <p className="mt-2 text-sm text-[color:var(--color-ink-muted)]">
        That address does not match a route in this application.{' '}
        <Link to="/" className="underline underline-offset-4">
          Go to the run index
        </Link>
        .
      </p>
    </div>
  )
}
