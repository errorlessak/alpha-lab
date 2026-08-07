/**
 * A horizontal scroll container for a wide data table.
 *
 * The tables in this app are ledgers: ten columns of dense figures, sized by
 * their content. Below about 1000px they are wider than the viewport, and
 * without a container the *page* scrolls sideways — which moves the masthead
 * and the disclaimer off screen along with the table. §13.5 forbids that, and
 * `the index is usable at 768px` in the end-to-end suite is the check.
 *
 * The container takes focus **only while it actually overflows**. A scrollable
 * region has to be reachable by keyboard (WCAG 2.1.1), but a permanent tab stop
 * in front of every table on a desktop viewport — where nothing scrolls — is a
 * cost paid by keyboard users for nothing. So the tab stop appears with the
 * scrollbar and leaves with it.
 */

import { useEffect, useRef, useState } from 'react'
import type { ReactNode } from 'react'

export function TableScroll({ label, children }: { label: string; children: ReactNode }) {
  const ref = useRef<HTMLDivElement>(null)
  const [scrollable, setScrollable] = useState(false)

  useEffect(() => {
    const element = ref.current
    if (!element) return undefined

    const measure = () => setScrollable(element.scrollWidth > element.clientWidth + 1)
    measure()

    // The table's own width changes when data arrives, and the container's when
    // the viewport does, so both are observed.
    const observer = new ResizeObserver(measure)
    observer.observe(element)
    if (element.firstElementChild) observer.observe(element.firstElementChild)
    return () => observer.disconnect()
  }, [children])

  return (
    <div
      ref={ref}
      className="overflow-x-auto"
      // Announced as a region only when it scrolls; an unscrollable wrapper is
      // a landmark with nothing in it as far as a screen reader is concerned.
      {...(scrollable
        ? { role: 'region' as const, 'aria-label': `${label} (scrolls horizontally)`, tabIndex: 0 }
        : {})}
    >
      {children}
    </div>
  )
}
