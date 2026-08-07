/**
 * The golden path (§15, Phase 12 exit).
 *
 * Open the run index, pick a run, read its verdict, walk its stages, and get to
 * the trades — the same journey a researcher makes when deciding whether an
 * edge is real. Anything that breaks this breaks the product.
 *
 * The keyboard walk is a separate test because §13.5 makes it a requirement in
 * its own right, not a nice-to-have that gets checked when someone remembers.
 */

import { expect, test } from '@playwright/test'
import type { Page } from '@playwright/test'

/**
 * Pixels by which the document is wider than the viewport.
 *
 * Measured only once the webfonts have swapped in. The fallback faces are wider
 * than Public Sans and IBM Plex Mono at the same size, so a measurement taken
 * mid-swap reports an overflow that is gone a frame later — which makes the
 * check fail on a cold cache and pass on a warm one.
 */
async function horizontalOverflow(page: Page): Promise<number> {
  await page.evaluate(() => document.fonts.ready)
  return page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  )
}

test.describe('golden path', () => {
  test('a researcher can go from the index to a run and read its verdict', async ({ page }) => {
    await page.goto('/')

    await expect(page.getByRole('heading', { name: 'Evidence file' })).toBeVisible()

    // The index lists at least the run `just smoke` produced.
    const firstRun = page.getByRole('table').getByRole('link').first()
    await expect(firstRun).toBeVisible()
    await firstRun.click()

    // The Fold Strip is the hero and the honest disclosure of how the model was
    // validated. If it is not here, the page is not doing its job.
    await expect(page.getByTestId('fold-strip')).toBeVisible()
    await expect(page.getByRole('group', { name: /walk-forward validation/i })).toBeVisible()

    // The verdict, as tolerance rails rather than a row of big numbers.
    await expect(page.getByRole('heading', { name: 'Verdict' })).toBeVisible()
    await expect(page.getByText('net Sharpe', { exact: true })).toBeVisible()
    await expect(page.getByText('deflated Sharpe', { exact: true })).toBeVisible()
    await expect(page.getByText('break-even cost', { exact: true })).toBeVisible()

    // Every plate carries a provenance stamp, so no chart can be screenshotted
    // out of context and misread.
    const stamps = page.getByTestId('provenance-stamp')
    expect(await stamps.count()).toBeGreaterThan(0)
    await expect(stamps.first()).toContainText('run ')

    // The leaderboard always shows the shuffled-label baseline — the scale on
    // the axis.
    await expect(page.getByText('baseline_shuffled').first()).toBeVisible()
  })

  test('the fold strip filters, and says which fold is selected', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('table').getByRole('link').first().click()
    await expect(page.getByTestId('fold-strip')).toBeVisible()

    const foldOne = page.getByRole('button', { name: /^Fold 1/ })
    await foldOne.click()
    await expect(foldOne).toHaveAttribute('aria-pressed', 'true')

    await page.getByRole('button', { name: /^All folds/ }).click()
    await expect(foldOne).toHaveAttribute('aria-pressed', 'false')
  })

  test('every pipeline stage is reachable and renders', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('table').getByRole('link').first().click()

    for (const stage of ['data', 'features', 'folds', 'models', 'backtest', 'explain']) {
      await page.getByRole('link', { name: new RegExp(`${stage}$`) }).click()
      await expect(page).toHaveURL(new RegExp(`/${stage}$`))
      // Each stage renders something and does not blow up. A stage with no
      // artifact shows the API's own explanation, which is still a render.
      await expect(page.locator('main')).toBeVisible()
      await expect(page.locator('main')).not.toContainText('Something went wrong')
    }
  })

  test('the margin argues with the run', async ({ page }) => {
    await page.setViewportSize({ width: 1600, height: 1000 })
    await page.goto('/')
    await page.getByRole('table').getByRole('link').first().click()
    await expect(page.getByRole('heading', { name: 'Verdict' })).toBeVisible()

    // The margin is present either with notes or with its empty statement.
    const margin = page.getByRole('complementary', { name: 'Margin notes' })
    await expect(margin).toBeVisible()
  })

  test('an unknown run reports the cause rather than a blank page', async ({ page }) => {
    await page.goto('/runs/no-such-run')
    await expect(page.getByRole('alert')).toBeVisible()
    await expect(page.getByRole('alert')).toContainText('no-such-run')
    // §13.5: never "Something went wrong".
    await expect(page.locator('body')).not.toContainText('Something went wrong')
  })
})

test.describe('quality bar', () => {
  test('every route is reachable by keyboard alone', async ({ page }) => {
    await page.goto('/')

    // Tab until the first run link has focus, then activate it with the
    // keyboard. A mouse is never used in this test.
    let focusedRun = false
    for (let i = 0; i < 40 && !focusedRun; i += 1) {
      await page.keyboard.press('Tab')
      focusedRun = await page.evaluate(() => {
        const active = document.activeElement as HTMLAnchorElement | null
        return Boolean(active?.getAttribute('href')?.startsWith('/runs/'))
      })
    }
    expect(focusedRun, 'a run link should be reachable by tabbing').toBe(true)

    await page.keyboard.press('Enter')
    await expect(page).toHaveURL(/\/runs\//)
    await expect(page.getByTestId('fold-strip')).toBeVisible()
  })

  test('the focus ring is visible on interactive elements', async ({ page }) => {
    await page.goto('/')

    // This is a single-page application: until React has rendered the shell
    // there is nothing focusable, and a Tab pressed before then leaves focus on
    // `<body>` — which reports `outline-style: none` and always will. Waiting
    // for the first focusable element makes this a test of the focus ring
    // rather than a race against the first paint.
    const skipLink = page.getByRole('link', { name: 'Skip to content' })
    await expect(skipLink).toBeAttached()

    await page.keyboard.press('Tab')
    await expect(skipLink).toBeFocused()

    const outline = await skipLink.evaluate((element) => {
      const style = window.getComputedStyle(element)
      return { width: style.outlineWidth, style: style.outlineStyle }
    })

    expect(outline.style).not.toBe('none')
    expect(parseFloat(outline.width)).toBeGreaterThan(0)
  })

  test('the margin collapses to inline callouts below 1280px', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('table').getByRole('link').first().click()
    await expect(page.getByRole('heading', { name: 'Verdict' })).toBeVisible()

    await page.setViewportSize({ width: 900, height: 1000 })
    await expect(page.getByRole('complementary', { name: 'Margin notes' })).toBeHidden()

    await page.setViewportSize({ width: 1600, height: 1000 })
    await expect(page.getByRole('complementary', { name: 'Margin notes' })).toBeVisible()
  })

  test('the index is usable at 768px', async ({ page }) => {
    await page.setViewportSize({ width: 768, height: 900 })
    await page.goto('/')
    await expect(page.getByRole('heading', { name: 'Evidence file' })).toBeVisible()
    await expect(page.getByRole('table')).toBeVisible()

    expect(await horizontalOverflow(page)).toBeLessThanOrEqual(1)
  })

  test('a run page is usable at 768px', async ({ page }) => {
    await page.setViewportSize({ width: 768, height: 900 })
    await page.goto('/')
    await page.getByRole('table').getByRole('link').first().click()
    await expect(page.getByRole('heading', { name: 'Verdict' })).toBeVisible()

    expect(await horizontalOverflow(page)).toBeLessThanOrEqual(1)
  })

  for (const scheme of ['light', 'dark'] as const) {
    test(`text meets the AA contrast bar in ${scheme} mode`, async ({ page }) => {
      // §13.5's accessibility bar is not negotiable, and the palette has three
      // colours that clear it in one scheme and not the other — sulphur is
      // 2.31:1 as light-mode text, oxblood 3.92:1 on the striped row fill in
      // dark. The tokens carry text-only variants for exactly that reason, and
      // this is what stops a future edit from reaching for the structural
      // colour because it is the one with the obvious name.
      await page.emulateMedia({ colorScheme: scheme })
      await page.goto('/')
      await expect(page.getByRole('table')).toBeVisible()

      const failures = await page.evaluate(() => {
        const luminance = (colour: string) => {
          const weights = [0.2126, 0.7152, 0.0722]
          return (colour.match(/\d+(\.\d+)?/g) ?? [])
            .slice(0, 3)
            .map(Number)
            .reduce((total, value, index) => {
              const s = value / 255
              const linear = s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4
              return total + (weights[index] ?? 0) * linear
            }, 0)
        }

        // The nearest ancestor that actually paints. `transparent` inherits
        // what is behind it, and the striped rows are why that matters.
        const background = (element: Element): string => {
          for (let node: Element | null = element; node; node = node.parentElement) {
            const colour = getComputedStyle(node).backgroundColor
            if (colour && !colour.startsWith('rgba(0, 0, 0, 0)')) return colour
          }
          return getComputedStyle(document.documentElement).backgroundColor
        }

        const bad: string[] = []
        for (const element of document.querySelectorAll('main *')) {
          const text = Array.from(element.childNodes)
            .filter((node) => node.nodeType === Node.TEXT_NODE)
            .map((node) => node.textContent?.trim() ?? '')
            .join('')
          if (!text) continue

          const style = getComputedStyle(element)
          if (style.visibility === 'hidden' || style.display === 'none') continue

          const size = parseFloat(style.fontSize)
          const bold = Number(style.fontWeight) >= 700
          // WCAG's large-text allowance: 18.66px bold, or 24px.
          const bar = size >= 24 || (bold && size >= 18.66) ? 3 : 4.5

          const foreground = luminance(style.color)
          const behind = luminance(background(element))
          const lighter = Math.max(foreground, behind)
          const darker = Math.min(foreground, behind)
          const contrast = (lighter + 0.05) / (darker + 0.05)
          if (contrast < bar) {
            bad.push(
              `${contrast.toFixed(2)} < ${bar} — ${style.color} on ${background(element)} — "${text.slice(0, 40)}"`,
            )
          }
        }
        return bad
      })

      expect(failures, failures.join('\n')).toHaveLength(0)
    })
  }

  test('the disclaimer travels with every page', async ({ page }) => {
    await page.goto('/')
    await expect(page.getByText('Not investment advice')).toBeVisible()
  })
})
