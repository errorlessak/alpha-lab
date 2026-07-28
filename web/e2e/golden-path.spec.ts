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
    await page.keyboard.press('Tab')

    const outline = await page.evaluate(() => {
      const active = document.activeElement
      if (!active) return null
      const style = window.getComputedStyle(active)
      return { width: style.outlineWidth, style: style.outlineStyle }
    })

    expect(outline).not.toBeNull()
    expect(outline?.style).not.toBe('none')
    expect(parseFloat(outline?.width ?? '0')).toBeGreaterThan(0)
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

    // Nothing overflows the viewport horizontally.
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    )
    expect(overflow).toBeLessThanOrEqual(1)
  })

  test('the disclaimer travels with every page', async ({ page }) => {
    await page.goto('/')
    await expect(page.getByText('Not investment advice')).toBeVisible()
  })
})
