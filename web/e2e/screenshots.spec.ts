/**
 * Screenshots for `docs/` (§15, Phase 12 exit).
 *
 * Run with `npx playwright test screenshots` after a `just smoke`, so the run
 * they show is a real one. They are committed, which is why they are written
 * with a fixed viewport and no cursor artefacts.
 */

import { expect, test } from '@playwright/test'

const DOCS = '../docs/screenshots'

test.use({ viewport: { width: 1600, height: 1000 } })

test('capture the run index', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Evidence file' })).toBeVisible()
  await page.screenshot({ path: `${DOCS}/run-index.png`, fullPage: false })
})

test('capture the run overview', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('table').getByRole('link').first().click()
  await expect(page.getByRole('heading', { name: 'Verdict' })).toBeVisible()
  // Let the fold strip finish its one orchestrated moment.
  await page.waitForTimeout(900)
  await page.screenshot({ path: `${DOCS}/run-overview.png`, fullPage: false })
})

test('capture the leaderboard', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('table').getByRole('link').first().click()
  await page.getByRole('link', { name: /models$/ }).click()
  await expect(page.getByRole('heading', { name: 'Leaderboard' })).toBeVisible()
  await page.waitForTimeout(400)
  await page.screenshot({ path: `${DOCS}/leaderboard.png`, fullPage: false })
})

test('capture the backtest stage', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('table').getByRole('link').first().click()
  await page.getByRole('link', { name: /backtest$/ }).click()
  await expect(page.getByRole('heading', { name: 'Cost sensitivity' })).toBeVisible()
  await page.waitForTimeout(600)
  await page.screenshot({ path: `${DOCS}/backtest.png`, fullPage: false })
})
