/**
 * Lighthouse against the built app (§15, Phase 12 exit: ≥ 95 on performance,
 * accessibility and best-practices).
 *
 * Three runs, median taken. A single Lighthouse run on a developer machine
 * swings several points with whatever else is compiling at the time, so one
 * number is not evidence either way — a pass can be luck and a fail can be a
 * background process. The median of three is the smallest thing that is not.
 *
 * Assumes `npm run preview` is already serving on 4173; `just e2e` and the
 * justfile's `lighthouse` recipe both arrange that.
 */

import { execSync } from 'node:child_process'
import { readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const URL_UNDER_TEST = process.env.LIGHTHOUSE_URL ?? 'http://127.0.0.1:4173'
const CATEGORIES = ['performance', 'accessibility', 'best-practices']
const THRESHOLD = 95
const RUNS = 3

function runOnce(index) {
  const output = join(tmpdir(), `alpha-lab-lh-${index}.json`)
  execSync(
    [
      'npx lighthouse',
      URL_UNDER_TEST,
      `--only-categories=${CATEGORIES.join(',')}`,
      '--output=json',
      `--output-path="${output}"`,
      '--chrome-flags="--headless=new --no-sandbox"',
      '--quiet',
    ].join(' '),
    { stdio: ['ignore', 'ignore', 'inherit'] },
  )
  const report = JSON.parse(readFileSync(output, 'utf8'))
  rmSync(output, { force: true })
  return Object.fromEntries(
    CATEGORIES.map((name) => [name, Math.round((report.categories[name]?.score ?? 0) * 100)]),
  )
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b)
  return sorted[Math.floor(sorted.length / 2)]
}

const results = []
for (let index = 0; index < RUNS; index += 1) {
  const scores = runOnce(index)
  results.push(scores)
  console.log(
    `  run ${index + 1}  ` + CATEGORIES.map((name) => `${name} ${scores[name]}`).join('  ·  '),
  )
}

let failed = false
console.log('')
for (const name of CATEGORIES) {
  const score = median(results.map((result) => result[name]))
  const verdict = score >= THRESHOLD ? 'pass' : 'FAIL'
  console.log(`  ${name.padEnd(16)} median ${String(score).padStart(3)}  ${verdict}`)
  if (score < THRESHOLD) failed = true
}

if (failed) {
  console.error(`\nBelow the ${THRESHOLD} bar §13.5 sets.`)
  process.exit(1)
}
