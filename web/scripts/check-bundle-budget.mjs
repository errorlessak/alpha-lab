/**
 * The bundle budget (§13.5).
 *
 * "Initial JS ≤ 200 KB gzipped." A budget that warns is a budget nobody keeps,
 * so this exits non-zero and takes CI with it.
 *
 * "Initial" means what the browser must download before the first route can
 * render: the entry chunk plus anything it statically imports. Lazily loaded
 * route chunks and the charting library are excluded, which is the whole point
 * of splitting them out — but they are reported so a regression that moves
 * something into the entry is visible rather than merely absent.
 */

import { gzipSync } from 'node:zlib'
import { readFileSync, readdirSync } from 'node:fs'
import { join } from 'node:path'

const BUDGET_BYTES = 200 * 1024
const DIST = new URL('../dist/assets/', import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1')
const MANIFEST = new URL('../dist/.vite/manifest.json', import.meta.url).pathname.replace(
  /^\/([A-Za-z]:)/,
  '$1',
)

function gzippedSize(path) {
  return gzipSync(readFileSync(path), { level: 9 }).length
}

function entryChunks() {
  // Prefer the manifest, which knows which chunks the entry statically imports.
  try {
    const manifest = JSON.parse(readFileSync(MANIFEST, 'utf8'))
    const entry = Object.values(manifest).find((chunk) => chunk.isEntry)
    if (entry) {
      const names = new Set([
        entry.file,
        ...(entry.imports ?? []).map((key) => manifest[key]?.file),
      ])
      return [...names].filter(Boolean)
    }
  } catch {
    // No manifest: fall back to the naming convention below.
  }
  return readdirSync(DIST)
    .filter((name) => name.startsWith('index-') && name.endsWith('.js'))
    .map((name) => `assets/${name}`)
}

const dist = DIST.replace(/assets\/$/, '')
const initial = entryChunks()

if (initial.length === 0) {
  console.error('No entry chunk found in dist/. Run `npm run build` first.')
  process.exit(1)
}

let total = 0
for (const file of initial) {
  const size = gzippedSize(join(dist, file))
  total += size
  console.log(`  ${file.padEnd(48)} ${(size / 1024).toFixed(1).padStart(7)} KB gz`)
}

const lazy = readdirSync(DIST)
  .filter((name) => name.endsWith('.js') && !initial.includes(`assets/${name}`))
  .map((name) => ({ name, size: gzippedSize(join(DIST, name)) }))
  .sort((a, b) => b.size - a.size)

if (lazy.length > 0) {
  console.log('\n  lazily loaded, outside the budget:')
  for (const chunk of lazy.slice(0, 8)) {
    console.log(`  ${chunk.name.padEnd(48)} ${(chunk.size / 1024).toFixed(1).padStart(7)} KB gz`)
  }
}

const used = ((total / BUDGET_BYTES) * 100).toFixed(0)
console.log(
  `\ninitial JS ${(total / 1024).toFixed(1)} KB gzipped against a ${BUDGET_BYTES / 1024} KB budget (${used}%)`,
)

if (total > BUDGET_BYTES) {
  console.error(
    `\nOver budget by ${((total - BUDGET_BYTES) / 1024).toFixed(1)} KB. Move something behind a route split, or make the case for raising §13.5's number.`,
  )
  process.exit(1)
}
