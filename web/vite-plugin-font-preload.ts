import type { Plugin } from 'vite'

/**
 * Preload the two typefaces that paint above the fold.
 *
 * First contentful paint waits on the font the masthead is set in. The
 * `@font-face` rules live inside the bundled stylesheet, so the browser does
 * not discover them until it has fetched and parsed that CSS — one round trip
 * later than it needs to. A `<link rel="preload">` in the head starts both
 * downloads with the document.
 *
 * Only the two faces used above the fold. Preloading everything would compete
 * with the JS for bandwidth and make first paint worse, which is the usual way
 * this optimisation backfires.
 */
export function fontPreload(): Plugin {
  return {
    name: 'alpha-lab:font-preload',
    apply: 'build',
    enforce: 'post',
    transformIndexHtml(html, context) {
      const fonts = Object.keys(context.bundle ?? {}).filter(
        (name) =>
          name.endsWith('.woff2') &&
          (name.includes('public-sans-latin-wght-normal') ||
            name.includes('bricolage-grotesque-latin-wdth-normal')),
      )

      return {
        html,
        tags: fonts.map((file) => ({
          tag: 'link',
          attrs: {
            rel: 'preload',
            as: 'font',
            type: 'font/woff2',
            href: `/${file}`,
            crossorigin: 'anonymous',
          },
          injectTo: 'head-prepend' as const,
        })),
      }
    },
  }
}
