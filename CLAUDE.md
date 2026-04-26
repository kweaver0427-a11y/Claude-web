# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **static HTML/CSS website** for K&M Car Care ("Neighborhood Chemist"), a mobile car detailing/preservation service. There is no build toolchain, no JavaScript framework, no package manager, and no dependencies — the entire site is a single `index.html` file with embedded CSS.

## Development Workflow

Since there is no build step, development is direct file editing:

- **Preview locally:** Open `index.html` directly in a browser, or serve with any static file server:
  ```bash
  python3 -m http.server 8080
  # or
  npx serve .
  ```
- **Deploy:** Upload `index.html` (and `logo.png` once added) to any static host (GitHub Pages, Netlify, Vercel, S3).

There are no lint, test, or build commands.

## Architecture

The entire application lives in `index.html` (895 lines). Structure:

1. **`<head>`** — Google Fonts preconnect + import (Oswald, Open Sans, Courier Prime), all CSS in a single `<style>` block
2. **`<nav class="nav">`** — Fixed header; hamburger toggle targets `.nav__links` visibility at mobile breakpoints
3. **`<section class="hero">`** — Two-column grid (text left, stats right); collapses to single column at 1024px
4. **Trust bar** — Four micro-stat callouts below the hero

**Sections linked in nav but not yet built:** `#services`, `#process`, `#conservation`, `#contact`

**Missing asset:** `logo.png` is referenced in the nav (`<img src="logo.png" ...>`) but not committed. It should be a ~50×50px circular tire badge.

## Design System

All design tokens are CSS custom properties on `:root`:

| Category | Key Variables |
|----------|--------------|
| Colors | `--color-primary` (#000), `--color-white` (#fff), `--color-accent` (#C41230 Signal Red), `--color-accent-blue` (#00AEEF Science Blue), `--color-accent-yellow` (#FFD700 Safety Yellow) |
| Surfaces | `--color-surface` (#0a0a0a), `--color-surface-2` (#111), `--color-surface-3` (#141414) |
| Typography | `--font-headline` (Oswald), `--font-body` (Open Sans), `--font-mono` (Courier Prime) |
| Spacing | `--space-xs` through `--space-3xl` (0.25rem → 4.5rem) |
| Layout | `--max-width` (1280px), `--nav-height` (72px) |

## CSS Conventions

- **Naming:** BEM with double-underscore elements and double-hyphen modifiers (e.g., `.water-badge__label`, `.btn--primary`, `.btn--ghost`)
- **Responsive breakpoints:** `max-width: 1024px` (tablet), `768px` (mobile), `420px` (small mobile)
- **Typography scaling:** Uses `clamp()` for fluid headline sizes
- **Visual effects:** Blueprint grid via repeating `linear-gradient` on `body::before`; radial gradient glows via `body::after`; nav uses `backdrop-filter: blur(12px)`

## Brand & Content Conventions

- Brand voice is technical/clinical — short declarative phrases, chemical/lab terminology
- Monospace font (`--font-mono`) is used for eyebrow labels, badges, and "// comment"-style annotations
- Color accent hierarchy: red for primary CTAs, blue for science/data callouts, yellow used sparingly
- Dark-first design — all surfaces are near-black; white text throughout
