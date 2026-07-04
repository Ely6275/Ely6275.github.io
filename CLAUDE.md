# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A static personal portfolio site for Ely Herman, hosted on GitHub Pages at `Ely6275.github.io`. There is no build step, no package manager, and no framework — every page is a single self-contained HTML file with inline `<style>` and (where needed) inline `<script>`.

## Development

There are no build/lint/test commands — this is plain static HTML/CSS. To preview locally, just open a file in a browser, or use the VS Code Live Preview extension (`.vscode/settings.json` sets the default preview path to `/index.html`).

Changes are published simply by committing and pushing to `main`; GitHub Pages serves directly from the repo root.

## Structure

- `index.html` — the landing page: nav, hero, about, a grid of featured project cards, a smaller grid of earlier/high-school project cards, and a contact section.
- `project1.html`, `project2.html`, `project3.html` — one standalone detail page per featured project, linked from the corresponding card in `index.html` via `onclick="location.href='projectN.html'"`.
- Images/GIFs/PNGs at the repo root are referenced directly by relative path from whichever HTML file uses them (no `assets/` or `images/` subfolder — new media just gets dropped at the root).

## Page pattern

The site uses a "Schematic" (engineering blueprint) visual identity. Every page (`index.html` and each `projectN.html`) repeats the same structure rather than sharing a stylesheet or template:

- Same Google Fonts import (`Archivo` for headings/body, `IBM Plex Mono` for nav, labels, tags, buttons, and data).
- Same CSS custom properties block (`--ground` navy `#0e2a4a`, `--ground-2`, `--panel`, `--ink`, `--muted`, `--accent` green `#7fe7c4`, `--accent-dim`, `--amber` `#ffb454`, `--line`, `--sans`, `--mono`) redeclared at the top of each `<style>` block — this is the site's only "theme," so a palette/typography change must be repeated in every file. The body carries a 28px grid-paper background built from two `linear-gradient`s.
- Drafting-sheet conventions carry the identity: amber mono `SHEET NN / NN` or numbered `NN — LABEL` section eyebrows, `DWG. EH-NN` labels floating on project card borders, `.fig-frame` image frames with green corner tick marks and `FIG. NN` captions, and `.spec-panel` datasheet grids for project metadata.
- Same fixed, blurred nav bar (`EH // INDEX` logo). On project pages the nav uses a `.back-link` to `index.html#projects` instead of section links.
- Project detail pages follow numbered sections: hero + spec panel → `.cap-grid` At a Glance (2×2 capability cells) → highlights/derivations/media → roadmap or takeaways → links row. Check an existing `projectN.html` for the section markup before adding a new one rather than inventing new class names. Writeups are recruiter-oriented: keyword-forward hero sentences, bolded action-verb bullets, quantified results.

When adding a new featured project: add a card to the `.projects-grid` in `index.html` (image, tags, title, description, links) and create a new `projectN.html` by copying the closest existing project page's structure, styles, and nav pattern.

When adding an "earlier work" / high-school project: add a card to the smaller `.projects-grid--small` grid at the bottom of the Projects section in `index.html` — these don't have their own detail pages.
