# Authoring Guide — Obsidian + Claude

How to write and scale notes for this repository fast, while keeping everything
compatible with the MkDocs documentation site.

The repo is plain Markdown in Git, so it doubles as an **Obsidian vault** and an
AI-assisted "second brain." This guide sets up that workflow.

## Table of Contents
- [Why this workflow](#why-this-workflow)
- [Open the repo as an Obsidian vault](#open-the-repo-as-an-obsidian-vault)
- [Obsidian settings that keep notes MkDocs-safe](#obsidian-settings-that-keep-notes-mkdocs-safe)
- [Link & image conventions](#link--image-conventions)
- [AI-assisted note generation](#ai-assisted-note-generation)
- [Preview before you commit](#preview-before-you-commit)
- [Checklist](#checklist)

## Why this workflow

- **Markdown-native**: no export/import step — the files you edit in Obsidian are
  the files the site builds from.
- **Git-backed**: full history, PR review, and GitHub discoverability stay intact.
- **AI-assisted**: generate structured notes from a prompt template, then refine
  and cross-link in Obsidian.

## Open the repo as an Obsidian vault

1. Clone the repo (or use your existing clone).
2. In Obsidian: **Open folder as vault** → select the repo root.
3. Obsidian creates a local `.obsidian/` folder for your settings. This is
   personal/local — do not commit it (it's editor state, not content).

## Obsidian settings that keep notes MkDocs-safe

Obsidian defaults to `[[wikilinks]]`, which **MkDocs does not resolve**. Change
these under **Settings → Files & Links** so links work in both Obsidian and the
built site:

| Setting | Value | Why |
|---|---|---|
| **Use [[Wikilinks]]** | **Off** | Forces standard `[text](path.md)` links that MkDocs understands |
| **New link format** | **Relative path to file** | Relative links resolve on the site (absolute vault paths don't) |
| **Default location for new attachments** | **Same folder as current file** (or a per-topic `resources/`) | Keeps images next to their note, matching the repo layout |
| **Automatically update internal links** | **On** | Renames/moves fix links across the vault automatically |

> Tip: if you prefer wikilinks while drafting, run a converter before committing,
> or keep MkDocs happy by switching the two link settings above.

## Link & image conventions

- **Internal links**: standard relative Markdown — `[CNN notes](../deep-learning-notes/dl-notes.md)`.
- **Images**: relative paths, stored near the note (e.g. `./resources/images/foo.png`).
- **Avoid**: Windows-style backslashes (`..\foo\bar.md`) and absolute vault paths.
- Follow the folder + naming conventions in [CONTRIBUTING.md](../../CONTRIBUTING.md)
  (tiered `-core-basics/-intermediate/-advanced` files, `core/` vs `industry/` vs `meta/`).

## AI-assisted note generation

Use the prompt template in [CONTRIBUTING.md](../../CONTRIBUTING.md#ai-assisted-content-generation)
with any capable model (Claude, or the planned note-generator Space). Workflow:

1. **Generate** a draft from the template: give it the topic, audience level
   (beginner/intermediate/advanced), and focus area (core concept vs industry).
2. **Place** the draft in the correct folder using the naming convention.
3. **Refine & cross-link** in Obsidian — connect it to related notes.
4. **Verify** technical claims and code before committing.

### With Claude Code

This repo ships a `CLAUDE.md` with project conventions, so Claude Code understands
the structure. Typical prompts:

- *"Generate an intermediate-level note on <topic> for `core/<area>/` following the
  CONTRIBUTING template, and place it in the right folder."*
- *"Add cross-references between this note and related notes in `core/ai-ml/`."*
- *"Convert any wikilinks in this file to standard relative Markdown links."*

## Preview before you commit

Always build the site locally to catch broken links/images before pushing:

```bash
pip install -r requirements-docs.txt
mkdocs serve          # live preview at http://127.0.0.1:8000
# or one-off:
mkdocs build          # warns about any broken internal links
```

Fix any `is not found among documentation files` warnings before opening a PR.

## Checklist

- [ ] Wikilinks off, relative link format on
- [ ] File in the correct `core/` / `industry/` / `meta/` folder
- [ ] Naming convention followed (tiered files where applicable)
- [ ] Internal links and images resolve (`mkdocs build` is clean)
- [ ] Technical claims and code verified
- [ ] Cross-referenced with related notes
