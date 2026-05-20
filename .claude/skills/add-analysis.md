# Skill: add-analysis

Add a new analysis page to this Quarto website and publish it to GitHub Pages.

## Repo layout

```
oliche-quarto/
├── _quarto.yml          ← explicit render list — MUST add new entry here
├── index.qmd            ← listing page, auto-discovers analyses/
├── analyses/
│   └── <yyyy-mm-slug>/
│       └── index.qmd   ← one file per analysis
├── _freeze/             ← execution cache, commit alongside source
└── _site/              ← built output, committed and deployed directly
```

## Workflow

### 1. Work on a fresh branch

```bash
git checkout -b analysis/<slug>
```

### 2. Create the analysis directory and source file

```bash
mkdir analyses/<yyyy-mm-slug>
```

`analyses/<yyyy-mm-slug>/index.qmd` frontmatter template:

```yaml
---
title: "..."
description: "One sentence shown in the listing grid."
date: "yyyy-mm-dd"
author: "Olivier Winter"
categories: [tag1, tag2]
---
```

- Use `{python}` code cells; figures render inline via matplotlib `plt.show()`
- Do **not** call `matplotlib.use(...)` — Quarto handles the backend
- Math: `$inline$` and `$$display$$` (MathJax, works in HTML output)

### 3. Register the new file in `_quarto.yml`

Add a line under `project.render`:

```yaml
project:
  render:
    - index.qmd
    - analyses/existing/index.qmd
    - analyses/<yyyy-mm-slug>/index.qmd   # ← add this
```

**This step is mandatory.** Quarto will silently skip any `.qmd` not in this list.

### 4. Render and check

```bash
quarto render
open _site/analyses/<yyyy-mm-slug>/index.html
```

### 5. Commit and push

Stage source + freeze cache + built site:

```bash
git add analyses/<yyyy-mm-slug>/ _freeze/analyses/<yyyy-mm-slug>/ _site/ _quarto.yml
git commit -m "Add <slug> analysis"
git push -u origin analysis/<slug>
```

Then open a PR or merge to `main`. GitHub Actions deploys `_site/` directly to Pages — no CI build step.

## Gotchas

- **Always pull before starting** — there may be remote commits adding new analyses
- **Full `quarto render` after pull** cleans `_site/`; anything not in the render list disappears
- **`freeze: auto`** skips re-execution if inputs are unchanged; delete `_freeze/analyses/<slug>/` to force a re-run
- The bootstrap CSS filename in `_site/site_libs/` changes between quarto versions — stage all of `_site/` to avoid deleted-file noise