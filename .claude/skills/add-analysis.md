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

- Use `{python}` code cells
- Math: `$inline$` and `$$display$$` (MathJax, works in HTML output)
- **Figures:** save explicitly to `figures/` and reference with `![caption](figures/foo.png)`.
  Do NOT rely on `plt.show()` inline — `.gitignore` blocks `*_files/` so Quarto's default
  `index_files/figure-html/` output path is ignored by git and the image won't deploy.
  `analyses/*/figures/` is also gitignored — images are tracked only via `_site/` after rendering.
  Pattern:
  ```python
  #| output: false
  from pathlib import Path
  Path("figures").mkdir(exist_ok=True)
  plt.savefig("figures/foo.png", dpi=150)
  plt.close()
  ```
  Then in markdown: `![caption](figures/foo.png)`

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
quarto render analyses/<yyyy-mm-slug>/index.qmd
open _site/analyses/<yyyy-mm-slug>/index.html
```

### 5. Commit and push

`analyses/*/figures/` is gitignored — figures are tracked exclusively via `_site/` after rendering.
Stage source + freeze cache + built site (which now contains the images):

```bash
git add analyses/<yyyy-mm-slug>/ _freeze/analyses/<yyyy-mm-slug>/ _site/ _quarto.yml
git commit -m "Add <slug> analysis"
git push -u origin analysis/<slug>
```

Then open a PR or merge to `main`. GitHub Actions deploys `_site/` directly to Pages — no CI build step.

### Updating figures on an existing page

After saving new figures locally and updating the `.qmd`:

```bash
quarto render analyses/<yyyy-mm-slug>/index.qmd
git add _site/analyses/<yyyy-mm-slug>/ analyses/<yyyy-mm-slug>/index.qmd
git commit -m "Update <slug> figures"
git push
```

Do **not** stage `analyses/<yyyy-mm-slug>/figures/` — it is gitignored and the images live in `_site/`.

## Gotchas

- **Always pull before starting** — there may be remote commits adding new analyses
- **Full `quarto render` after pull** cleans `_site/`; anything not in the render list disappears
- **`freeze: auto`** skips re-execution if inputs are unchanged; delete `_freeze/analyses/<slug>/` to force a re-run
- The bootstrap CSS filename in `_site/site_libs/` changes between quarto versions — stage all of `_site/` to avoid deleted-file noise