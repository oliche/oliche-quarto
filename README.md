# oliche-quarto

## Adding a new analysis

1. Create a folder under `analyses/` named `YYYY-MM-<short-slug>`, e.g. `analyses/2026-05-neuron-tuning/`
2. Add an `index.qmd` inside it with this front matter:
   ```yaml
   ---
   title: "Your Analysis Title"
   description: "One sentence shown on the hub card."
   date: "YYYY-MM-DD"
   author: "Olivier Winter"
   categories: [tag1, tag2]
   ---
   ```
3. Write your analysis below the front matter (markdown + code cells)
4. Render and publish — **always from the project root:**

   - [ ] `quarto render` — re-renders all pages with a consistent CSS; unchanged notebooks use cached outputs (`_freeze/`) so this is fast
   - [ ] `git add _site/ && git commit -m "Add <analysis-name>" && git push`

   > While writing, use `quarto preview` (project root) for live reload. Run `quarto render` once before committing to sync all pages.

   > **Avoid** `quarto render analyses/my-page/` — rendering a single page in isolation can update the shared bootstrap CSS hash without updating the other pages, breaking their styles on GitHub. A pre-commit hook will catch this if it happens.

## Publishing to GitHub Pages

**One-time setup**

1. Go to **github.com/oliche/oliche-quarto → Settings → Pages**
2. Set source to **"GitHub Actions"**
3. Save

Every push to `main` now triggers a build and deploy automatically.
The site is live at: **https://oliche.github.io/oliche-quarto/**
