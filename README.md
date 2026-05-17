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
4. Render and publish:
   ```bash
   quarto render
   git add -A && git commit -m "Add <analysis-name>" && git push
   ```
   Only analyses whose source has changed are re-executed (outputs are cached in `_freeze/`).
   GitHub Actions picks up the pre-built `_site/` and deploys it automatically.

   To render a single analysis without touching the rest:
   ```bash
   quarto render analyses/2026-05-neuron-tuning/
   ```

> Preview while writing: `quarto preview` starts a live-reload server at http://localhost:5851/

## Publishing to GitHub Pages

**One-time setup**

1. Go to **github.com/oliche/oliche-quarto → Settings → Pages**
2. Set source to **"GitHub Actions"**
3. Save

Every push to `main` now triggers a build and deploy automatically.
The site is live at: **https://oliche.github.io/oliche-quarto/**
