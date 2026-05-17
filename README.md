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
4. Run `quarto render` from the repo root — your card appears on the home page automatically

> Preview while writing: `quarto preview` starts a live-reload server at http://localhost:5851/

## Publishing to GitHub Pages

**One-time setup**

1. Go to **github.com/oliche/oliche-quarto → Settings → Pages**
2. Set source to **"GitHub Actions"**
3. Save

Every push to `main` now triggers a build and deploy automatically.
The site is live at: **https://oliche.github.io/oliche-quarto/**
