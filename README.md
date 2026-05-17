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
2. Set source to **"Deploy from a branch"**, branch **`gh-pages`**, folder **`/ (root)`**
3. Save

**Publish**

```bash
quarto publish gh-pages
```

This renders the site and pushes it to the `gh-pages` branch. GitHub Pages picks it up automatically within a minute or two.

The site will be live at: **https://oliche.github.io/oliche-quarto/**

> Run the same command whenever you want to push updates.
