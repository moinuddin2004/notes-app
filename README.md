# My Notes

MkDocs Material se bana hosted notes site, custom theme ke saath. Notes `docs/` ke andar `.md` files hain — GitHub pe daalo, site khud build ho jati hai.

```
docs/                        →  saare notes
  stylesheets/extra.css      →  poora custom design (colors, cards, hero)
  javascripts/extra.js       →  reading progress bar
  assets/favicon.svg         →  favicon
  index.md                   →  homepage (hero + cards)
  cheatsheet.md              →  markdown features ka live demo
  tags.md                    →  auto-generated tag index
mkdocs.yml                   →  site ki settings
.github/                     →  auto-deploy workflow
TEMPLATE.md                  →  naya note banane ka template
```

## Setup — ek dafa

### 1. Repo banao

GitHub pe naya repo banao, phir:

```bash
git init
git add .
git commit -m "notes site"
git branch -M main
git remote add origin https://github.com/USERNAME/my-notes.git
git push -u origin main
```

### 2. Pages on karo

Repo → **Settings → Pages → Build and deployment → Source: GitHub Actions**

Push karte hi Actions build kar dega. Site:

```
https://USERNAME.github.io/my-notes/
```

### 3. site_url theek karo

`mkdocs.yml` mein `site_url` aur `extra.social` ke `USERNAME` ko apne GitHub username se badal do. Search aur sitemap iske bina theek kaam nahi karte.

## Naya note daalna

**Browser se (mobile pe bhi chalta hai):**

GitHub pe repo kholo → `docs/` mein jao → **Add file → Upload files** se `.md` drop karo, ya **Create new file** se wahin likh lo → Commit.

Naya folder chahiye? Filename mein `javascript/promises.md` likh do — GitHub khud folder bana dega.

**Laptop se:**

```bash
git pull
# docs/ mein file daalo
git add . && git commit -m "add promises note" && git push
```

Dono soorat mein ~40 second baad note live aur searchable hota hai. Nav khud update hoti hai — `mkdocs.yml` chhune ki zarurat nahi.

## Local preview

```bash
pip install -r requirements.txt
mkdocs serve
```

`site_url` mein path hai, is liye site `http://localhost:8000/my-notes/` pe khulegi (root pe nahi). Save karte hi live reload ho jati hai.

## Note ka format

`TEMPLATE.md` copy karo aur bhar do. Frontmatter zaroori hai — usi se title aur tags bante hain:

```markdown
---
title: Python — requests library
tags:
  - python
  - library
date: 2026-08-30
---
```

Saare markdown features ka live demo `docs/cheatsheet.md` mein hai — site pe **Writing cheatsheet** page. `docs/python/requests.md` ek poora bhara hua asal note hai.

## Design badalna

Sab kuch `docs/stylesheets/extra.css` ke sab se upar wale token block se chalta hai. Sirf wahan colors badlo, poori site ke saath badal jayegi:

| Token | Kya control karta hai |
|---|---|
| `--nt-brand` | main purple — links, buttons, active nav, tags |
| `--nt-accent` | pink — card hover line, progress bar ka doosra sira |
| `--nt-bg` / `--nt-surface` | page ka background aur cards/sheets ka background |
| `--nt-line` | saari borders |
| `--nt-ink` / `--nt-ink-soft` | text ka gehra aur halka rang |
| `--nt-r-sm/md/lg` | corners kitne round hain |

Light aur dark, dono ke apne token blocks hain — `[data-md-color-scheme="default"]` aur `[data-md-color-scheme="slate"]`.

Homepage ke hero aur cards ka markup `docs/index.md` mein hai (`.nt-hero`, `.nt-grid`, `.nt-card`). Nayi card grid banani ho to wahi structure copy kar lo — plain `<div class="nt-card">` pe `markdown="span"` lagana mat bhoolna, warna `:material-icon:` icons render nahi honge.

## Kaam ke markdown features

| Cheez | Kaise |
|---|---|
| Warning box | `!!! warning` phir 4 space indent |
| Collapsible | `??? note "Title"` |
| Code tabs | `=== "Tab ka naam"` phir 4 space indent |
| Code ka title | ` ```python title="app.py" ` |
| Line highlight | ` ```python hl_lines="2 3" ` |
| Checklist | `- [x] done` |
| Keyboard key | `++ctrl+c++` |
| Highlight | `==zaroori==` |
| Icon | `:material-heart:` |
| Footnote | `text[^1]` aur neeche `[^1]: detail` |
