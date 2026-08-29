# My Notes

MkDocs Material se bana hosted notes site. Notes `docs/` ke andar `.md` files hain — GitHub pe daalo, site khud build ho jati hai.

```
docs/            →  saare notes
mkdocs.yml       →  site ki settings
.github/         →  auto-deploy workflow
TEMPLATE.md      →  naya note banane ka template
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

`mkdocs.yml` mein `site_url` ko apne asal link se badal do. Search aur sitemap iske bina theek kaam nahi karte.

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
mkdocs serve       # localhost:8000, save karte hi live reload
```

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

# requests

## Ek line mein
## Install
## Core concepts
## Examples
## Gotchas
## Cheatsheet
```

`docs/python/requests.md` mein poora bhara hua example hai — tabs, admonitions, cheatsheet table sab ke saath.

## Kaam ke markdown features

| Cheez | Kaise |
|---|---|
| Warning box | `!!! warning` phir 4 space indent |
| Collapsible | `??? note "Title"` |
| Code tabs | `=== "Tab ka naam"` phir 4 space indent |
| Checklist | `- [x] done` |
| Footnote | `text[^1]` aur neeche `[^1]: detail` |

## Customise

- Color: `mkdocs.yml` → `palette` → `primary` (deep purple, indigo, teal, etc.)
- Site ka naam: `site_name`
- Naye markdown features: `markdown_extensions` mein add karo
