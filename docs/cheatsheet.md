---
title: Writing cheatsheet
tags:
  - meta
  - markdown
date: 2026-08-30
---

# Writing cheatsheet

!!! abstract "Ek line mein"
    Notes likhte waqt jo bhi cheez chahiye — box, tab, table, key — uska syntax aur live demo, dono yahan.

## Boxes (admonitions)

`!!!` likho, phir type, phir 4 space indent kar ke content.

```markdown
!!! tip "Title yahan"
    Content 4 space indent pe.
```

!!! note "Note"
    Aam maloomat ke liye — neutral.

!!! tip "Tip"
    Koi shortcut ya behtar tareeqa.

!!! warning "Warning"
    Aisi cheez jo galti se bigaad sakti hai.

!!! danger "Danger"
    Data ya security ka masla.

!!! example "Example"
    Chhota sa demo ya sample.

!!! quote "Quote"
    Kisi kitab ya docs ka hawala.

### Collapsible

`???` use karo `!!!` ki jagah — box band shuru hoga. `???+` se khula shuru hoga.

??? note "Click karo — khulega"
    Lambi details yahan chhupa do taake page saaf rahe.

## Code

### Simple block

````markdown
```python
print("hello")
```
````

```python
def greet(name: str) -> str:
    return f"Salam, {name}!"
```

### Title ke saath

````markdown
```python title="app.py"
...
```
````

```python title="app.py" linenums="1" hl_lines="2 3"
import requests

r = requests.get("https://api.github.com", timeout=10)
r.raise_for_status()
print(r.json())
```

`linenums="1"` line numbers deta hai, `hl_lines="2 3"` un lines ko highlight karta hai.

### Tabs

```markdown
=== "Tab ka naam"

    content 4 space indent pe
```

=== "pip"

    ```bash
    pip install requests
    ```

=== "poetry"

    ```bash
    poetry add requests
    ```

=== "uv"

    ```bash
    uv add requests
    ```

## Tables

```markdown
| Cheez | Matlab |
|---|---|
| `a` | pehli |
```

| Syntax | Kya karta hai | Kab use karo |
|---|---|---|
| `**bold**` | **bold** | zor dene ke liye |
| `*italic*` | *italic* | naya lafz introduce karte waqt |
| `==mark==` | ==highlight== | scan karte waqt aankh yahan rukni chahiye |
| `~~strike~~` | ~~kaat do~~ | purani cheez jo ab galat hai |
| `` `code` `` | `code` | koi bhi literal cheez |

## Lists

```markdown
- [x] ho gaya
- [ ] baaqi hai
```

- [x] Theme set karna
- [x] Pehla note likhna
- [ ] JavaScript section shuru karna
- [ ] SQL cheatsheet

## Keyboard keys

`++ctrl+shift+p++` likho:

- Search kholne ke liye ++slash++ ya ++s++
- Search band karne ke liye ++esc++
- Palette switch: header ke top-right icon se

## Definition list

```markdown
Term
:   Uski tafseel
```

`Session`
:   Connection reuse karta hai aur cookies yaad rakhta hai.

`Response`
:   Server ka poora jawab — status, headers, body.

## Chhoti cheezein

| Cheez | Syntax |
|---|---|
| Link | `[text](url)` |
| Button | `[text](url){ .md-button }` |
| Primary button | `[text](url){ .md-button .md-button--primary }` |
| Footnote | `text[^1]` aur neeche `[^1]: detail` |
| Emoji / icon | `:material-heart:` → :material-heart: |
| Superscript | `x^2^` → x^2^ |
| Subscript | `H~2~O` → H~2~O |
| Abbreviation | `*[HTML]: HyperText Markup Language` |

Abbreviation ka asar: HTML pe hover karo.[^1]

*[HTML]: HyperText Markup Language

[^1]: Footnote aise dikhta hai — page ke bilkul neeche chala jata hai, aur wapas jaane ka arrow bhi milta hai.

## Frontmatter

Har note ke bilkul upar. Isi se title, tags aur search chalte hain.

```yaml
---
title: Python — requests library
tags:
  - python
  - library
date: 2026-08-30
---
```

!!! warning "Tags lowercase rakho"
    `Python` aur `python` do alag tags ban jate hain — [Tags](tags.md) page bikhar jayega.
