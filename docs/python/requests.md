---
title: Python — requests library
tags:
  - python
  - library
  - http
date: 2026-08-30
---

# requests

!!! abstract "Ek line mein"
    HTTP requests bhejne ki sab se aasan Python library — `urllib` ka insaan-dost version.

## Install

```bash
pip install requests
```

## Core concepts

| Cheez | Matlab |
|---|---|
| `Response` | Server ka jawab — status, headers, body sab isme |
| `Session` | Connection reuse + cookies yaad rakhta hai |
| `raise_for_status()` | 4xx/5xx pe exception phenkta hai |

## Examples

=== "GET"

    ```python
    import requests

    r = requests.get("https://api.github.com/users/octocat", timeout=10)
    r.raise_for_status()
    print(r.json()["name"])
    ```

=== "POST"

    ```python
    r = requests.post(
        "https://httpbin.org/post",
        json={"name": "Ali", "city": "Karachi"},
        timeout=10,
    )
    print(r.status_code, r.json()["json"])
    ```

=== "Session"

    ```python
    with requests.Session() as s:
        s.headers.update({"Authorization": "Bearer TOKEN"})
        a = s.get("https://api.example.com/me")
        b = s.get("https://api.example.com/projects")
    ```

## Gotchas

!!! warning "Timeout hamesha do"
    Default mein koi timeout nahi hota. Agar server jawab na de to script hamesha ke liye hang ho jayegi.

!!! danger "`r.text` vs `r.content`"
    `r.text` decoded string deta hai (encoding ka guess laga ke), `r.content` raw bytes. Image ya PDF download karte waqt hamesha `r.content`.

- `r.json()` fail ho sakta hai agar response JSON na ho — `try/except` mein rakho
- `params=` use karo, URL mein khud query string mat jodo
- Bohat saari requests? `Session` use karo, warna har baar naya TCP connection banega

## Cheatsheet

| Kaam | Code |
|---|---|
| Query params | `requests.get(url, params={"q": "python"})` |
| Custom headers | `requests.get(url, headers={"X-Key": "abc"})` |
| JSON body | `requests.post(url, json=payload)` |
| Form body | `requests.post(url, data=payload)` |
| File upload | `requests.post(url, files={"f": open("a.png","rb")})` |
| Status check | `r.raise_for_status()` |
| Redirects band | `requests.get(url, allow_redirects=False)` |
| Stream download | `requests.get(url, stream=True)` |
