#!/usr/bin/env python3
"""build_deck.py — turn a simple slide-spec Markdown file into a single,
self-contained, offline HTML presentation. Standard library only.

Usage:
    python build_deck.py --input slides.md --output deck.html \
        --title "Weekly Update" --subtitle "Frame-stride + adapters" \
        --date 2026-05-25 --author "Lukas Bierling" --deliverable "D2/D3"

The title slide is generated from the CLI flags; the slide-spec contains
only the CONTENT slides. Slides are separated by a line containing only
`---`. See SKILL.md for the full slide-spec mini-syntax. The theme is read
from --theme (default: theme.css next to this script) and inlined, so the
output HTML has zero external dependencies.
"""
from __future__ import annotations

import argparse
import base64
import html
import mimetypes
import re
from pathlib import Path

# ----------------------------------------------------------------------------
# Inline formatting: **bold**, *italic*, `code`, [text](url). HTML-escape first,
# protect code spans, then apply the rest.
# ----------------------------------------------------------------------------
_CODE_RE = re.compile(r"`([^`]+)`")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)([^*]+)\*(?!\*)")


def inline(text: str) -> str:
    text = html.escape(text, quote=False)
    codes: list[str] = []

    def _stash(m: re.Match) -> str:
        codes.append(f"<code>{m.group(1)}</code>")
        return f"\x00{len(codes) - 1}\x00"

    text = _CODE_RE.sub(_stash, text)
    text = _LINK_RE.sub(r'<a href="\2">\1</a>', text)
    text = _BOLD_RE.sub(r"<strong>\1</strong>", text)
    text = _ITALIC_RE.sub(r"<em>\1</em>", text)
    text = re.sub(r"\x00(\d+)\x00", lambda m: codes[int(m.group(1))], text)
    return text


def embed_image(src: str, base_dir: Path) -> str:
    if src.startswith(("http://", "https://", "data:")):
        return src
    p = Path(src) if Path(src).is_absolute() else (base_dir / src)
    if p.exists():
        mime = mimetypes.guess_type(str(p))[0] or "image/png"
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{b64}"
    return src  # leave as-is; explicit broken ref beats a silent guess


# ----------------------------------------------------------------------------
# Block rendering for one slide's lines.
# ----------------------------------------------------------------------------
def render_blocks(lines: list[str], base_dir: Path) -> str:
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        # fenced code block
        if stripped.startswith("```"):
            lang = stripped[3:].strip()
            body: list[str] = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                body.append(lines[i])
                i += 1
            i += 1  # closing fence
            cls = f' class="language-{html.escape(lang)}"' if lang else ""
            code = html.escape("\n".join(body), quote=False)
            out.append(f"<pre><code{cls}>{code}</code></pre>")
            continue

        # ::: fenced regions (metrics / columns / note)
        m = re.match(r":::\s*(\w+)", stripped)
        if m:
            kind = m.group(1)
            body, i = _collect_until(lines, i + 1, ":::")
            out.append(_render_region(kind, body, base_dir))
            continue

        # headings
        if stripped.startswith("### "):
            out.append(f"<h3>{inline(stripped[4:])}</h3>"); i += 1; continue
        if stripped.startswith("## "):
            out.append(f"<h2>{inline(stripped[3:])}</h2>"); i += 1; continue
        if stripped.startswith("# "):
            out.append(f"<h1>{inline(stripped[2:])}</h1>"); i += 1; continue

        # speaker note (single line)
        if stripped.startswith("<!-- notes:"):
            note = stripped[len("<!-- notes:"):].rstrip("->").strip()
            out.append(f'<aside class="notes">{inline(note)}</aside>'); i += 1; continue

        # blockquote / callout (consecutive > lines)
        if stripped.startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip()[1:].strip()); i += 1
            out.append(f'<div class="callout">{inline(" ".join(buf))}</div>')
            continue

        # image or video (same ![caption](path) syntax; video by extension)
        im = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
        if im:
            alt, src = im.group(1), im.group(2)
            embedded = embed_image(src, base_dir)  # base64 data URI (mime guessed)
            cap = f"<figcaption>{inline(alt)}</figcaption>" if alt else ""
            stem = src.lower().split("?", 1)[0]
            if stem.endswith((".mp4", ".webm", ".ogg", ".mov", ".m4v")):
                out.append(
                    f'<figure><video src="{embedded}" controls autoplay muted loop '
                    f'playsinline></video>{cap}</figure>'
                )
            else:
                out.append(f'<figure><img src="{embedded}" alt="{html.escape(alt)}">{cap}</figure>')
            i += 1; continue

        # GFM table: a header row of |-separated cells followed by a delimiter row
        if "|" in stripped and i + 1 < n and _is_table_delim(lines[i + 1]):
            header = _split_row(stripped)
            i += 2  # consume header + delimiter row
            rows: list[list[str]] = []
            while i < n and lines[i].strip() and "|" in lines[i] \
                    and lines[i].strip() != "|||" and not lines[i].strip().startswith(":::"):
                rows.append(_split_row(lines[i].strip()))
                i += 1
            out.append(_render_table(header, rows))
            continue

        # list (bullets or ordered) — collect consecutive list lines
        if re.match(r"\s*([-*]|\d+\.)\s+", line):
            items: list[tuple[int, str, bool]] = []
            while i < n and re.match(r"\s*([-*]|\d+\.)\s+", lines[i]):
                lm = re.match(r"(\s*)([-*]|\d+\.)\s+(.*)", lines[i])
                indent = len(lm.group(1)) // 2
                ordered = bool(re.match(r"\d+\.", lm.group(2)))
                items.append((indent, inline(lm.group(3)), ordered))
                i += 1
            out.append(_render_list(items))
            continue

        # blank line
        if not stripped:
            i += 1; continue

        # paragraph
        out.append(f"<p>{inline(stripped)}</p>"); i += 1

    return "\n".join(out)


def _collect_until(lines: list[str], start: int, marker: str) -> tuple[list[str], int]:
    body: list[str] = []
    i = start
    while i < len(lines) and lines[i].strip() != marker:
        body.append(lines[i]); i += 1
    return body, i + 1  # skip closing marker


def _render_region(kind: str, body: list[str], base_dir: Path) -> str:
    if kind == "metrics":
        cards = []
        for row in body:
            if not row.strip():
                continue
            parts = [p.strip() for p in row.split("|")]
            label = parts[0] if len(parts) > 0 else ""
            value = parts[1] if len(parts) > 1 else ""
            source = parts[2] if len(parts) > 2 else ""
            src_html = f'<div class="metric-source">{inline(source)}</div>' if source else ""
            cards.append(
                f'<div class="metric-card"><div class="metric-value">{inline(value)}</div>'
                f'<div class="metric-label">{inline(label)}</div>{src_html}</div>'
            )
        return f'<div class="metric-grid">{"".join(cards)}</div>'
    if kind == "columns":
        # split on a line that is exactly |||
        cols: list[list[str]] = [[]]
        for row in body:
            if row.strip() == "|||":
                cols.append([])
            else:
                cols[-1].append(row)
        rendered = "".join(f'<div class="col">{render_blocks(c, base_dir)}</div>' for c in cols)
        return f'<div class="columns">{rendered}</div>'
    if kind == "note":
        return f'<aside class="notes">{render_blocks(body, base_dir)}</aside>'
    # unknown region → render as plain blocks
    return render_blocks(body, base_dir)


def _split_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _is_table_delim(line: str) -> bool:
    cells = _split_row(line)
    seen = False
    for c in cells:
        c2 = c.replace(" ", "")
        if c2 == "":
            continue
        if not re.fullmatch(r":?-+:?", c2):
            return False
        seen = True
    return seen


def _render_table(header: list[str], rows: list[list[str]]) -> str:
    ncol = len(header)
    thead = "".join(f"<th>{inline(c)}</th>" for c in header)
    body = []
    for r in rows:
        cells = (r + [""] * ncol)[:ncol]
        body.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells) + "</tr>")
    return (
        f'<table class="md-table"><thead><tr>{thead}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table>'
    )


def _render_list(items: list[tuple[int, str, bool]]) -> str:
    # build nested list from (indent, html, ordered) tuples
    def build(idx: int, level: int) -> tuple[str, int]:
        tag = "ol" if items[idx][2] else "ul"
        parts = [f"<{tag}>"]
        while idx < len(items) and items[idx][0] == level:
            parts.append(f"<li>{items[idx][1]}")
            # nested children?
            if idx + 1 < len(items) and items[idx + 1][0] > level:
                child, idx = build(idx + 1, level + 1)
                parts.append(child)
            parts.append("</li>")
            idx += 1
        parts.append(f"</{tag}>")
        return "".join(parts), idx

    html_out, _ = build(0, items[0][0])
    return html_out


# ----------------------------------------------------------------------------
# Assemble the deck.
# ----------------------------------------------------------------------------
NAV_JS = """
(function () {
  var slides = Array.prototype.slice.call(document.querySelectorAll('.slide'));
  var i = 0;
  var counter = document.getElementById('counter');
  var footcount = document.getElementById('footcount');
  var pbar = document.getElementById('pbar');
  function show(n) {
    i = Math.max(0, Math.min(slides.length - 1, n));
    slides.forEach(function (s, k) { s.classList.toggle('active', k === i); });
    counter.textContent = (i + 1) + ' / ' + slides.length;
    if (footcount) footcount.textContent = (i + 1) + ' / ' + slides.length;
    pbar.style.width = ((i + 1) / slides.length * 100) + '%';
    location.hash = i + 1;
  }
  document.getElementById('next').onclick = function () { show(i + 1); };
  document.getElementById('prev').onclick = function () { show(i - 1); };
  document.addEventListener('keydown', function (e) {
    if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') { show(i + 1); e.preventDefault(); }
    else if (e.key === 'ArrowLeft' || e.key === 'PageUp') { show(i - 1); }
    else if (e.key === 'Home') { show(0); }
    else if (e.key === 'End') { show(slides.length - 1); }
  });
  var start = parseInt(location.hash.replace('#', ''), 10);
  show(isNaN(start) ? 0 : start - 1);
})();
"""


def build_title_slide(args) -> str:
    parts = ['<section class="slide title-slide">']
    parts.append('<div class="kicker">Weekly Thesis Update</div>')
    parts.append(f"<h1>{inline(args.title)}</h1>")
    if args.subtitle:
        parts.append(f'<div class="subtitle">{inline(args.subtitle)}</div>')
    meta = []
    if args.author:
        meta.append(f"<strong>{inline(args.author)}</strong>")
    if args.date:
        meta.append(inline(args.date))
    if args.deliverable:
        meta.append(f"Focus: {inline(args.deliverable)}")
    if meta:
        parts.append(f'<div class="meta">{" &nbsp;·&nbsp; ".join(meta)}</div>')
    parts.append("</section>")
    return "\n".join(parts)


def split_slides(text: str) -> list[list[str]]:
    slides: list[list[str]] = [[]]
    for line in text.splitlines():
        if line.strip() == "---":
            slides.append([])
        else:
            slides[-1].append(line)
    return [s for s in slides if any(l.strip() for l in s)]


def render_slide(lines: list[str], base_dir: Path) -> str:
    content = render_blocks(lines, base_dir)
    # section divider = slide whose only element is an <h1>
    is_divider = content.count("<h1>") == 1 and re.sub(r"<h1>.*?</h1>", "", content, flags=re.S).strip() == ""
    cls = "slide section-divider" if is_divider else "slide"
    return f'<section class="{cls}">{content}</section>'


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a self-contained HTML deck from a slide-spec markdown.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--date", default="")
    ap.add_argument("--author", default="")
    ap.add_argument("--deliverable", default="")
    ap.add_argument("--theme", default=str(Path(__file__).parent / "theme.css"))
    args = ap.parse_args()

    input_path = Path(args.input)
    base_dir = input_path.parent
    spec = input_path.read_text(encoding="utf-8")

    theme_path = Path(args.theme)
    css = theme_path.read_text(encoding="utf-8") if theme_path.exists() else ""

    slides_html = [build_title_slide(args)]
    for slide_lines in split_slides(spec):
        slides_html.append(render_slide(slide_lines, base_dir))

    footer_bits = [b for b in (args.date, args.deliverable) if b]
    footer = " · ".join(html.escape(b) for b in footer_bits)

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(args.title)}</title>
<style>
{css}
</style>
</head>
<body>
<div class="deck">
<div class="progress"><div class="progress-bar" id="pbar"></div></div>
{chr(10).join(slides_html)}
<div class="deck-footer">{footer}{' · ' if footer else ''}<span id="footcount"></span></div>
<div class="nav"><button id="prev" aria-label="Previous">‹</button><span id="counter"></span><button id="next" aria-label="Next">›</button></div>
</div>
<script>{NAV_JS}</script>
</body>
</html>
"""
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    n = len(slides_html)
    print(f"Wrote {out_path} ({n} slides, self-contained, {len(doc)} bytes)")


if __name__ == "__main__":
    main()
