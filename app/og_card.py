"""The share card, drawn per question.

The card was a static PNG with an example question baked into its mock
search box. Sharing a search put two different questions in front of the
reader -- "how did he turn $500 into $40 million?" printed in the image,
and the real one overlaid by X from og:title. It read as a mistake
because it was one.

So the box gets the question that was actually asked. Pillow rather than
a headless browser: the card is 1200x630 of flat shapes and four lines of
text, and the alternative is shipping Chrome in the image to draw it.

The two faces are the site's own, checked into demo/fonts because
python:3.12-slim has no fonts at all -- not even DejaVu. Both are OFL.
"""

from __future__ import annotations

import hashlib
import io
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_ROOT = Path(__file__).resolve().parent.parent
_FONTS = _ROOT / "demo" / "fonts"

W, H = 1200, 630
BG = (11, 14, 17)
PANEL = (18, 22, 28)
LINE = (35, 42, 51)
TEXT = (230, 232, 234)
MUTED = (138, 147, 158)
GREEN = (22, 199, 132)
INK = (4, 18, 12)

# Inter ships as a variable font; Pillow reaches the weights through the
# axes rather than separate files.
_INTER = _FONTS / "Inter-Variable.ttf"
_SERIF = _FONTS / "InstrumentSerif-Regular.ttf"


@lru_cache(maxsize=32)
def _inter(size: int, weight: int = 400) -> ImageFont.FreeTypeFont:
    font = ImageFont.truetype(str(_INTER), size)
    try:
        font.set_variation_by_axes([14.0, float(weight)])
    except OSError:                      # a static build, or no axes
        pass
    return font


@lru_cache(maxsize=8)
def _serif(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(_SERIF), size)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, width: int,
          max_lines: int) -> list[str]:
    """Greedy wrap, with an ellipsis rather than a hard cut.

    A question arrives from a URL anybody can craft, so it can be one
    word of six hundred characters. Falling back to a character split
    keeps that from running off the card.
    """
    words, lines, current = text.split(), [], ""
    for word in words:
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) <= width or not current:
            current = trial
            # One unbroken token wider than the card.
            while draw.textlength(current, font=font) > width and len(current) > 1:
                lines.append(current[:-1])
                current = current[-1:]
                if len(lines) >= max_lines:
                    break
        else:
            lines.append(current)
            current = word
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    if len(lines) == max_lines and current not in lines[-1:]:
        pass
    # Trim the tail if there is more text than room.
    joined = " ".join(lines)
    if len(joined) < len(text.strip()):
        last = lines[-1]
        while (lines and draw.textlength(last + "…", font=font) > width
               and len(last) > 1):
            last = last[:-1]
        lines[-1] = last.rstrip() + "…"
    return lines


def _rounded(draw, box, radius, fill=None, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill,
                           outline=outline, width=width)


def render(question: str | None = None) -> bytes:
    """A 1200x630 PNG. With a question, it sits in the search box."""
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # The glow the site carries. Pillow has no radial gradient, so it is
    # built from stacked translucent ellipses on their own layer --
    # compositing once keeps the falloff smooth instead of banding.
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    for i in range(30, 0, -1):
        r = 90 + i * 20
        gd.ellipse((W - 300 - r, -180 - r, W - 300 + r, -180 + r),
                   fill=(*GREEN, 5))
    img = Image.alpha_composite(img.convert("RGBA"), glow).convert("RGB")
    draw = ImageDraw.Draw(img)

    x = 76
    draw.text((x, 62), "MARKET", font=_inter(25, 800), fill=TEXT)
    bw = draw.textlength("MARKET ", font=_inter(25, 800))
    draw.text((x + bw, 62), "BUBBLE", font=_inter(25, 800), fill=GREEN)
    bw += draw.textlength("BUBBLE ", font=_inter(25, 800))
    draw.text((x + bw, 62), "· SEARCH", font=_inter(25, 800), fill=MUTED)

    head = _serif(72)
    draw.text((x, 122), "Ask the ", font=head, fill=TEXT)
    hw = draw.textlength("Ask the ", font=head)
    draw.text((x + hw, 122), "Market Bubble", font=head, fill=GREEN)
    hw += draw.textlength("Market Bubble ", font=head)
    draw.text((x + hw, 122), "broadcast", font=head, fill=TEXT)
    draw.text((x, 206), "anything.", font=head, fill=TEXT)

    # The search box. Green border when it holds a real question, so the
    # card reads as the product mid-use rather than a screenshot of an
    # empty page.
    top, bottom = 320, 404
    box_w = W - x * 2
    asked = (question or "").strip()
    _rounded(draw, (x, top, x + box_w, bottom), 18, fill=PANEL,
             outline=GREEN if asked else LINE, width=2)

    btn_w = 168
    _rounded(draw, (x + box_w - btn_w, top, x + box_w, bottom), 18, fill=GREEN)
    bf = _inter(26, 700)
    draw.text((x + box_w - btn_w + (btn_w - draw.textlength("Search", font=bf)) / 2,
               top + (bottom - top - 30) / 2), "Search", font=bf, fill=INK)

    qf = _inter(28, 500)
    inner = box_w - btn_w - 64
    if asked:
        lines = _wrap(draw, asked, qf, inner, 2)
        y = top + (bottom - top - len(lines) * 36) / 2
        for line in lines:
            draw.text((x + 30, y), line, font=qf, fill=TEXT)
            y += 36
    else:
        draw.text((x + 30, top + 26),
                  "e.g. what did they say about liquidation?",
                  font=qf, fill=(96, 105, 116))

    sub = ("Answered from the transcripts, with the moment it was said."
           if asked else
           "Search every episode by meaning — and jump to the exact moment.")
    draw.text((x, 440), sub, font=_inter(25), fill=MUTED)

    bx = x
    for label in ("▶ jumps to the timestamp", "no login", "open source"):
        f = _inter(21, 600)
        tw = draw.textlength(label, font=f)
        _rounded(draw, (bx, 508, bx + tw + 40, 556), 24,
                 outline=(24, 78, 60), width=2)
        draw.text((bx + 20, 519), label, font=f, fill=GREEN)
        bx += tw + 56

    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()


def etag(question: str | None) -> str:
    """Stable per question, so a crawler that has seen it does not refetch."""
    return hashlib.sha256((question or "").encode()).hexdigest()[:16]
