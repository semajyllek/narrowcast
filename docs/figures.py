"""Generate the deep dive's figures as inline SVG, from the measured data.

Inline SVG rather than raster: it inherits the page's CSS custom properties, so
one file renders correctly in both themes without shipping two images, and it
stays sharp at any width. No external assets, which the artifact CSP forbids
anyway.

Colour is a **sequential ramp, not a categorical palette**, because the three
outcomes are ordered by how much they tell the user: naming a label > naming its
group > declining. A single hue stepped by lightness encodes that order, and is
robust under colour-vision deficiency by construction — the steps differ in
lightness, not only in hue.

Checked rather than eyeballed, on the page's own surfaces:

    light  #1e3a8a / #3b82f6 / #bfd3f2   pairwise OKLab dE 25.0, 27.7, 49.1
    dark   #93c5fd / #3b82f6 / #1e3a5f   pairwise OKLab dE 20.9, 30.0, 46.3

all clear of the 15 floor, and lightness is monotonic in both. The lightest step
sits below 3:1 against its surface, so the relief rule applies and every segment
carries a direct label; the same numbers also appear in the adjacent tables.

Usage:  python docs/figures.py --plantid ../narrowcast-plantid --derm ../narrowcast-derm
It rewrites the FIG:<name> blocks in deep_dive.html in place.
"""

import argparse
import re
from pathlib import Path

import pandas as pd

W = 720                      # viewBox width; the page scales it to full column
INK, INK2, INK3 = "var(--ink)", "var(--ink2)", "var(--ink3)"
RULE, SURF = "var(--rule)", "var(--panel)"
C1, C2, C3 = "var(--c1)", "var(--c2)", "var(--c3)"        # label / group / decline
MONO = "ui-monospace,SFMono-Regular,Menlo,monospace"
SANS = "ui-sans-serif,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif"


def _open(w, h, title, desc):
    # No `height` attribute: `height="auto"` is not a valid SVG length, and a
    # browser that ignores it falls back to the default replaced-element size,
    # which rendered all four figures at an identical wrong aspect ratio. width
    # plus a viewBox plus CSS `height:auto` gives the intrinsic ratio.
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" '
            f'role="img" aria-labelledby="t{abs(hash(title))%9999} d{abs(hash(desc))%9999}" '
            f'style="display:block;width:100%;height:auto;max-width:100%">'
            f'<title id="t{abs(hash(title))%9999}">{title}</title>'
            f'<desc id="d{abs(hash(desc))%9999}">{desc}</desc>')


def _txt(x, y, s, size=12, fill=INK2, anchor="start", weight=400, mono=False):
    fam = MONO if mono else SANS
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}" font-family="{fam}">{s}</text>')


def _legend(x, y, items):
    """Identity is never colour alone: a swatch plus its name, always present."""
    out, cx = [], x
    for colour, name in items:
        out.append(f'<rect x="{cx}" y="{y-9}" width="11" height="11" rx="2.5" fill="{colour}"/>')
        out.append(_txt(cx + 16, y, name, 12, INK2))
        cx += 20 + 7.4 * len(name)
    return "".join(out)


# ---------------------------------------------------------------- figure A

def fig_trap():
    """The published 14-label comparison, four metrics, two label sets.

    Grouped bars rather than stacked: these are four different measures, not
    parts of one whole. One axis, because all four are proportions.
    """
    rows = [("Coverage", .618, .806, "answers at all"),
            ("Precision", .985, .978, "answers that are true"),
            ("Names a species", .761, .476, "the one that falls"),
            ("Right when it names", .970, .817, "and it is worse at it")]
    left, top, rowh, barh, plot = 186, 46, 62, 19, W - 186 - 92
    h = top + rowh * len(rows) + 20
    s = [_open(W, h, "Crowded label sets inflate the reported metrics",
               "Four measures for two 14-label sets. Coverage rises and precision "
               "holds, while the share named to species and the accuracy of those "
               "names both fall sharply.")]
    s.append(_legend(left, 22, [(C1, "14 distinct genera"), (C2, "14 species from 2 genera")]))

    for i, (name, a, b, note) in enumerate(rows):
        y = top + i * rowh
        s.append(_txt(left - 12, y + 13, name, 12.5, INK, "end", 600))
        s.append(_txt(left - 12, y + 29, note, 10.5, INK3, "end"))
        for j, (v, c) in enumerate(((a, C1), (b, C2))):
            by = y + j * (barh + 3)
            s.append(f'<rect x="{left}" y="{by}" width="{plot*v:.1f}" height="{barh}" '
                     f'rx="4" fill="{c}"/>')
            s.append(_txt(left + plot * v + 7, by + barh - 5.5,
                          f"{v*100:.1f}%", 11.5, INK2, "start", 600, mono=True))
        # the delta is the point of the row, so state it rather than leave it to subtraction
        d = (b - a) * 100
        s.append(_txt(W - 8, y + 22, f"{d:+.1f}pp", 12, INK3, "end", 600, mono=True))
    s.append("</svg>")
    return "".join(s)


# ---------------------------------------------------------------- figure B

def fig_domains(df):
    """Where the group answers come from, per domain. The audio result lives here.

    100% stacked, because the three outcomes partition every in-set observation.
    """
    pairs = [("text", "text-varied", "text-crowded", "20 Newsgroups"),
             ("audio", "kws-sem-varied", "kws-sem-crowded", "Speech Commands, semantic groups"),
             ("audio", "kws-ac-varied", "kws-ac-crowded", "Speech Commands, acoustic groups"),
             ("audio", "esc50-varied", "esc50-crowded", "ESC-50"),
             ]
    idx = {r.label_set: r for r in df.itertuples()}
    left, top, gap, barh, plot = 96, 76, 30, 17, W - 96 - 62
    h = top + len(pairs) * (2 * barh + 3 + gap) + 34
    s = [_open(W, h, "Where the group answers come from",
               "Three-way outcome split for a varied and a crowded label set in each "
               "of four corpora. In text the group answers come out of label answers; "
               "in the audio corpora they come out of declines.")]
    s.append(_legend(0, 22, [(C1, "names a label"), (C2, "group only"), (C3, "declines")]))
    s.append(_txt(0, 40, "each bar is 100% of in-set observations", 10.5, INK3))

    y = top
    for _, va, ca, title in pairs:
        # The corpus name sits ABOVE its pair of bars; the arm names sit beside
        # the bar each one describes. An earlier version put all three on the
        # left at one-row offsets, which rendered the title across the first bar.
        s.append(_txt(left, y - 6, title, 12, INK, "start", 600))
        for j, key in enumerate((va, ca)):
            r = idx[key]
            by = y + j * (barh + 3)
            s.append(_txt(left - 10, by + barh - 4.5,
                          "varied" if j == 0 else "crowded", 10.5, INK3, "end"))
            x = left
            for v, c in ((r.label_share, C1), (r.group_share, C2), (r.decline_share, C3)):
                w = plot * v
                if w > 0.7:
                    # 2px surface gap between adjacent fills; the lightest step is
                    # under 3:1 on the panel, so it also carries a hairline edge
                    # or its extent is invisible against the surface.
                    edge = (f' stroke="{INK3}" stroke-width="0.6" stroke-opacity="0.45"'
                            if c == C3 else "")
                    s.append(f'<rect x="{x:.1f}" y="{by}" width="{max(w-2,0.8):.1f}" '
                             f'height="{barh}" rx="2.5" fill="{c}"{edge}/>')
                if w > 42:
                    # carry the unit on the mark: the caption calls these
                    # percentages and a bare "72" beside a "0.72" elsewhere in
                    # the document is a unit ambiguity, not a saving.
                    ink = SURF if c != C3 else INK2
                    s.append(_txt(x + w / 2 - 1, by + barh - 4.5, f"{v*100:.0f}%",
                                  10.5, ink, "middle", 700, mono=True))
                x += w
        # the number that matters: how the group share was paid for
        d = (idx[ca].label_share - idx[va].label_share) * 100
        s.append(_txt(W - 6, y + barh + 2, f"{d:+.1f} pp", 11.5,
                      INK3, "end", 700, mono=True))
        y += 2 * barh + 3 + gap
    s.append(_txt(W - 6, y + 4, "change in label share", 10, INK3, "end"))
    s.append("</svg>")
    return "".join(s)


# ---------------------------------------------------------------- figure C

def fig_scatter(df):
    """Headroom against realised group share, every arm. The rule, and its spread."""
    left, top, plot_w, plot_h = 56, 34, W - 56 - 22, 260
    x0, x1, y0, y1 = -0.05, 0.50, 0.0, 0.90
    def X(v): return left + plot_w * (v - x0) / (x1 - x0)
    def Y(v): return top + plot_h * (1 - (v - y0) / (y1 - y0))
    h = top + plot_h + 62
    s = [_open(W, h, "Headroom against the group-answer share it predicts",
               "1,409 arms. The line is the 1.8x rule of thumb. Points below it are "
               "arms that retreat less than the rule predicts.")]
    # grid + axes, recessive
    for gv in (0.0, 0.2, 0.4, 0.6, 0.8):
        s.append(f'<line x1="{left}" y1="{Y(gv):.1f}" x2="{left+plot_w}" y2="{Y(gv):.1f}" '
                 f'stroke="{RULE}" stroke-width="1"/>')
        s.append(_txt(left - 8, Y(gv) + 4, f"{gv:.1f}", 10.5, INK3, "end", mono=True))
    for gv in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        s.append(_txt(X(gv), top + plot_h + 18, f"{gv:.1f}", 10.5, INK3, "middle", mono=True))
    s.append(_txt(left - 8, top - 10, "group share", 11, INK2, "start"))
    s.append(_txt(left + plot_w, top + plot_h + 36, "headroom  (coarse accuracy − fine accuracy)",
                  11, INK2, "end"))

    pl = df[df.domain == "plants"]
    s.append('<g opacity="0.30">')
    for r in pl.itertuples():
        s.append(f'<circle cx="{X(r.headroom):.1f}" cy="{Y(r.group_share):.1f}" r="2.1" fill="{C2}"/>')
    s.append("</g>")
    # the nine non-plant arms, ringed so they read against the cloud
    for r in df[df.domain != "plants"].itertuples():
        s.append(f'<circle cx="{X(r.headroom):.1f}" cy="{Y(r.group_share):.1f}" r="5.2" '
                 f'fill="{C1}" stroke="{SURF}" stroke-width="2"/>')
    s.append(f'<line x1="{X(0):.1f}" y1="{Y(0):.1f}" x2="{X(0.5):.1f}" y2="{Y(0.9):.1f}" '
             f'stroke="{INK}" stroke-width="2" stroke-dasharray="5 4"/>')
    s.append(_txt(X(0.42), Y(0.83), "group share = 1.8 × headroom", 11, INK, "end", 600))
    s.append(_legend(left, h - 10, [(C2, "1,400 plant arms"), (C1, "9 audio / text / bird arms")]))
    s.append("</svg>")
    return "".join(s)


# ---------------------------------------------------------------- figure D

def fig_ood(sweep):
    """The dermatology sweep: headroom pinned, only the declared OOD rate moving."""
    g = sweep.groupby("p_ood")[["label_share", "group_share", "decline_share",
                                "t_group", "headroom"]].mean()
    left, top, colw, gap, plot_h = 74, 74, 96, 26, 190
    h = top + plot_h + 92
    s = [_open(W, h, "The same model at five declared out-of-set rates",
               "Headroom is identical in all five columns. As the declared "
               "out-of-set rate rises the decline threshold rises with it and the "
               "group answers are absorbed into declines.")]
    s.append(_legend(left, 22, [(C1, "names a label"), (C2, "group only"), (C3, "declines")]))
    s.append(_txt(left, 40, "crowded dermatology arm · headroom fixed at 0.1237 throughout",
                  10.5, INK3))
    for i, (p, r) in enumerate(g.iterrows()):
        x = left + i * (colw + gap)
        y = top
        for v, c in ((r.label_share, C1), (r.group_share, C2), (r.decline_share, C3)):
            bh = plot_h * v
            if bh > 0.6:
                edge = (f' stroke="{INK3}" stroke-width="0.6" stroke-opacity="0.45"'
                        if c == C3 else "")
                s.append(f'<rect x="{x}" y="{y:.1f}" width="{colw}" '
                         f'height="{max(bh-2,0.8):.1f}" rx="3" fill="{c}"{edge}/>')
            if bh > 15:
                # the lightest step needs ink-coloured text; surface-on-surface is
                # unreadable and this segment is where the failure actually shows
                ink = SURF if c != C3 else INK2
                s.append(_txt(x + colw / 2, y + bh / 2 + 4,
                              f"{v*100:.0f}%", 11.5, ink, "middle", 700, mono=True))
            y += bh
        s.append(_txt(x + colw / 2, top + plot_h + 20, f"{p:.2f}", 12.5, INK, "middle", 700, mono=True))
        s.append(_txt(x + colw / 2, top + plot_h + 36, f"t_group {r.t_group:.2f}",
                      10, INK3, "middle", mono=True))
        # the quantity the rule predicts, which never moves
        s.append(_txt(x + colw / 2, top + plot_h + 54, f"group {r.group_share*100:.1f}%",
                      10.5, INK2, "middle", 600, mono=True))
    s.append(_txt(left + 2 * (colw + gap) + colw / 2, top + plot_h + 78,
                  "declared out-of-set rate  p_ood", 11.5, INK2, "middle"))
    s.append(_txt(left - 10, top + 6, "100%", 10.5, INK3, "end", mono=True))
    s.append(_txt(left - 10, top + plot_h, "0", 10.5, INK3, "end", mono=True))
    s.append("</svg>")
    return "".join(s)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plantid", type=Path, default=Path("../narrowcast-plantid"))
    ap.add_argument("--derm", type=Path, default=Path("../narrowcast-derm"))
    ap.add_argument("--html", type=Path, default=Path("docs/deep_dive.html"))
    a = ap.parse_args()

    arms = pd.read_csv(a.plantid / "data/processed/headroom_arms.csv")
    sweep = pd.read_csv(a.derm / "results/ood_sweep.csv")

    figs = {"trap": fig_trap(), "domains": fig_domains(arms),
            "scatter": fig_scatter(arms), "ood": fig_ood(sweep)}

    html = a.html.read_text()
    for name, svg in figs.items():
        pat = re.compile(rf"(<!--FIG:{name}-->).*?(<!--/FIG:{name}-->)", re.S)
        if not pat.search(html):
            print(f"  no placeholder for {name}; skipping")
            continue
        html = pat.sub(lambda m: m.group(1) + svg + m.group(2), html)
        print(f"  {name}: {len(svg):,} bytes")
    a.html.write_text(html)
    print(f"-> {a.html}")


if __name__ == "__main__":
    main()
