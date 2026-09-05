"""Projecting what a labels list will give you, before spending any compute.

`plan` has to answer "what will I get" without fetching images or fitting a head,
so it interpolates the measured grid in `frontier.json` rather than running
anything. Two axes, both of which the measurements show matter:

  K                   catalogue size -- the encoder gap shrinks as K falls
  sibling fraction   share of chosen labels sharing a group with another
                      chosen labels -- drives the label-level share

The sibling anchors are **measured, not assumed**. The "unrelated" arm is not
at 0.0: random draws from a 530-labels catalogue over 172 groups collide often
enough to sit at 0.10 (K=10), 0.23 (K=20) and 0.44 (K=50). Interpolating from a
notional 0.0 would make well-separated sets look better than anything measured.

This is a projection from measurements on *a* catalogue, not a guarantee about
the user's. `build` measures the real thing and the card reports that instead.
"""

import json
from bisect import bisect_left
from pathlib import Path

PROFILES = Path(__file__).parent / "profiles"
METRICS = ("top1", "coverage", "precision", "label_share")
DEFAULT_PROFILE = "plants-bioclip2"


def available() -> list[str]:
    return sorted(p.stem for p in PROFILES.glob("*.json"))


def load_profile(name: str | None):
    """The measured grid for one domain+encoder pair, or None if there isn't one.

    Projection is **disabled** without a profile rather than falling back to the
    plant grid. `BIRDS_FINDINGS.md` licenses stating the coarse-answer trap as a
    property of hierarchical label sets in general -- but it licenses the
    *warnings*, not the numbers. These were measured on 530 plant labels through
    BioCLIP-2 and mean nothing for fungi, SKUs or weld defects.
    """
    if name is None:
        return None
    f = PROFILES / f"{name}.json"
    return json.loads(f.read_text()) if f.exists() else None


GRID = load_profile(DEFAULT_PROFILE)


def _interp(x, x0, x1, y0, y1):
    if x1 == x0:
        return y0
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def _cells(encoder, K, GRID):
    """The easy/hard pair measured at this K, with their sibling anchors."""
    rows = [r for r in GRID["cells"] if r["encoder"] == encoder and r["K"] == K]
    by_arm = {r["arm"]: r for r in rows}
    anchors = GRID["sibling_anchors"][str(K)]
    return by_arm, anchors


def _at_K(encoder, K, cfrac, p_ood, GRID):
    """Interpolate between the easy and hard arms at a measured K."""
    by_arm, anchors = _cells(encoder, K, GRID)
    if "easy" not in by_arm or "hard" not in by_arm:
        return None
    out = {}
    for m in METRICS:
        lo = by_arm["easy"][m] if m == "top1" else by_arm["easy"]["p_ood"][p_ood][m]
        hi = by_arm["hard"][m] if m == "top1" else by_arm["hard"]["p_ood"][p_ood][m]
        c = min(max(cfrac, anchors["easy"]), anchors["hard"])
        out[m] = _interp(c, anchors["easy"], anchors["hard"], lo, hi)
    return out


def project(encoder: str, n_labels: int, sibling_frac: float, p_ood: float = 0.2,
            profile: str | None = DEFAULT_PROFILE) -> dict:
    """Projected top-1, coverage, precision and label-level share.

    Clamped to the measured range of K rather than extrapolated: outside 10-50
    labels there is no measurement, and a projection that silently continues a
    trend past its evidence is the thing this project's conventions exist to
    prevent. `extrapolated` says which way it was clamped.
    """
    GRID = load_profile(profile)
    if GRID is None:
        raise ValueError(
            f"no measured profile {profile!r}; projection is disabled without one. "
            f"Available: {available() or 'none'}. `plan` still reports structure.")
    ks = sorted({r["K"] for r in GRID["cells"] if r["encoder"] == encoder})
    if not ks:
        raise ValueError(f"no measurements for encoder {encoder!r}")
    key = f"{p_ood:g}"
    if key not in GRID["p_ood_measured"]:
        raise ValueError(f"p_ood {p_ood} not measured; have {GRID['p_ood_measured']}")

    K = min(max(n_labels, ks[0]), ks[-1])
    extrapolated = None if K == n_labels else ("below" if n_labels < ks[0] else "above")

    i = bisect_left(ks, K)
    if i < len(ks) and ks[i] == K:
        vals = _at_K(encoder, K, sibling_frac, key, GRID)
    else:
        lo, hi = ks[i - 1], ks[i]
        a = _at_K(encoder, lo, sibling_frac, key, GRID)
        b = _at_K(encoder, hi, sibling_frac, key, GRID)
        vals = {m: _interp(K, lo, hi, a[m], b[m]) for m in METRICS}

    return {**vals, "K_used": K, "extrapolated": extrapolated, "profile": profile,
            "encoder": encoder, "p_ood": p_ood, "sibling_frac": sibling_frac}


def measured_p_ood(profile: str | None = DEFAULT_PROFILE) -> list[float]:
    g = load_profile(profile)
    return [float(p) for p in g["p_ood_measured"]] if g else []
