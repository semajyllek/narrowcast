"""Run several encoders through the existing pipeline and report the frontier.

Knows nothing about config files and nothing about how to embed — it takes a
list of encoder names and a loaded dataset, and calls the same `build` functions
a single-encoder run uses. Every number here is produced by the code that
produces a card, so the frontier and the card cannot disagree.

The decision rule is **smallest that qualifies**, not best. Size is the
constraint the user declared; accuracy above the floor they set is not worth
paying bytes for.
"""

from dataclasses import dataclass

from narrowcast import build as B


@dataclass
class Result:
    encoder: str
    size_mb: float | None
    metrics: dict | None
    error: str | None = None

    def value(self, metric: str):
        return None if not self.metrics else self.metrics.get(metric)

    def qualifies(self, metric: str, minimum: float) -> bool:
        v = self.value(metric)
        return v is not None and v >= minimum


def evaluate(encoder: str, size_mb, rows, background, ood_rate, hazards, seed=0) -> Result:
    """One encoder, end to end, through the same path `build` uses."""
    try:
        ds = B.load_rows(rows, encoder, background=background, seed=seed)
        clf = B.fit_head(ds)
        frame = B.score_frame(clf, ds)
        metrics = B.fit_and_measure(frame, p_ood=ood_rate, hazards=hazards)
        return Result(encoder, size_mb, metrics)
    except Exception as e:                     # one bad candidate must not end the sweep
        return Result(encoder, size_mb, None, f"{type(e).__name__}: {e}")


def run(candidates, rows, background, metric, ood_rate=0.2, hazards=(), on_result=None):
    """`candidates` is [(encoder_name, size_mb)]. Returns results, largest metric first."""
    out = []
    for name, size in candidates:
        r = evaluate(name, size, rows, background, ood_rate, hazards)
        out.append(r)
        if on_result:
            on_result(r)
    return sorted(out, key=lambda r: (r.value(metric) is None, -(r.value(metric) or 0)))


def choose(results, metric: str, minimum: float):
    """Smallest candidate clearing the floor, or None. Ties broken by metric."""
    ok = [r for r in results if r.qualifies(metric, minimum)]
    if not ok:
        return None
    return sorted(ok, key=lambda r: (r.size_mb if r.size_mb is not None else 1e9,
                                     -(r.value(metric) or 0)))[0]


def shortfall(results, metric: str, minimum: float):
    """The closest candidate and how far short it fell — for the refusal message."""
    scored = [r for r in results if r.value(metric) is not None]
    if not scored:
        return None, None
    best = max(scored, key=lambda r: r.value(metric))
    return best, minimum - best.value(metric)


def render(results, metric: str, minimum: float, max_size_mb=None) -> str:
    """The frontier — printed whether the run succeeds or refuses."""
    L = [f"  {'encoder':32s} {'int4':>9s} {metric:>14s}  {'other':>28s}",
         f"  {'-'*32} {'-'*9} {'-'*14}  {'-'*28}"]
    for r in results:
        size = "?" if r.size_mb is None else f"{r.size_mb:.1f}M"
        if r.error:
            L.append(f"  {r.encoder[:32]:32s} {size:>9s} {'failed':>14s}  {r.error[:28]}")
            continue
        v = r.value(metric)
        mark = "  ok" if v is not None and v >= minimum else "    "
        other = (f"cov {r.metrics['coverage']:.3f} prec {r.metrics['precision']:.3f}"
                 if r.metrics.get("precision") is not None else "")
        L.append(f"  {r.encoder[:32]:32s} {size:>9s} {v:14.4f}{mark} {other:>28s}")
    budget = "" if max_size_mb is None else f", budget {max_size_mb:.0f} MB"
    L.append(f"\n  floor: {metric} >= {minimum:.4f}{budget}")
    return "\n".join(L)
