"""Parse and validate a task config. Knows nothing about models or data loading.

    task: oregon-plants
    data:        { images: ./photos, background: ./negatives }
    objective:   { metric: label_share, minimum: 0.90 }
    constraints: { max_size_mb: 50, domain: [plant, flora] }
    compute:     { device: mps, max_candidates: 6 }

Every field is validated up front and the errors name the field, because a
config typo that silently becomes a default is how a user ends up believing a
constraint was honoured when it was not.
"""

from dataclasses import dataclass, field
from pathlib import Path

# Metrics `fit` can be asked to satisfy. Each maps to a key `build.fit_and_measure`
# already returns, so adding one here does not mean computing anything new.
METRICS = ("label_share", "coverage", "precision", "closed_set_top1")
SOURCE_KEYS = ("images", "manifest", "embeddings")


@dataclass
class Config:
    task: str
    data: dict
    metric: str
    minimum: float
    max_size_mb: float | None = None
    domain: tuple = ()
    device: str | None = None
    max_candidates: int = 6
    ood_rate: float = 0.2
    hazards: tuple = ()
    encoders: tuple = ()                      # explicit override; skips the search
    background: dict = field(default_factory=dict)


def _require(d, key, where):
    if key not in d:
        raise ValueError(f"config: `{where}` needs a `{key}` field")
    return d[key]


def _source(d, where):
    """One of images/manifest/embeddings, as a dict ready for `sources.load`."""
    given = {k: d[k] for k in SOURCE_KEYS if d.get(k)}
    if len(given) != 1:
        raise ValueError(f"config: `{where}` needs exactly one of "
                         f"{', '.join(SOURCE_KEYS)}; got {list(given) or 'none'}")
    return given


def parse(raw: dict) -> Config:
    """Validate a already-loaded mapping. Separate from `load` so it is testable
    without touching the filesystem."""
    data = _require(raw, "data", "config")
    obj = _require(raw, "objective", "config")
    con = raw.get("constraints") or {}
    comp = raw.get("compute") or {}

    metric = _require(obj, "metric", "objective")
    if metric not in METRICS:
        raise ValueError(f"config: unknown metric {metric!r}; have {', '.join(METRICS)}")
    minimum = float(_require(obj, "minimum", "objective"))
    if not 0 <= minimum <= 1:
        raise ValueError(f"config: `objective.minimum` is a fraction in [0,1], got {minimum}")

    size = con.get("max_size_mb")
    if size is not None and float(size) <= 0:
        raise ValueError(f"config: `constraints.max_size_mb` must be positive, got {size}")

    return Config(
        task=raw.get("task", "task"),
        data=_source(data, "data"),
        background=_source(data["background"], "data.background")
        if isinstance(data.get("background"), dict) else
        ({"images": data["background"]} if data.get("background") else {}),
        metric=metric,
        minimum=minimum,
        max_size_mb=None if size is None else float(size),
        domain=tuple(con.get("domain") or ()),
        device=comp.get("device"),
        max_candidates=int(comp.get("max_candidates", 6)),
        ood_rate=float(obj.get("ood_rate", 0.2)),
        hazards=tuple(obj.get("hazards") or ()),
        encoders=tuple(con.get("encoders") or ()),
    )


def load(path) -> Config:
    """Read YAML (or JSON — YAML is a superset) from disk and validate it."""
    import json
    text = Path(path).read_text()
    try:
        import yaml
        raw = yaml.safe_load(text)
    except ImportError:
        raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError(f"config: {path} did not parse to a mapping")
    return parse(raw)
