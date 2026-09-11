"""Where the tool gets data from. Nothing here knows what a plant is.

The tool takes a dataset; it does not go and find one. Fetching from
iNaturalist, GBIF or anywhere else lives in whatever project owns that domain
(`analysis/` here, an app repo elsewhere), because the choice of corpus, its
licensing and its taxonomy are domain decisions and the tool has no business
making them.

Three ways in, in increasing order of "I have already done the work":

    --images DIR          DIR/<label>/*.jpg
    --manifest FILE       parquet/csv with columns: label, path [, group, cluster, origin]
    --embeddings FILE     .npz with arrays: descriptor, label [, group, cluster, origin]

`cluster` is the unit that must not straddle a train/test split -- several
photographs of one individual, one specimen, one manufacturing run. Supply it
whenever the data has that structure; without it every row is treated as
independent, which is the assumption `CLAUDE.md`'s first convention exists to
warn about. `group` is the coarse rank the cascade can fall back to; it defaults
to the first whitespace-delimited token of the label, which is exactly right for
Linnaean binomials and often right elsewhere.

`origin` is which acquisition source or population a row came from -- a corpus, a
device, a skin-type band, a speaker group. It is optional and nothing requires
it. Supply it when your rows come from more than one, because labels that have
training rows from the *deployment* origin and labels that do not are not
comparable, and the ones that do not are measurably worse off. `build` measures
that cost when `--deployment-origin` names which one you will actually see; see
`build.origin_cost`.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class Rows:
    """A dataset before embedding: labels, groups, clusters, and where to find pixels."""
    label: np.ndarray
    group: np.ndarray
    cluster: np.ndarray
    path: np.ndarray | None = None          # None when embeddings were supplied
    descriptor: np.ndarray | None = None
    origin: np.ndarray | None = None        # acquisition source / population, optional
    has_clusters: bool = True
    notes: list[str] = field(default_factory=list)

    def __len__(self):
        return len(self.label)

    @property
    def labels(self) -> list[str]:
        return sorted(set(self.label.tolist()))


def default_group(label: str) -> str:
    return str(label).split()[0] if str(label).split() else str(label)


def _finish(label, path=None, descriptor=None, group=None, cluster=None, notes=None,
            origin=None) -> Rows:
    label = np.asarray(label, dtype=str)
    notes = list(notes or [])
    if group is None:
        group = np.array([default_group(x) for x in label])
        notes.append("group inferred from the first token of each label")
    else:
        group = np.asarray(group, dtype=str)
    has_clusters = cluster is not None
    if not has_clusters:
        cluster = np.arange(len(label)).astype(str)
        notes.append("no cluster column supplied: every row treated as independent, "
                     "so intervals are anticonservative if several rows share a subject")
    else:
        # A cluster column of unique ids is arithmetically identical to no cluster
        # column at all -- every cluster has one row, so resampling clusters is
        # resampling rows. It is worse than supplying nothing, because it looks
        # like the protection is on and suppresses the warning above.
        #
        # Found by running `fit` over a real mixed corpus: Pl@ntNet has no
        # observation grouping, so its rows were keyed by image id and 93% of
        # clusters came out singleton, while the card reported honest-looking
        # clustered intervals.
        c = np.asarray(cluster, dtype=str)
        _, counts = np.unique(c, return_counts=True)
        singleton = float((counts == 1).sum()) / max(len(counts), 1)
        if singleton > 0.5:
            notes.append(
                f"{100 * singleton:.0f}% of clusters contain a single row, so for "
                f"those rows the bootstrap is row-level and the intervals are "
                f"anticonservative — a unique id per row is the same as supplying "
                f"no cluster column")
    if origin is not None:
        origin = np.asarray(origin, dtype=str)
        notes.append(f"origin supplied: {len(set(origin.tolist()))} distinct "
                     "(pass --deployment-origin to measure what it costs)")
    return Rows(label, group, np.asarray(cluster, dtype=str),
                None if path is None else np.asarray(path, dtype=str),
                descriptor, origin, has_clusters, notes)


def from_images(root) -> Rows:
    """DIR/<label>/*.jpg — the least-effort layout."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a directory")
    labels, paths = [], []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(sub.iterdir()):
            if f.suffix.lower() in IMAGE_SUFFIXES:
                labels.append(sub.name.replace("_", " "))
                paths.append(str(f))
    if not labels:
        raise ValueError(f"no images under {root} — expected {root}/<label>/*.jpg")
    return _finish(labels, path=paths,
                   notes=[f"{len(set(labels))} labels from subdirectory names under {root}"])


def from_manifest(path) -> Rows:
    """A table with `label` and `path`, optionally `group`, `cluster` and `origin`."""
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    for required in ("label", "path"):
        if required not in cols:
            raise ValueError(f"{path} needs a '{required}' column; found {list(df.columns)}")
    return _finish(df[cols["label"]], path=df[cols["path"]],
                   group=df[cols["group"]] if "group" in cols else None,
                   cluster=df[cols["cluster"]] if "cluster" in cols else None,
                   origin=df[cols["origin"]] if "origin" in cols else None,
                   notes=[f"{len(df)} rows from {path.name}"])


def from_embeddings(path) -> Rows:
    """Precomputed vectors: skips the encoder entirely."""
    z = np.load(Path(path), allow_pickle=True)
    if "descriptor" not in z.files:
        raise ValueError(f"{path} has no 'descriptor' array; found {z.files}")
    label = z["label"] if "label" in z.files else z.get("species_name")
    if label is None:
        raise ValueError(f"{path} has no 'label' array; found {z.files}")
    cluster = z["cluster"] if "cluster" in z.files else (
        z["obs_id"] if "obs_id" in z.files else None)
    return _finish(label, descriptor=z["descriptor"],
                   group=z["group"] if "group" in z.files else None,
                   cluster=cluster,
                   origin=z["origin"] if "origin" in z.files else None,
                   notes=[f"{len(z['descriptor'])} precomputed embeddings from {Path(path).name}"])


def load(images=None, manifest=None, embeddings=None) -> Rows:
    """Exactly one source, or a clear error saying so."""
    given = [(k, v) for k, v in
             (("--images", images), ("--manifest", manifest), ("--embeddings", embeddings)) if v]
    if len(given) != 1:
        raise ValueError("give exactly one of --images DIR, --manifest FILE or "
                         "--embeddings FILE" +
                         (f"; got {', '.join(k for k, _ in given)}" if given else ""))
    kind, value = given[0]
    return {"--images": from_images, "--manifest": from_manifest,
            "--embeddings": from_embeddings}[kind](value)
