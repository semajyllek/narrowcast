"""Where the tool gets data from. Nothing here knows what a plant is.

narrowcast audits a classifier; it does not build an encoder and it never reads
a pixel. Both ways in are self-contained `.npz` files, in increasing order of
"I have already done the work":

    --embeddings FILE     descriptor, label [, group, cluster, origin]
                          Vectors from whatever encoder you chose. A linear head
                          is fitted over them, then the thresholds.
    --scores FILE         proba, classes, label [, group, cluster, origin]
                          Per-row posteriors from a model you already have. Only
                          the thresholds are fitted. Nothing about your model is
                          assumed beyond "it returns a distribution over labels".

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
comparable, and the ones that do not are measurably worse off. `audit` measures
that cost when `--deployment-origin` names which one you will actually see; see
`build.origin_cost`.

There used to be `--images DIR` and `--manifest FILE` here. Both existed to point
at pixels for an encoder this tool no longer carries, and a manifest of paths
without an encoder promises something that cannot happen -- so they are gone
rather than left to fail late.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

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
    proba: np.ndarray | None = None     # set by `from_scores` only
    classes: np.ndarray | None = None   # column order of `proba`
    # Which out-of-list rows are deployment-plausible. See `from_scores`.
    regional: np.ndarray | None = None

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


def _regional(z) -> np.ndarray | None:
    """The caller's `regional` flag: which out-of-list rows a user could actually
    put in front of this model.

    narrowcast cannot derive this and will not guess. A background pool drawn at
    random from everything is dominated by inputs the deployment would never see
    — plantid's is "mosses, ferns and tropical flora a Europe/NA app would never
    be shown" — which makes the reject decision look easier than it is. The
    caller knows which of their negatives are plausible; this is the column that
    says so, and without it the tool simply has no regional bucket rather than
    inventing one.
    """
    if "regional" not in z.files:
        return None
    return np.asarray(z["regional"]).astype(bool)


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
    r = _finish(label, descriptor=z["descriptor"],
                   group=z["group"] if "group" in z.files else None,
                   cluster=cluster,
                   origin=z["origin"] if "origin" in z.files else None,
                   notes=[f"{len(z['descriptor'])} precomputed embeddings from {Path(path).name}"])
    r.regional = _regional(z)
    return r


def from_scores(path) -> Rows:
    """Per-row posteriors from a model that is not ours.

    This is the audit path. The caller has a classifier; what they do not have is
    an honest account of what it will do in front of a user. `proba` is one row
    per observation and one column per entry of `classes`, and nothing beyond
    "the columns are a distribution over `classes`" is assumed about how it was
    produced -- a logistic head, a fine-tuned network, an ensemble, a vendor API.

    Rows are *not* required to sum to 1. A model that abstains by leaving mass
    unassigned, or one whose scores are calibrated to something other than a
    simplex, is still auditable; the cascade reads the largest label mass and the
    largest group mass, and both are order-preserving under a positive rescale.
    """
    z = np.load(Path(path), allow_pickle=True)
    for required in ("proba", "classes"):
        if required not in z.files:
            raise ValueError(f"{path} has no {required!r} array; found {z.files}")
    label = z["label"] if "label" in z.files else None
    if label is None:
        raise ValueError(f"{path} has no 'label' array (the truth column); "
                         f"found {z.files}")
    proba, classes = np.asarray(z["proba"], float), np.asarray(z["classes"], dtype=str)
    if proba.ndim != 2:
        raise ValueError(f"'proba' must be 2-D (rows x classes); got shape {proba.shape}")
    if proba.shape[0] != len(label):
        raise ValueError(f"'proba' has {proba.shape[0]} rows but 'label' has "
                         f"{len(label)}; they must line up")
    if proba.shape[1] != len(classes):
        raise ValueError(f"'proba' has {proba.shape[1]} columns but 'classes' has "
                         f"{len(classes)} entries; they must line up")
    if (proba < 0).any():
        raise ValueError("'proba' contains negative values; the cascade reads these "
                         "as label and group mass, which must be non-negative")
    unknown = sorted(set(np.asarray(label, dtype=str).tolist()) - set(classes.tolist()))
    cluster = z["cluster"] if "cluster" in z.files else (
        z["obs_id"] if "obs_id" in z.files else None)
    notes = [f"{len(proba)} scored rows over {len(classes)} classes from {Path(path).name}"]
    if unknown:
        # Not an error: rows whose truth is outside `classes` are exactly the
        # out-of-list observations the decline threshold is fitted to reject, and
        # an audit without them cannot measure declining at all.
        notes.append(f"{len(unknown)} label(s) not among `classes` — treated as "
                     f"out-of-list: {', '.join(unknown[:4])}"
                     + (" ..." if len(unknown) > 4 else ""))
    r = _finish(label, descriptor=None,
                group=z["group"] if "group" in z.files else None,
                cluster=cluster,
                origin=z["origin"] if "origin" in z.files else None,
                notes=notes)
    r.proba, r.classes = proba, classes
    r.regional = _regional(z)
    if r.regional is not None:
        if len(r.regional) != len(label):
            raise ValueError(f"'regional' has {len(r.regional)} entries but "
                             f"'label' has {len(label)}; they must line up")
        r.notes.append(f"{int(r.regional.sum())} row(s) flagged regional: "
                       "out-of-list inputs the deployment could plausibly see")
    return r


def load(embeddings=None, scores=None) -> Rows:
    """Exactly one source, or a clear error saying so."""
    given = [(k, v) for k, v in
             (("--embeddings", embeddings), ("--scores", scores)) if v]
    if len(given) != 1:
        raise ValueError("give exactly one of --embeddings FILE or --scores FILE" +
                         (f"; got {', '.join(k for k, _ in given)}" if given else ""))
    kind, value = given[0]
    return {"--embeddings": from_embeddings, "--scores": from_scores}[kind](value)
