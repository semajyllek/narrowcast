"""Fit a head for a chosen label set, measure it honestly, write a bundle.

The encoder is frozen and shared; what gets built per user is a logistic head
plus two thresholds. That is the whole personalisation story, and it is why this
costs CPU-seconds rather than GPU-hours: at 20 labels and 512 dimensions the
head is ~40 KB against an encoder of 17.9 MB.

Measurement here is not the same as `plan`'s projection. `plan` interpolates a
grid measured on someone else's catalogue; `build` fits on the user's actual data
and evaluates on held-out rows of it, so the card reports the real thing.

Three evaluation buckets, matching `eval/rejection.py`:

  in_catalog   held-out rows of the chosen labels
  near_ood     rows of pool labels outside the set that share a group with it
  distant_ood  held-out background rows

near_ood is built from the relatives the user did not choose, which is the
failure mode a narrow catalogue actually has. A build whose pool contains no
such relatives reports that, rather than quietly scoring rejection on easy
negatives alone.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from narrowcast.cascade import (
    DECLINE,
    GROUP,
    LABEL,
    UTILITY,
    cluster_bootstrap,
    decide,
    deployment_weights,
    fit_thresholds,
    group_matrix,
    make_splits,
)
from narrowcast.labels import group_of

OTHER = "__OTHER__"
BG_TRAIN_FRAC = 0.6
OOD_MIX = {"near_ood": 0.32, "distant_ood": 0.68}
BUNDLE_VERSION = 1


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    frame: pd.DataFrame          # per-observation scores are added after fitting
    X_eval: np.ndarray
    truth: np.ndarray
    bucket: np.ndarray
    counts: dict
    cluster: np.ndarray          # real label per eval row, background included
    group: np.ndarray | None = None   # caller-supplied coarse rank, per eval row


def _l2(X):
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def load_rows(rows, encoder_variant: str, background=None, seed: int = 0) -> Dataset:
    """Assemble a Dataset from a caller-supplied source (`tool/sources.py`).

    The tool does not fetch. It is handed images, a manifest or precomputed
    vectors, and the domain that produced them keeps ownership of how.

    `background` is an optional second source of negatives. Without it there is
    no reject class: the model is closed-set, cannot decline, and the card says
    so rather than implying a rejection capability that was never fitted.
    """
    def _vecs(r):
        if r.descriptor is not None:
            return _l2(np.asarray(r.descriptor, dtype="float32"))
        from narrowcast.encode import embed_images, load_encoder
        model, preprocess, device = load_encoder(encoder_variant)
        return _l2(embed_images(list(r.path), model, preprocess, device))

    X = _vecs(rows)
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(rows.cluster)))
    rng.shuffle(uniq)
    tr = np.isin(rows.cluster, uniq[: len(uniq) // 2])

    Xtr, ytr = [X[tr]], [rows.label[tr]]
    ev, truth, cluster, bucket = [X[~tr]], [rows.label[~tr]], [rows.cluster[~tr]], \
        ["in_catalog"] * int((~tr).sum())
    group = [rows.group[~tr]]
    counts = {"in_catalog": int((~tr).sum()), "near_ood": 0, "distant_ood": 0,
              "train": int(tr.sum())}
    notes = list(rows.notes)

    if background is not None:
        B = _vecs(background)
        cut = rng.permutation(len(B))
        n = int(BG_TRAIN_FRAC * len(B))
        Xtr.append(B[cut[:n]]); ytr.append(np.full(n, OTHER))
        far = cut[n:]
        ev.append(B[far]); truth.append(np.full(len(far), OTHER))
        cluster.append(background.cluster[far])
        group.append(background.group[far])
        bucket += ["distant_ood"] * len(far)
        counts["distant_ood"] = len(far)
        counts["train"] += n
    else:
        notes.append("no background supplied: closed-set only, the model cannot decline")

    counts["notes"] = notes
    counts["has_clusters"] = bool(rows.has_clusters)
    return Dataset(np.vstack(Xtr), np.concatenate(ytr), pd.DataFrame(),
                   np.vstack(ev), np.concatenate(truth), np.array(bucket), counts,
                   np.concatenate(cluster), np.concatenate(group))


def fit_head(ds: Dataset, C: float = 10.0) -> LogisticRegression:
    return LogisticRegression(max_iter=3000, C=C, class_weight="balanced").fit(
        ds.X_train, ds.y_train
    )


def score_frame(clf, ds: Dataset) -> pd.DataFrame:
    """Per-observation cascade inputs and outcomes."""
    classes = np.array(clf.classes_)
    mask = classes != OTHER
    gmap = (dict(zip(ds.truth.tolist(), ds.group.tolist()))
            if ds.group is not None else None)
    gmat, ug = group_matrix(classes, mask, gmap)

    cata = clf.predict_proba(ds.X_eval)[:, mask]
    gscore = cata @ gmat.T
    sp_pred = classes[mask][cata.argmax(1)]
    gp_pred = ug[gscore.argmax(1)]
    # The caller's `group` column wins. Deriving it from the label with the
    # default whitespace rule silently made every label its own group for any
    # domain that does not use Latin binomials -- on 20 Newsgroups it produced
    # group accuracy exactly equal to label accuracy, i.e. a group rank carrying
    # no information, and the cascade correctly refused to ever use it.
    supplied = ds.group if ds.group is not None else None
    true_group = np.array([
        (OTHER if t == OTHER else (supplied[i] if supplied is not None else group_of(t)))
        for i, t in enumerate(ds.truth)])

    return pd.DataFrame({
        "label_conf": cata.max(1),
        "group_conf": gscore.max(1),
        "label_ok": sp_pred == ds.truth,
        "group_ok": gp_pred == true_group,
        "pred_label": sp_pred,
        "pred_group": gp_pred,
        "truth": ds.truth,
        "in_catalog": ds.bucket == "in_catalog",
        "bucket": ds.bucket,
        # Clustering identity, not the label: `make_splits` and the bootstrap
        # both key on these, and both need background rows to carry their real
        # labels rather than collapsing into a single __OTHER__ cluster.
        "label": ds.cluster,
        "group": (ds.group if ds.group is not None
                  else np.array([group_of(c) for c in ds.cluster])),
    })


def _ci(numer, denom, clusters, n=2000, seed=0):
    """Cluster-bootstrapped 95% interval for the ratio sum(numer)/sum(denom).

    A ratio, not a mean, because coverage and precision are *prevalence-weighted*
    across buckets. Bootstrapping the unweighted mean of the same rows estimates
    a different quantity entirely -- it put precision's interval at 22-77% around
    a point estimate of 96%, since unweighted it is dominated by the OOD rows
    that `deployment_weights` deliberately down-weights.
    """
    numer, denom, clusters = (np.asarray(numer, float), np.asarray(denom, float),
                              np.asarray(clusters))
    if len(numer) == 0 or denom.sum() <= 0 or len(set(clusters)) < 2:
        return None
    rng = np.random.RandomState(seed)
    uniq = np.array(sorted(set(clusters)))
    index = {c: np.flatnonzero(clusters == c) for c in uniq}
    out = []
    for _ in range(n):
        idx = np.concatenate([index[c] for c in rng.choice(uniq, len(uniq), replace=True)])
        d = denom[idx].sum()
        if d > 0:
            out.append(numer[idx].sum() / d)
    if not out:
        return None
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def hazard_metrics(te, lv, hazards, seed=0) -> dict:
    """For each consequential label: how often is it given a *non*-consequential name?

    This is the union over every wrong answer, and it is not optional. Measured on
    Oregon's lethal plants (`OREGON_SAFETY_FINDINGS.md`), no single confusion
    exceeded 2.5% while the union reached **6.7%** -- because the errors scatter
    across many different harmless-looking labels. A per-pair report passes a
    model that a union report fails.

    Being named as *another* consequential label is a wrong answer but not a
    dangerous one: the user still does not eat it. Those are counted separately
    rather than folded in.

    **Group answers count.** A coarse answer naming a group that contains no
    consequential label is just as actionable as a wrong label -- "it is a
    Lomatium" for poison hemlock is precisely the error that kills foragers. Only
    declining, or answering with the hazard's own group, is safe.
    """
    if not hazards:
        return {}
    hz = set(hazards)
    hz_groups = {h.split()[0] for h in hz}
    truth = te["truth"].to_numpy()
    pred = te["pred_label"].to_numpy()
    pgen = te["pred_group"].to_numpy()
    named = lv == LABEL
    group_only = (lv != LABEL) & (lv != DECLINE)
    out = {}
    for label in sorted(hz):
        m = truth == label
        if not m.any():
            continue
        # answered as something the user would treat as harmless
        sp_safe = named[m] & (pred[m] != label) & ~np.isin(pred[m], list(hz))
        gn_safe = group_only[m] & ~np.isin(pgen[m], list(hz_groups))
        wrong_safe = sp_safe | gn_safe
        wrong_haz = (named[m] & (pred[m] != label) & np.isin(pred[m], list(hz))) | \
                    (group_only[m] & np.isin(pgen[m], list(hz_groups)))
        ci = _ci(wrong_safe.astype(float), np.ones(m.sum()),
                 te["label"].to_numpy()[m], seed=seed)
        out[label] = {
            "n": int(m.sum()),
            "declined": float((lv[m] == DECLINE).mean()),
            "named_correctly": float((named[m] & (pred[m] == label)).mean()),
            "named_other_hazard": float(wrong_haz.mean()),
            "named_non_hazard": float(wrong_safe.mean()),
            "ci": ci,
            # No interval when the catalogue offers no cluster inside one label:
            # its images are not grouped by individual plant, so a row-level
            # bootstrap would treat several photographs of one plant as
            # independent -- the error CLAUDE.md's first convention exists to
            # prevent. Sources that carry observation ids (iNaturalist) do get one.
            "ci_unavailable_reason": None if ci else "no cluster within a single label",
        }
    return out


def fit_and_measure(df: pd.DataFrame, p_ood: float, seed: int = 0,
                    hazards=None) -> dict:
    """Fit thresholds on a clustered calibration half, report on the other."""
    fold = make_splits(df, seed=seed)
    cal, te = df[fold == "calib"], df[fold == "test"]
    if cal.empty or te.empty:
        raise ValueError("calibration or test split is empty; too few observations")

    w_cal = deployment_weights(cal["bucket"].to_numpy(), p_ood=p_ood, ood_mix=OOD_MIX)
    (tg, ts), _ = fit_thresholds(
        cal["label_conf"].to_numpy(), cal["group_conf"].to_numpy(),
        cal["label_ok"].to_numpy(), cal["group_ok"].to_numpy(),
        cal["in_catalog"].to_numpy(), sample_weight=w_cal,
    )

    lv = decide(te["label_conf"].to_numpy(), te["group_conf"].to_numpy(), tg, ts)
    w = deployment_weights(te["bucket"].to_numpy(), p_ood=p_ood, ood_mix=OOD_MIX)
    answered = lv != DECLINE
    correct = ((lv == LABEL) & te["label_ok"].to_numpy()) | \
              ((lv == GROUP) & te["group_ok"].to_numpy())
    inc = te["in_catalog"].to_numpy()

    per_bucket = {}
    for b in ("in_catalog", "near_ood", "distant_ood"):
        bm = te["bucket"].to_numpy() == b
        if bm.any():
            per_bucket[b] = {
                "n": int(bm.sum()),
                "answered": float((lv[bm] != DECLINE).mean()),
                "correct_when_answered": float(
                    correct[bm & answered].mean()) if (bm & answered).any() else None,
            }

    # Cluster bootstrap over labels, never over rows -- CLAUDE.md's first
    # convention, and it exists because row-level intervals have twice produced
    # effects here that failed to replicate. A card at 14 labels rests on very
    # few clusters, so a wide interval is itself the finding the user needs.
    clusters = te["label"].to_numpy()
    ones = np.ones(len(te))
    ci = {
        "coverage": _ci(w * answered, w, clusters),
        "precision": _ci(w * answered * correct, w * answered, clusters),
        "label_share": _ci((lv == LABEL) & inc, inc * ones, clusters),
        "closed_set_top1": _ci(te["label_ok"].to_numpy() & inc, inc * ones, clusters),
    }

    return {
        "t_group": float(tg), "t_label": float(ts), "p_ood": p_ood,
        "coverage": float(w[answered].sum() / w.sum()),
        "precision": float(w[answered & correct].sum() / w[answered].sum())
        if answered.any() else None,
        "label_share": float((lv[inc] == LABEL).mean()) if inc.any() else None,
        "closed_set_top1": float(te["label_ok"].to_numpy()[inc].mean()) if inc.any() else None,
        "ci": ci,
        "n_label_clusters": int(len(set(clusters[inc]))),
        "per_bucket": per_bucket,
        "hazard": hazard_metrics(te, lv, hazards, seed=seed),
        "n_calib": int(len(cal)), "n_test": int(len(te)),
    }


def save_bundle(out: Path, clf, chosen, encoder, metrics, composition, counts,
                source: str, hazards=None) -> Path:
    """Head weights, thresholds, and everything needed to reproduce the claim."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "head.npz", coef=clf.coef_, intercept=clf.intercept_,
                        classes=np.asarray(clf.classes_, dtype=str))
    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "encoder": encoder,
        "labels": chosen,
        "source": source,
        "hazards": sorted(hazards or []),
        "counts": counts,
        "composition": {k: v for k, v in composition.items()
                        if k != "outside_siblings" and not k.startswith("_")},
        "outside_siblings": composition.get("outside_siblings", {}),
        "metrics": metrics,
        "utility": UTILITY,
        "ood_mix": OOD_MIX,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out


def load_bundle(path: Path) -> dict:
    return json.loads((Path(path) / "manifest.json").read_text())
