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
    fit_novelty_threshold,
    fit_thresholds,
    group_matrix,
    hazard_rows,
    make_splits,
    suppress,
)
from narrowcast.labels import group_of

OTHER = "__OTHER__"
BG_TRAIN_FRAC = 0.6
BG_ORIGIN = "__BACKGROUND__"
OOD_MIX = {"near_ood": 0.32, "distant_ood": 0.68}
# The deployment-realistic mix, used when the caller flags which of their
# out-of-list rows a user could plausibly supply. `distant_ood` drawn at random
# is dominated by inputs the deployment never sees and makes rejection look
# easier than it is; `regional_ood` is the same rule restricted to what actually
# turns up. Rows left in `distant_ood` then carry weight zero -- they are
# reported, but they are not evidence about the operating point.
OOD_MIX_REGIONAL = {"near_ood": 0.32, "regional_ood": 0.68}
BUNDLE_VERSION = 2


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
    origin_train: np.ndarray | None = None   # acquisition source per training row
    origin_eval: np.ndarray | None = None    # ... and per eval row
    # The real species of every eval row, in-list or not. `truth` collapses every
    # out-of-list row to OTHER and `cluster` may be an arbitrary id, so without
    # this the identity of an out-of-list row is not recoverable from the frame --
    # which makes "how often is *this particular* unlisted plant misnamed"
    # unanswerable. Needed by `outside_hazard_metrics`.
    species: np.ndarray | None = None


def _l2(X):
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


# Two embedding spaces from different encoders are, to a very good approximation,
# independent random rotations of each other: a cosine between vectors drawn from
# them concentrates at 0 with spread ~1/sqrt(D). Vectors from the *same* encoder
# share a common cone and sit well above that, whatever their subject matter. So
# `4 / sqrt(D)` separates "same space" from "unrelated space" at p < 1e-4, and it
# is a statistical bound rather than a tuned constant.
ORTHOGONAL_Z = 4.0
# Below this the bound is too wide to separate anything -- at D = 16 it is already
# 1.0, and a cosine cannot exceed that -- so the test has no power and is skipped
# rather than being applied where it cannot discriminate. Real encoders are
# 256-1024 dimensional; this only excludes toy vectors.
MIN_SPACE_CHECK_DIM = 64


def space_coherence(X, centroid) -> float:
    """Mean cosine of L2-normalised rows to a reference direction."""
    c = np.asarray(centroid, dtype="float64").reshape(1, -1)
    c = c / np.clip(np.linalg.norm(c), 1e-12, None)
    return float((np.asarray(X, dtype="float64") @ c.T).mean())


def check_same_space(fg, bg, what="the background pool"):
    """Two vector pools that cannot have come from one encoder.

    The failure this exists for was silent and cost three points: a bundle
    embedded with a Core ML export was measured against a background pool embedded
    with the torch original, and the negatives -- living in an unrelated space --
    were trivially rejected, so label share came out flattered. Nothing errored,
    and the number went into a table. `manifest["encoder"]` cannot catch it: it is
    a string the caller declares, and both pools carry the same model name.

    **A dimension mismatch is refused. Orthogonal geometry is only warned about**,
    and the asymmetry is deliberate. Different widths are proof. The geometry test
    rests on a premise -- that embeddings from one encoder share a common cone, so
    a cross-pool cosine near zero means two encoders -- which holds for every real
    contrastive encoder the authors have seen and is **not validated here**, since
    nothing in this package can load one. On synthetic vectors with independently
    drawn class centroids it false-positives, because such vectors share no cone
    to begin with. Refusing on an untested premise would be the mistake this
    project has already recorded twice: a partial measurement read as a verdict.

    So: returns a list of notes, and raises only on the certain case. A domain
    repo that *does* have encoders should measure the false-positive rate on real
    pairs; until then this stays a warning.
    """
    if fg is None or bg is None or len(fg) == 0 or len(bg) == 0:
        return []
    if fg.shape[1] != bg.shape[1]:
        raise SystemExit(
            f"{what} has {bg.shape[1]}-dimensional vectors and your rows have "
            f"{fg.shape[1]}. These cannot be from one encoder.")
    d = fg.shape[1]
    if d < MIN_SPACE_CHECK_DIM:
        return []
    bound = ORTHOGONAL_Z / np.sqrt(d)
    within = min(space_coherence(fg, fg.mean(0)), space_coherence(bg, bg.mean(0)))
    cross = space_coherence(bg, fg.mean(0))
    if abs(cross) < bound <= within:
        return [f"{what} may not live in the same embedding space as your rows: "
                f"mean cosine across the two pools is {cross:+.4f}, "
                f"indistinguishable from unrelated (|cos| < {bound:.4f} at "
                f"D={d}), while within each pool it is {within:.4f}. Two "
                f"encoders, or one encoder and an export of it, would look like "
                f"this -- and so would genuinely unrelated subject matter. Worth "
                f"checking: negatives in an unrelated space are trivially "
                f"rejected and label share comes out flattered, which is silent."]
    return []


def load_rows(rows, encoder_variant: str, background=None, seed: int = 0) -> Dataset:
    """Assemble a Dataset from a caller-supplied source (`narrowcast/sources.py`).

    The tool does not fetch, and no longer encodes. It is handed vectors that
    some encoder already produced, and the domain that produced them keeps
    ownership of which encoder and how. `encoder_variant` is therefore a label
    the caller declares for the record, never something resolved against a
    registry -- the card prints it back and states no size for it.

    `background` is an optional second source of negatives. Without it there is
    no reject class: the model is closed-set, cannot decline, and the card says
    so rather than implying a rejection capability that was never fitted.
    """
    def _vecs(r):
        if r.descriptor is None:
            raise ValueError("no 'descriptor' array: this path needs vectors. "
                             "Supply --embeddings, or --scores if what you have "
                             "is posteriors from a model you already fitted.")
        return _l2(np.asarray(r.descriptor, dtype="float32"))

    X = _vecs(rows)
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(rows.cluster)))
    rng.shuffle(uniq)
    tr = np.isin(rows.cluster, uniq[: len(uniq) // 2])

    Xtr, ytr = [X[tr]], [rows.label[tr]]
    ev, truth, cluster, bucket = [X[~tr]], [rows.label[~tr]], [rows.cluster[~tr]], \
        ["in_catalog"] * int((~tr).sum())
    group = [rows.group[~tr]]
    has_origin = rows.origin is not None
    o_tr = [rows.origin[tr]] if has_origin else None
    o_ev = [rows.origin[~tr]] if has_origin else None
    counts = {"in_catalog": int((~tr).sum()), "near_ood": 0, "distant_ood": 0,
              "train": int(tr.sum())}
    notes = list(rows.notes)

    if background is not None:
        B = _vecs(background)
        notes += check_same_space(X, B)
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
        if has_origin:
            # Background rows have no origin in the caller's sense; they are
            # negatives, not members of either population, and labelling them
            # with one would make them look like evidence about it.
            o_tr.append(np.full(n, BG_ORIGIN))
            o_ev.append(np.full(len(far), BG_ORIGIN))
    else:
        notes.append("no background supplied: closed-set only, the model cannot decline")

    # How many training rows each label actually got. The card cannot otherwise
    # tell a head fitted on eight photographs per label from one fitted on eight
    # hundred, and they are very different products: on a group-crowded list the
    # label-level share is still climbing steeply at 64 rows per label, while on a
    # separated list with a strong encoder it is saturated by 8. Reported so a low
    # share can be read as "needs more data" rather than "needs a different list".
    # `rows.label[tr]`, not anything derived from `ytr`: by this point `ytr` may
    # carry an appended block of OTHER for the background negatives, which are not
    # a label the user chose and would distort both the median and the minimum.
    per_label = pd.Series(rows.label[tr]).value_counts()
    counts["rows_per_label"] = {
        "median": int(per_label.median()) if len(per_label) else 0,
        "min": int(per_label.min()) if len(per_label) else 0,
        "n_labels": int(len(per_label)),
        "n_below_32": int((per_label < 32).sum()),
    }
    counts["notes"] = notes
    counts["has_clusters"] = bool(rows.has_clusters)
    return Dataset(np.vstack(Xtr), np.concatenate(ytr), pd.DataFrame(),
                   np.vstack(ev), np.concatenate(truth), np.array(bucket), counts,
                   np.concatenate(cluster), np.concatenate(group),
                   np.concatenate(o_tr) if has_origin else None,
                   np.concatenate(o_ev) if has_origin else None,
                   species=np.concatenate(cluster))


def load_scored(rows, seed: int = 0) -> Dataset:
    """Assemble a Dataset from posteriors a caller already has (`--scores`).

    No head is fitted and no vectors exist, so every row is an evaluation row and
    the train side is empty. What this path *can* do that `--embeddings` cannot is
    bucket the out-of-list rows properly: a row whose truth is not among `classes`
    is out-of-list by construction, and whether it is `near_ood` or `distant_ood`
    follows from whether its group is one the label set already contains. That
    distinction matters -- `deployment_weights` mixes the two at declared shares,
    and near-OOD is reliably the weakest bucket because a relative of a listed
    label is exactly what a closed-set score cannot say "none of these" about.

    Out-of-list rows carry `OTHER` as truth, so they can never score a correct
    label or group, and keep their *real* label as the clustering identity. That
    second half is not cosmetic: giving them `OTHER` as a cluster leaves
    `make_splits` one cluster for the whole bucket, puts every negative on one
    side of the split, and fits thresholds on a calibration set with no negatives
    in it. That mistake silently broke a published table once already.
    """
    label = np.asarray(rows.label, dtype=str)
    classes = np.asarray(rows.classes, dtype=str)
    in_list = np.isin(label, classes)

    group = np.asarray(rows.group, dtype=str)
    listed_groups = set(group[in_list].tolist())
    bucket = np.where(
        in_list, "in_catalog",
        np.where(np.isin(group, sorted(listed_groups)), "near_ood", "distant_ood"))
    if rows.regional is not None:
        # Only an out-of-list row can be regional: an in-list row is not a
        # rejection case at all, and promoting one would put it in a bucket where
        # no correct answer exists.
        bucket = np.where(np.asarray(rows.regional, bool) & (bucket == "distant_ood"),
                          "regional_ood", bucket)

    truth = np.where(in_list, label, OTHER)
    counts = {b: int((bucket == b).sum())
              for b in ("in_catalog", "near_ood", "distant_ood", "regional_ood")}
    counts["train"] = 0
    # The caller's model was trained on something we were not shown, so the
    # rows-per-label warning has no denominator here. Saying nothing is correct;
    # inventing one from the evaluation rows would describe the wrong set.
    counts["rows_per_label"] = None

    notes = list(rows.notes)
    if counts["near_ood"] + counts["distant_ood"] == 0:
        notes.append("every row's label is among `classes`: no out-of-list rows, "
                     "so the decline threshold has nothing to reject and coverage "
                     "is not a measurement of rejection")
    else:
        notes.append(f"out-of-list rows bucketed by group: {counts['near_ood']} "
                     f"near_ood, {counts['distant_ood']} distant_ood")
    counts["notes"] = notes
    counts["has_clusters"] = bool(rows.has_clusters)

    empty = np.zeros((0, 1), dtype="float32")
    return Dataset(empty, np.asarray([], dtype=str), pd.DataFrame(),
                   empty, truth, bucket, counts,
                   np.asarray(rows.cluster, dtype=str), group,
                   None, None if rows.origin is None else np.asarray(rows.origin, dtype=str),
                   species=label)


def fit_head(ds: Dataset, C: float = 10.0) -> LogisticRegression:
    return LogisticRegression(max_iter=3000, C=C, class_weight="balanced").fit(
        ds.X_train, ds.y_train
    )



def origin_cost(ds: Dataset, deployment: str, C: float = 10.0) -> dict | None:
    """What it costs a label to have no training rows from the deployment origin.

    When rows carry an `origin` -- two corpora, two devices, two populations --
    the labels that have training rows from the origin you will actually deploy
    against and the labels that do not are not comparable. The ones that do not
    are *worse off than if no label had them*: the head is one multinomial and
    one argmax, so rows from the deployment origin move the boundaries of the
    labels that got them, and a label still represented only by the other origin
    loses ties it used to win.

    This is measured, not predicted, because its size is domain-dependent and
    nothing about the label set tells you what it will be. Two heads are fitted
    on the same label set -- one on everything, one with every deployment-origin
    training row removed -- and both are scored on the same held-out
    deployment-origin rows. The difference is what that data did, to the labels
    that got it and to the labels that did not.

    Measured across three domains in the research repos: on plants the cost is
    ~0 once the label set is small, and on dermatology and keyword spotting it is
    10-20 points and does not shrink with label count. It tracks the accuracy of
    the build rather than the number of labels, so **do not infer it from K**.

    **A label with no deployment-origin rows at all cannot be scored here** --
    there is nothing to score it on -- and that is the common case in practice
    rather than an edge case. Those labels are counted and named as `unmeasured`
    rather than quietly folded into an average that would then understate the
    problem. What is measured is the labels that have held-out deployment-origin
    rows but no deployment-origin training rows.

    Returns None when there is no origin column at all.
    """
    if ds.origin_train is None or ds.origin_eval is None:
        return None

    out = {"deployment_origin": deployment, "measurable": False}
    dep_tr = ds.origin_train == deployment
    if not dep_tr.any():
        return {**out, "why": f"no training rows carry origin {deployment!r}"}

    labels = sorted(set(ds.y_train.tolist()) - {OTHER})
    trained_on = set(ds.y_train[dep_tr].tolist())
    te_dep = (ds.origin_eval == deployment) & (ds.bucket == "in_catalog")
    have_eval = set(ds.truth[te_dep].tolist())

    lacking = [l for l in labels if l not in trained_on and l in have_eval]
    unmeasured = [l for l in labels if l not in trained_on and l not in have_eval]
    covered = [l for l in labels if l in trained_on]
    out.update(n_lacking=len(lacking), n_covered=len(covered),
               n_unmeasured=len(unmeasured), labels_unmeasured=unmeasured[:12])

    if not lacking:
        why = ("every label has training rows from the deployment origin"
               if not unmeasured else
               "no label has both held-out rows from the deployment origin and no "
               "training rows from it, so there is no comparison to draw")
        return {**out, "why": why}
    if not covered:
        return {**out, "why": "no label has training rows from the deployment origin"}
    if len(set(ds.y_train[~dep_tr].tolist())) < 2:
        return {**out, "why": "removing the deployment-origin rows leaves fewer than "
                              "two labels to fit a head on"}

    without = LogisticRegression(max_iter=3000, C=C, class_weight="balanced").fit(
        ds.X_train[~dep_tr], ds.y_train[~dep_tr])
    with_ = fit_head(ds, C=C)
    truth, X = ds.truth[te_dep], ds.X_eval[te_dep]

    def macro(clf, group):
        keep = np.isin(truth, list(group))
        if not keep.any():
            return None
        hit = pd.Series((clf.predict(X[keep]) == truth[keep]).astype(float))
        return float(hit.groupby(pd.Series(truth[keep])).mean().mean())

    out.update(measurable=True, labels_lacking=lacking[:12],
               n_eval_rows=int(te_dep.sum()))
    for name, group in (("lacking", lacking), ("covered", covered)):
        a, b = macro(without, group), macro(with_, group)
        out[f"{name}_without"], out[f"{name}_with"] = a, b
        out[f"{name}_delta"] = None if (a is None or b is None) else round(b - a, 4)
    return out


def score_frame(clf, ds: Dataset) -> pd.DataFrame:
    """Per-observation cascade inputs and outcomes, from a head we fitted."""
    return frame_from_posteriors(clf.predict_proba(ds.X_eval),
                                 np.array(clf.classes_), ds)


def frame_from_posteriors(proba, classes, ds: Dataset) -> pd.DataFrame:
    """Per-observation cascade inputs and outcomes, from posteriors of any origin.

    Split out of `score_frame` so that a model we did not fit is measured by
    exactly the same code as one we did. Everything downstream -- threshold
    fitting, the three-way split, headroom, the intervals -- reads this frame and
    nothing else, so the audit path cannot drift from the build path by accident.

    `proba` is rows x len(classes) and need not sum to 1; `decide` compares the
    largest label mass and the largest group mass against two thresholds, and
    both are order-preserving under any positive rescale of a row.
    """
    proba, classes = np.asarray(proba, float), np.asarray(classes)
    mask = classes != OTHER
    gmap = (dict(zip(ds.truth.tolist(), ds.group.tolist()))
            if ds.group is not None else None)
    gmat, ug = group_matrix(classes, mask, gmap)

    cata = proba[:, mask]
    # The near-OOD gate's input: the share of mass that stayed inside the label
    # set, i.e. 1 - P(__OTHER__). Computed here and nowhere else, because this is
    # the single seam an audited model and a fitted one share -- recomputing it in
    # `predict` or in the fit would be exactly the fork CLAUDE.md pins with a test.
    # A ratio rather than a subtraction because `proba` need not sum to 1. With no
    # reject class the share is 1 for every row and the gate is inert, which is
    # the right degradation: `predict.Bundle.notes` already says there is nothing
    # to reject with.
    total = np.clip(proba.sum(1), 1e-12, None)
    novelty = cata.sum(1) / total
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
        "novelty": novelty,
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
        # The real species, in-list or not; see `Dataset.species`.
        "species": ds.species if ds.species is not None else ds.cluster,
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


def hazard_metrics(te, lv, hazards, seed=0, groups=None) -> dict:
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
    # The caller's group map, not the first whitespace token. That default is a
    # Latin-binomial convention and this is the fourth place it has been wrong --
    # `labels.py:126` calls an earlier one "the third place this default has
    # broken a non-binomial domain". It matters most here: a foraging list groups
    # by *family*, so poison hemlock's group is `Apiaceae`, and deriving "Conium"
    # from the name means the hazard's own group is never recognised. Every
    # coarse answer then counts as dangerous, including the one that is actually
    # a warning.
    hz_groups = ({groups[h] for h in hz if h in groups} if groups
                 else {h.split()[0] for h in hz})
    pred = te["pred_label"].to_numpy()
    pgen = te["pred_group"].to_numpy()
    named = lv == LABEL
    group_only = (lv != LABEL) & (lv != DECLINE)
    out = {}
    for label in sorted(hz):
        m = hazard_rows(te, label)
        if not m.any():
            # A declared hazard with no test rows used to `continue`, so it
            # vanished from the gate while the card counted the survivors and
            # said "all N consequential labels are under the bar". With clustered
            # splits over a large catalogue a rare hazard can disappear this way.
            # Record it as unmeasured instead: absence of evidence is reported,
            # not converted into a pass.
            out[label] = {
                "n": 0, "declined": None, "named_correctly": None,
                "named_other_hazard": None, "named_non_hazard": None,
                "named_as": {}, "ci": None, "unmeasured": True,
                "ci_unavailable_reason": "no test rows for this label",
            }
            continue
        # answered as something the user would treat as harmless
        sp_safe = named[m] & (pred[m] != label) & ~np.isin(pred[m], list(hz))
        gn_safe = group_only[m] & ~np.isin(pgen[m], list(hz_groups))
        wrong_safe = sp_safe | gn_safe
        wrong_haz = (named[m] & (pred[m] != label) & np.isin(pred[m], list(hz))) | \
                    (group_only[m] & np.isin(pgen[m], list(hz_groups)))
        ci = _ci(wrong_safe.astype(float), np.ones(m.sum()),
                 te["label"].to_numpy()[m], seed=seed)
        # Which harmless labels it was actually given. The absent path has always
        # recorded this; without it here the card can tell an in-list caller that
        # the gate failed but not which label to pass to `--never-answer`, which
        # makes the remedy unusable on the path that has shipped longest.
        got = pd.Series(pred[m][sp_safe]).value_counts().head(3).to_dict()
        out[label] = {
            "n": int(m.sum()),
            "declined": float((lv[m] == DECLINE).mean()),
            "named_correctly": float((named[m] & (pred[m] == label)).mean()),
            "named_other_hazard": float(wrong_haz.mean()),
            "named_non_hazard": float(wrong_safe.mean()),
            "named_as": {str(k): int(v) for k, v in got.items()},
            "ci": ci,
            "unmeasured": False,
            # No interval when the catalogue offers no cluster inside one label:
            # its images are not grouped by individual plant, so a row-level
            # bootstrap would treat several photographs of one plant as
            # independent -- the error CLAUDE.md's first convention exists to
            # prevent. Sources that carry observation ids (iNaturalist) do get one.
            "ci_unavailable_reason": None if ci else "no cluster within a single label",
        }
    return out


def outside_hazard_metrics(te, lv, hazards, groups=None, seed=0) -> dict:
    """For a dangerous species the user deliberately did **not** list: how often
    does it receive the name of something they did?

    `hazard_metrics` answers the opposite question -- "my list contains something
    dangerous, how often is it given a harmless name" -- and requires the hazard to
    be a label. That is the right model for a catalogue that includes hazards on
    purpose. It cannot express a forager's, where the dangerous plant is absent by
    design: nobody lists poison hemlock among things they intend to eat, so
    `--hazard "Conium maculatum"` is correctly refused, and the risk goes
    unmeasured.

    Here every in-list label is by construction something the user believes they
    can use, so **any label answer is dangerous**. The asymmetry with the in-list
    case is deliberate: there, being named as *another* hazard is wrong but not
    dangerous; here there is no such escape, because everything nameable is
    something the user wants.

    A group answer is safe only when the named group contains a declared hazard --
    "it is an umbellifer" genuinely warns the person holding the root, where "it is
    a *Lomatium*" is a species-level claim wearing the clothes of caution. That
    distinction only works if the bundle groups by something coarse enough to
    contain the hazard, which is why `groups` matters here as much as it does in
    `hazard_metrics`.

    Rows are found by `cascade.hazard_rows`, shared with `hazard_metrics` and with
    the split itself. Here it is the `species` column that matches, holding an
    out-of-list row's real name -- `truth` is `OTHER` for all of them by
    construction, so the predicate this function needs is not the one the in-list
    case needs, and `make_splits` has to stratify on both.
    """
    if not hazards:
        return {}
    hz = set(hazards)
    hz_groups = ({groups[h] for h in hz if h in groups} if groups
                 else {h.split()[0] for h in hz})
    pgen = te["pred_group"].to_numpy()
    pred = te["pred_label"].to_numpy()
    named = lv == LABEL
    group_only = (lv != LABEL) & (lv != DECLINE)

    out = {}
    for label in sorted(hz):
        m = hazard_rows(te, label)
        if not m.any():
            out[label] = {"n": 0, "declined": None, "named_in_list": None,
                          "warned_at_group": None, "dangerous": None,
                          "named_as": {}, "ci": None, "unmeasured": True,
                          "ci_unavailable_reason": "no out-of-list rows for this label"}
            continue
        # every in-list name is something the user means to use
        danger_sp = named[m]
        danger_gn = group_only[m] & ~np.isin(pgen[m], list(hz_groups))
        dangerous = danger_sp | danger_gn
        ci = _ci(dangerous.astype(float), np.ones(int(m.sum())),
                 te["label"].to_numpy()[m], seed=seed)
        top = pd.Series(pred[m & (lv == LABEL)]).value_counts().head(3).to_dict()
        out[label] = {
            "n": int(m.sum()),
            "declined": float((lv[m] == DECLINE).mean()),
            "named_in_list": float(danger_sp.mean()),
            "warned_at_group": float((group_only[m] &
                                      np.isin(pgen[m], list(hz_groups))).mean()),
            "dangerous": float(dangerous.mean()),
            "named_as": {str(k): int(v) for k, v in top.items()},
            "ci": ci,
            "unmeasured": False,
            "ci_unavailable_reason": None if ci else "no cluster within a single label",
        }
    return out


def _group_members(labels, groups) -> dict | None:
    """group -> the labels in it, over the label set the model can emit.

    `cascade.suppress` needs this to tell "it is an umbellifer" -- a warning worth
    keeping -- from "it is a *Daucus*" where *Daucus carota* is the only listed
    *Daucus*, which is the suppressed claim under another name.
    """
    if not labels:
        return None
    out = {}
    for lab in labels:
        g = (groups or {}).get(lab) or group_of(lab)
        out.setdefault(g, set()).add(lab)
    return out


def fit_and_measure(df: pd.DataFrame, p_ood: float, seed: int = 0,
                    hazards=None, groups=None, utility=None,
                    hazards_absent=None, never_answer=None, labels=None,
                    gate=False) -> dict:
    """Fit thresholds on a clustered calibration half, report on the other.

    `groups` is the caller's label -> group map, needed by `hazard_metrics` so a
    hazard's own group is recognised under a non-genus grouping.

    `utility` overrides `cascade.UTILITY`. The override has always existed --
    `utility()` merges it and `fit_thresholds()` threads it -- and nothing in the
    package passed it, so the declared payoffs were effectively hard-coded. They
    are still *declared*: a caller picks a profile written down in advance, which
    is the discipline `CLAUDE.md` asks for. What it must never be is fitted to
    the outcome.
    """
    # Both threat models' hazards, because `make_splits` stratifies on the union
    # and measuring a hazard the split never routed into the test half reports it
    # as "unmeasured" -- which is the honest answer to the wrong question.
    declared = set(hazards or []) | set(hazards_absent or [])
    fold = make_splits(df, seed=seed, hazards=declared)
    cal, te = df[fold == "calib"], df[fold == "test"]
    if cal.empty or te.empty:
        raise ValueError("calibration or test split is empty; too few observations")

    # `deployment_weights` divides each bucket's share by the sum over the *whole*
    # mix, so a bucket that is absent leaves its share unclaimed and the weighting
    # renormalises to a lower effective prevalence than the caller asked for. A
    # source with no in-pool relatives has no near_ood bucket, and `--ood-rate 0.2`
    # then silently scored at 0.145 -- while the card printed "an assumed 20.0%
    # out-of-list rate". Restricting the mix to the buckets actually present, per
    # side, makes the stated prevalence the real one.
    def _mix(sub):
        # The regional mix wherever the caller flagged regional rows, because it
        # is the deployment-realistic one and the whole point of the flag is that
        # `distant_ood` overstates how easy rejection is.
        base = (OOD_MIX_REGIONAL if (sub["bucket"] == "regional_ood").any()
                else OOD_MIX)
        present = {b: s for b, s in base.items() if (sub["bucket"] == b).any()}
        return present or base

    w_cal = deployment_weights(cal["bucket"].to_numpy(), p_ood=p_ood, ood_mix=_mix(cal))
    (tg, ts), _ = fit_thresholds(
        cal["label_conf"].to_numpy(), cal["group_conf"].to_numpy(),
        cal["label_ok"].to_numpy(), cal["group_ok"].to_numpy(),
        cal["in_catalog"].to_numpy(), sample_weight=w_cal, weights=utility,
    )

    # Stage two: the near-OOD gate, fitted alone with the first two fixed. See
    # `cascade.fit_novelty_threshold` for why it is staged rather than joint, and
    # for the guarantee that makes staging safe -- the sweep can always choose the
    # ungated baseline, so this cannot score below the fit above.
    # Resolved here, before any decision is taken, so that `tn` is None from the
    # start when the fit declines the gate. Deciding it further down -- after
    # `lv_open` had already been computed with a threshold -- left every headline
    # number reflecting a decline rule the manifest then said did not exist: a
    # `t_novel` at the bottom of the calibration grid gates nothing on calib and
    # can still catch a test row below that minimum.
    tn, gate_gain, gate_unreadable = None, None, None
    if gate:
        if "novelty" not in cal:
            gate_unreadable = "no novelty column in the frame"
        elif float(np.ptp(te["novelty"].to_numpy())) < 1e-9:
            gate_unreadable = ("the posteriors carry no out-of-list class, so "
                               "1 - P(__OTHER__) is 1 on every row and there is "
                               "nothing for the gate to read")
        else:
            fitted_tn, gate_gain = fit_novelty_threshold(
                cal["label_conf"].to_numpy(), cal["group_conf"].to_numpy(),
                cal["novelty"].to_numpy(), cal["label_ok"].to_numpy(),
                cal["group_ok"].to_numpy(), cal["in_catalog"].to_numpy(),
                tg, ts, weights=utility, sample_weight=w_cal)
            # Adopted only if it earned something. A gate the fit turned off is
            # not applied, not measured against, and not written to the bundle.
            tn = float(fitted_tn) if gate_gain > 0 else None

    # Headroom -- coarse-rank accuracy minus fine-rank accuracy -- governs whether
    # the cascade retreats to the group rank at all (plantid's
    # HEADROOM_FINDINGS.md: CV R^2 0.883 over 1,409 arms, against 0.362 for fine
    # accuracy alone; group_share ~= 1.8 x headroom). These are the same two
    # arrays `fit_thresholds` just consumed, and they were discarded until now.
    #
    # Measured on the CALIBRATION half, so it is a property of the label set and
    # encoder rather than of the thresholds those metrics are scored under, and
    # restricted to in-catalogue rows because a group answer for a background row
    # is a rejection outcome, not a rank-retreat one.
    #
    # In-catalogue rows all carry the identical deployment weight
    # ((1-p_ood)/n_in, see cascade.deployment_weights), so weighting this mean
    # would change nothing. It is left unweighted deliberately.
    cal_inc = cal["in_catalog"].to_numpy()
    if cal_inc.any():
        calib_fine = float(cal["label_ok"].to_numpy()[cal_inc].mean())
        calib_coarse = float(cal["group_ok"].to_numpy()[cal_inc].mean())
        headroom = calib_coarse - calib_fine
    else:
        calib_fine = calib_coarse = headroom = None

    # Fitted first, suppressed after. `cascade.suppress` explains why the override
    # is kept out of `fit_thresholds`: leaving it out is what makes its cost a
    # measurable delta rather than something the operating point absorbs.
    lv_open = decide(te["label_conf"].to_numpy(), te["group_conf"].to_numpy(), tg, ts,
                     te["novelty"].to_numpy() if tn is not None else None, tn)
    if never_answer and not labels:
        # Without the label set `_group_members` is None, so the measurement would
        # suppress label answers only while `Bundle.predict` — which always builds
        # the members from its own classes — also kills hollow-group answers. Two
        # different models from one bundle, which is the single invariant this
        # tool will not break. Refuse rather than degrade.
        raise ValueError(
            "never_answer needs `labels` (the label set the model can emit). "
            "Without it a group answer whose group holds nothing but suppressed "
            "labels cannot be recognised, and the measurement would describe a "
            "model that `predict` does not run.")
    lv = lv_open if not never_answer else suppress(
        lv_open, te["pred_label"].to_numpy(), never_answer,
        te["pred_group"].to_numpy(), _group_members(labels, groups))
    w = deployment_weights(te["bucket"].to_numpy(), p_ood=p_ood, ood_mix=_mix(te))
    answered = lv != DECLINE
    correct = ((lv == LABEL) & te["label_ok"].to_numpy()) | \
              ((lv == GROUP) & te["group_ok"].to_numpy())
    inc = te["in_catalog"].to_numpy()

    per_bucket = {}
    for b in ("in_catalog", "near_ood", "distant_ood", "regional_ood"):
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
        "group_share": _ci((lv == GROUP) & inc, inc * ones, clusters),
        "closed_set_top1": _ci(te["label_ok"].to_numpy() & inc, inc * ones, clusters),
    }

    # What the override costs, in the currency the card already leads with. A
    # suppression whose price is not printed is exactly the kind of number this
    # tool refuses to report: a caller suppressing `Daucus carota` gives up every
    # correct wild-carrot answer, and that trade is theirs to make with the figure
    # in front of them.
    # What the gate did, measured rather than asserted. plantid fitted this and
    # found a utility null at its own payoffs, so a card that implied the gate was
    # an established win would be overstating a result its own source doc
    # declines to make. If the fit turned it off, that is what gets printed.
    gate_report = None
    if gate:
        if gate_unreadable:
            gate_report = {"fitted": False, "reason": gate_unreadable}
        elif tn is None:
            gate_report = {"fitted": True, "t_novel": None,
                           "calib_utility_gained": float(gate_gain),
                           "fit_turned_it_off": True, "rows_declined": 0,
                           "near_ood_wrong_ungated": None,
                           "near_ood_wrong_gated": None,
                           "label_share_ungated": None}
        else:
            # `lv_open` is the gated decision; `ungated` is what it would have been
            # without. Both are needed and neither may be derived from the other.
            ungated = decide(te["label_conf"].to_numpy(),
                             te["group_conf"].to_numpy(), tg, ts)
            nm = te["bucket"].to_numpy() == "near_ood"

            def _wrong(levels, m):
                if not m.any():
                    return None
                ans = levels[m] != DECLINE
                ok = ((levels[m] == LABEL) & te["label_ok"].to_numpy()[m]) | \
                     ((levels[m] == GROUP) & te["group_ok"].to_numpy()[m])
                return float((ans & ~ok).mean())

            gate_report = {
                "fitted": True,
                "t_novel": float(tn),
                "calib_utility_gained": float(gate_gain),
                "fit_turned_it_off": False,
                "rows_declined": int(((ungated != DECLINE) & (lv_open == DECLINE)).sum()),
                "near_ood_wrong_ungated": _wrong(ungated, nm),
                "near_ood_wrong_gated": _wrong(lv_open, nm),
                "label_share_ungated": float((ungated[inc] == LABEL).mean())
                if inc.any() else None,
                # The gate's own effect on label share, not the headline: with
                # `--never-answer` also in play the headline carries both costs
                # and attributing all of it here would overstate the gate's price.
                "label_share_gated": float((lv_open[inc] == LABEL).mean())
                if inc.any() else None,
            }

    suppression = None
    if never_answer:
        lost = (lv_open != DECLINE) & (lv == DECLINE)
        was_right = ((lv_open == LABEL) & te["label_ok"].to_numpy()) | \
                    ((lv_open == GROUP) & te["group_ok"].to_numpy())
        suppression = {
            "labels": sorted(never_answer),
            "label_share_without": float((lv_open[inc] == LABEL).mean())
            if inc.any() else None,
            "label_share_with": float((lv[inc] == LABEL).mean()) if inc.any() else None,
            "answers_removed": int(lost.sum()),
            "correct_answers_removed": int((lost & was_right).sum()),
            "n_test": int(len(te)),
        }

    return {
        "t_group": float(tg), "t_label": float(ts), "p_ood": p_ood,
        "ood_mix": _mix(te),
        "t_novel": None if tn is None else float(tn),
        "novelty_gate": gate_report,
        "suppression": suppression,
        "coverage": float(w[answered].sum() / w.sum()),
        "precision": float(w[answered & correct].sum() / w[answered].sum())
        if answered.any() else None,
        "label_share": float((lv[inc] == LABEL).mean()) if inc.any() else None,
        # The other two thirds of the same three-way split. Without them a report
        # cannot tell retreat to the group from declining, and the card asserted
        # the former while measuring neither.
        "group_share": float((lv[inc] == GROUP).mean()) if inc.any() else None,
        "decline_share": float((lv[inc] == DECLINE).mean()) if inc.any() else None,
        "calib_fine": calib_fine, "calib_coarse": calib_coarse,
        "headroom": headroom,
        "closed_set_top1": float(te["label_ok"].to_numpy()[inc].mean()) if inc.any() else None,
        "ci": ci,
        "n_label_clusters": int(len(set(clusters[inc]))),
        "per_bucket": per_bucket,
        "hazard": hazard_metrics(te, lv, hazards, seed=seed, groups=groups),
        "hazard_absent": outside_hazard_metrics(te, lv, hazards_absent,
                                                groups=groups, seed=seed),
        "n_calib": int(len(cal)), "n_test": int(len(te)),
    }


def save_bundle(out: Path, clf, chosen, encoder, metrics, composition, counts,
                source: str, hazards=None, groups=None, utility=None,
                never_answer=None, space=None) -> Path:
    """Head weights, thresholds, and everything needed to reproduce the claim.

    `groups` is the label -> group map, and storing it is what makes `predict`
    agree with the card. Without it a bundle can only re-derive the coarse rank
    from the label's first whitespace token, which is a Latin-binomial convention:
    on a domain like `comp.sys.mac.hardware` it makes every label its own group,
    so the cascade silently loses the rank it was measured with. The caller's
    group column wins at build time and has to keep winning at predict time.

    Bundle format 2 adds it. A v1 bundle has no map and `predict` says so rather
    than guessing.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # `clf` is None when the posteriors came from a model that is not ours
    # (`--scores`). There is then no head to write, and `predict` cannot run the
    # bundle -- which is honest: we measured someone else's model, we did not
    # obtain a copy of it. The manifest records the absence so `predict` can say
    # so rather than failing on a missing file.
    if clf is not None:
        # float32 deliberately: `_vecs` casts descriptors to float32 and sklearn
        # keeps the dtype, so this is already what the fit produced. Written
        # explicitly so that a later change upstream cannot silently double every
        # head on disk, and pinned by a test.
        np.savez_compressed(out / "head.npz",
                            coef=np.asarray(clf.coef_, dtype="float32"),
                            intercept=np.asarray(clf.intercept_, dtype="float32"),
                            classes=np.asarray(clf.classes_, dtype=str),
                            # The direction the training vectors point in. Lets
                            # `predict` refuse vectors from a different encoder,
                            # which the declared `encoder` string cannot catch.
                            space=np.zeros(0, dtype="float32") if space is None
                            else np.asarray(space, dtype="float32").ravel())
    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "has_head": clf is not None,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "encoder": encoder,
        "labels": chosen,
        "source": source,
        "hazards": sorted(hazards or []),
        "never_answer": sorted(never_answer or []),
        "counts": counts,
        "composition": {k: v for k, v in composition.items()
                        if k != "outside_siblings" and not k.startswith("_")},
        "outside_siblings": composition.get("outside_siblings", {}),
        "metrics": metrics,
        # The payoffs actually fitted against, not the module default. Recording
        # UTILITY unconditionally would make the manifest lie about any bundle
        # built with an overriding profile -- and the payoffs are the one input
        # that must be legible after the fact, since they are what makes the
        # thresholds reproducible rather than tuned.
        "utility": {**UTILITY, **(utility or {})},
        "ood_mix": (metrics or {}).get("ood_mix") or OOD_MIX,
        "groups": {str(k): str(v) for k, v in (groups or {}).items()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out


def load_bundle(path: Path) -> dict:
    return json.loads((Path(path) / "manifest.json").read_text())
