"""The three-way decision: name the label, name its group, or decline.

Lifted from the project this grew out of, where it was species/genus/decline over
plants. The vocabulary is `label` and `group` here because nothing about the
machinery is biological -- a group is whatever coarser rank you can retreat to
when you cannot defend the fine one.

Two properties make this the right shape. The scores are **nested by
construction**:

    max_c P(c)  <=  max_g sum_{c in g} P(c)  <=  sum_{c != OTHER} P(c)

so the cascade is well-ordered and "confident at label, unsure at group" is
unreachable. And thresholds are fitted by maximising an **explicitly declared**
utility rather than read off a metric: an "is the answer useful" objective is
degenerate, because group accuracy always beats label accuracy and "always answer
group" would win it while deleting the product.

`UTILITY` is declared before anything is fitted. Change it deliberately and write
down why -- a utility silently tuned against a test set is the failure mode the
whole design exists to prevent.
"""

import numpy as np
import pandas as pd

LABEL, GROUP, DECLINE = "label", "group", "decline"
IN_CATALOG = "in_catalog"
SPLIT_CLUSTER = {"in_catalog": "label", "near_ood": "group",
                 "distant_ood": "label", "regional_ood": "label"}
OOD_MIX_GLOBAL = {"near_ood": 0.32, "distant_ood": 0.68}

UTILITY = {"label_correct": 1.0, "group_correct": 0.5, "wrong": -4.0,
           "decline_ood": 1.0, "decline_in_catalog": 0.0}


# Declared profiles. `wrong` is the stakes dial: it is what the threshold fit
# trades coverage against, so raising its magnitude buys abstention.
#
# These are written down here, in advance and with reasons, which is the
# discipline `CLAUDE.md` asks for -- "declare utilities before fitting; changing
# it is a deliberate act with a written reason". Selecting *among* them from
# properties of the label set is legitimate, because the label set is known
# before any row is scored. Selecting from measured outcomes would be reading the
# payoffs off the test set, which is the degeneracy the rule exists to prevent.
#
# The magnitudes come from the frontier recorded in plantid's `eval/rejection.py`:
# mu = 2 gives precision 0.944 at coverage 0.797, mu = 4 gives 0.965 / 0.747,
# mu = 8 gives 0.967 / 0.671, mu = 32 gives 0.991 / 0.645. Past about 8 the
# precision gain flattens and only coverage is spent.
PROFILES = {
    # Being wrong is cheap: a misnamed garden plant costs curiosity, not health.
    "identify": {**UTILITY, "wrong": -2.0},
    # The declared default, and the one every published number here was fitted at.
    "standard": dict(UTILITY),
    # Someone may eat it. Abstention is worth far more than an answer, and the
    # bar the card gates on is 1% of consequential labels given a harmless name.
    "forage": {**UTILITY, "wrong": -20.0, "decline_ood": 1.0},
    # A false positive costs an expert's time rather than a life, but still costs.
    "conserve": {**UTILITY, "wrong": -6.0},
}

def group_matrix(classes, mask, group_map=None):
    """(n_groups, n_labels) indicator G, and the group name per row.

    G[j, i] = 1 iff label i belongs to group j. Right-multiplying the per-label
    posterior by G.T sums probability mass within each group, which is what makes
    the cascade's scores nested (see `decide`).

    `group_map` is a label -> group dict supplied by the caller. Without it the
    group is the label's first whitespace token, which is a Latin-binomial
    convention: on a domain like `comp.sys.mac.hardware` it makes every label its
    own group, the group rank carries no information, and the cascade correctly
    but uselessly refuses to ever use it.
    """
    if group_map:
        groups = np.array([group_map.get(c, c.split()[0]) for c in classes[mask]])
    else:
        groups = np.array([c.split()[0] for c in classes[mask]])
    ug = np.unique(groups)
    return np.stack([(groups == g).astype(float) for g in ug]), ug


def decide(label_conf, group_conf, t_group, t_label):
    """Vectorised cascade -> array of LABEL / GROUP / DECLINE."""
    out = np.full(len(label_conf), LABEL, dtype=object)
    out[label_conf < t_label] = GROUP
    out[group_conf < t_group] = DECLINE
    return out


def utility(levels, label_ok, group_ok, in_catalog, weights=None):
    """Per-observation utility of the decision taken. Vectorised: threshold
    fitting evaluates this tens of thousands of times."""
    w = {**UTILITY, **(weights or {})}
    levels = np.asarray(levels, dtype=object)
    label_ok = np.asarray(label_ok, bool)
    group_ok = np.asarray(group_ok, bool)
    in_catalog = np.asarray(in_catalog, bool)

    is_dec, is_sp = levels == DECLINE, levels == LABEL
    is_gn = ~is_dec & ~is_sp
    return (
        is_dec * np.where(in_catalog, w["decline_in_catalog"], w["decline_ood"])
        + is_sp * np.where(label_ok, w["label_correct"], w["wrong"])
        + is_gn * np.where(group_ok, w["group_correct"], w["wrong"])
    )


def deployment_weights(buckets, p_ood=None, ood_mix=None):
    """Per-observation weights that reweight the evaluation buckets to an assumed
    deployment mix.

    Without this the operating point is set by however many observations each
    bucket happens to contain. That is an accident of sampling, and it moves the
    product: expanding the in-catalogue bucket from 750 to 2,283 shifted the
    calibration set from 59.5% out-of-catalogue to 44.8%, which moved
    `t_label` from 0.897 to 0.552 and took in-catalogue species answers from
    9% to 67% — a completely different product, from adding data alone.

    `p_ood=None` leaves the raw counts (uniform weights).
    """
    buckets = np.asarray(buckets)
    if p_ood is None:
        return np.ones(len(buckets), float)
    mix = ood_mix or OOD_MIX_GLOBAL
    w = np.zeros(len(buckets), float)
    n_in = max((buckets == IN_CATALOG).sum(), 1)
    w[buckets == IN_CATALOG] = (1 - p_ood) / n_in
    total = sum(mix.values())
    for bucket, share in mix.items():
        m = buckets == bucket
        if m.any():
            w[m] = p_ood * (share / total) / m.sum()
    return w * len(buckets) / w.sum()


def fit_thresholds(label_conf, group_conf, label_ok, group_ok, in_catalog,
                   weights=None, n_grid=60, sample_weight=None):
    """Grid-search (t_group, t_label) maximising expected utility. Calibration
    only. `sample_weight` reweights buckets to an assumed deployment prevalence
    — see `deployment_weights`."""
    sw = np.ones(len(label_conf)) if sample_weight is None else np.asarray(sample_weight, float)
    sw = sw / sw.sum()
    g_grid = np.quantile(group_conf, np.linspace(0, 1, n_grid))
    s_grid = np.quantile(label_conf, np.linspace(0, 1, n_grid))
    best, best_u = (0.0, 0.0), -np.inf
    for tg in g_grid:
        for ts in s_grid:
            u = float(np.dot(utility(decide(label_conf, group_conf, tg, ts),
                                     label_ok, group_ok, in_catalog, weights), sw))
            if u > best_u:
                best, best_u = (float(tg), float(ts)), u
    return best, best_u


# A split key is too coarse when one cluster can swallow a whole side of the
# split. Counting clusters is not the test -- five groups sounds like plenty and
# still puts a third of the bucket in one of them.
MAX_CLUSTER_SHARE = 0.34


def _too_coarse(group, key) -> bool:
    """True when one cluster under `key` is big enough to take a whole side."""
    if key not in group or group[key].nunique() < 2:
        return True
    return float(group[key].value_counts().iloc[0]) / len(group) > MAX_CLUSTER_SHARE


def make_splits(df, seed=0):
    """Assign 'calib'/'test' per row, splitting on the cluster for each bucket.

    Clustered because ~6 observations share a species and species difficulty is
    the dominant variance component; an observation-level split would put the
    same difficulty on both sides.

    `near_ood` keys on the *group* so a whole genus falls on one side. That is
    right when the group is a genus and **fails when it is coarser**: a bundle
    grouped by family gives every unlisted umbellifer the same key, so all of them
    land together and none reaches the test half. The failure is silent -- the
    bucket simply has no test rows, and anything measured over it reports "no
    data" rather than an error, which is how a safety check came back "not
    measured" with every photograph present.

    So a key that lets one cluster take more than `MAX_CLUSTER_SHARE` of a bucket
    is rejected in favour of the finest identity that still prevents leakage: the
    real species. Counting clusters is not enough -- five families sounds like
    plenty and still puts a third of the bucket in one of them.
    """
    rng = np.random.RandomState(seed)
    fold = pd.Series("test", index=df.index, dtype=object)
    for bucket, group in df.groupby("bucket"):
        key = SPLIT_CLUSTER.get(bucket, "label")
        if _too_coarse(group, key) and "species" in group:
            if not _too_coarse(group, "species"):
                key = "species"
        clusters = np.array(sorted(group[key].unique()))
        rng.shuffle(clusters)
        calib = set(clusters[: len(clusters) // 2])
        fold[group.index[group[key].isin(calib)]] = "calib"
    return fold


def cluster_bootstrap(values, clusters, n=2000, seed=0):
    """Resample *clusters*, not rows. Unclustered CIs have twice given this
    project effects that failed to replicate."""
    rng = np.random.RandomState(seed)
    uniq = np.array(sorted(set(clusters)))
    index = {c: np.flatnonzero(np.asarray(clusters) == c) for c in uniq}
    out = []
    for _ in range(n):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([index[c] for c in pick])
        out.append(np.mean(values[idx]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

