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

import zlib

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


def hazard_rows(df, hazard):
    """Mask of the rows a declared hazard owns, under *either* threat model.

    `build.hazard_metrics` keys on `truth`, because a listed hazard is a label the
    model can name. `build.outside_hazard_metrics` keys on `species` and
    out-of-list, because a forager's hazard is absent from the label set by design
    and its `truth` is `OTHER` for every row it has.

    The two must disagree, because `species` is not one thing across the source
    paths: `load_scored` sets it to the real label, `load_embeddings` to the
    clustering id. So the union lives here, once. A hazard stratified into the
    test half under one predicate and then measured under the other would still be
    reported *unmeasured* -- and unmeasured is the outcome the stratification
    exists to stop, so the miss would be invisible in exactly the place the card
    promises to be loud.
    """
    m = np.zeros(len(df), bool)
    if "truth" in df:
        m |= df["truth"].to_numpy() == hazard
    if "species" in df:
        s = df["species"].to_numpy() == hazard
        if "in_catalog" in df:
            s &= ~df["in_catalog"].to_numpy().astype(bool)
        m |= s
    return m


def make_splits(df, seed=0, hazards=None):
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

    **Declared hazards are stratified rather than shuffled.** Everything above
    resamples whole clusters, which is right on average and wrong for the one
    label a caller has told us they are afraid of. With the 23 near-OOD species of
    an Oregon foraging list, `near_ood` keys on the genus, each hazard is its own
    genus, and reaching the test half is a coin flip per seed: *Conium* made it in
    6 of 8 splits and *Cicuta* in 2 of 8. The card reports the misses honestly as
    unmeasured -- but a single audit then decides by shuffle whether the declared
    hazard was checked at all, which is not a property a safety report may have.

    So each hazard's own rows are re-assigned after the bucket loop, halved at the
    *cluster* identity so several photographs of one plant still cannot straddle
    the split. This deliberately breaks the whole-genus rule for that one genus:
    the hazard is by declaration the thing being measured, not a member of the
    background it was drawn from. `MAX_CLUSTER_SHARE` does not reach this -- with
    23 genera no single one exceeds 0.34, so `_too_coarse` is content and the coin
    flip survives.

    A hazard with a single cluster goes wholly to `test`, because halving it would
    straddle a cluster and that is the convention that does not bend. It loses its
    interval, and `_ci` already reports that as "no cluster within a single label"
    rather than inventing one.

    The cost is real and worth stating: a hazard that the shuffle happened to send
    wholly to `test` now has about half as many test rows, so its interval widens.
    That is the trade -- a wider interval on every audit, against a point estimate
    that exists on only some of them. Sending it wholly to `test` instead would
    keep the rows and pull negatives out of the calibration set, which is the
    mistake `build.load_scored` records as having silently broken a published
    table.
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

    # Each hazard draws from its own generator, keyed on the seed and on its own
    # name, so its halving depends on neither the bucket shuffle nor on what else
    # was declared. Sharing `rng` would let adding `--hazard` move the whole
    # split; sharing one stream across this loop would make Conium's halves
    # depend on whether Cicuta was also named.
    #
    # What this does *not* buy is comparable headline numbers. A declared hazard's
    # rows move between the halves, the calibration set's composition changes with
    # them, and `fit_thresholds` therefore returns a different operating point --
    # on the fixture in the tests, up to 8 points of label share. Only the split
    # assignment is stable; anything fitted from it is not.
    for hazard in sorted(hazards or []):
        m = hazard_rows(df, hazard)
        if not m.any():
            continue
        clusters = np.array(sorted(df.loc[m, "label"].unique()))
        if len(clusters) < 2:
            fold[df.index[m]] = "test"
            continue
        hz_rng = np.random.RandomState([seed, zlib.crc32(hazard.encode()) % 2**31])
        hz_rng.shuffle(clusters)
        to_calib = m & df["label"].isin(clusters[: len(clusters) // 2]).to_numpy()
        fold[df.index[to_calib]] = "calib"
        fold[df.index[m & ~to_calib]] = "test"
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

