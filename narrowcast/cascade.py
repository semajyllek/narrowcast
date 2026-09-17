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


def decide(label_conf, group_conf, t_group, t_label, novelty=None, t_novel=None):
    """Vectorised cascade -> array of LABEL / GROUP / DECLINE.

    The optional third threshold is the **near-OOD gate**: a row whose mass sits
    mostly outside the label set is declined outright, whatever the other two
    thresholds said about it.

    **It is the reject arm, and plantid preferred the retreat arm.** That is not a
    departure from `NEAR_OOD_FINDINGS.md` but the consequence of reading it here.
    There, retreating beat rejecting (near-OOD wrong -0.0995 against -0.0695)
    because a near-OOD congener answered at the genus rank scored *correct* --
    an unlisted *Lomatium* called "Lomatium" is right. narrowcast scores no
    out-of-list row as correct at any rank: `frame_from_posteriors` sets
    `true_group` to `__OTHER__` for them, so `group_ok` is False by construction
    and retreating moves a row from `wrong` to `wrong`. **The retreat arm cannot
    pay here at any declared payoffs** -- it is arithmetic, not a fit that judged
    it unhelpful. Declining can: `decline_ood` is +1.0 against `wrong` at -4.0,
    and at -20.0 under `forage`.

    **Order is load-bearing, and the decline is unconditional.** The gate is
    applied last, so a gated row declines whatever `label_conf` and `group_conf`
    said. A gate that only downgraded the rank would be the dead arm above.

    `novelty` here is the in-list mass share, `1 - P(__OTHER__)` -- not the
    centroid geometry plantid's primary arm used. The two are statistically level
    (-0.0901 against -0.0995, overlapping intervals) and this one is free: the
    column is already computed and was being discarded. Where no reject class
    exists the share is 1 everywhere, the gate is constant, and it gates nothing.
    """
    out = np.full(len(label_conf), LABEL, dtype=object)
    out[label_conf < t_label] = GROUP
    out[group_conf < t_group] = DECLINE
    if novelty is not None and t_novel is not None:
        out[np.asarray(novelty, float) < t_novel] = DECLINE
    return out


def suppress(levels, pred_label, never, pred_group=None, group_members=None):
    """Force DECLINE wherever the cascade would answer with a suppressed label.

    `UTILITY["wrong"]` is one scalar over every label, so a profile can make the
    whole model cautious and cannot make it cautious about one thing. This is the
    per-label dial: a list of labels the bundle will never emit.

    **It suppresses the look-alike, not the hazard.** On a forager's list the
    dangerous plant is out-of-list and the harm is it being named *Daucus carota*,
    so `--never-answer "Daucus carota"` is what removes the harm. Suppressing the
    hazard itself is not merely weaker, it is inert: `build.hazard_metrics` counts
    rows where the prediction is *not* the hazard, so rows whose argmax is the
    hazard were never in the numerator. Suppressing it moves those rows from LABEL
    to DECLINE, leaves numerator and denominator alone, and drops
    `named_correctly` to zero -- the same danger for less utility.

    **After `decide`, never inside it, and never inside the fit.** `decide` takes
    confidences alone, and threading the predicted label through it would pull
    this into `fit_thresholds`'s grid search. The thresholds are fitted as though
    no label were suppressed, and suppression is applied to the result, so what
    the override costs stays a clean measurable delta instead of being absorbed
    into the operating point. Folding it into the fit would delete that number.

    **Decline, not retreat to the group.** A group answer looks like the softer
    option and is conditionally wrong: "it is an umbellifer" genuinely warns the
    person holding the root, "it is a *Lomatium*" is a species-level claim wearing
    the clothes of caution, and which one you get depends on how the caller
    grouped. Declining is safe under every grouping.

    **A group answer is suppressed only when the group has nothing else in it.**
    Suppressing on the argmax alone would kill "it is an umbellifer" on any row
    that happened to lean towards *Daucus carota*, which is a warning the forager
    wanted. But where the caller grouped by genus and *Daucus carota* is the only
    listed *Daucus*, "it is a *Daucus*" is the suppressed claim with a different
    name on it, so that one goes too. `group_members` is the label set per group;
    without it only label answers are suppressed, and the genus-grouped hole
    stays open.
    """
    if not never:
        return levels
    never = set(never)
    out = np.asarray(levels, dtype=object).copy()
    kill = (out == LABEL) & np.isin(np.asarray(pred_label, dtype=str), sorted(never))
    if pred_group is not None and group_members:
        hollow = sorted(g for g, members in group_members.items()
                        if members and set(members) <= never)
        if hollow:
            kill |= (out == GROUP) & np.isin(np.asarray(pred_group, dtype=str), hollow)
    out[kill] = DECLINE
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


def fit_novelty_threshold(label_conf, group_conf, novelty, label_ok, group_ok,
                          in_catalog, t_group, t_label, weights=None, n_grid=60,
                          sample_weight=None):
    """Fit the near-OOD gate alone, with the other two thresholds already fixed.

    The gate declines; see `decide` for why the retreat arm plantid preferred is
    structurally unable to pay here. That difference traces to an asymmetry worth
    knowing about: `build.outside_hazard_metrics` treats a group answer naming an
    out-of-list row's own group as a *warning* and therefore safe, while `utility`
    scores the identical answer as `wrong`. Two parts of one tool disagree about
    whether a true coarse statement about an out-of-list row is worth anything.
    Reconciling them is a declared-utility change and wants its own pass with a
    written reason; it is flagged here, not resolved.

    **Staged, not joint, and that is a deliberate trade.** A 3-D grid at the same
    resolution is 216,000 evaluations against 3,600 -- about 90 seconds on a real
    audit where the current fit takes 1.5 -- and reaching it would mean rewriting
    `fit_thresholds`, whose tie-breaking produced every threshold on file across
    four repositories. Fitting the gate as a second stage costs `3600 + 60`
    instead, and leaves the first two thresholds byte-identical to what they were
    when no gate is asked for.

    What the staging costs, stated rather than discovered later: the joint optimum
    over three thresholds is not reached, and `t_group`/`t_label` were fitted
    against a decision rule this gate then changes. Both are acceptable because
    the sweep **can always choose the baseline** -- the grid starts at the minimum
    of `novelty`, and a strict `<` against the minimum gates no row -- so the
    fitted gate cannot score below the ungated fit on the calibration half.

    The concrete shape of that cost: **the gate helps where the first stage
    answers too much.** Where the declared payoffs already drive `t_group` up to
    decline nearly everything -- a high `p_ood` over buckets the closed-set scores
    cannot separate -- there is no surviving error for it to remove, and it
    reports a gain of zero. On the test fixture it takes near-OOD wrong from 0.958
    to 0.000 at `p_ood = 0.10` **without moving the in-list label share at all**,
    and is inert at 0.30.

    Returns `(t_novel, utility_gained)`; a gain of 0.0 means the fit turned the
    gate off, which is a result and not a failure. plantid measured this gate as a
    utility **null** at `wrong = -4` and `p_ood = 0.20` and did not ship it. It is
    fitted here rather than assumed precisely so a caller declaring different
    payoffs -- `forage` costs a wrong answer -20, and near-OOD is where a
    forager's hazard lives -- gets the answer for their stakes, not plantid's.
    """
    sw = np.ones(len(label_conf)) if sample_weight is None else np.asarray(sample_weight, float)
    sw = sw / sw.sum()
    novelty = np.asarray(novelty, float)

    def u_at(tn):
        lv = decide(label_conf, group_conf, t_group, t_label, novelty, tn)
        return float(np.dot(utility(lv, label_ok, group_ok, in_catalog, weights), sw))

    grid = np.quantile(novelty, np.linspace(0, 1, n_grid))
    base = u_at(grid[0])                    # gates nothing: strict < against the min
    best, best_u = float(grid[0]), base
    for tn in grid[1:]:
        u = u_at(float(tn))
        if u > best_u:
            best, best_u = float(tn), u
    return best, best_u - base


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
    fold = pd.Series("test", index=df.index, dtype=object)
    for bucket, group in df.groupby("bucket"):
        key = SPLIT_CLUSTER.get(bucket, "label")
        if _too_coarse(group, key) and "species" in group:
            if not _too_coarse(group, "species"):
                key = "species"
        clusters = np.array(sorted(group[key].unique()))
        # One generator per bucket, keyed on the seed and the bucket's *contents*
        # -- the clusters it actually holds under the key actually chosen. Not on
        # its name, and emphatically not a single stream shared across this loop.
        #
        # Shared, every bucket's split depended on the *alphabetical order* of the
        # bucket names, so flagging out-of-list rows as `regional_ood` -- which
        # renames a bucket and changes nothing else about the data -- reshuffled
        # `in_catalog` and `near_ood` as a side effect and moved coverage by 12
        # points on one seed. Keyed on the name, that side effect goes but the
        # renamed bucket still resplits, so a pure relabelling still moves the
        # headline. Keyed on contents, identical rows give an identical split
        # whatever the bucket is called.
        #
        # What this buys is **locality, not stability**. Adding one row still
        # reshuffles its own bucket completely -- measured at 0.50 agreement,
        # no better than chance -- because the fingerprint changes. That is
        # defensible: that bucket's data did change. What it no longer does is
        # disturb the buckets that did not change, which sit at 1.000. The
        # shared stream disturbed every one of them.
        fingerprint = zlib.crc32("\x00".join(clusters.tolist()).encode()) % 2**31
        rng = np.random.RandomState([seed, fingerprint])
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

