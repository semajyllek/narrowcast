import json
import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from narrowcast import build, cascade, card, labels, sources


# ---- labels list parsing -------------------------------------------------

def test_canonical_strips_authority_and_comments():
    assert labels.canonical("Sedum acre L.") == "Sedum acre"
    assert labels.canonical("  Trifolium repens  # in the lawn ") == "Trifolium repens"
    assert labels.canonical("# just a comment") is None


def test_canonical_matches_the_repo_join_key():
    """Hybrids normalise to 'x', not '×' -- the tool must not spell the key its own way."""
    from narrowcast.labels import canonical_name
    for raw in ("Fragaria × ananassa Duchesne", "Pelargonium x hortorum L.H. Bailey",
                "Sedum acre L."):
        assert labels.canonical(raw) == canonical_name(raw)


def test_read_list_dedupes_and_keeps_order(tmp_path):
    p = tmp_path / "s.txt"
    p.write_text("Sedum acre L.\n\n# comment\nTrifolium repens\nSedum acre\n")
    assert labels.read_list(p) == ["Sedum acre", "Trifolium repens"]


def test_read_list_raises_rather_than_dropping_when_binomial_is_declared(tmp_path):
    """A silently ignored labels is a model that cannot see a plant the user asked for."""
    p = tmp_path / "s.txt"
    p.write_text("Sedum acre\nnot a binomial\n")
    with pytest.raises(ValueError, match="could not parse"):
        labels.read_list(p, binomial=True)


def test_mixed_list_falls_back_to_raw_labels_rather_than_rejecting(tmp_path):
    """Auto-detect: only a list that is wholly binomial gets the Linnaean key."""
    p = tmp_path / "s.txt"
    p.write_text("Sedum acre\nnot a binomial\n")
    assert labels.read_list(p) == ["Sedum acre", "not a binomial"]


# ---- composition ----------------------------------------------------------

POOL = ["Sedum acre", "Sedum album", "Sedum dasyphyllum", "Sedum rupestre",
        "Trifolium repens", "Trifolium pratense", "Bellis perennis"]


def test_analyse_sibling_fraction_and_crowding():
    a = labels.analyse(["Sedum acre", "Sedum album", "Bellis perennis"], pool=POOL)
    assert a["n_labels"] == 3 and a["n_groups"] == 2
    assert a["in_set_sibling_frac"] == pytest.approx(2 / 3)
    assert a["crowded_groups"] == {"Sedum": 2}


def test_analyse_groups_outside_siblings_by_genus():
    a = labels.analyse(["Sedum acre", "Sedum album"], pool=POOL)
    assert a["outside_siblings"] == {"Sedum": ["Sedum dasyphyllum", "Sedum rupestre"]}
    assert a["n_labels_exposed"] == 2


def test_analyse_no_siblings_when_set_is_separated():
    a = labels.analyse(["Bellis perennis"], pool=POOL)
    assert a["in_set_sibling_frac"] == 0.0
    assert a["crowded_groups"] == {} and a["outside_siblings"] == {}


# ---- card -----------------------------------------------------------------

def _manifest(label_share):
    return {
        "bundle_version": 1, "created": "2026-01-01T00:00:00",
        "encoder": "mobileclip2_s2", "source": "local-catalogue",
        "labels": ["Sedum acre", "Sedum album"],
        "counts": {"train": 100},
        "composition": {"n_labels": 2, "crowded_groups": {"Sedum": 2}},
        "outside_siblings": {"Sedum": ["Sedum dasyphyllum"]},
        "utility": {"species_correct": 1.0},
        "metrics": {
            "t_group": 0.5, "t_label": 0.9, "p_ood": 0.2,
            "coverage": 0.84, "precision": 0.97, "label_share": label_share,
            "closed_set_top1": 0.81, "n_calib": 10, "n_test": 20,
            "per_bucket": {"in_catalog": {"n": 20, "answered": 0.9,
                                          "correct_when_answered": 0.95}},
        },
    }


def test_card_flags_a_low_label_share():
    out = card.render(_manifest(0.31))
    assert "Read the label-level share, not the coverage" in out


def test_card_omits_the_flag_when_label_share_is_healthy():
    assert "Read the label-level share" not in card.render(_manifest(0.85))


def test_card_always_carries_the_safety_line():
    for share in (0.31, 0.85):
        assert "A correct-looking answer is not verification" in card.render(_manifest(share))


def test_card_singularises_a_lone_relative():
    out = card.render(_manifest(0.85))
    assert "1 relative not on your list" in out


def test_card_shows_cluster_bootstrapped_intervals():
    m = _manifest(0.85)
    m["metrics"]["ci"] = {"label_share": [0.61, 0.96], "precision": [0.90, 0.99],
                          "closed_set_top1": [0.70, 0.92]}
    m["metrics"]["n_label_clusters"] = 7
    out = card.render(m)
    assert "61.0–96.0%" in out
    # "clusters", not "labels". `_ci` resamples the cluster column, which equals
    # the label only when the caller supplied no finer grouping. Saying "labels"
    # overstated the protection for any dataset that does supply `cluster`, and
    # was actively wrong where those clusters are singletons.
    assert "over **clusters**, not rows" in out


def test_card_dashes_a_missing_interval_rather_than_inventing_one():
    out = card.render(_manifest(0.85))   # no "ci" key at all
    assert "—" in out and "None" not in out


# ---- interval computation -------------------------------------------------

def test_ci_needs_at_least_two_clusters():
    one = np.ones(2)
    assert build._ci(np.array([1.0, 0.0]), one, np.array(["a", "a"])) is None
    assert build._ci(np.array([]), np.array([]), np.array([])) is None


def test_ci_brackets_the_point_estimate():
    vals = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    clusters = np.array(["a", "a", "b", "b", "c", "c"])
    lo, hi = build._ci(vals, np.ones(6), clusters)
    assert lo <= vals.mean() <= hi


def test_ci_tracks_the_weighted_ratio_not_the_unweighted_mean():
    """The bug this replaced: precision 96% with a 22-77% interval around it."""
    correct = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    w = np.array([10.0, 10.0, 0.1, 0.1, 0.1, 0.1])   # in-list rows dominate
    clusters = np.array(["a", "b", "c", "d", "e", "f"])
    lo, hi = build._ci(correct * w, w, clusters)
    point = (correct * w).sum() / w.sum()
    assert lo <= point <= hi
    assert hi > 0.5   # nowhere near the 0.33 unweighted mean


# ---- consequential labels (the union rate) --------------------------------

def _haz_frame(pred_label, pred_group, truth):
    return pd.DataFrame({"pred_label": pred_label, "pred_group": pred_group,
                         "truth": truth, "label": truth})


def test_union_counts_every_harmless_name_not_just_one_confusion():
    """The Oregon finding: no single pair exceeded 2.5% while the union hit 6.7%."""
    truth = ["Conium maculatum"] * 4
    pred = ["Daucus carota", "Anthriscus caucalis", "Foeniculum vulgare", "Conium maculatum"]
    te = _haz_frame(pred, [p.split()[0] for p in pred], truth)
    lv = np.array([build.LABEL] * 4)
    h = build.hazard_metrics(te, lv, {"Conium maculatum"})["Conium maculatum"]
    assert h["named_non_hazard"] == pytest.approx(0.75)   # 3 different harmless names
    assert h["named_correctly"] == pytest.approx(0.25)


def test_being_named_another_hazard_is_not_counted_as_dangerous():
    truth = ["Conium maculatum"] * 2
    pred = ["Cicuta douglasii", "Daucus carota"]
    te = _haz_frame(pred, [p.split()[0] for p in pred], truth)
    lv = np.array([build.LABEL] * 2)
    h = build.hazard_metrics(te, lv, {"Conium maculatum", "Cicuta douglasii"})["Conium maculatum"]
    assert h["named_other_hazard"] == pytest.approx(0.5)
    assert h["named_non_hazard"] == pytest.approx(0.5)


def test_a_genus_answer_naming_a_harmless_group_is_dangerous():
    """'It is a Lomatium' for poison hemlock is as actionable as a wrong labels."""
    te = _haz_frame(["x", "x"], ["Lomatium", "Conium"], ["Conium maculatum"] * 2)
    lv = np.array([build.GROUP, build.GROUP])
    h = build.hazard_metrics(te, lv, {"Conium maculatum"})["Conium maculatum"]
    assert h["named_non_hazard"] == pytest.approx(0.5)     # the Lomatium answer
    assert h["named_other_hazard"] == pytest.approx(0.5)   # its own group: safe


def test_declining_is_never_counted_as_dangerous():
    te = _haz_frame(["Daucus carota"], ["Daucus"], ["Conium maculatum"])
    lv = np.array([build.DECLINE])
    h = build.hazard_metrics(te, lv, {"Conium maculatum"})["Conium maculatum"]
    assert h["named_non_hazard"] == 0.0 and h["declined"] == 1.0


def test_no_hazards_declared_yields_no_section():
    assert build.hazard_metrics(_haz_frame(["a"], ["a"], ["b"]), np.array([build.LABEL]), None) == {}
    assert "Consequential labels" not in card.render(_manifest(0.85))


def test_card_gate_fires_above_the_declared_bar():
    m = _manifest(0.85)
    m["metrics"]["hazard"] = {"Conium maculatum": {
        "n": 40, "declined": 0.10, "named_correctly": 0.83,
        "named_other_hazard": 0.0, "named_non_hazard": 0.067, "ci": [0.017, 0.128]}}
    out = card.render(m)
    assert "Do not rely on this model" in out and "6.7%" in out


def test_card_gate_passes_below_the_bar():
    m = _manifest(0.85)
    m["metrics"]["hazard"] = {"Conium maculatum": {
        "n": 40, "declined": 0.25, "named_correctly": 0.99,
        "named_other_hazard": 0.0, "named_non_hazard": 0.008, "ci": None}}
    out = card.render(m)
    assert "Do not rely on this model" not in out
    assert "under the 1.0% bar" in out


# ---- bundle round trip ----------------------------------------------------

class _Clf:
    coef_ = np.zeros((2, 4))
    intercept_ = np.zeros(2)
    classes_ = np.array(["Sedum acre", "__OTHER__"])


def test_bundle_round_trip(tmp_path):
    m = _manifest(0.5)
    out = build.save_bundle(tmp_path / "b", _Clf(), m["labels"], "mobileclip2_s2",
                            m["metrics"], labels.analyse(m["labels"], pool=POOL),
                            {"train": 100}, source="local-catalogue")
    loaded = build.load_bundle(out)
    assert loaded["labels"] == m["labels"]
    assert loaded["encoder"] == "mobileclip2_s2"
    assert loaded["metrics"]["coverage"] == 0.84
    assert json.loads((out / "manifest.json").read_text())["bundle_version"] == 2
    assert (out / "head.npz").exists()


def test_bundle_stores_the_group_map_for_predict(tmp_path):
    """Format 2 persists label -> group. Without it `predict` can only re-derive
    the coarse rank from the first whitespace token, which is a Latin-binomial
    convention: on `comp.sys.mac.hardware` every label becomes its own group and
    the cascade silently loses the rank it was measured with."""
    m = _manifest(0.5)
    gmap = {"comp.graphics": "comp", "comp.windows.x": "comp", "rec.autos": "rec"}
    out = build.save_bundle(tmp_path / "b", _Clf(), list(gmap), "mobileclip2_s2",
                            m["metrics"], labels.analyse(m["labels"], pool=POOL),
                            {"train": 100}, source="x", groups=gmap)
    assert build.load_bundle(out)["groups"] == gmap


# ---- data sources: the tool takes a dataset, it does not fetch one ---------

def test_missing_cluster_is_recorded_not_silently_assumed(tmp_path):
    f = tmp_path / "v.npz"
    np.savez(f, descriptor=np.zeros((2, 4), dtype="float32"),
             label=np.array(["a", "b"]))
    r = sources.from_embeddings(f)
    assert r.has_clusters is False
    assert any("treated as independent" in n for n in r.notes)


def test_group_defaults_to_first_token_and_is_overridable(tmp_path):
    f = tmp_path / "v.npz"
    np.savez(f, descriptor=np.zeros((1, 4), dtype="float32"),
             label=np.array(["Sedum acre"]), group=np.array(["Crassulaceae"]))
    assert list(sources.from_embeddings(f).group) == ["Crassulaceae"]
    assert sources.default_group("Sedum acre") == "Sedum"


def test_exactly_one_source_required():
    with pytest.raises(ValueError, match="exactly one"):
        sources.load()
    with pytest.raises(ValueError, match="exactly one"):
        sources.load(embeddings="a", scores="b")


# ---- non-binomial labels are first-class ----------------------------------

def test_read_list_accepts_labels_that_are_not_binomials(tmp_path):
    f = tmp_path / "l.txt"
    f.write_text("weld_porosity\ncrack_lateral\n# comment\n")
    assert labels.read_list(f) == ["weld_porosity", "crack_lateral"]


def test_read_list_still_normalises_a_pure_binomial_list(tmp_path):
    f = tmp_path / "l.txt"
    f.write_text("Sedum acre L.\nTrifolium repens\n")
    assert labels.read_list(f) == ["Sedum acre", "Trifolium repens"]


_MIN = {"data": {"images": "./x"}, "objective": {"metric": "label_share", "minimum": 0.9}}


def _r(name, size, val, metric="label_share"):
    return SW.Result(name, size, {metric: val, "coverage": 0.8, "precision": 0.9})


# ---- the supplied group column must actually reach the cascade -------------

def _rows(labels, groups, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(hash(g) % 7, 0.3, (1, dim)) for g in groups]).astype("float32")
    return sources._finish(labels, descriptor=X, group=groups,
                           cluster=[f"c{i}" for i in range(len(labels))])


def test_supplied_group_reaches_the_frame_and_is_not_recomputed():
    """`sources` read the group column, `Rows` carried it, and `score_frame` used
    to discard it and re-derive from whitespace — which made every dotted label
    its own group and silently disabled the group rank for non-binomial domains."""
    labels = ["comp.graphics", "comp.windows.x", "rec.autos", "rec.motorcycles"]
    groups = ["comp", "comp", "rec", "rec"]
    ds = build.load_rows(_rows(labels * 6, groups * 6), "unused")
    clf = build.fit_head(ds)
    frame = build.score_frame(clf, ds)
    assert set(frame["group"]) <= {"comp", "rec"}, "group column was re-derived"
    assert "comp.graphics" not in set(frame["group"])


def test_group_matrix_honours_a_supplied_mapping():
    classes = np.array(["comp.graphics", "comp.windows.x", "rec.autos"])
    mask = np.ones(3, bool)
    _, ug = cascade.group_matrix(classes, mask)
    assert len(ug) == 3                      # default rule: every label its own group
    _, ug2 = cascade.group_matrix(classes, mask,
                                  {"comp.graphics": "comp", "comp.windows.x": "comp",
                                   "rec.autos": "rec"})
    assert sorted(ug2) == ["comp", "rec"]


def test_cascade_scores_are_nested_whatever_the_posterior():
    """`max_c P(c) <= max_g sum_{c in g} P(c) <= sum_{c != OTHER} P(c)`.

    `cascade.py` calls this structural and reasons from it -- it is what makes
    "confident at label, unsure at group" unreachable and the three-way decision
    well-ordered. Nothing pinned it. It matters beyond tidiness because the
    property has to survive score distributions the tool did not produce: a
    distilled or otherwise miscalibrated model can be arbitrarily
    temperature-shifted and still must not break the ordering the thresholds are
    fitted against. Dirichlet draws stand in for "any posterior at all".
    """
    classes = np.array(["Sedum acre", "Sedum album", "Sedum dasyphyllum",
                        "Trifolium repens", "Trifolium pratense", build.OTHER])
    mask = classes != build.OTHER
    gmat, _ = cascade.group_matrix(classes, mask)
    rng = np.random.default_rng(0)
    for alpha in (0.05, 1.0, 20.0):          # spiky, uniform, and flat posteriors
        cata = rng.dirichlet(np.full(len(classes), alpha), size=2000)[:, mask]
        label_conf = cata.max(1)
        group_conf = (cata @ gmat.T).max(1)
        assert (label_conf <= group_conf + 1e-12).all()
        assert (group_conf <= cata.sum(1) + 1e-12).all()


def test_binomial_labels_are_unaffected_by_the_fix():
    """Every previously committed result used whitespace-separated binomials, where
    the default rule was already correct."""
    from narrowcast.labels import group_of
    assert group_of("Sedum acre") == "Sedum"
    assert group_of("Larus occidentalis") == "Larus"


# ---- headroom: the quantity that predicts retreat to the group -------------

def _cascade_frame(n_labels=8, per_label=6, n_bg=40, label_ok=True, group_ok=True):
    """A frame shaped like `score_frame`'s output, with the two rank outcomes
    controllable independently. That independence is the whole point: headroom is
    coarse-rank accuracy minus fine-rank accuracy, and nothing else may move it."""
    rng = np.random.default_rng(0)
    lab = [f"G{i // 2} sp{i}" for i in range(n_labels) for _ in range(per_label)]
    rows = {
        "label_conf": rng.uniform(0.4, 0.99, len(lab)),
        "group_conf": rng.uniform(0.4, 0.99, len(lab)),
        "label_ok": np.full(len(lab), label_ok),
        "group_ok": np.full(len(lab), group_ok),
        "in_catalog": np.full(len(lab), True),
        "bucket": ["in_catalog"] * len(lab),
        "label": lab,
        "group": [l.split()[0] for l in lab],
    }
    bg = {
        "label_conf": rng.uniform(0.0, 0.5, n_bg),
        "group_conf": rng.uniform(0.0, 0.5, n_bg),
        "label_ok": np.full(n_bg, False),
        "group_ok": np.full(n_bg, False),
        "in_catalog": np.full(n_bg, False),
        "bucket": ["distant_ood"] * n_bg,
        "label": [f"bg{i}" for i in range(n_bg)],
        "group": [f"bg{i}" for i in range(n_bg)],
    }
    return pd.concat([pd.DataFrame(rows), pd.DataFrame(bg)], ignore_index=True)


def test_headroom_is_the_gap_between_the_two_ranks():
    """Coarse right where fine is wrong is headroom 1.0; the value must not depend
    on which half `make_splits` happened to choose."""
    m = build.fit_and_measure(_cascade_frame(label_ok=False, group_ok=True), p_ood=0.2)
    assert m["calib_fine"] == pytest.approx(0.0)
    assert m["calib_coarse"] == pytest.approx(1.0)
    assert m["headroom"] == pytest.approx(1.0)


def test_headroom_is_zero_when_the_group_rank_adds_nothing():
    """One label per group -- a group answer *is* a label answer, so no retreat is
    possible and none should be reported. This is the varied arm in every
    published table (plantid HEADROOM_FINDINGS.md, P1)."""
    m = build.fit_and_measure(_cascade_frame(label_ok=True, group_ok=True), p_ood=0.2)
    assert m["headroom"] == pytest.approx(0.0)


def test_the_three_in_list_shares_partition():
    """label / group / decline are the whole of the cascade's in-list behaviour.
    The card reasons about where answers went, so they must actually sum."""
    m = build.fit_and_measure(_cascade_frame(label_ok=False, group_ok=True), p_ood=0.2)
    total = m["label_share"] + m["group_share"] + m["decline_share"]
    assert total == pytest.approx(1.0)


def test_headroom_is_measured_on_calibration_not_test():
    """It has to be usable before the test numbers are trusted, and it must be a
    property of the label set rather than of the thresholds."""
    df = _cascade_frame(label_ok=False, group_ok=True)
    a = build.fit_and_measure(df, p_ood=0.2)
    b = build.fit_and_measure(df, p_ood=0.6)      # different thresholds entirely
    assert a["headroom"] == pytest.approx(b["headroom"])


# ---- the card must not assert where the answers went ----------------------

def _retreat_manifest(label_share, group_share, decline_share, headroom=0.0):
    m = _manifest(label_share)
    m["metrics"].update({"group_share": group_share, "decline_share": decline_share,
                         "headroom": headroom})
    return m


def test_card_names_group_retreat_when_that_is_what_happened():
    out = card.render(_retreat_manifest(0.30, group_share=0.60, decline_share=0.10))
    assert "60.0% are answered at group" in out
    assert "because* of those group answers" in out


def test_card_says_declining_when_the_model_is_declining():
    """The card used to assert 'the rest are answered at group' while measuring no
    such thing. On a model that declines instead, that sentence was simply false."""
    out = card.render(_retreat_manifest(0.30, group_share=0.05, decline_share=0.65))
    assert "65.0% are declined outright" in out
    assert "mostly declining rather than retreating" in out


def test_card_reports_benign_retreat_that_cost_nothing():
    """Retreat is not harm: group answers drawn from declines inflate coverage while
    quality holds (the kws-acoustic case), and that was invisible before. Gated on
    *measured* retreat, not on headroom, which only predicts it."""
    out = card.render(_retreat_manifest(0.75, group_share=0.20, decline_share=0.05,
                                        headroom=0.15))
    assert "retreats to the group, and it has not cost you" in out
    assert "15.0pp" in out


def test_card_stays_quiet_when_there_is_no_retreat():
    out = card.render(_retreat_manifest(0.85, group_share=0.02, decline_share=0.13,
                                        headroom=0.01))
    assert "retreats to the group" not in out
    assert "Read the label-level share" not in out


def test_card_renders_a_bundle_built_before_headroom_existed():
    """Old manifests carry none of these keys; the card must degrade, not raise."""
    out = card.render(_manifest(0.31))
    assert "Read the label-level share, not the coverage" in out
    assert "n/a" in out


def test_absent_bucket_does_not_lower_the_stated_prevalence():
    """`deployment_weights` leaves an absent bucket's share unclaimed and
    renormalises. A source with no in-pool relatives has no near_ood, so
    `--ood-rate 0.2` scored at an effective 0.145 while the card printed "an
    assumed 20.0% out-of-list rate"."""
    df = _cascade_frame(label_ok=False, group_ok=True)   # in_catalog + distant_ood only
    assert "near_ood" not in set(df["bucket"])
    m = build.fit_and_measure(df, p_ood=0.2)
    # per_bucket answered rates are unweighted, so check the weighting directly
    w = cascade.deployment_weights(df["bucket"].to_numpy(), p_ood=0.2,
                                   ood_mix={"distant_ood": 1.0})
    inc = df["in_catalog"].to_numpy()
    assert w[~inc].sum() / w.sum() == pytest.approx(0.2)
    assert m["p_ood"] == 0.2


# ---- acquisition origin, clustering, and rows per label -------------------

class _Cfg:
    """Minimal stand-in for a parsed config; `_candidates` reads only these."""
    def __init__(self, encoders=(), domain=(), max_size_mb=None, max_candidates=8):
        self.encoders, self.domain = encoders, domain
        self.max_size_mb, self.max_candidates = max_size_mb, max_candidates


def _origin_rows(labels, groups, origin, dim=8, seed=0, shift=0.0):
    """Like `_rows`, but rows from origin 'B' are displaced in feature space, so
    a head fitted without them genuinely does worse on them."""
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(hash(g) % 7, 0.3, (1, dim)) for g in groups]).astype("float32")
    X[np.asarray(origin) == "B"] += shift
    return sources._finish(labels, descriptor=X, group=groups, origin=origin,
                           cluster=[f"c{i}" for i in range(len(labels))])


def test_origin_column_is_optional_and_absent_by_default():
    ds = build.load_rows(_rows(["a x", "b y"] * 8, ["a", "b"] * 8), "unused")
    assert ds.origin_train is None and ds.origin_eval is None
    assert build.origin_cost(ds, "B") is None


def test_origin_cost_refuses_rather_than_guessing_when_nothing_to_compare():
    labels = ["a x", "b y"] * 8
    groups = ["a", "b"] * 8
    ds = build.load_rows(_origin_rows(labels, groups, ["A"] * 16), "unused")
    out = build.origin_cost(ds, "B")
    assert out["measurable"] is False and "no training rows" in out["why"]

    # every label covered -> nothing is being denied anything.  Origin must
    # alternate *within* each label; ["A","B"]*8 against ["a x","b y"]*8 makes
    # origin a perfect proxy for the label, which is a different situation.
    origin = ["A" if i % 4 < 2 else "B" for i in range(16)]
    ds2 = build.load_rows(_origin_rows(labels, groups, origin), "unused")
    out2 = build.origin_cost(ds2, "B")
    assert out2["measurable"] is False and out2["n_lacking"] == 0


def test_origin_cost_measures_the_labels_denied_the_deployment_origin():
    """One label gets no rows from the deployment origin; the report must name it
    and score it separately from the labels that did get them. Reporting one
    average would hide the subgroup being harmed, which is the failure the
    label-level share exists to prevent."""
    labels, groups, origin = [], [], []
    for i in range(60):
        lab = ["a x", "b y", "c z"][i % 3]
        labels.append(lab)
        groups.append(lab.split()[0])
        origin.append("A" if i % 2 else "B")
    rows = _origin_rows(labels, groups, origin, shift=1.5)
    ds = build.load_rows(rows, "unused")
    # 'c z' keeps its B *eval* rows but loses every B *training* row, which is
    # the situation the measurement exists for: scoreable, and denied the data.
    drop = (ds.y_train == "c z") & (ds.origin_train == "B")
    ds.X_train, ds.y_train = ds.X_train[~drop], ds.y_train[~drop]
    ds.origin_train = ds.origin_train[~drop]

    out = build.origin_cost(ds, "B")
    assert out["measurable"] is True
    assert out["n_lacking"] == 1 and out["labels_lacking"] == ["c z"]
    assert out["n_covered"] == 2
    # both groups are scored, and separately
    assert out["lacking_delta"] is not None and out["covered_delta"] is not None


def test_card_reports_origin_cost_per_group_never_as_one_number():
    oc = {"deployment_origin": "clinic", "measurable": True,
          "n_lacking": 2, "n_covered": 18, "n_eval_rows": 300,
          "labels_lacking": ["psoriasis", "lichen planus"],
          "covered_with": 0.87, "covered_without": 0.79, "covered_delta": 0.08,
          "lacking_with": 0.71, "lacking_without": 0.85, "lacking_delta": -0.14}
    text = "\n".join(card._origin_section(oc))
    assert "clinic" in text
    assert "0.080" in text or "+0.080" in text     # the gain
    assert "0.140" in text                          # the damage, reported beside it
    assert "psoriasis" in text                      # named, so it can be acted on
    assert "does **not** shrink" in text            # the K warning travels with it


def test_card_says_nothing_when_there_is_no_origin_column():
    assert card._origin_section(None) == []
    assert card._origin_section({}) == []


def test_origin_cost_counts_labels_it_cannot_score_instead_of_averaging_them_away():
    """A label with no deployment-origin rows *at all* is the common real case and
    the most exposed. It must be surfaced, not folded into a mean that would then
    understate the problem."""
    labels, groups, origin = [], [], []
    for i in range(60):
        lab = ["a x", "b y", "c z"][i % 3]
        labels.append(lab)
        groups.append(lab.split()[0])
        origin.append("A" if lab == "c z" else ("A" if i % 2 else "B"))
    ds = build.load_rows(_origin_rows(labels, groups, origin, shift=1.5), "unused")
    out = build.origin_cost(ds, "B")
    assert out["n_unmeasured"] == 1 and out["labels_unmeasured"] == ["c z"]
    text = "\n".join(card._origin_section(out))
    assert "c z" in text and "no `B` rows at" in text


def test_singleton_clusters_are_flagged_as_no_clustering():
    """A unique id per row is arithmetically the same as supplying no cluster
    column, but it suppresses the "no cluster column" warning — so the card
    reported clustered intervals for what was a row-level bootstrap.

    Found by running `fit` over a mixed Pl@ntNet/iNaturalist corpus: Pl@ntNet has
    no observation grouping, its rows were keyed by image id, and 93% of the
    resulting clusters held exactly one row.
    """
    labels = ["Sedum acre", "Sedum album"] * 10
    rows = sources._finish(labels, descriptor=np.zeros((20, 4)),
                       cluster=[str(i) for i in range(20)])
    assert rows.has_clusters
    assert any("single row" in n for n in rows.notes), rows.notes

    grouped = sources._finish(labels, descriptor=np.zeros((20, 4)),
                          cluster=[str(i // 5) for i in range(20)])
    assert not any("single row" in n for n in grouped.notes), grouped.notes


def _counts(median, minimum=None, n_labels=20, n_below=0):
    return {"rows_per_label": {"median": median, "min": minimum or median,
                               "n_labels": n_labels, "n_below_32": n_below}}


def test_thin_data_advice_inverts_on_whether_the_model_is_retreating():
    """Eight rows per label is fine on a separated list and badly short on a
    crowded one, and the two call for opposite responses — take more photographs,
    or change the list. The card gave the same advice for both because it never
    reported how much data the head was fitted on.

    Thresholds and wording come from TINY_FINDINGS.md §2: on a separated list the
    label-level share is saturated by ~32 rows per label, and on a crowded one 64
    rows still buys under half of what unlimited data buys.
    """
    thin_crowded = card._data_limited_section(
        {"label_share": 0.02, "group_share": 0.31}, _counts(8))
    thin_separated = card._data_limited_section(
        {"label_share": 0.45, "group_share": 0.03}, _counts(8))

    assert "8 training rows per label" in "".join(thin_crowded)
    assert "helps *least*" in "".join(thin_crowded)
    assert "helps *most*" in "".join(thin_separated)
    assert "helps *least*" not in "".join(thin_separated)


def test_a_healthy_model_on_thin_data_is_told_so_without_alarm():
    out = "".join(card._data_limited_section(
        {"label_share": 0.88, "group_share": 0.01}, _counts(12)))
    assert "not a problem" in out and "rebuild" in out


def test_plentiful_data_says_nothing_and_names_only_the_thin_labels():
    assert card._data_limited_section(
        {"label_share": 0.88, "group_share": 0.01}, _counts(400)) == []
    out = "".join(card._data_limited_section(
        {"label_share": 0.88, "group_share": 0.01},
        _counts(400, minimum=9, n_below=3)))
    assert "3 of 20 labels" in out and "fewest: 9" in out


def test_rows_per_label_excludes_the_background_negatives():
    """`ytr` carries an appended block of __OTHER__ once negatives are supplied.
    Counting it would put OTHER in the per-label table and drag the median."""
    rows = _rows(["Sedum acre", "Sedum album"] * 20, ["Sedum"] * 40)
    bg = _rows(["Bellis perennis"] * 40, ["Bellis"] * 40)
    ds = build.load_rows(rows, "unused", background=bg)
    rpl = ds.counts["rows_per_label"]
    assert rpl["n_labels"] == 2, rpl
    assert rpl["median"] > 0


# ---- predict ---------------------------------------------------------------

def _built(tmp_path, labels_, groups_, n=12):
    rows = _rows(labels_ * n, groups_ * n)
    bg = _rows(["Bellis perennis"] * (4 * n), ["Bellis"] * (4 * n))
    ds = build.load_rows(rows, "unused", background=bg)
    clf = build.fit_head(ds)
    frame = build.score_frame(clf, ds)
    m = build.fit_and_measure(frame, p_ood=0.2)
    gmap = dict(zip(labels_, groups_))
    out = build.save_bundle(tmp_path / "b", clf, labels_, "mobileclip2_s2", m,
                            labels.analyse(labels_, pool=labels_), ds.counts,
                            source="test", groups=gmap, space=ds.X_train.mean(0))
    return out, ds, frame, m


def test_predict_reproduces_the_scores_the_card_was_measured_from(tmp_path):
    """The contract: a prediction and the card cannot disagree.

    `predict` recomputes the posterior from saved weights rather than calling
    sklearn, masks `__OTHER__` out of both scores, and sums group mass the same
    way. If any of that drifts, the card describes a model the user is not
    running.
    """
    from narrowcast import predict as P
    out, ds, frame, _ = _built(tmp_path, ["Sedum acre", "Sedum album", "Bellis annua"],
                               ["Sedum", "Sedum", "Bellis"])
    res = P.Bundle(out).predict(ds.X_eval)
    assert np.allclose([r["label_conf"] for r in res],
                       frame["label_conf"].to_numpy(), atol=1e-9)
    assert np.allclose([r["group_conf"] for r in res],
                       frame["group_conf"].to_numpy(), atol=1e-9)


def test_predict_decisions_match_the_fitted_cascade(tmp_path):
    from narrowcast import predict as P
    out, ds, frame, _ = _built(tmp_path, ["Sedum acre", "Sedum album", "Bellis annua"],
                               ["Sedum", "Sedum", "Bellis"])
    b = P.Bundle(out)
    got = np.array([r["rank"] for r in b.predict(ds.X_eval)], dtype=object)
    want = cascade.decide(frame["label_conf"].to_numpy(),
                          frame["group_conf"].to_numpy(), b.t_group, b.t_label)
    assert (got == want).all()
    assert set(got) <= {cascade.LABEL, cascade.GROUP, cascade.DECLINE}


def test_predict_answers_at_the_group_the_caller_declared(tmp_path):
    """Non-binomial labels: the coarse rank must come from the stored map, not
    from the first whitespace token, which would make every label its own group."""
    from narrowcast import predict as P
    out, ds, _, _ = _built(tmp_path, ["comp.graphics", "comp.windows.x", "rec.autos"],
                           ["comp", "comp", "rec"])
    b = P.Bundle(out)
    assert set(b.ugroups) == {"comp", "rec"}
    assert {r["group"] for r in b.predict(ds.X_eval)} <= {"comp", "rec"}
    assert not b.notes, b.notes


def test_a_format_1_bundle_says_its_group_map_is_missing(tmp_path):
    """Older bundles have no map. `predict` still runs and warns, rather than
    silently answering at a coarse rank the model was never measured with."""
    from narrowcast import predict as P
    out, _, _, _ = _built(tmp_path, ["comp.graphics", "comp.windows.x", "rec.autos"],
                          ["comp", "comp", "rec"])
    mf = json.loads((out / "manifest.json").read_text())
    mf["bundle_version"], mf["groups"] = 1, {}
    (out / "manifest.json").write_text(json.dumps(mf))
    assert any("first whitespace token" in n for n in P.Bundle(out).notes)


def test_render_reports_the_declines_not_just_the_answers(tmp_path):
    """A run that declines most of its input is working as fitted; a caller shown
    only the answered rows would never know."""
    from narrowcast import predict as P
    out, ds, _, _ = _built(tmp_path, ["Sedum acre", "Sedum album", "Bellis annua"],
                           ["Sedum", "Sedum", "Bellis"])
    rows = _rows(["Sedum acre"] * len(ds.X_eval), ["Sedum"] * len(ds.X_eval))
    text = P.render(P.Bundle(out).predict(ds.X_eval), rows, limit=3)
    assert "named to a label" in text and "declined" in text


# ---- multi-modal honesty ---------------------------------------------------

def _mf(labels_, groups_=None, encoder="mobileclip2_s2"):
    """A manifest whose composition is derived from the group map, as `cmd_build`
    now derives it -- a fixture that hardcodes one and varies the other tests a
    combination the tool cannot produce."""
    m = _manifest(0.5)
    gmap = groups_ or {l: l for l in labels_}
    m["labels"] = labels_
    m["encoder"] = encoder
    m["groups"] = gmap
    m["composition"] = labels.analyse(labels_, pool=labels_, groups=gmap)
    m["metrics"]["group_share"] = 0.0
    m["metrics"]["decline_share"] = 0.5
    return m


def test_card_states_no_size_for_an_encoder_it_never_ran():
    """`--embeddings` means the encoder ran elsewhere. Quoting the registry's
    bytes would put a fabricated number on the artifact whose job is being
    checkable — a wav2vec2 audio model was carded as `mobileclip2_s0`, 5.7 MB.

    Every build is now this case: the encoder registry is gone and nothing here
    loads a model, so the card can never state a size."""
    out = card.render(_mf(["yes", "no"], encoder="wav2vec2-base"))
    assert "size not stated" in out and "scored outside this tool" in out
    assert "5.7 MB" not in out and "MB int4" not in out


def test_card_says_when_the_group_rank_is_inert():
    """Single-word labels make the default rule the identity map, so the cascade
    is two-way and the 0% group share is construction rather than measurement."""
    out = card.render(_mf(["yes", "no", "up", "down"]))
    assert "group rank is inert" in out
    assert "all 4 labels are their own group" in out


def test_card_is_silent_when_a_real_coarse_rank_was_supplied():
    out = card.render(_mf(["yes", "no", "up", "down"],
                          {"yes": "g6", "no": "g5", "up": "g5", "down": "g1"}))
    assert "group rank is inert" not in out


def test_inert_check_needs_the_stored_map_and_stays_quiet_without_it():
    """A format 1 bundle has no map; absence is not evidence the rank is inert."""
    m = _mf(["yes", "no"])
    m["groups"] = {}
    assert "group rank is inert" not in card.render(m)


def test_analyse_honours_the_supplied_group_map():
    """Third place the first-whitespace-token default has broken a non-binomial
    domain, after `score_frame` and `predict`. Without the map, `comp.graphics`
    and `comp.sys.mac.hardware` are reported as different groups -- so the card
    asserted "your list is group-crowded" while its own composition block said
    seven groups for seven labels."""
    news = ["comp.graphics", "comp.sys.mac.hardware", "rec.autos"]
    bare = labels.analyse(news, pool=news)
    assert bare["n_groups"] == 3 and bare["crowded_groups"] == {}

    gmap = {"comp.graphics": "comp", "comp.sys.mac.hardware": "comp",
            "rec.autos": "rec"}
    mapped = labels.analyse(news, pool=news, groups=gmap)
    assert mapped["n_groups"] == 2
    assert mapped["crowded_groups"] == {"comp": 2}
    assert mapped["in_set_sibling_frac"] == pytest.approx(2 / 3)


def test_card_does_not_speak_about_plants_or_photographs():
    """The tool is domain-general; a text or audio card that describes
    "photographs you take" and "when the plant is on your list" is wrong and
    costs exactly the trust the card exists to build."""
    out = card.render(_mf(["comp.graphics", "comp.sys.mac.hardware"],
                          {"comp.graphics": "comp", "comp.sys.mac.hardware": "comp"},
                          encoder="all-MiniLM-L6-v2"))
    low = out.lower()
    assert "photograph" not in low and "plant" not in low


def test_crowded_example_names_a_real_group_or_rephrases():
    """With a crowded group to name the card quotes it; without one it used to
    emit the sentence "it is a group", which reads as a bug because it is one."""
    named = card.render(_mf(["comp.graphics", "comp.sys.mac.hardware"],
                            {"comp.graphics": "comp", "comp.sys.mac.hardware": "comp"}))
    assert '"it is a comp"' in named
    bare = card.render(_mf(["alpha", "beta"]))
    assert "it is a group" not in bare


# ---- the audit path: posteriors from a model this tool did not fit ----------

def _scores_npz(tmp_path, n_per=8, ood=0):
    """Two groups of two labels each, plus optional out-of-list rows.

    One out-of-list label shares a group with the list (`near_ood`), the other
    does not (`distant_ood`), so the bucketing rule has both cases to get right.
    """
    classes = np.array(["Alpha one", "Alpha two", "Beta one", "Beta two"])
    lab, grp, clu, rows = [], [], [], []
    for c in classes:
        for k in range(n_per):
            p = np.full(len(classes), 0.05)
            p[list(classes).index(c)] = 0.85
            rows.append(p); lab.append(c); grp.append(c.split()[0])
            clu.append(f"{c}-{k // 2}")
    for k in range(ood):
        near = k % 2 == 0
        rows.append(np.full(len(classes), 0.25))
        lab.append("Alpha absent" if near else "Zeta absent")
        grp.append("Alpha" if near else "Zeta")
        clu.append(f"ood{k // 2}")
    f = tmp_path / "s.npz"
    np.savez(f, proba=np.vstack(rows), classes=classes, label=np.array(lab),
             group=np.array(grp), cluster=np.array(clu))
    return f, classes


def test_scores_path_buckets_out_of_list_rows_by_group(tmp_path):
    """A row whose truth is not among `classes` is out-of-list by construction.
    Whether it is a *relative* follows from its group, and the two are weighted
    differently by `deployment_weights` -- so getting this wrong moves the
    operating point, not just a label."""
    f, _ = _scores_npz(tmp_path, ood=8)
    ds = build.load_scored(sources.from_scores(f))
    assert ds.counts["in_catalog"] == 32
    assert ds.counts["near_ood"] == 4 and ds.counts["distant_ood"] == 4
    assert set(ds.truth[ds.bucket != "in_catalog"]) == {build.OTHER}


def test_scores_path_keeps_the_real_label_as_the_clustering_identity(tmp_path):
    """Out-of-list rows score as __OTHER__ but must not *cluster* as it. Giving
    them one identity leaves `make_splits` a single cluster for the whole bucket
    and fits thresholds on a calibration set with no negatives in it -- the
    mistake that silently broke a published table."""
    f, _ = _scores_npz(tmp_path, ood=8)
    ds = build.load_scored(sources.from_scores(f))
    ood = ds.cluster[ds.bucket != "in_catalog"]
    assert build.OTHER not in set(ood)
    assert len(set(ood)) > 1


def test_audited_and_fitted_models_go_through_the_same_measurement(tmp_path):
    """`frame_from_posteriors` is the one path. If `score_frame` ever stops
    delegating to it, a model we fitted and a model we audited would be measured
    by two different code paths and could drift apart silently."""
    f, classes = _scores_npz(tmp_path, ood=8)
    rows = sources.from_scores(f)
    ds = build.load_scored(rows)
    frame = build.frame_from_posteriors(rows.proba, rows.classes, ds)
    assert set(frame.columns) >= {"label_conf", "group_conf", "label_ok",
                                  "group_ok", "in_catalog", "bucket"}
    # in-list rows put 0.85 on the truth and share a group with one other label
    inl = frame[frame["in_catalog"]]
    assert inl["label_ok"].all() and inl["group_ok"].all()
    assert inl["label_conf"].max() == pytest.approx(0.85)
    assert inl["group_conf"].max() == pytest.approx(0.90)   # 0.85 + its sibling


def test_scores_need_not_sum_to_one(tmp_path):
    """A model that abstains by leaving mass unassigned is still auditable: the
    cascade compares the largest label mass and largest group mass against two
    thresholds, and both survive a positive rescale."""
    f, classes = _scores_npz(tmp_path, ood=4)
    z = dict(np.load(f, allow_pickle=True))
    z["proba"] = z["proba"] * 0.5
    g = tmp_path / "half.npz"
    np.savez(g, **z)
    rows = sources.from_scores(g)
    ds = build.load_scored(rows)
    frame = build.frame_from_posteriors(rows.proba, rows.classes, ds)
    assert frame[frame["in_catalog"]]["label_ok"].all()


def test_scores_reject_a_shape_that_cannot_line_up(tmp_path):
    f = tmp_path / "bad.npz"
    np.savez(f, proba=np.zeros((3, 4)), classes=np.array(["a", "b"]),
             label=np.array(["a", "a", "b"]))
    with pytest.raises(ValueError, match="columns"):
        sources.from_scores(f)


def test_scores_reject_negative_mass(tmp_path):
    f = tmp_path / "neg.npz"
    np.savez(f, proba=np.array([[-0.1, 1.1]]), classes=np.array(["a", "b"]),
             label=np.array(["a"]))
    with pytest.raises(ValueError, match="negative"):
        sources.from_scores(f)


def test_an_audit_with_no_out_of_list_rows_says_so(tmp_path):
    """Coverage is only a measurement of rejection if there is something to
    reject. Without out-of-list rows it is a different number wearing the same
    name, and the card has to be able to say which."""
    f, _ = _scores_npz(tmp_path, ood=0)
    ds = build.load_scored(sources.from_scores(f))
    assert ds.counts["near_ood"] == 0 and ds.counts["distant_ood"] == 0
    assert any("no out-of-list rows" in n for n in ds.counts["notes"])


def test_audited_bundle_carries_no_head_and_predict_refuses_it(tmp_path):
    """We measured someone else's model; we did not obtain a copy of it."""
    from narrowcast import predict as PRED
    f, _ = _scores_npz(tmp_path, ood=8)
    rows = sources.from_scores(f)
    ds = build.load_scored(rows)
    frame = build.frame_from_posteriors(rows.proba, rows.classes, ds)
    metrics = build.fit_and_measure(frame, p_ood=0.2)
    out = build.save_bundle(tmp_path / "b", None, sorted(rows.classes.tolist()),
                            "unstated", metrics, labels.analyse(sorted(rows.classes.tolist())),
                            ds.counts, source=str(f))
    assert not (out / "head.npz").exists()
    assert json.loads((out / "manifest.json").read_text())["has_head"] is False
    with pytest.raises(ValueError, match="did not fit"):
        PRED.Bundle(out)


def test_the_card_does_not_invent_a_training_set_it_was_never_shown(tmp_path):
    """`rows_per_label` is None on an audit. The thin-data warning has no
    denominator, and saying nothing is correct -- counting the evaluation rows
    would describe the wrong set."""
    f, _ = _scores_npz(tmp_path, ood=8)
    rows = sources.from_scores(f)
    ds = build.load_scored(rows)
    frame = build.frame_from_posteriors(rows.proba, rows.classes, ds)
    metrics = build.fit_and_measure(frame, p_ood=0.2)
    chosen = sorted(rows.classes.tolist())
    out = build.save_bundle(tmp_path / "b", None, chosen, "unstated", metrics,
                            labels.analyse(chosen), ds.counts, source=str(f))
    text = card.render(json.loads((out / "manifest.json").read_text()))
    assert "Training rows not known to this tool" in text
    assert "size not stated" in text


# ---- hazard: the caller's group map, and hazards that vanish -----------------

def _haz_frame2(pred_label, pred_group, truth):
    return pd.DataFrame({"pred_label": pred_label, "pred_group": pred_group,
                         "truth": truth, "label": truth})


def test_hazard_groups_come_from_the_supplied_map_not_the_first_token():
    """A foraging list groups by family, so poison hemlock's group is `Apiaceae`.

    Deriving "Conium" from the name means the hazard's own group is never
    recognised, so the one coarse answer that is actually a *warning* --
    "it is an umbellifer" -- gets counted as a dangerous mistake.
    """
    truth = ["Conium maculatum"] * 4
    # every answer is the group, and the group contains the hazard
    te = _haz_frame2(["-"] * 4, ["Apiaceae"] * 4, truth)
    lv = np.array([build.GROUP] * 4)
    groups = {"Conium maculatum": "Apiaceae", "Daucus carota": "Apiaceae"}

    with_map = build.hazard_metrics(te, lv, {"Conium maculatum"}, groups=groups)
    assert with_map["Conium maculatum"]["named_non_hazard"] == 0.0

    # without the map the same answers read as four dangerous errors
    without = build.hazard_metrics(te, lv, {"Conium maculatum"})
    assert without["Conium maculatum"]["named_non_hazard"] == 1.0


def test_a_hazard_with_no_test_rows_is_reported_unmeasured_not_skipped():
    """It used to `continue`, so the label vanished and the card counted the
    survivors and said "all N are under the bar"."""
    te = _haz_frame2(["Daucus carota"], ["Daucus"], ["Daucus carota"])
    lv = np.array([build.LABEL])
    out = build.hazard_metrics(te, lv, {"Conium maculatum"})
    assert "Conium maculatum" in out
    assert out["Conium maculatum"]["unmeasured"] is True
    assert out["Conium maculatum"]["n"] == 0
    assert out["Conium maculatum"]["named_non_hazard"] is None


# ---- declared utility profiles ----------------------------------------------

def test_profiles_are_declared_and_ordered_by_stakes():
    p = cascade.PROFILES
    assert p["standard"] == cascade.UTILITY
    assert (p["identify"]["wrong"] > p["standard"]["wrong"]
            > p["conserve"]["wrong"] > p["forage"]["wrong"])


def test_a_harsher_profile_buys_abstention():
    """`wrong` is the stakes dial: raising its magnitude should not increase the
    share of answers given."""
    rng = np.random.default_rng(0)
    n = 400
    sc = rng.uniform(0.2, 1.0, n)
    ok = rng.uniform(size=n) < sc
    gc = np.clip(sc + 0.1, 0, 1)
    args = (sc, gc, ok, ok, np.ones(n, bool))
    (tg_i, tl_i), _ = cascade.fit_thresholds(*args, weights=cascade.PROFILES["identify"])
    (tg_f, tl_f), _ = cascade.fit_thresholds(*args, weights=cascade.PROFILES["forage"])
    answered_i = (cascade.decide(sc, gc, tg_i, tl_i) != cascade.DECLINE).mean()
    answered_f = (cascade.decide(sc, gc, tg_f, tl_f) != cascade.DECLINE).mean()
    assert answered_f <= answered_i


def test_the_manifest_records_the_payoffs_actually_used(tmp_path):
    """Recording the module default would make a bundle lie about how its
    thresholds were fitted, and the payoffs are the one input that has to stay
    legible after the fact."""
    import json as _json

    from sklearn.linear_model import LogisticRegression
    X = np.random.RandomState(0).normal(size=(40, 4))
    y = np.array(["a"] * 20 + ["b"] * 20)
    clf = LogisticRegression(max_iter=500).fit(X, y)
    out = build.save_bundle(tmp_path / "b", clf, ["a", "b"], "enc", {}, {}, {},
                            source="t", utility=cascade.PROFILES["forage"])
    man = _json.loads((out / "manifest.json").read_text())
    assert man["utility"]["wrong"] == -20.0


# ---- the forager's threat model: a hazard deliberately NOT on the list -------

def _absent_frame(pred_label, pred_group, real, in_cat):
    """`species` carries the real identity; `label` is the clustering column and
    may be an arbitrary id, which is exactly why `species` had to be added."""
    return pd.DataFrame({"pred_label": pred_label, "pred_group": pred_group,
                         "truth": [build.OTHER] * len(real), "species": real,
                         "label": [f"cl{i}" for i in range(len(real))],
                         "in_catalog": in_cat})


def test_any_in_list_name_for_an_unlisted_hazard_is_dangerous():
    """Every listed label is something the user means to use, so unlike the
    in-list case there is no "named as another hazard" escape."""
    real = ["Conium maculatum"] * 4
    te = _absent_frame(["Daucus carota"] * 4, ["Apiaceae"] * 4, real, [False] * 4)
    lv = np.array([build.LABEL] * 4)
    out = build.outside_hazard_metrics(te, lv, {"Conium maculatum"})
    assert out["Conium maculatum"]["dangerous"] == 1.0
    assert out["Conium maculatum"]["named_as"] == {"Daucus carota": 4}


def test_a_group_answer_naming_the_hazards_own_group_is_a_warning_not_an_error():
    """"It is an umbellifer" warns the person holding the root. "It is a Lomatium"
    is a species-level claim wearing the clothes of caution."""
    real = ["Conium maculatum"] * 4
    groups = {"Conium maculatum": "Apiaceae"}
    te = _absent_frame(["-"] * 4, ["Apiaceae"] * 4, real, [False] * 4)
    lv = np.array([build.GROUP] * 4)
    safe = build.outside_hazard_metrics(te, lv, {"Conium maculatum"}, groups=groups)
    assert safe["Conium maculatum"]["dangerous"] == 0.0
    assert safe["Conium maculatum"]["warned_at_group"] == 1.0

    te2 = _absent_frame(["-"] * 4, ["Lomatium"] * 4, real, [False] * 4)
    unsafe = build.outside_hazard_metrics(te2, lv, {"Conium maculatum"}, groups=groups)
    assert unsafe["Conium maculatum"]["dangerous"] == 1.0


def test_declining_an_unlisted_hazard_is_always_safe():
    real = ["Conium maculatum"] * 3
    te = _absent_frame(["-"] * 3, ["-"] * 3, real, [False] * 3)
    lv = np.array([build.DECLINE] * 3)
    out = build.outside_hazard_metrics(te, lv, {"Conium maculatum"})
    assert out["Conium maculatum"]["dangerous"] == 0.0
    assert out["Conium maculatum"]["declined"] == 1.0


def test_in_list_rows_are_not_counted_for_an_absent_hazard():
    """Only out-of-list rows can be this kind of error."""
    real = ["Conium maculatum"] * 2 + ["Daucus carota"] * 2
    te = _absent_frame(["Daucus carota"] * 4, ["Apiaceae"] * 4, real,
                       [False, False, True, True])
    lv = np.array([build.LABEL] * 4)
    out = build.outside_hazard_metrics(te, lv, {"Conium maculatum"})
    assert out["Conium maculatum"]["n"] == 2


def test_an_absent_hazard_with_no_rows_is_unmeasured_not_passed():
    te = _absent_frame(["x"], ["y"], ["Daucus carota"], [True])
    lv = np.array([build.DECLINE])
    out = build.outside_hazard_metrics(te, lv, {"Conium maculatum"})
    assert out["Conium maculatum"]["unmeasured"] is True


def test_card_reports_an_unmeasured_absent_hazard_rather_than_staying_quiet():
    """A declared risk with no data must not read as a passed check."""
    from narrowcast.card import _absent_hazard_section
    txt = "\n".join(_absent_hazard_section(
        {"Conium maculatum": {"unmeasured": True, "n": 0, "dangerous": None}}))
    assert "none of this was measured" in txt
    assert "not a passed check" in txt


def test_card_fires_on_an_absent_hazard_over_the_bar():
    from narrowcast.card import _absent_hazard_section
    txt = "\n".join(_absent_hazard_section({"Conium maculatum": {
        "unmeasured": False, "n": 100, "dangerous": 0.045, "named_in_list": 0.045,
        "warned_at_group": 0.0, "declined": 0.955}}))
    assert "⚠" in txt and "4.5%" in txt
    assert "--ood-rate" in txt


# ---- a declared hazard must not reach the test half by coin flip -------------

_HZ_GENERA = ["Conium", "Cicuta", "Daucus", "Anthriscus", "Foeniculum", "Heracleum",
              "Pastinaca", "Torilis", "Osmorhiza", "Sanicula", "Angelica",
              "Ligusticum", "Perideridia", "Cynoglossum", "Digitalis", "Aconitum",
              "Delphinium", "Veratrum", "Nicotiana", "Datura", "Solanum",
              "Ranunculus", "Aquilegia"]


def _near_ood_frame(n_obs=6):
    """The shape that produced the failure: 23 unlisted species, each its own
    genus, in the bucket `make_splits` keys on the *group*. `label` is the
    clustering id (one plant, several photographs), as it is in a real frame."""
    rec = [{"bucket": "near_ood", "label": f"{g} sp1-obs{o}", "group": g,
            "species": f"{g} sp1", "truth": build.OTHER, "in_catalog": False}
           for g in _HZ_GENERA for o in range(n_obs) for _ in range(2)]
    rec += [{"bucket": "in_catalog", "label": f"in{i}", "group": f"G{i % 5}",
             "species": f"in{i}", "truth": f"in{i}", "in_catalog": True}
            for i in range(20)]
    return pd.DataFrame(rec)


def _reaches(df, target, hazards, fold="test", seeds=8):
    return sum(bool((cascade.make_splits(df, seed=s, hazards=hazards) == fold)[
        df["species"] == target].any()) for s in range(seeds))


def _e2e_frame(n_genera=12, n_obs=4):
    """A frame `fit_and_measure` will accept, with enough near-OOD genera that the
    genus-keyed shuffle can put any one of them wholly in the calibration half."""
    rng = np.random.default_rng(0)
    base = _cascade_frame(label_ok=True, group_ok=True)
    base["pred_label"] = base["label"]
    base["pred_group"] = base["group"]
    base["truth"] = np.where(base["in_catalog"], base["label"], build.OTHER)
    base["species"] = base["label"]
    near = [{"label_conf": float(rng.uniform(0.0, 0.5)),
             "group_conf": float(rng.uniform(0.0, 0.5)),
             "label_ok": False, "group_ok": False, "in_catalog": False,
             "bucket": "near_ood", "label": f"{g} sp1-obs{o}", "group": g,
             "pred_label": "G0 sp0", "pred_group": "G0", "truth": build.OTHER,
             "species": f"{g} sp1"}
            for g in (["Conium"] + [f"Rel{i}" for i in range(n_genera - 1)])
            for o in range(n_obs) for _ in range(2)]
    return pd.concat([base, pd.DataFrame(near)], ignore_index=True)

def test_an_undeclared_hazard_reaches_the_test_half_only_by_coin_flip():
    """The failure this fixes, pinned so it cannot come back as a default. With
    23 near-OOD genera no single one exceeds `MAX_CLUSTER_SHARE`, so `_too_coarse`
    is content and the genus stays the key -- the shuffle then decides. Measured
    on the real Oregon list: Conium in 6 of 8 splits, Cicuta in 2 of 8."""
    df = _near_ood_frame()
    assert 1 <= _reaches(df, "Conium sp1", hazards=None) <= 7


def test_a_declared_hazard_lands_in_both_halves_at_every_seed():
    """Not "usually measured". A card whose safety section depends on the seed is
    a coin flip wearing a measurement's clothes."""
    df = _near_ood_frame()
    hz = {"Conium sp1", "Cicuta sp1"}
    for target in hz:
        assert _reaches(df, target, hz, fold="test") == 8
        # calib too: forcing it wholly into test would pull negatives out of the
        # calibration set, which is the mistake `load_scored` records as having
        # silently broken a published table.
        assert _reaches(df, target, hz, fold="calib") == 8


def test_declaring_a_hazard_moves_no_other_row():
    """Stratification draws from its own generator, so `--hazard` re-assigns that
    hazard's rows and nothing else. Note the scope: *split assignment* is what is
    stable. The thresholds fitted from it still move, because the hazard's own
    rows changed sides and the calibration composition changed with them."""
    df = _near_ood_frame()
    hz = {"Conium sp1", "Cicuta sp1"}
    other = ~df["species"].isin(hz).to_numpy()
    for seed in range(8):
        plain = cascade.make_splits(df, seed=seed)
        with_hz = cascade.make_splits(df, seed=seed, hazards=hz)
        assert (plain[other] == with_hz[other]).all()


def test_one_hazards_halving_does_not_depend_on_another_being_declared():
    """Each hazard is keyed on its own name, so declaring Cicuta cannot change
    which of Conium's observations are held out."""
    df = _near_ood_frame()
    alone = cascade.make_splits(df, seed=0, hazards={"Conium sp1"})
    with_other = cascade.make_splits(df, seed=0,
                                     hazards={"Conium sp1", "Cicuta sp1"})
    m = (df["species"] == "Conium sp1").to_numpy()
    assert (alone[m] == with_other[m]).all()
    assert set(alone[m]) == {"calib", "test"}


def test_a_single_cluster_hazard_goes_wholly_to_test_rather_than_straddling():
    """Cluster, never row -- the one convention that does not bend for this. All
    its rows are one plant, so it is measured without an interval instead of
    being split down the middle of a cluster."""
    df = _near_ood_frame()
    df.loc[df["species"] == "Cicuta sp1", "label"] = "Cicuta sp1-single"
    fold = cascade.make_splits(df, seed=3, hazards={"Cicuta sp1"})
    assert set(fold[df["species"] == "Cicuta sp1"]) == {"test"}


def test_a_hazard_with_no_rows_leaves_the_split_untouched():
    """Declaring a hazard the source never saw is not an error here -- the metrics
    report it unmeasured. It must not perturb the split on the way."""
    df = _near_ood_frame()
    assert (cascade.make_splits(df, seed=0, hazards={"Not present"})
            == cascade.make_splits(df, seed=0)).all()


def test_both_threat_models_find_their_hazard_through_one_predicate():
    """`hazard_metrics` keys on `truth`, `outside_hazard_metrics` on `species` and
    out-of-list. A hazard stratified under one predicate and measured under the
    other still reports "unmeasured", and the miss is invisible. One helper, used
    by the split and by both metrics."""
    df = pd.DataFrame({
        "truth": ["Amanita phalloides", build.OTHER, build.OTHER],
        "species": ["Amanita phalloides", "Conium maculatum", "Daucus carota"],
        "in_catalog": [True, False, False],
    })
    assert list(cascade.hazard_rows(df, "Amanita phalloides")) == [True, False, False]
    assert list(cascade.hazard_rows(df, "Conium maculatum")) == [False, True, False]


def test_card_survives_an_unmeasured_consequential_label():
    """`hazard_metrics` records a hazard with no test rows rather than dropping it
    — and the section that reads those records compared `None > 0.01` and raised.
    The crash was reachable from the exact case the recording exists for."""
    from narrowcast.card import _hazard_section
    txt = "\n".join(_hazard_section({
        "Conium maculatum": {"n": 0, "declined": None, "named_correctly": None,
                             "named_other_hazard": None, "named_non_hazard": None,
                             "ci": None, "unmeasured": True},
        "Cicuta douglasii": {"n": 40, "declined": 0.9, "named_correctly": 0.1,
                             "named_other_hazard": 0.0, "named_non_hazard": 0.0,
                             "ci": [0.0, 0.05], "unmeasured": False}}))
    assert "Conium maculatum" in txt
    assert "1 declared and not measured" in txt
    # and it must not count the missing one among those that passed
    assert "All 1 consequential labels are under" in txt


def test_card_does_not_call_a_wholly_unmeasured_gate_a_pass():
    from narrowcast.card import _hazard_section
    txt = "\n".join(_hazard_section({"Conium maculatum": {
        "n": 0, "named_non_hazard": None, "unmeasured": True}}))
    assert "none of this was measured" in txt
    assert "not a passed check" in txt


def test_card_discloses_that_hazards_were_forced_into_both_halves():
    """The bucket counts otherwise read as a plain sample, and after
    stratification they are not one."""
    man = _manifest(0.85)
    man["metrics"]["per_bucket"]["near_ood"] = {
        "n": 50, "answered": 0.2, "correct_when_answered": 0.5}
    man["metrics"]["hazard_absent"] = {
        "Conium maculatum": {"unmeasured": True, "n": 0, "dangerous": None}}
    assert "forced into both halves" in card.render(man)
    # and stays quiet when nothing was declared, because then it is a plain sample
    assert "forced into both halves" not in card.render(_manifest(0.85))


def test_a_declared_hazard_is_measured_end_to_end_at_every_seed():
    """The whole point, through `fit_and_measure` rather than `make_splits`: the
    stratified rows must land where the metric actually looks for them."""
    df = _e2e_frame()
    # Undeclared, the shuffle decides. Asserted first so that if the fixture ever
    # stops being able to miss, this test fails loudly instead of passing vacuously.
    def _missed(seed):
        te = df[cascade.make_splits(df, seed=seed) == "test"]
        lv = np.full(len(te), build.DECLINE)
        return build.outside_hazard_metrics(te, lv, {"Conium sp1"}
                                            )["Conium sp1"]["unmeasured"]

    assert any(_missed(s) for s in range(6)), \
        "the unstratified split never missed it; nothing is pinned"

    for seed in range(6):
        m = build.fit_and_measure(df, p_ood=0.2, seed=seed,
                                  hazards_absent=["Conium sp1"])
        assert m["hazard_absent"]["Conium sp1"]["unmeasured"] is False, seed
        assert m["hazard_absent"]["Conium sp1"]["n"] > 0


# ---- per-label cost: labels the model will never answer ----------------------

def test_suppression_declines_the_label_and_spares_an_informative_group():
    """The look-alike, not the hazard. And "it is an umbellifer" is a warning the
    forager wanted, so a group answer survives while the label answer does not."""
    lv = np.array([build.LABEL, build.GROUP, build.LABEL], dtype=object)
    pred = np.array(["Daucus carota", "Daucus carota", "Osmorhiza berteroi"])
    pgen = np.array(["Apiaceae", "Apiaceae", "Apiaceae"])
    members = {"Apiaceae": {"Daucus carota", "Osmorhiza berteroi"}}
    out = cascade.suppress(lv, pred, {"Daucus carota"}, pgen, members)
    assert list(out) == [build.DECLINE, build.GROUP, build.LABEL]


def test_a_group_holding_only_suppressed_labels_is_suppressed_too():
    """Grouped by genus with one listed Daucus, "it is a *Daucus*" is the
    suppressed claim under another name. Grouped by family it is not."""
    lv = np.array([build.GROUP], dtype=object)
    args = (np.array(["Daucus carota"]), {"Daucus carota"}, np.array(["Daucus"]))
    assert list(cascade.suppress(lv, *args, {"Daucus": {"Daucus carota"}})) \
        == [build.DECLINE]
    assert list(cascade.suppress(lv, *args,
                                 {"Daucus": {"Daucus carota", "Daucus pusillus"}})) \
        == [build.GROUP]


def test_suppression_without_a_group_map_still_suppresses_labels():
    lv = np.array([build.LABEL, build.GROUP], dtype=object)
    out = cascade.suppress(lv, np.array(["a", "a"]), {"a"})
    assert list(out) == [build.DECLINE, build.GROUP]


def test_no_suppression_is_the_identity():
    lv = np.array([build.LABEL, build.GROUP, build.DECLINE], dtype=object)
    assert list(cascade.suppress(lv, np.array(["a", "b", "c"]), None)) == list(lv)
    assert list(cascade.suppress(lv, np.array(["a", "b", "c"]), set())) == list(lv)


def test_the_threshold_fit_does_not_see_the_suppression():
    """Keeping the override out of `fit_thresholds` is what makes its cost a
    measurable delta instead of something the operating point absorbs. If someone
    folds it into the fit, the thresholds move and this fails."""
    df = _e2e_frame()
    plain = build.fit_and_measure(df, p_ood=0.2)
    sup = build.fit_and_measure(df, p_ood=0.2, never_answer=["G0 sp0"],
                                labels=sorted(set(df.loc[df["in_catalog"], "label"])))
    assert sup["t_label"] == plain["t_label"]
    assert sup["t_group"] == plain["t_group"]


def test_the_card_prints_what_the_suppression_cost():
    """A per-label safety dial whose price is not stated is the kind of number
    this tool exists to refuse."""
    df = _e2e_frame()
    labels = sorted(set(df.loc[df["in_catalog"], "label"]))
    # a label the model actually answers *in the test half*, so the assertion is
    # about the suppression rather than about which half the split chose
    te = df[cascade.make_splits(df, seed=0) == "test"]
    victim = te.loc[te["in_catalog"], "pred_label"].value_counts().index[0]
    m = build.fit_and_measure(df, p_ood=0.2, never_answer=[victim], labels=labels)
    s = m["suppression"]
    assert s["labels"] == [victim]
    assert s["label_share_with"] <= s["label_share_without"]
    assert s["answers_removed"] > 0
    man = _manifest(0.85)
    man["metrics"]["suppression"] = s
    txt = card.render(man)
    assert "never answer" in txt.lower()
    assert victim in txt


def test_the_card_points_at_the_lookalike_not_the_consequential_label():
    """It used to advise treating the consequential labels as always-decline,
    which is inert: the rate counts rows named as something *else*, so those rows
    were never in the numerator. Shipping `--never-answer` beside that advice
    would send the reader to do the thing that does not work."""
    from narrowcast.card import _hazard_section
    txt = "\n".join(_hazard_section({"Conium maculatum": {
        "n": 40, "declined": 0.1, "named_correctly": 0.5,
        "named_other_hazard": 0.0, "named_non_hazard": 0.4,
        "named_as": {"Daucus carota": 12, "Osmorhiza berteroi": 4},
        "ci": None, "unmeasured": False}}))
    assert '--never-answer "Daucus carota"' in txt
    assert "Suppressing the consequential label itself does nothing" in txt


def test_the_in_list_path_records_what_the_hazard_was_named():
    """Without `named_as` the card can say the gate failed but not which label to
    suppress, which makes the remedy unusable on the oldest path."""
    truth = ["Conium maculatum"] * 4
    pred = ["Daucus carota", "Daucus carota", "Foeniculum vulgare", "Conium maculatum"]
    te = _haz_frame(pred, [p.split()[0] for p in pred], truth)
    h = build.hazard_metrics(te, np.array([build.LABEL] * 4),
                             {"Conium maculatum"})["Conium maculatum"]
    assert h["named_as"] == {"Daucus carota": 2, "Foeniculum vulgare": 1}


def test_a_bundle_predicts_under_the_suppression_the_card_was_measured_with(tmp_path):
    """The contract is that a prediction and the card cannot disagree. A manifest
    field written but never read would pass every other test here."""
    from narrowcast import predict as P
    labels = [f"G{i // 3} sp{i}" for i in range(6)]
    rows = _rows(labels * 8, [l.split()[0] for l in labels] * 8, dim=16)
    ds = build.load_rows(rows, "test-encoder")
    clf = build.fit_head(ds)
    frame = build.score_frame(clf, ds)
    victim = sorted(rows.labels)[0]
    metrics = build.fit_and_measure(frame, p_ood=0.1, never_answer=[victim],
                                    labels=rows.labels)
    out = build.save_bundle(tmp_path / "b", clf, rows.labels, "test-encoder",
                            metrics, {}, ds.counts, source="t",
                            never_answer=[victim])
    b = P.Bundle(out)
    assert b.never_answer == {victim}
    X = np.asarray(rows.descriptor, dtype="float32")
    res = b.predict(X)
    assert not any(r["answer"] == victim for r in res)
    assert all(r["rank"] != build.LABEL
               for r in res if r["label"] == victim)
    assert any("suppressed at predict time" in n for n in b.notes)


def test_cli_refuses_to_suppress_a_label_that_is_not_on_the_list(tmp_path):
    """You can only suppress what the model can emit. The sibling flags refuse
    their own wrong side too — a flag that silently does nothing is worse than one
    that will not start.

    A real source file, so the run reaches the validation rather than dying on the
    npz and passing for the wrong reason."""
    import subprocess, sys as _sys
    f, classes = _scores_npz(tmp_path, ood=4)
    def run(label):
        return subprocess.run(
            [_sys.executable, "-m", "narrowcast.cli", "audit", "--scores", str(f),
             "--out", str(tmp_path / "b"), "--never-answer", label],
            capture_output=True, text=True)

    bad = run("Conium maculatum")
    assert bad.returncode != 0
    assert "not in the label set" in bad.stderr

    ok = run(str(classes[0]))
    assert ok.returncode == 0, ok.stderr


def test_measuring_a_suppression_without_the_label_set_is_refused():
    """`_group_members` needs the label set; without it the measurement suppresses
    label answers only while `predict` also kills hollow-group answers — two
    different models from one bundle."""
    df = _e2e_frame()
    with pytest.raises(ValueError, match="needs `labels`"):
        build.fit_and_measure(df, p_ood=0.2, never_answer=["G0 sp0"])


# ---- the third threshold: retreat when the row looks out-of-list -------------

def test_the_gate_declines_unconditionally_whatever_the_other_two_said():
    """The one place the three-way order can silently invert. The gate is applied
    last and declines outright: a gated row must not come back as a group answer
    just because its group score was high."""
    confident = cascade.decide(np.array([0.99]), np.array([0.99]),
                               t_group=0.5, t_label=0.5,
                               novelty=np.array([0.1]), t_novel=0.9)
    assert list(confident) == [build.DECLINE]
    already_declining = cascade.decide(np.array([0.9]), np.array([0.1]),
                                       t_group=0.5, t_label=0.5,
                                       novelty=np.array([0.0]), t_novel=0.9)
    assert list(already_declining) == [build.DECLINE]


def test_the_retreat_arm_could_not_pay_here_which_is_why_the_gate_rejects():
    """plantid preferred retreating to the group rank. That arm is arithmetically
    dead in narrowcast: no out-of-list row scores correct at any rank, so
    retreating one moves it from `wrong` to `wrong`. Declining is worth
    `decline_ood` instead. Pinned so the "improvement" back to retreat is caught."""
    u = cascade.utility([build.LABEL, build.GROUP, build.DECLINE],
                        label_ok=[False] * 3, group_ok=[False] * 3,
                        in_catalog=[False] * 3)
    assert u[0] == u[1] == cascade.UTILITY["wrong"]      # retreating changes nothing
    assert u[2] == cascade.UTILITY["decline_ood"] > u[1]  # declining is the lever


def test_without_a_gate_decide_is_exactly_what_it_was():
    rng = np.random.default_rng(0)
    lc, gc = rng.random(200), rng.random(200)
    base = cascade.decide(lc, gc, 0.4, 0.6)
    assert list(cascade.decide(lc, gc, 0.4, 0.6, novelty=None, t_novel=None)) \
        == list(base)
    # and a threshold at or below the minimum gates nothing
    nov = rng.random(200)
    assert list(cascade.decide(lc, gc, 0.4, 0.6, nov, nov.min())) == list(base)


def test_a_reject_class_in_the_posteriors_is_not_one_of_the_users_labels(tmp_path):
    """`__OTHER__` is what the model says when it means none of these. Counting it
    among the labels put it on the card, fed it to `analyse` as a group member,
    and would have let `--never-answer "__OTHER__"` validate."""
    import subprocess, sys as _sys
    rng = np.random.default_rng(0)
    classes = np.array(["Sedum acre", "Sedum album", "Bellis annua", build.OTHER])
    lab = np.repeat(classes[:3], 12)
    proba = rng.random((len(lab), 4))
    f = tmp_path / "s.npz"
    np.savez(f, proba=proba / proba.sum(1, keepdims=True), classes=classes,
             label=lab, group=np.array([l.split()[0] for l in lab]),
             cluster=np.array([f"c{i // 2}" for i in range(len(lab))]))
    r = subprocess.run([_sys.executable, "-m", "narrowcast.cli", "audit",
                        "--scores", str(f), "--out", str(tmp_path / "b")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "3 labels" in r.stderr, r.stderr
    man = json.loads((tmp_path / "b" / "manifest.json").read_text())
    assert build.OTHER not in man["labels"]


def _gate_frame(n_in=10, n_near=8, obs=3):
    """The configuration the gate exists for, and the premise of the finding it
    comes from: the closed-set confidences **cannot separate the buckets** — a
    posterior over the user's labels has no way to say "none of these" — so the
    near-OOD rows are named exactly as confidently as the in-list ones. Only the
    reject mass tells them apart. If the confidences did separate them, `t_group`
    would already be doing this job and the gate would have nothing to add."""
    rng = np.random.default_rng(0)
    rec = []
    for i in range(n_in):
        lab = f"G{i // 2} sp{i}"
        for o in range(obs):
            for _ in range(2):
                rec.append({"label_conf": 0.90 + rng.uniform(0, .08),
                            "group_conf": 0.93 + rng.uniform(0, .06),
                            "label_ok": True, "group_ok": True, "in_catalog": True,
                            "bucket": "in_catalog", "label": f"{lab}-o{o}",
                            "group": lab.split()[0], "species": lab, "truth": lab,
                            "pred_label": lab, "pred_group": lab.split()[0],
                            "novelty": 0.90 + rng.uniform(0, .09)})
    for k in range(n_near):
        sp = f"G{k % 3} rel{k}"
        for o in range(obs):
            for _ in range(2):
                rec.append({"label_conf": 0.90 + rng.uniform(0, .08),
                            "group_conf": 0.93 + rng.uniform(0, .06),
                            "label_ok": False, "group_ok": False, "in_catalog": False,
                            "bucket": "near_ood", "label": f"{sp}-o{o}",
                            "group": f"G{k % 3}", "species": sp, "truth": build.OTHER,
                            "pred_label": f"G{k % 3} sp0", "pred_group": f"G{k % 3}",
                            "novelty": 0.25 + rng.uniform(0, .15)})
    return pd.DataFrame(rec)


def test_the_gate_fires_and_pays_when_the_reject_mass_separates_the_buckets():
    """Without this the suite cannot tell "the fit correctly turned it off" from
    "the gate never works" — which is exactly how the dead retreat arm survived
    two fixtures."""
    m = build.fit_and_measure(_gate_frame(), p_ood=0.05, gate=True)
    g = m["novelty_gate"]
    assert g["fitted"] and not g["fit_turned_it_off"]
    assert g["calib_utility_gained"] > 0
    assert g["rows_declined"] > 0
    assert g["near_ood_wrong_gated"] < g["near_ood_wrong_ungated"]
    # and on this frame it removes exactly the rows that were wrong: near-OOD
    # errors go to zero while the in-list label share does not move at all
    assert g["near_ood_wrong_gated"] == 0.0
    assert m["label_share"] == g["label_share_ungated"]


def test_the_gate_has_nothing_to_add_where_the_fit_already_declines_everything():
    """The cost of fitting it as a greedy second stage, made concrete. The gate
    helps where stage one *answers* too much. Where the declared payoffs already
    push stage one to decline nearly everything — a high `p_ood` with buckets the
    closed-set scores cannot separate — there is no error left for it to remove,
    and it correctly reports a gain of zero rather than adding declines."""
    g = build.fit_and_measure(_gate_frame(), p_ood=0.3, gate=True)["novelty_gate"]
    assert g["fit_turned_it_off"] is True
    assert g["rows_declined"] == 0


def test_the_gate_never_scores_below_the_ungated_fit():
    """The sweep starts at the minimum of novelty, and a strict `<` against the
    minimum gates no row — so the fit can always choose the baseline. That is what
    makes fitting this as a greedy second stage safe."""
    noise = _e2e_frame()
    # novelty that carries no signal about the buckets: the gate must still not
    # be able to lose, it simply finds nothing worth thresholding
    noise["novelty"] = np.random.default_rng(0).uniform(0.2, 1.0, len(noise))
    for frame in (_gate_frame(), noise):
        for p in (0.05, 0.1, 0.2, 0.3):
            m = build.fit_and_measure(frame, p_ood=p, gate=True)
            assert m["novelty_gate"]["calib_utility_gained"] >= 0.0


def test_a_frame_with_no_reject_class_reports_the_gate_as_unreadable():
    """Constant novelty means the posteriors carry no out-of-list class. Saying so
    beats reporting a threshold that read a constant."""
    m = build.fit_and_measure(_e2e_frame().assign(novelty=1.0), p_ood=0.3, gate=True)
    assert m["novelty_gate"]["fitted"] is False
    assert "no out-of-list class" in m["novelty_gate"]["reason"]
    assert m["t_novel"] is None


def test_no_gate_requested_leaves_the_decision_byte_identical():
    """The whole point of staging: asking for no gate must produce exactly the
    numbers the tool produced before there was one."""
    f = _gate_frame()
    a = build.fit_and_measure(f, p_ood=0.3)
    b = build.fit_and_measure(f, p_ood=0.3, gate=True)
    assert a["t_group"] == b["t_group"] and a["t_label"] == b["t_label"]
    assert a["novelty_gate"] is None and a["t_novel"] is None


def test_a_gate_the_fit_turned_off_is_not_written_into_the_bundle():
    """A threshold at the bottom of the calibration grid gates nothing *there* and
    can still catch a test row below that minimum. `predict` would then apply a
    decline rule the fit had explicitly declined to adopt."""
    m = build.fit_and_measure(_gate_frame(), p_ood=0.3, gate=True)
    assert m["novelty_gate"]["fit_turned_it_off"] is True
    assert m["t_novel"] is None


def test_the_card_does_not_call_a_turned_off_gate_a_win():
    """plantid measured this gate as a utility null and did not ship it. A card
    that implied otherwise would overstate a result its own source declines."""
    from narrowcast.card import _gate_section
    off = "\n".join(_gate_section({"fitted": True, "fit_turned_it_off": True,
                                   "t_novel": 0.1, "calib_utility_gained": 0.0}))
    assert "turned off by the fit" in off and "utility null" in off
    unread = "\n".join(_gate_section({"fitted": False, "reason": "no out-of-list class"}))
    assert "not fitted" in unread
    on = "\n".join(_gate_section({
        "fitted": True, "fit_turned_it_off": False, "t_novel": 0.94,
        "calib_utility_gained": 0.12, "rows_declined": 27,
        "near_ood_wrong_ungated": 0.516, "near_ood_wrong_gated": 0.094,
        "label_share_ungated": 0.981}))
    assert "51.6% → 9.4%" in on
    assert "declines** rather than retreating" in on


def test_predict_applies_the_gate_and_says_so(tmp_path):
    """Fourth place in the seam. A `t_novel` in the manifest that `predict`
    ignored would leave the card describing a model nobody runs."""
    from narrowcast import predict as P
    rng = np.random.default_rng(4)
    cent = {f"G{i // 2} sp{i}": rng.normal(size=24) for i in range(6)}
    def blk(names, n, scale):
        v, l, g, c = [], [], [], []
        for nm in names:
            mu = cent.get(nm, rng.normal(scale=3.0, size=24))
            for o in range(n):
                v.append(mu + rng.normal(scale=scale, size=24))
                l.append(nm); g.append(nm.split()[0]); c.append(f"{nm}-o{o}")
        return sources._finish(l, descriptor=np.array(v, "float32"), group=g, cluster=c)
    fg = blk(list(cent), 10, 0.3)
    bg = blk([f"Far{i} sp{i}" for i in range(8)], 8, 0.5)
    ds = build.load_rows(fg, "enc", background=bg)
    assert build.OTHER in set(ds.y_train.tolist()), "no reject class to gate with"
    clf = build.fit_head(ds)
    metrics = build.fit_and_measure(build.score_frame(clf, ds), p_ood=0.1, gate=True)
    # The fit is free to turn the gate off on an easy fixture, and on this one it
    # does. What is under test here is the *seam* — that a threshold in the
    # manifest reaches the decision — so it is set explicitly. Whether the fit
    # chooses one is pinned separately, on a frame where it pays.
    assert metrics["novelty_gate"]["fitted"]
    nov = np.sort(np.concatenate([
        build.score_frame(clf, ds)["novelty"].to_numpy()]))
    forced = float(nov[len(nov) // 2])
    out = build.save_bundle(tmp_path / "b", clf, fg.labels, "enc",
                            {**metrics, "t_novel": forced}, {}, ds.counts, source="t")

    b = P.Bundle(out)
    assert b.t_novel == forced
    assert any("declined outright" in n for n in b.notes)
    res = b.predict(np.vstack([np.asarray(fg.descriptor, "float32"),
                               np.asarray(bg.descriptor, "float32")]))
    low = [r for r in res if r["novelty"] < forced]
    high = [r for r in res if r["novelty"] >= forced]
    assert low and high, "the threshold splits nothing; the assertions are vacuous"
    assert all(r["rank"] == build.DECLINE for r in low)
    assert any(r["rank"] != build.DECLINE for r in high)


def test_a_gate_the_fit_turned_off_changes_no_number_on_the_card():
    """The bug this pins: the turned-off reset used to run *after* every metric
    had been computed from the gated decision. A `t_novel` at the bottom of the
    calibration grid gates nothing there and can still catch a test row below that
    minimum, so the card could report a decline rule the manifest denied and
    `predict` would not run."""
    f = _gate_frame()
    for p in (0.2, 0.3, 0.4, 0.5):
        gated = build.fit_and_measure(f, p_ood=p, gate=True)
        if not gated["novelty_gate"]["fit_turned_it_off"]:
            continue
        plain = build.fit_and_measure(f, p_ood=p)
        for k in ("coverage", "precision", "label_share", "group_share",
                  "decline_share", "closed_set_top1", "t_group", "t_label"):
            assert gated[k] == plain[k], f"{k} moved at p_ood={p}"
        assert gated["per_bucket"] == plain["per_bucket"]
        assert gated["ci"] == plain["ci"]
        assert gated["t_novel"] is None


def test_the_gates_declines_are_not_charged_to_the_suppression():
    """Both flags at once. The suppression delta is measured against the decision
    *including* the gate, or the gate's declines get billed to `--never-answer`
    and the card prints a cost the suppression did not incur."""
    f = _gate_frame()
    labs = sorted(set(f.loc[f["in_catalog"], "truth"]))
    both = build.fit_and_measure(f, p_ood=0.1, gate=True, labels=labs,
                                 never_answer=[labs[0]])
    only_gate = build.fit_and_measure(f, p_ood=0.1, gate=True)
    assert both["novelty_gate"]["rows_declined"] == \
        only_gate["novelty_gate"]["rows_declined"]
    # every answer the suppression is charged with must name a suppressed label
    assert both["suppression"]["answers_removed"] >= 0
    assert both["suppression"]["answers_removed"] <= int(
        (f["pred_label"] == labs[0]).sum())


# ---- Phase 3 leftovers: regional bucket, encoder binding, head dtype ---------

def test_the_head_is_float32_on_disk(tmp_path):
    """`_vecs` casts descriptors to float32 and sklearn keeps the dtype, so this
    is already what the fit produces — pinned so an upstream change cannot
    silently double every head on disk."""
    out, ds, _, _ = _built(tmp_path, ["Sedum acre", "Sedum album"], ["Sedum"] * 2)
    z = np.load(out / "head.npz")
    assert z["coef"].dtype == np.float32 and z["intercept"].dtype == np.float32


def test_two_pools_from_different_encoders_are_flagged_not_silently_measured():
    """The failure this exists for was silent and cost three points: negatives
    living in an unrelated space are trivially rejected, so label share came out
    flattered and the number went into a table. The declared `encoder` string
    cannot catch it — both pools carry the same model name.

    A warning rather than a refusal, because the premise (embeddings from one
    encoder share a common cone) is not validated here — nothing in the package
    can load an encoder to check it against."""
    rng = np.random.default_rng(0)
    d = 256
    base = rng.normal(size=(8, d))
    def pool(n, rot=None):
        X = np.vstack([base[i % 8] + rng.normal(scale=0.6, size=d) for i in range(n)])
        if rot is not None:
            X = X @ rot
        return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype("float32")
    fg = pool(200)
    other, _ = np.linalg.qr(rng.normal(size=(d, d)))
    assert build.check_same_space(fg, pool(120, other)), "rotation not flagged"
    # a shared cone is what makes the test discriminate; without one it cannot,
    # and the docstring says so rather than the code pretending otherwise
    cone = rng.normal(size=d)
    same = np.vstack([pool(120) + 3 * cone, ])
    assert not build.check_same_space(np.vstack([fg + 3 * cone]), same)


def test_a_different_dimension_is_refused_because_that_one_is_certain():
    """The asymmetry: different widths are proof and are refused; orthogonal
    geometry rests on an unvalidated premise and only warns."""
    rng = np.random.default_rng(0)
    with pytest.raises(SystemExit, match="cannot be from one encoder"):
        build.check_same_space(rng.normal(size=(20, 64)), rng.normal(size=(20, 128)))


def test_predict_flags_vectors_from_another_space(tmp_path):
    """Fourth seam again: the bundle stores the direction its training vectors
    pointed in, so `predict` can make the same refusal `audit` makes."""
    from narrowcast import predict as P
    labs = ["Sedum acre", "Sedum album", "Bellis annua"]
    rows = _rows(labs * 20, [l.split()[0] for l in labs] * 20, dim=256)
    ds = build.load_rows(rows, "enc")
    clf = build.fit_head(ds)
    out = build.save_bundle(tmp_path / "b", clf, labs, "enc",
                            build.fit_and_measure(build.score_frame(clf, ds),
                                                  p_ood=0.2),
                            {}, ds.counts, source="t", space=ds.X_train.mean(0))
    b = P.Bundle(out)
    assert b.space is not None
    b.predict(ds.X_eval)
    assert not b.space_warning, b.space_warning
    rng = np.random.default_rng(1)
    rot, _ = np.linalg.qr(rng.normal(size=(256, 256)))
    b.predict(np.asarray(ds.X_eval @ rot, dtype="float32"))
    assert any("unrelated encoder" in n for n in b.space_warning)


def test_a_flagged_regional_row_moves_the_mix_to_the_deployment_realistic_one():
    """narrowcast cannot derive geography and will not guess. When the caller says
    which negatives a user could actually supply, the operating point anchors to
    those — and the unrelated rows stay in the report carrying no weight."""
    f = _e2e_frame()
    f.loc[f["bucket"] == "near_ood", "bucket"] = "regional_ood"
    m = build.fit_and_measure(f, p_ood=0.2)
    assert "regional_ood" in m["ood_mix"]
    assert "regional_ood" in m["per_bucket"]


def test_only_an_out_of_list_row_can_be_regional(tmp_path):
    """Promoting an in-list row would put it in a bucket where no correct answer
    exists, and it would then count against the model for being recognised."""
    rng = np.random.default_rng(0)
    classes = np.array(["Sedum acre", "Sedum album"])
    lab = np.array(["Sedum acre"] * 8 + ["Bellis annua"] * 8)
    proba = rng.random((16, 2))
    f = tmp_path / "r.npz"
    np.savez(f, proba=proba / proba.sum(1, keepdims=True), classes=classes,
             label=lab, group=np.array([l.split()[0] for l in lab]),
             cluster=np.array([f"c{i // 2}" for i in range(16)]),
             regional=np.ones(16, bool))           # flags everything, in-list too
    ds = build.load_scored(sources.from_scores(f))
    assert ds.counts["in_catalog"] == 8
    assert ds.counts["regional_ood"] == 8


def test_space_coherence_is_a_cosine_and_cannot_exceed_one():
    """It normalised only the reference, making it a mean *projection*: on raw
    descriptors it returned 5.4, the comparison against the bound was meaningless,
    and the warning never fired on the one path that passes raw vectors."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 128)) * 40.0        # nowhere near unit norm
    assert abs(build.space_coherence(X, X.mean(0))) <= 1.0
    scaled = build.space_coherence(X * 7.0, X.mean(0))
    assert scaled == pytest.approx(build.space_coherence(X, X.mean(0)))


def test_predict_warns_on_raw_unnormalised_vectors_from_another_space(tmp_path):
    """The path the CLI actually takes: `sources.from_embeddings` hands over raw
    descriptors and `Bundle.predict` normalises inside `proba`, not before. The
    earlier test passed pre-normalised vectors and so could not see this."""
    from narrowcast import predict as P
    labs = ["Sedum acre", "Sedum album", "Bellis annua"]
    rows = _rows(labs * 20, [l.split()[0] for l in labs] * 20, dim=256)
    ds = build.load_rows(rows, "enc")
    clf = build.fit_head(ds)
    out = build.save_bundle(tmp_path / "b", clf, labs, "enc",
                            build.fit_and_measure(build.score_frame(clf, ds),
                                                  p_ood=0.2),
                            {}, ds.counts, source="t", space=ds.X_train.mean(0))
    b = P.Bundle(out)
    raw = np.asarray(rows.descriptor, dtype="float32") * 25.0    # unnormalised
    b.predict(raw)
    assert not b.space_warning, b.space_warning
    rng = np.random.default_rng(1)
    rot, _ = np.linalg.qr(rng.normal(size=(256, 256)))
    b.predict(np.asarray(raw @ rot, dtype="float32"))
    assert any("unrelated encoder" in n for n in b.space_warning)


def test_an_audit_bundle_stores_no_embedding_space():
    """`--scores` leaves `X_train` empty, and `.mean(0)` on it is NaN with a
    warning rather than an error. The invariant is local to `save_bundle` so it
    cannot depend on the caller remembering."""
    import tempfile
    d = pathlib.Path(tempfile.mkdtemp())
    out = build.save_bundle(d / "b", None, ["a", "b"], "enc", {}, {}, {},
                            source="t", space=np.full(8, np.nan))
    assert not (out / "head.npz").exists()
    assert json.loads((out / "manifest.json").read_text())["has_head"] is False


def test_two_files_declaring_different_encoders_are_refused():
    """The measured conclusion: the geometry test catches 0 of 21 export pairs,
    including the recorded failure. Comparing what the two files *say* is the only
    thing here that sees a Core ML export against its own torch original."""
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(40, 128)), rng.normal(size=(40, 128))
    with pytest.raises(SystemExit, match="different models"):
        build.check_same_space(a, b, declared=["bioclip2", "bioclip2_cml4"])
    assert build.check_same_space(a, b, declared=["bioclip2", "bioclip2"]) == []


def test_the_geometry_warning_says_what_it_cannot_see():
    """A warning that overstated its coverage would be worse than none: a reader
    who saw no warning would conclude the pools matched."""
    rng = np.random.default_rng(0)
    d = 256
    cone = rng.normal(size=d)
    def pool(rot=None):
        X = rng.normal(scale=0.5, size=(150, d)) + cone
        if rot is not None:
            X = X @ rot
        return X / np.linalg.norm(X, axis=1, keepdims=True)
    rot, _ = np.linalg.qr(rng.normal(size=(d, d)))
    notes = build.check_same_space(pool(), pool(rot))
    assert notes and "0 of 21" in notes[0]
    assert "name the encoder in both files" in notes[0]


def test_a_declared_encoder_is_read_from_the_file(tmp_path):
    f = tmp_path / "e.npz"
    np.savez(f, descriptor=np.zeros((4, 8), "float32"),
             label=np.array(["a"] * 4), encoder="bioclip2_cml4")
    rows = sources.from_embeddings(f)
    assert rows.encoder == "bioclip2_cml4"
    assert any("declares encoder" in n for n in rows.notes)


def test_the_files_encoder_declaration_beats_the_flag(tmp_path):
    """Two channels for one fact. The file's wins because it is attached to the
    vectors, and the disagreement is said out loud rather than resolved quietly."""
    import subprocess, sys as _sys
    rng = np.random.default_rng(0)
    labs = np.repeat(["Sedum acre", "Sedum album"], 12)
    f = tmp_path / "e.npz"
    np.savez(f, descriptor=rng.normal(size=(24, 16)).astype("float32"), label=labs,
             group=np.array(["Sedum"] * 24),
             cluster=np.array([f"c{i // 2}" for i in range(24)]),
             encoder="bioclip2_cml4")
    r = subprocess.run([_sys.executable, "-m", "narrowcast.cli", "audit",
                        "--embeddings", str(f), "--out", str(tmp_path / "b"),
                        "--encoder-name", "bioclip2"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "disagrees with" in r.stderr
    man = json.loads((tmp_path / "b" / "manifest.json").read_text())
    assert man["encoder"] == "bioclip2_cml4"


def test_renaming_one_bucket_does_not_reshuffle_the_others():
    """`make_splits` shared one generator across `df.groupby("bucket")`, so every
    bucket's split depended on the alphabetical order of the bucket *names*.
    Flagging out-of-list rows as `regional_ood` renames a bucket and nothing else,
    and it moved coverage by 12 points on one seed by reshuffling `in_catalog`
    and `near_ood` as a side effect."""
    f = _e2e_frame()
    f.loc[f["bucket"] == "near_ood", "species"] = "shared sp"
    renamed = f.copy()
    renamed["bucket"] = renamed["bucket"].replace({"near_ood": "regional_ood"})
    a, b = cascade.make_splits(f, seed=0), cascade.make_splits(renamed, seed=0)
    untouched = (f["bucket"] == "in_catalog").to_numpy()
    assert untouched.any()
    assert (a[untouched].values == b[untouched].values).all()


def test_each_bucket_still_splits_about_in_half():
    """The per-bucket generator must not cost the property the loop existed for."""
    f = _e2e_frame()
    fold = cascade.make_splits(f, seed=0)
    for b, g in f.groupby("bucket"):
        share = (fold[g.index] == "calib").mean()
        assert 0.2 <= share <= 0.8, (b, share)


def test_relabelling_a_bucket_does_not_change_the_split_at_all():
    """A pure relabelling — the same rows, the same clusters, a different bucket
    name — must produce the same split. It did not: keyed on nothing it moved
    every bucket, keyed on the name it still moved the renamed one, and coverage
    swung 12 points on real Oregon data for no reason but the label."""
    f = _e2e_frame()
    f.loc[f["bucket"] == "near_ood", "bucket"] = "distant_ood"   # both key on `label`
    renamed = f.copy()
    renamed["bucket"] = renamed["bucket"].replace({"distant_ood": "regional_ood"})
    assert (cascade.make_splits(f, seed=0).values ==
            cascade.make_splits(renamed, seed=0).values).all()
    # and a rename that genuinely changes the split *key* is allowed to differ:
    # `near_ood` clusters on the group, `regional_ood` on the label
    assert cascade.SPLIT_CLUSTER["near_ood"] != cascade.SPLIT_CLUSTER["regional_ood"]
