import json
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
                            source="test", groups=gmap)
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
