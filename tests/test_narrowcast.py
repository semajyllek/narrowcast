import json

import numpy as np
import pandas as pd
import pytest

from narrowcast import build, card, encoders, labels, plan, projection, sources


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


# ---- encoder choice -------------------------------------------------------

def test_choose_takes_largest_that_fits():
    assert encoders.choose(20).variant == "mobileclip2_s2"
    assert encoders.choose(10).variant == "mobileclip2_s0"
    assert encoders.choose(1000).variant == "bioclip2"
    assert encoders.choose(None).variant == "bioclip2"


def test_choose_falls_back_to_smallest_when_nothing_fits():
    assert encoders.choose(1).variant == "mobileclip2_s0"


def test_bioclip2_int4_reconciles_with_shipped_artifact():
    """152 MB against the 160 MB build is the check that the counts are image-tower only."""
    assert encoders.BY_VARIANT["bioclip2"].size_mb(4) == pytest.approx(152.0)


# ---- projection -----------------------------------------------------------

def test_project_matches_measured_cell_at_an_anchor():
    anchors = projection.GRID["sibling_anchors"]["20"]
    p = projection.project("mobileclip2_s2", 20, anchors["easy"], p_ood=0.2)
    assert p["coverage"] == pytest.approx(0.771, abs=1e-6)
    assert p["label_share"] == pytest.approx(0.833, abs=1e-6)


def test_sibling_dense_sets_project_lower_label_share():
    a = projection.GRID["sibling_anchors"]["20"]
    easy = projection.project("bioclip2", 20, a["easy"])
    hard = projection.project("bioclip2", 20, a["hard"])
    assert hard["label_share"] < easy["label_share"]
    # ...while coverage looks *better*, which is the trap the tool exists to flag
    assert hard["coverage"] > easy["coverage"]


def test_project_clamps_rather_than_extrapolating():
    p = projection.project("bioclip2", 3, 0.0)
    assert p["K_used"] == 10 and p["extrapolated"] == "below"
    p = projection.project("bioclip2", 400, 0.0)
    assert p["K_used"] == 50 and p["extrapolated"] == "above"


def test_project_rejects_unmeasured_prevalence():
    with pytest.raises(ValueError, match="not measured"):
        projection.project("bioclip2", 20, 0.2, p_ood=0.33)


# ---- plan -----------------------------------------------------------------

def test_plan_warns_on_crowded_groups():
    pl = plan.make_plan(["Sedum acre", "Sedum album", "Sedum dasyphyllum"],
                        budget_mb=20, pool=POOL)
    kinds = {w.kind for w in pl["warnings"]}
    assert "crowded" in kinds and "outside_siblings" in kinds
    assert "label-level" in plan.render(pl)


def test_plan_warns_about_label_it_has_no_images_for():
    """A confident projection for a list `build` will drop is the worst failure here."""
    pl = plan.make_plan(["Sedum acre", "Conium maculatum"], budget_mb=20, pool=POOL)
    w = {x.kind: x for x in pl["warnings"]}
    assert "missing" in w and "Conium maculatum" in w["missing"].detail
    assert pl["n_available"] == 1


def test_plan_refuses_to_project_when_most_of_the_list_is_absent():
    pl = plan.make_plan(["Conium maculatum", "Cicuta virosa", "Sedum acre"],
                        budget_mb=20, pool=POOL)
    assert pl["projection"] is None
    assert "unprojectable" in {x.kind for x in pl["warnings"]}
    assert "No projection" in plan.render(pl)


def test_plan_reports_budget_shortfall():
    """Names the next step up, which is PlantCLEF2024 at 43.3 MB, not BioCLIP-2."""
    pl = plan.make_plan(["Bellis perennis"], budget_mb=20, pool=POOL)
    assert "43.3 MB" in pl["budget_note"]
    assert pl["encoder"].variant == "mobileclip2_s2"


def test_choose_ranks_storage_and_cannot_see_latency():
    """PlantCLEF2024 is a third of BioCLIP-2's bytes and twice its latency."""
    assert encoders.choose(50).variant == "plantclef24"
    assert encoders.choose(200).variant == "bioclip2"
    pc = encoders.BY_VARIANT["plantclef24"]
    bc = encoders.BY_VARIANT["bioclip2"]
    assert pc.size_mb() < bc.size_mb() and pc.ms_per_image > bc.ms_per_image


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
    assert "over **labels**, not rows" in out


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
    assert json.loads((out / "manifest.json").read_text())["bundle_version"] == 1
    assert (out / "head.npz").exists()


# ---- data sources: the tool takes a dataset, it does not fetch one ---------

def test_images_layout_reads_labels_from_directory_names(tmp_path):
    for label in ("Sedum acre", "weld porosity"):
        d = tmp_path / label.replace(" ", "_")
        d.mkdir()
        (d / "a.jpg").write_bytes(b"x")
    r = sources.from_images(tmp_path)
    assert r.labels == ["Sedum acre", "weld porosity"] and len(r) == 2


def test_manifest_requires_label_and_path(tmp_path):
    f = tmp_path / "m.csv"
    f.write_text("label,notpath\na,b\n")
    with pytest.raises(ValueError, match="needs a 'path' column"):
        sources.from_manifest(f)


def test_missing_cluster_is_recorded_not_silently_assumed(tmp_path):
    f = tmp_path / "m.csv"
    f.write_text("label,path\na,/x.jpg\nb,/y.jpg\n")
    r = sources.from_manifest(f)
    assert r.has_clusters is False
    assert any("treated as independent" in n for n in r.notes)


def test_group_defaults_to_first_token_and_is_overridable(tmp_path):
    f = tmp_path / "m.csv"
    f.write_text("label,path,group\nSedum acre,/x.jpg,Crassulaceae\n")
    assert list(sources.from_manifest(f).group) == ["Crassulaceae"]
    assert sources.default_group("Sedum acre") == "Sedum"


def test_exactly_one_source_required():
    with pytest.raises(ValueError, match="exactly one"):
        sources.load()
    with pytest.raises(ValueError, match="exactly one"):
        sources.load(images="a", manifest="b")


# ---- projection is gated behind a measured profile -------------------------

def test_projection_refuses_without_a_profile_for_the_domain():
    """Birds licensed the warning, not the numbers."""
    with pytest.raises(ValueError, match="projection is disabled"):
        projection.project("bioclip2", 20, 0.2, profile="fungi-bioclip2")


def test_default_plant_profile_still_resolves():
    assert "plants-bioclip2" in projection.available()
    assert projection.project("bioclip2", 20, 0.229)["profile"] == "plants-bioclip2"


# ---- non-binomial labels are first-class ----------------------------------

def test_read_list_accepts_labels_that_are_not_binomials(tmp_path):
    f = tmp_path / "l.txt"
    f.write_text("weld_porosity\ncrack_lateral\n# comment\n")
    assert labels.read_list(f) == ["weld_porosity", "crack_lateral"]


def test_read_list_still_normalises_a_pure_binomial_list(tmp_path):
    f = tmp_path / "l.txt"
    f.write_text("Sedum acre L.\nTrifolium repens\n")
    assert labels.read_list(f) == ["Sedum acre", "Trifolium repens"]
