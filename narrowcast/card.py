"""The model card -- the artifact that makes a built model trustworthy.

The weights are the commodity here; this is not. A tool that lets anyone train a
narrow classifier and says nothing about its failure modes is a machine for
producing confident wrong answers at scale, and plant identification has
consequences that make that worse than usually.

So the card states, in order: what it was built from, what it actually scores on
held-out data, the label-level share alongside coverage (without which a
sibling-dense set reads as the best case rather than the worst), which relatives
it will confuse, and what it cannot do.
"""

import json
from pathlib import Path

from narrowcast.encoders import BY_VARIANT


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _ci(metrics, key):
    """Interval for `key`, or an honest dash when there was nothing to resample."""
    iv = (metrics.get("ci") or {}).get(key)
    return "—" if not iv else f"{100 * iv[0]:.1f}–{100 * iv[1]:.1f}%"


HAZARD_BAR = 0.01   # declared, not tuned: see OREGON_SAFETY_FINDINGS.md

# Declared from the shape of the measured space, not tuned to a metric. Across
# plantid's 1,409 arms (HEADROOM_FINDINGS.md), retreat and harm travel together:
# of arms answering >=18% of in-list observations at group, 99.1% also have a
# label-level share under 0.6, and above 35% retreat *none* keeps a healthy share.
# Benign retreat -- coverage bought without costing quality -- lives in a narrow
# band, so this gates on measured retreat rather than on headroom, which only
# predicts it. Below 0.10 retreat is negligible and the card stays quiet.
GROUP_RETREAT_BAR = 0.10


def _hazard_section(hz: dict) -> list:
    """The union rate, reported as a gate rather than a statistic.

    Per-confusion rates are individually reassuring and collectively misleading:
    on Oregon's lethal plants no single pair exceeded 2.5% while the union hit
    6.7%, because wrong answers scatter across many harmless-looking labels. So
    this section leads with the union and states a pass/fail against a bar fixed
    in advance.
    """
    if not hz:
        return []
    worst = max(v["named_non_hazard"] for v in hz.values())
    fails = [k for k, v in hz.items() if v["named_non_hazard"] > HAZARD_BAR]

    L = ["## Consequential labels", ""]
    if fails:
        L += [f"> ### ⚠ Do not rely on this model for {len(fails)} of "
              f"{len(hz)} consequential labels",
              f">",
              f"> The worst case is **{_pct(worst)}** — this model gives a "
              f"consequential thing a harmless name that often. The bar set in "
              f"advance is {_pct(HAZARD_BAR)}.",
              f">",
              f"> Reducing coverage is what fixes this: the same model measured "
              f"at lower coverage answers less and is wrong less. Rebuild with a "
              f"higher `--ood-rate`, or treat these labels as always-decline.", ""]
    else:
        L += [f"All {len(hz)} consequential labels are under the "
              f"{_pct(HAZARD_BAR)} bar; worst case {_pct(worst)}.", ""]

    L += ["The number that matters is **named as something harmless** — the union "
          "over every wrong answer, not any single confusion. A group-level answer "
          "counts if the group it names contains nothing consequential: \"it is a "
          "*Lomatium*\" for poison hemlock is as actionable as a wrong label. "
          "Being named as another consequential label is wrong but not dangerous, "
          "so it is counted separately.", "",
          "| label | n | correct | declined | named as another consequential label | "
          "**named as something harmless** | 95% CI |",
          "|---|---|---|---|---|---|---|"]
    for k, v in sorted(hz.items(), key=lambda kv: -kv[1]["named_non_hazard"]):
        ci = v.get("ci")
        cis = "—" if not ci else f"{100*ci[0]:.1f}–{100*ci[1]:.1f}%"
        mark = " ⚠" if v["named_non_hazard"] > HAZARD_BAR else ""
        L.append(f"| **{k}** | {v['n']} | {_pct(v['named_correctly'])} | "
                 f"{_pct(v['declined'])} | {_pct(v['named_other_hazard'])} | "
                 f"**{_pct(v['named_non_hazard'])}**{mark} | {cis} |")
    L.append("")
    if any(not v.get("ci") for v in hz.values()):
        L += ["_No interval where the data offers no grouping inside a single "
              "label — these images are not grouped by individual plant, and a "
              "row-level interval would treat several photographs of one plant as "
              "independent. Sources carrying observation ids do get intervals._", ""]
    return L


def _retreat_section(m: dict, comp: dict) -> list:
    """Where the non-label answers went, and whether that cost anything.

    The card used to say a low label-level share meant "the rest are answered at
    group". It measured no such thing, and it can be false: the model may be
    declining instead. plantid's HEADROOM_FINDINGS.md separates the two shadows --
    group answers drawn from declines inflate coverage while quality holds, group
    answers drawn from label answers collapse quality. Same mechanism, opposite
    consequences, so the card reports which one happened rather than assuming.

    Headroom predicts *retreat*, not *harm*, which is why the label-level share
    stays the headline and this section never replaces it.
    """
    share, group, decline = (m.get("label_share"), m.get("group_share"),
                             m.get("decline_share"))
    headroom = m.get("headroom")
    example = next(iter(comp.get("crowded_groups") or {}), "group")
    out = []

    if share is not None and share < 0.6:
        if group is not None and decline is not None:
            # Which pool the missing label answers actually went to.
            where = (f"{_pct(group)} are answered at group and {_pct(decline)} are "
                     f"declined outright"
                     if group >= decline else
                     f"{_pct(decline)} are declined outright and only {_pct(group)} "
                     f"are answered at group")
            cost = (f"Coverage and precision look healthy here *because* of those "
                    f"group answers, not despite them."
                    if group is not None and group >= 0.2 else
                    f"This model is mostly declining rather than retreating, so "
                    f"coverage is paying the price directly.")
        else:
            where, cost = "the rest are answered at group or declined", ""
        out += [
            f"> **Read the label-level share, not the coverage.** This model names "
            f"a label on only {_pct(share)} of in-list observations; {where}. "
            f"Because your list is group-crowded, a group answer may narrow "
            f"nothing — \"it is a {example}\" when most of your list is that "
            f"group. {cost}".rstrip(),
            "",
        ]
    elif group is not None and group >= GROUP_RETREAT_BAR:
        # The benign shadow, invisible to this card until now: the list retreats
        # to the group appreciably and quality held anyway. Rare -- see the bar's
        # note -- which is exactly why it is worth naming when it happens, rather
        # than letting the reader infer harm from the retreat.
        why = (f" Coarse accuracy exceeds label accuracy by {100 * headroom:.1f}pp "
               f"on the calibration split, which is what makes retreating "
               f"attractive to the cascade." if headroom else "")
        out += [
            f"> **This list retreats to the group, and it has not cost you.** "
            f"{_pct(group)} of in-list observations are answered at group rather "
            f"than at a label, yet the label-level share is still {_pct(share)} — "
            f"so those group answers came out of what would otherwise have been "
            f"declines, not out of label answers.{why} Coverage is higher than it "
            f"would be without them and quality is unharmed. Treat it as a "
            f"standing risk rather than a problem: the same retreat on a harder "
            f"list is what collapses the label-level share.",
            "",
        ]
    return out


def render(manifest: dict) -> str:
    m = manifest["metrics"]
    comp = manifest["composition"]
    enc = BY_VARIANT.get(manifest["encoder"])
    sizing = f"{enc.size_mb():.1f} MB int4" if enc else "size unknown"

    L = [
        f"# Model card — {comp['n_labels']} labels",
        "",
        f"Built {manifest['created']} · encoder `{manifest['encoder']}` ({sizing}) · "
        f"source `{manifest['source']}`",
        "",
        "## What it answers",
        "",
        f"Measured on held-out data at an assumed **{_pct(m['p_ood'])} out-of-list "
        f"rate** — the share of photographs you take that are of something not on "
        f"your list. That assumption is the single biggest lever on these numbers; "
        f"rebuild with `--ood-rate` if it is wrong for you.",
        "",
        "| | | 95% CI |",
        "|---|---|---|",
        f"| Coverage — queries it answers | **{_pct(m['coverage'])}** | "
        f"{_ci(m, 'coverage')} |",
        f"| Precision — answers that are correct | **{_pct(m['precision'])}** | "
        f"{_ci(m, 'precision')} |",
        f"| **Label-level share** — in-list observations named to labels | "
        f"**{_pct(m['label_share'])}** | {_ci(m, 'label_share')} |",
        f"| Group-level share — in-list observations answered at group only | "
        f"{_pct(m.get('group_share'))} | {_ci(m, 'group_share')} |",
        f"| Closed-set top-1 — accuracy when the plant is on your list | "
        f"{_pct(m['closed_set_top1'])} | {_ci(m, 'closed_set_top1')} |",
        "",
        f"Intervals are bootstrapped over **labels**, not rows, because "
        f"observations of one label are not independent. This model rests on "
        f"{m.get('n_label_clusters', '?')} labels in the test half, so they are "
        f"wide — that width is a fact about your list, not a formatting choice.",
        "",
    ]

    L += _retreat_section(m, comp)

    L += _hazard_section(m.get("hazard") or {})

    L += ["## Where it declines and where it errs", "", "| bucket | n | answered | correct when answered |",
          "|---|---|---|---|"]
    labels = {"in_catalog": "on your list", "near_ood": "relatives you did not choose",
              "distant_ood": "unrelated plants"}
    for b, v in m.get("per_bucket", {}).items():
        L.append(f"| {labels.get(b, b)} | {v['n']} | {_pct(v['answered'])} | "
                 f"{_pct(v['correct_when_answered'])} |")
    L.append("")

    oc = manifest.get("outside_siblings") or {}
    if oc:
        L += [
            "## Relatives it will confuse",
            "",
            "These labels are close relatives of ones on your list but are **not on "
            "it**, so no correct answer exists for them. This is the weakest "
            "rejection case measured.",
            "",
        ]
        for g, rel in list(oc.items())[:12]:
            noun = "relative" if len(rel) == 1 else "relatives"
            L.append(f"- **{g}** — {len(rel)} {noun} not on your list: "
                     f"{', '.join(rel[:8])}" + (" …" if len(rel) > 8 else ""))
        if len(oc) > 12:
            L.append(f"- _(+{len(oc) - 12} more groups)_")
        L.append("")

    L += [
        "## How the decision is made",
        "",
        "Three-way: name the labels, name the group, or decline. Thresholds were "
        "fitted by maximising expected utility on a calibration split held out from "
        "these numbers, with payoffs declared before fitting:",
        "",
        "```",
        json.dumps(manifest["utility"], indent=2),
        "```",
        "",
        f"Fitted thresholds: `t_group={m['t_group']:.4f}`, `t_label={m['t_label']:.4f}`. "
        f"Calibrated on {m['n_calib']} observations, reported on {m['n_test']}.",
        "",
        "## What it cannot do",
        "",
        f"- It knows {comp['n_labels']} labels. Everything else it can only decline "
        f"or get wrong — and the relatives listed above are the ones it will get "
        f"wrong confidently.",
        "- **A correct-looking answer is not verification.** Where being wrong is "
        "expensive, treat an answer as a candidate to check, never as a result. "
        "The within-group case is the measured weak point.",
        "- Numbers above are held-out but come from the same image source as training. "
        "Photographs taken differently — your phone, your light, your angles — will "
        "score lower.",
    ]
    counts = manifest.get("counts", {})
    if counts.get("missing_organs"):
        L.append(f"- No embeddings were available for: "
                 f"{', '.join(counts['missing_organs'])}. Built from the rest.")
    L += ["", "---", "",
          f"Training rows {counts.get('train', '?')} · evaluation rows "
          f"{sum(v['n'] for v in m.get('per_bucket', {}).values())} · "
          f"bundle format v{manifest['bundle_version']}"]
    return "\n".join(L)


def write(bundle_dir: Path) -> Path:
    manifest = json.loads((Path(bundle_dir) / "manifest.json").read_text())
    out = Path(bundle_dir) / "CARD.md"
    out.write_text(render(manifest))
    return out
