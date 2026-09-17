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


def _absent_hazard_section(hz: dict) -> list:
    """Dangerous species the user deliberately did not list.

    Separate from `_hazard_section` because it answers the opposite question and
    a reader must not confuse them: there, a listed hazard given a harmless name;
    here, an unlisted hazard given the name of something the user means to use.
    """
    if not hz:
        return []
    measured = {k: v for k, v in hz.items() if not v.get("unmeasured")}
    L = ["", "## Dangerous look-alikes you did not list", ""]
    if not measured:
        L += ["No rows for any of them, so **none of this was measured**. A "
              "declared risk with no data is not a passed check.", ""]
        for k in sorted(hz):
            L.append(f"- **{k}** — not measured")
        return L
    worst = max(v["dangerous"] for v in measured.values())
    fails = [k for k, v in measured.items() if v["dangerous"] > HAZARD_BAR]
    if fails:
        L += [f"> ### ⚠ {len(fails)} of {len(measured)} dangerous look-alikes get a "
              f"name from your list", "",
              f"> Worst case **{_pct(worst)}** — that often, this model tells the "
              f"user an unlisted dangerous plant is one of theirs. The bar set in "
              f"advance is {_pct(HAZARD_BAR)}.", "",
              "> Raising `--ood-rate` is what fixes this: it lifts the decline "
              "threshold, and declining is always safe here.", ""]
    else:
        L += [f"All {len(measured)} stay under the {_pct(HAZARD_BAR)} bar; worst "
              f"case {_pct(worst)}.", ""]
    L += ["| look-alike | n | named from your list | warned at group | declined |",
          "|---|---|---|---|---|"]
    for k, v in sorted(hz.items(), key=lambda kv: -(kv[1].get("dangerous") or -1)):
        if v.get("unmeasured"):
            L.append(f"| {k} | 0 | — | — | — |")
            continue
        mark = " ⚠" if v["dangerous"] > HAZARD_BAR else ""
        L.append(f"| {k}{mark} | {v['n']} | {_pct(v['named_in_list'])} | "
                 f"{_pct(v['warned_at_group'])} | {_pct(v['declined'])} |")
    unmeasured = [k for k, v in hz.items() if v.get("unmeasured")]
    if unmeasured:
        L += ["", f"**{len(unmeasured)} declared and not measured** (no rows): "
                  + ", ".join(sorted(unmeasured)) + ". Fetch data for them or "
                  "stop declaring them; an unmeasured hazard is not a safe one."]
    return L


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
              "label — these rows are not grouped by subject, and a "
              "row-level interval would treat several rows of one subject as "
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
    # Falls back to the *phrasing* rather than a placeholder word: with no crowded
    # group to name, the old default produced the sentence "it is a group", which
    # reads as a bug because it is one.
    example = next(iter(comp.get("crowded_groups") or {}), None)
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
            + (f"Because your list is group-crowded, a group answer may narrow "
               f"nothing — \"it is a {example}\" when most of your list is that "
               f"group. " if example else
               f"A group answer narrows the field only as far as the group goes, "
               f"which on a crowded list may be no distance at all. ")
            + f"{cost}".rstrip(),
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


# Below this many training rows per label, the label-level share was still
# climbing in every arm measured (narrowcast-plantid TINY_FINDINGS.md §2). Above
# it the picture splits: on a separated list with a strong encoder 8 rows already
# buys 100% of what unlimited data buys, while on a group-crowded list 64 rows
# buys 18-82% and the curve has not flattened. So the floor is where the warning
# starts, and whether the model is *retreating* decides how hard it lands.
THIN_ROWS_PER_LABEL = 32


def _inert_group_section(manifest: dict, m: dict) -> list:
    """Say so when every label is its own group, because the cascade is then two-way.

    The three-way decision needs a coarse rank to retreat to. If the group map is
    the identity -- which is what the default first-whitespace-token rule produces
    for single-word labels like keywords, and for dotted ones like
    `comp.sys.mac.hardware` -- then a group answer *is* a label answer, group mass
    never sums across labels, and the cascade can only name or decline.

    That is not a defect and the fitted thresholds handle it correctly. But it
    halves what the model can do, it is invisible in every number on this card,
    and it is usually an accident: the caller had a coarse rank available and did
    not supply it. Found by building a keyword model where `yes`, `no` and `up`
    each became their own group.
    """
    groups = manifest.get("groups") or {}
    labels = manifest.get("labels") or []
    if len(labels) < 2 or not groups:
        return []
    distinct = len({groups.get(l, l) for l in labels})
    if distinct < len(labels):
        return []
    return [
        f"> **The group rank is inert — all {len(labels)} labels are their own "
        f"group.** This model can only name a label or decline; there is no coarser "
        f"answer to retreat to, so the group share above is 0% by construction "
        f"rather than by measurement. If your labels do have a coarser rank — a "
        f"genus, a product family, a phoneme class — supply it as a `group` column "
        f"and rebuild, and the model gains a third answer. If they genuinely do "
        f"not, this is correct and nothing is wrong.",
        "",
    ]


def _data_limited_section(m: dict, counts: dict) -> list:
    """Whether a low label-level share is a data problem or a label-set problem.

    A user with eight rows per label and one with eight hundred otherwise
    receive identical cards, and the advice they need is opposite: the first
    should take more pictures, the second should change the list. Nothing in the
    card said which, and the numbers to tell them apart were already in the
    bundle.
    """
    rpl = counts.get("rows_per_label") or {}
    med = rpl.get("median")
    if not med:
        return []

    share, group = m.get("label_share"), m.get("group_share")
    thin = med < THIN_ROWS_PER_LABEL
    retreating = (group or 0) >= GROUP_RETREAT_BAR
    out = []

    if thin and share is not None and share < 0.6:
        # Both explanations are live and the card must not pick one silently.
        which = (
            "Your list is also group-crowded and this model is retreating to the "
            "group rank, which is the case where more data helps *least* — on a "
            "crowded list 64 rows per label still bought under half of what "
            "unlimited data bought. Expect more data to help, and not to "
            "be sufficient on their own."
            if retreating else
            "Your list is not retreating to the group rank, which is the case "
            "where more data helps *most* — on a separated list the label-level "
            "share is typically saturated by around 32 rows per label."
        )
        out += [
            f"> **This head was fitted on {med} training rows per label (median), "
            f"and that is thin.** A low label-level share here has two possible "
            f"causes — too little data, or a label set that cannot be told apart "
            f"— and they call for opposite responses. {which}",
            "",
        ]
    elif thin:
        out += [
            f"> **Fitted on {med} training rows per label (median).** The numbers "
            f"above are healthy, so this is not a problem — but they rest on thin "
            f"data, and a rebuild with more rows is the cheapest way to "
            f"confirm they hold.",
            "",
        ]
    n_thin = rpl.get("n_below_32", 0)
    if n_thin and n_thin < rpl.get("n_labels", 0):
        out += [
            f"{n_thin} of {rpl['n_labels']} labels have fewer than "
            f"{THIN_ROWS_PER_LABEL} training rows (fewest: {rpl['min']}). Those "
            f"labels are the least well fitted and are not broken out separately "
            f"above.",
            "",
        ]
    return out


def _unmeasured_lines(oc: dict) -> list:
    """Labels with no deployment-origin rows *at all* cannot be scored.

    They are the common case, not an edge case, and they are the labels most
    exposed to the effect. Folding them into an average would understate it, and
    omitting them would hide it, so they are counted and named.
    """
    n = oc.get("n_unmeasured") or 0
    if not n:
        return []
    names = oc.get("labels_unmeasured") or []
    shown = ", ".join(names[:6]) + (f" (+{n - min(6, len(names))} more)" if n > 6 else "")
    return ["",
            f"**{n} labels have no `{oc.get('deployment_origin')}` rows at all**, so "
            "there is nothing to score them on. They are the most exposed to this and "
            "the least measurable; read their accuracy as coming from a population "
            "this build was never tested against.",
            "",
            f"Unmeasured: {shown}."]


def _origin_section(oc: dict | None) -> list:
    """What the deployment-origin data did, and to whom.

    Reported per label group rather than as one number, because it is not one
    number: the labels that got deployment-origin training rows gain and the
    labels that did not are *worse off than if none had*. A single average hides
    a subgroup being harmed, which is the same failure the label-level share
    exists to prevent.

    Measured, never inferred. The size is domain-dependent -- ~0 on plants once
    the label set is narrow, 10-20 points on dermatology and keyword spotting at
    every label count tried -- so it cannot be read off the number of labels.
    """
    if not oc:
        return []
    dep = oc.get("deployment_origin")
    out = ["", f"## Origin composition — deployment origin `{dep}`", ""]

    if not oc.get("measurable"):
        out.append(f"Not measured: {oc.get('why', 'nothing to compare')}.")
        out += _unmeasured_lines(oc)
        return out

    n_lack, n_cov = oc.get("n_lacking"), oc.get("n_covered")
    lack_d, cov_d = oc.get("lacking_delta"), oc.get("covered_delta")
    out.append(f"**{n_lack} of {n_lack + n_cov} labels have no training data from "
               f"`{dep}`**, the origin this model will be used against.")
    out.append("")
    out.append("Two heads on the same label set — one fitted on everything, one with "
               f"every `{dep}` training row removed — scored on the same "
               f"{oc.get('n_eval_rows')} held-out `{dep}` rows:")
    out.append("")
    out.append("| labels | with the data | without it | measured effect |")
    out.append("|---|---|---|---|")
    for name, key, n in (("have it", "covered", n_cov), ("lack it", "lacking", n_lack)):
        w, wo = oc.get(f"{key}_with"), oc.get(f"{key}_without")
        d = oc.get(f"{key}_delta")
        if w is None or wo is None:
            continue
        out.append(f"| {name} ({n}) | {w:.3f} | {wo:.3f} | "
                   f"{'—' if d is None else f'{d:+.3f}'} |")
    out.append("")

    if lack_d is not None and lack_d < -0.01:
        out.append(f"**The {n_lack} labels without `{dep}` data are {abs(lack_d):.3f} "
                   "worse than if no label had it.** One multinomial means one "
                   f"argmax: `{dep}` rows move the boundaries of the labels that got "
                   "them, and a label represented only by the other origin loses ties "
                   "it used to win.")
        out.append("")
        names = oc.get("labels_lacking") or []
        if names:
            shown = ", ".join(names[:6])
            more = f" (+{n_lack - min(6, len(names))} more)" if n_lack > 6 else ""
            out.append(f"Affected: {shown}{more}.")
            out.append("")
        out.append("This does **not** shrink as the label set narrows. It tracks the "
                   "accuracy of the build, and on two of the three domains measured it "
                   "was undiminished at 5 labels.")
    elif lack_d is not None:
        out.append(f"The {n_lack} labels without `{dep}` data are not measurably worse "
                   f"off ({lack_d:+.3f}).")
    out += _unmeasured_lines(oc)
    return out


def render(manifest: dict) -> str:
    m = manifest["metrics"]
    comp = manifest["composition"]
    # An encoder this tool never ran has no size it can state, and it never runs
    # one: `encoder` is a string the caller declared for the record. There used to
    # be a registry lookup here with a fallback for precomputed vectors; every
    # build is now that case, so the fallback is the only branch and the registry
    # is gone. Quoting bytes for a model we did not load would attach a fabricated
    # number to the artifact whose whole job is being checkable.
    sizing = "size not stated — scored outside this tool"

    L = [
        f"# Model card — {comp['n_labels']} labels",
        "",
        f"Built {manifest['created']} · encoder `{manifest['encoder']}` ({sizing}) · "
        f"source `{manifest['source']}`",
        "",
        "## What it answers",
        "",
        f"Measured on held-out data at an assumed **{_pct(m['p_ood'])} out-of-list "
        f"rate** — the share of inputs you will show it that are of something not on "
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
        f"| Closed-set top-1 — accuracy when the answer is on your list | "
        f"{_pct(m['closed_set_top1'])} | {_ci(m, 'closed_set_top1')} |",
        "",
        # "clusters", not "labels". The bootstrap resamples the cluster column,
        # which is the label only when no finer grouping was supplied. Calling
        # them labels overstated the protection on any dataset that supplies a
        # `cluster` -- and badly so when those clusters are singletons, where
        # resampling them *is* resampling rows. `sources` notes that case and the
        # note is printed below.
        f"Intervals are bootstrapped over **clusters**, not rows, because rows "
        f"sharing a subject are not independent. This model rests on "
        f"{m.get('n_label_clusters', '?')} clusters in the test half, so they are "
        f"wide — that width is a fact about your data, not a formatting choice.",
        "",
    ]

    L += _retreat_section(m, comp)
    L += _inert_group_section(manifest, m)
    L += _data_limited_section(m, manifest.get("counts", {}))
    L += _origin_section(m.get("origin_cost"))

    L += _hazard_section(m.get("hazard") or {})
    L += _absent_hazard_section(m.get("hazard_absent") or {})

    L += ["## Where it declines and where it errs", "", "| bucket | n | answered | correct when answered |",
          "|---|---|---|---|"]
    labels = {"in_catalog": "on your list", "near_ood": "relatives you did not choose",
              "distant_ood": "unrelated inputs"}
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
        "- Numbers above are held-out but come from the **same source** as training. "
        "Inputs acquired differently — another camera, another microphone, another "
        "corpus — score lower, and how much lower depends on the encoder: measured "
        "across one such change it cost a 152 MB encoder 0.1pp and a 17.9 MB one "
        "17.9pp. A small encoder is the case to re-check before trusting these "
        "numbers on your own acquisition.",
    ]
    counts = manifest.get("counts", {})
    if counts.get("missing_organs"):
        L.append(f"- No embeddings were available for: "
                 f"{', '.join(counts['missing_organs'])}. Built from the rest.")
    # `rows_per_label` is None on an audited model -- we were handed posteriors,
    # never its training set -- so this must survive the key being present and
    # null, not merely absent.
    rpl = counts.get("rows_per_label") or {}
    L += ["", "---", "",
          (f"Training rows {counts['train']}" if counts.get("train")
           else "Training rows not known to this tool")
          + (f" ({rpl['median']}/label median)" if rpl.get("median") else "")
          + f" · evaluation rows "
          f"{sum(v['n'] for v in m.get('per_bucket', {}).values())} · "
          f"bundle format v{manifest['bundle_version']}"]
    return "\n".join(L)


def write(bundle_dir: Path) -> Path:
    manifest = json.loads((Path(bundle_dir) / "manifest.json").read_text())
    out = Path(bundle_dir) / "CARD.md"
    out.write_text(render(manifest))
    return out
