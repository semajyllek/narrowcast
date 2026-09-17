# narrowcast — orientation for a new session

**Audit a classifier over a narrow label set, and the truth about how it will
fail.** Public, MIT, pip-installable, 141 tests, CI on 3.10/3.12/3.13.

Extracted from [narrowcast-plantid](https://github.com/semajyllek/narrowcast-plantid), which remains
the research record — **every number in the README traces to a findings doc
there**, and that is where new measurements belong. `DISPOSITION.md` there argues
why this package is now the shape it is.

## The one thing that is easy to get wrong

**This measures a model; it does not build one.** As of 0.2.0 there is no
encoder, no encoder registry, no Hub search, no candidate sweep, no task config
and no projection — 833 lines removed, and every failure in the project's history
lived in them. The caller brings posteriors (`--scores`) or vectors
(`--embeddings`); what this fits is at most a logistic head plus two thresholds.
Nothing here trains a backbone, and `PRUNE_FINDINGS.md` in narrowcast-plantid is
the measurement saying it should not try.

The older framing — "an evaluation tool that happens to build a model" — was
already the honest half of what shipped. The half that built things is gone.

## Why it exists

A crowded label set buys **coverage** with coarse answers that narrow nothing, so
coverage and precision go *up* while the model gets worse. Two 14-label sets, same
data, same encoder: crowded scores 0.806 coverage against 0.618 at the same
precision, and is much worse — label-level 0.476 against 0.761.

So no report ever prints coverage without the label-level share beside it.

## Architecture

| module | does only |
|---|---|
| `sources.py` | embeddings / scores → `Rows`, including the caller's optional `regional` flag. **The tool never fetches and never encodes.** |
| `cascade.py` | label/group/decline, the optional near-OOD gate, declared `UTILITY`, threshold fitting, clustered splits with declared hazards stratified into both halves, per-label `suppress`, cluster bootstrap |
| `build.py` | head (embeddings path only), per-row scores, measurement, hazard union, bundle |
| `card.py` | the report, the consequential-label gate, what a suppression cost, and the origin-composition section |
| `labels.py` | label-list parsing and composition analysis |
| `predict.py` | run a bundle *this tool fitted*, under the same suppression the card was measured with; refuses an audit bundle by name |
| `cli.py` | `audit` / `card` / `predict` |

**One measurement path, deliberately.** `build.score_frame` delegates to
`build.frame_from_posteriors`, which is also what the `--scores` path calls. A
model we fitted and a model we merely audited therefore go through identical code;
if that ever forks, the two can drift apart silently and the card stops meaning
one thing. Pinned by a test.

**An audit bundle has no head.** `save_bundle(clf=None)` writes no `head.npz` and
records `has_head: false`, and `predict.Bundle` refuses it with a message saying
why. We measured someone else's model; we did not obtain a copy of it.

## Conventions that are load-bearing

- **Cluster, never row.** Splits and bootstraps resample the *subject* (several
  photos of one plant). Row-level intervals have twice produced effects that
  failed to replicate.
  **With one declared exception**: a hazard named by `--hazard` or
  `--hazard-absent` is stratified into *both* halves, halved at the cluster. This
  breaks the whole-genus rule for that one genus on purpose — the hazard is the
  thing being measured, not a member of the background it came from. Without it,
  reaching the test half was a coin flip per seed (Conium 6 of 8 splits, Cicuta 2
  of 8) and a safety report decided by shuffle whether it checked anything. The
  cluster itself is still never split: a single-cluster hazard goes wholly to
  test and loses its interval instead.
- **Bootstrap the ratio, not the mean.** Coverage and precision are
  prevalence-weighted; bootstrapping the unweighted mean of the same rows once
  gave a 22–77% interval around a 96.1% point estimate.
- **Declare utilities before fitting.** `cascade.UTILITY` is fixed in source.
  Changing it is a deliberate act with a written reason.
- **Refuse rather than mislead.** `audit` exits on a source with no in-list rows.
  `--never-answer` refuses a label that is *not* on the list (you can only
  suppress what the model can emit) and refuses to suppress all of them;
  `fit_and_measure` refuses `never_answer` without `labels`, because without the
  label set it would suppress label answers only while `predict` also suppresses
  hollow-group ones — two different models out of one bundle.
  `predict` refuses an audit bundle by name instead of failing on a missing
  `head.npz`. `--deployment-origin` prints that it was *not measured* under
  `--scores` rather than reporting a silent null — it refits a head twice and
  there is no head. `--background-embeddings` is rejected under `--scores`
  instead of ignored, because the negatives are already in the file.
- **The embedding-space check is blind to the failure it was written for.**
  Measured in narrowcast-plantid's `SPACE_CHECK_FINDINGS.md` over 13 encoder
  variants on the same photographs: the geometry test catches 74 of 105
  different-family pairs, **0 of 21** export/quantization pairs, and
  false-positives on 2 of 39 same-encoder pairs. The 0 of 21 includes the exact
  recorded failure — torch BioCLIP-2 against its Core ML int4 export, cross-pool
  cosine 0.79–0.82 at every organ. A faithful export lands in nearly the same
  space, and no threshold separates the cases. So it warns, names its own rates,
  and says what it cannot see. **Do not promote it to a refusal**; that was tried
  and its own first fixture false-positived.
  What catches the real thing is **declaration**: both npz files naming their
  `encoder`, compared and refused on mismatch. narrowcast cannot verify either
  claim and does not try — comparing two declarations is strictly better than
  comparing none.
- **`make_splits` keys each bucket's shuffle on its *contents*, not its name and
  not a shared stream.** Shared, every bucket's split depended on the alphabetical
  order of the bucket names: flagging out-of-list rows as `regional_ood` renames a
  bucket and changes nothing else about the data, and it reshuffled `in_catalog`
  and `near_ood` as a side effect — 12 points of coverage on real Oregon data, from
  a relabelling. Keyed on the name that side effect goes but the renamed bucket
  still resplits. Keyed on the clusters it actually holds, identical rows give an
  identical split whatever the bucket is called. Pinned by a test. A rename that
  changes the split *key* (`near_ood` clusters on the group, the others on the
  label) is still allowed to differ, because then the data really did change.
- **`regional_ood` is the caller's to declare and is never derived.** An optional
  `regional` boolean column marks which out-of-list rows the deployment could
  plausibly be shown. With it present the mix becomes `OOD_MIX_REGIONAL` and the
  remaining `distant_ood` rows carry weight **zero** — reported, but not evidence
  about the operating point. Without it there is no regional bucket at all, which
  is right: narrowcast has no geography and will not guess one.
- **Nothing here touches the network, and nothing reads a pixel.** That used to
  need saying about `fit` and `build`; now it is structural. There is no encoder
  to load, so there is no registry to resolve against, no Hub to search, and no
  size to claim on the card. `--encoder-name` is a string the caller declares for
  the record, and the card prints it back with "size not stated".
- **The caller's `group` column wins.** The default first-whitespace-token rule is
  a Latin-binomial convention; it silently disabled the group rank for every
  non-binomial domain until fixed. Pinned by a test.

## Traps that have already bitten

- Torch is not a dependency at any level and there is no `[encode]` extra. CI
  installs `[dev]` only, which is now the whole story rather than a discipline.
- BSD `sed` does not support `\b`, which silently half-completed a bulk rename.
- `git merge -F -` does not read stdin; it fails and a following `push` succeeds
  as a no-op.
- **An out-of-list row is wrong at *every* rank, and two parts of this tool
  disagree about that.** `frame_from_posteriors` sets `true_group` to `__OTHER__`
  for out-of-list rows, so `group_ok` is False by construction and `utility`
  scores a group answer for one as `wrong` — the same payoff as a label answer.
  But `outside_hazard_metrics` counts that identical answer as a **warning** and
  therefore safe, because "it is an umbellifer" is a true statement that helps the
  person holding the root. Both readings are defensible; holding both at once is
  not. Reconciling them changes what `coverage` and `precision` mean on every card
  ever printed, so it is a declared-utility decision needing its own pass and a
  written reason — flagged, not resolved.
  The first thing it cost: plantid's preferred **retreat** arm of the near-OOD
  gate is arithmetically dead here. Retreating an out-of-list row moves it from
  `wrong` to `wrong`, so `--gate-near-ood` declines instead. That is not a
  departure from `NEAR_OOD_FINDINGS.md` but the same finding read under different
  metric semantics — and the reject arm was preregistered and measured there too.
- **Suppressing the hazard instead of the look-alike measures nothing.**
  `--never-answer` exists to make the model cautious about one thing, and the
  thing to name is the *harmless* label the hazard gets called — the wild carrot,
  not the hemlock. Suppressing the hazard looks obviously right and is inert:
  `hazard_metrics` counts rows whose prediction is *not* the hazard, so rows whose
  argmax is the hazard were never in the numerator. Suppress it and they move
  LABEL → DECLINE, the rate is unchanged, and `named_correctly` goes to zero —
  same danger, less utility. The card used to recommend exactly this.
- **The near-OOD gate is a greedy second stage, and it helps only where the
  first stage answers too much.** Fitting three thresholds jointly at the same
  resolution is 216,000 evaluations against 3,600 — about 90 seconds on a real
  audit against 1.5 — and would mean rewriting `fit_thresholds`, whose
  tie-breaking produced every threshold on file across four repos. So `t_group`
  and `t_label` are fitted exactly as before and the gate is swept alone
  afterwards. Safe because the sweep can always choose the baseline. The cost is
  that where the declared payoffs already push `t_group` up to decline nearly
  everything, there is no surviving error to remove and the gate reports zero.
- **The suppression is applied after the fit, never inside it.** Thresholds are
  fitted as though nothing were suppressed, which is what makes the override's
  cost a delta the card can print. Folding it into `fit_thresholds` would absorb
  that number into the operating point and delete it.
- **`deployment_weights` renormalises around an absent bucket.** A source with no
  in-pool relatives has no `near_ood`, so its 0.32 share went unclaimed and
  `--ood-rate 0.2` scored at an effective **0.145** — while the card printed "an
  assumed 20.0% out-of-list rate". `fit_and_measure` now restricts the mix to the
  buckets actually present, per side. Pinned by a test.

## Open

- **Domain shift.** Every number comes from iNaturalist photos, one text corpus,
  two audio corpora. Nobody has pointed a different camera at anything.
*(The headroom rule is adopted — see "What the card knows" below.)*

## What the card knows about retreat

`build` measures **headroom** (coarse-rank minus label-rank accuracy, on the
in-catalogue *calibration* rows) and the full three-way split of in-list
behaviour: `label_share`, `group_share`, `decline_share`. Established in
narrowcast-plantid's `HEADROOM_FINDINGS.md` over 1,409 arms — headroom predicts group-answer
share at CV R² 0.883 against 0.362 for label accuracy alone.

**Headroom is `coarse − fine`, not `1 − top-1`.** Both get called headroom across
the four repos and they govern different things: `coarse − fine` governs
**retreat**, `1 − top-1` governs **sensitivity to a training-data intervention**
(narrowcast-derm's `K_FINDINGS.md`). They coincide only where coarse accuracy
sits near 1, which is the easy small-K regime — and the whole point of the
1,409-arm result is that it was established by breaking exactly that
collinearity. Anything keyed on K rather than on measured top-1 is keyed to a
proxy: at K=10 plants show *exactly zero* intervention damage and dermatology
loses 13.5pp. **"Small label sets are safe" is false in general.**

**Headroom needs data, which is why `plan` is gone.** `plan` took a list of
label strings — it never loaded an image, a vector or a fitted head, and
`projection` interpolated a shipped grid measured on 530 plant species. An
earlier version of this file wrongly claimed `plan` could compute headroom; it
could not, because headroom needs scores and a calibration split, which exist
only inside `fit_and_measure`. A structural warning issued before any data
arrives was the weakest thing here and the easiest to mistake for a measurement,
so 0.2.0 drops it rather than keeping two grades of warning that read alike.

- **Headroom predicts *retreat*, not *harm*.** Group answers come out of
  declines (coverage inflates, quality holds) or out of label answers (quality
  collapses). The card reports which, instead of asserting — it used to say "the
  rest are answered at group" while measuring no such thing, which is false on a
  model that is declining.
- **The card gates on measured retreat, not on headroom.** Post-fit the
  observation is in hand, so gating on its predictor would be backwards.
  `GROUP_RETREAT_BAR = 0.10`, declared from the shape of the measured space: of
  arms retreating on ≥18% of in-list observations, 99.1% also have a label-level
  share under 0.6, and above 35% retreat *none* stays healthy. Benign retreat is
  real but narrow.
- The label-level share stays the headline regardless.

## Open

- **Domain shift.** Every number comes from iNaturalist photos, one text corpus,
  two audio corpora. Nobody has pointed a different camera at anything.
- Not on PyPI. `pip install /path/to/narrowcast` for now.
