# narrowcast — orientation for a new session

**Audit a classifier over a narrow label set, and the truth about how it will
fail.** Public, MIT, pip-installable, 77 tests, CI on 3.10/3.12/3.13.

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
| `sources.py` | embeddings / scores → `Rows`. **The tool never fetches and never encodes.** |
| `cascade.py` | label/group/decline, declared `UTILITY`, threshold fitting, clustered splits, cluster bootstrap |
| `build.py` | head (embeddings path only), per-row scores, measurement, hazard union, bundle |
| `card.py` | the report, the consequential-label gate, and the origin-composition section |
| `labels.py` | label-list parsing and composition analysis |
| `predict.py` | run a bundle *this tool fitted*; refuses an audit bundle by name |
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
- **Bootstrap the ratio, not the mean.** Coverage and precision are
  prevalence-weighted; bootstrapping the unweighted mean of the same rows once
  gave a 22–77% interval around a 96.1% point estimate.
- **Declare utilities before fitting.** `cascade.UTILITY` is fixed in source.
  Changing it is a deliberate act with a written reason.
- **Refuse rather than mislead.** `audit` exits on a source with no in-list rows.
  `predict` refuses an audit bundle by name instead of failing on a missing
  `head.npz`. `--deployment-origin` prints that it was *not measured* under
  `--scores` rather than reporting a silent null — it refits a head twice and
  there is no head. `--background-embeddings` is rejected under `--scores`
  instead of ignored, because the negatives are already in the file.
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
