# narrowcast — orientation for a new session

A small classifier over a narrow label set, and the truth about how it will fail.
Public, MIT, pip-installable, 65 tests, CI on 3.10/3.12/3.13.

Extracted from [plantid](https://github.com/semajyllek/plantid), which remains
the research record — **every number in the README traces to a findings doc
there**, and that is where new measurements belong.

## The one thing that is easy to get wrong

**This is an evaluation tool that happens to build a model, not a training
framework.** The encoder is always frozen and always someone else's. What gets
built is a logistic head plus two thresholds — ~40 KB against an encoder of
17.9–152 MB. Personalisation is the head; nothing here trains a backbone, and
`PRUNE_FINDINGS.md` in plantid is the measurement saying it should not try.

## Why it exists

A crowded label set buys **coverage** with coarse answers that narrow nothing, so
coverage and precision go *up* while the model gets worse. Two 14-label sets, same
data, same encoder: crowded scores 0.806 coverage against 0.618 at the same
precision, and is much worse — label-level 0.476 against 0.761.

So no report ever prints coverage without the label-level share beside it.

## Architecture

| module | does only |
|---|---|
| `sources.py` | images / manifest / embeddings → `Rows`. **The tool never fetches.** |
| `encode.py` | frozen encoder loading, batched embedding. Torch is an optional extra. |
| `cascade.py` | label/group/decline, declared `UTILITY`, threshold fitting, clustered splits, cluster bootstrap |
| `build.py` | head, per-row scores, measurement, hazard union, bundle |
| `card.py` | the report, and the consequential-label gate |
| `sweep.py` | run N candidates, return a frontier, decide or refuse |
| `config.py` | parse/validate a task config |
| `hub.py` | find candidate encoders on HF, **size-verified locally**. Maintenance only — no build path calls it |
| `plan.py` / `projection.py` | pre-compute warnings; projection gated behind `profiles/` |

## Conventions that are load-bearing

- **Cluster, never row.** Splits and bootstraps resample the *subject* (several
  photos of one plant). Row-level intervals have twice produced effects that
  failed to replicate.
- **Bootstrap the ratio, not the mean.** Coverage and precision are
  prevalence-weighted; bootstrapping the unweighted mean of the same rows once
  gave a 22–77% interval around a 96.1% point estimate.
- **Declare utilities before fitting.** `cascade.UTILITY` is fixed in source.
  Changing it is a deliberate act with a written reason.
- **Refuse rather than mislead.** `fit` exits 1 and writes no bundle when nothing
  clears the floor. `plan` will not project without a domain profile. `fit` will
  not sweep encoders over an `--embeddings` source, because that scores identical
  numbers N times and calls it a comparison.
- **`fit` and `build` never touch the network.** The encoder is whatever the
  config names, else the built-in registry within budget, so a candidate set is
  reproducible tomorrow. `constraints.domain` used to trigger a Hub search here
  and could never contribute a usable candidate — `encode.load_encoder` resolves
  variants against its own table and raises on a repo id, so every discovered
  candidate failed inside `sweep.evaluate` and was swallowed by its per-candidate
  handler. Discovery is now `narrowcast encoders`, a maintenance command;
  adopting a model is a deliberate edit to `encode.ENCODERS` and
  `encoders.ENCODERS`. A config still carrying `domain` gets a printed note
  rather than a silent no-op.
- **The caller's `group` column wins.** The default first-whitespace-token rule is
  a Latin-binomial convention; it silently disabled the group rank for every
  non-binomial domain until fixed. Pinned by a test.

## Traps that have already bitten

- Torch must stay optional. CI installs `[dev]` and **not** `[encode]` on purpose.
- The Hub's `num_parameters` filter leaks over-budget models. Verify size
  client-side from `safetensors.total`; unknown size is not "fits".
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
plantid's `HEADROOM_FINDINGS.md` over 1,409 arms — headroom predicts group-answer
share at CV R² 0.883 against 0.362 for label accuracy alone.

**`plan` cannot do this, and an earlier version of this file wrongly said it
could.** `cmd_plan` takes a list of label strings; it never loads an image, a
vector or a fitted head, and `projection` interpolates a shipped grid. Headroom
needs a fitted head and a calibration split, which exist only inside
`fit_and_measure`. `plan`'s warning is structural and says so; the card's is
measured.

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
