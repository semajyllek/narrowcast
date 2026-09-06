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
| `hub.py` | find candidate encoders on HF, **size-verified locally** |
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

## Open

- **Domain shift.** Every number comes from iNaturalist photos, one text corpus,
  two audio corpora. Nobody has pointed a different camera at anything.
- **Adopt the headroom rule.** No longer a hypothesis: plantid's
  `HEADROOM_FINDINGS.md` measures it over 1,409 arms — headroom (coarse-rank
  accuracy minus fine-rank accuracy) predicts group-answer share at CV R² 0.883,
  against 0.362 for fine accuracy alone, and generalises off-domain at MAE 0.033.
  Roughly `group_share ≈ 1.8 × headroom`, computable on the same calibration
  split that fits the thresholds.

  `plan`'s crowded-set warning still fires on label-set *structure*. It could
  fire on measured headroom instead, which predicts the mechanism rather than a
  proxy for it. Deliberately **not** changed by the run that measured it —
  adopting it is a separate act, and needs a test.

  Caveat to carry into any such change: headroom predicts *retreat*, not *harm*.
  Group answers can come out of declines (coverage inflates, quality holds) or
  out of label answers (quality collapses). So the label-level share stays
  mandatory in every report regardless.
- Not on PyPI. `pip install /path/to/narrowcast` for now.
