# narrowcast

**A small classifier over a narrow label set, and the truth about how it will fail.**

```bash
narrowcast plan  --species my.txt --budget 20
narrowcast build --images ./photos --background-images ./other --out models/mine
narrowcast card  models/mine
```

Training a classifier on your own classes is commodity — a dozen tools do it.
None of them tell you *what you are about to get*, and for narrow label sets the
metrics everyone publishes are actively misleading.

## The finding this exists for

Two 14-class sets, same data, same encoder, both built and measured:

| | crowded (8 *Sedum*, 6 *Trifolium*) | separated (14 distinct groups) |
|---|---|---|
| coverage | **80.6%** | 61.8% |
| precision | 97.8% | 98.5% |
| **label-level share** | **47.6%** | **76.1%** |
| closed-set top-1 | 81.7% | 97.0% |

The crowded set answers a third more queries at the same precision **and is much
worse.** It buys the coverage with group answers that narrow nothing — "it is a
*Sedum*" when eight of your fourteen classes are *Sedum*.

Report coverage and precision alone and a user reads their worst case as their
best. So `plan` runs before any compute, and no report prints coverage without
the label-level share beside it.

Reproduced outside biology: on birds, a *Larus*/*Calidris* set scored **higher**
coverage than 13 distinct genera while label-level fell 0.958 → 0.718. The trap
is a property of hierarchical label sets, not of plants.

## It takes a dataset; it does not fetch one

| flag | shape |
|---|---|
| `--images DIR` | `DIR/<label>/*.jpg` |
| `--manifest FILE` | parquet/csv: `label`, `path` [, `group`, `cluster`] |
| `--embeddings FILE` | npz: `descriptor`, `label` [, `group`, `cluster`] |

`--background-*` takes the same three forms and supplies negatives. Without it
there is no reject class: the model is closed-set, cannot decline, and the card
says so rather than implying a capability that was never fitted.

**`cluster`** is the unit that must not straddle a train/test split — several
photographs of one subject, one specimen, one production run. Without it every
row is treated as independent and the card records that its intervals are
anticonservative.

**`group`** is the coarse rank the cascade retreats to, defaulting to the label's
first whitespace token. Right for Linnaean binomials, overridable everywhere else.

Where the data came from — which corpus, under what licence, reconciled against
whose taxonomy — is a domain decision, so it lives in your project, not here.

## Three commands

**`plan`** — seconds, no training, no downloads. Finds crowded groups and names
the siblings you left *outside* the set (the weakest rejection case, since no
correct answer exists for them), sizes an encoder to your byte budget, and
projects — *only if a measured profile exists for your domain*.

**`build`** — fits a logistic head over frozen embeddings, fits label/group/decline
thresholds by expected-utility maximisation on a clustered calibration split,
evaluates against held-out data, and writes a bundle plus a card.

**`card`** — the report. Coverage, precision, label-level share, per-bucket
behaviour, cluster-bootstrapped intervals, and a **gate** on labels you declared
consequential.

## Consequential labels

```bash
narrowcast build --images ./photos --hazard "Conium maculatum" --out models/mine
```

For labels where being mistaken for a harmless one is the costly error, the card
reports the **union** — how often the label is given *any* harmless name — and
fails it against a bar fixed in advance.

Per-confusion reporting is not enough. On poison hemlock, no single confusion
exceeded 2.5% while the union reached **6.7%**, because wrong answers scatter
across many harmless-looking labels. A per-pair card passes a model a union card
fails.

Group answers count: naming a group that contains nothing consequential is as
actionable as a wrong label.

## Projection is gated

`profiles/plants-bioclip2.json` is a grid measured on 530 plant species through
BioCLIP-2. **There is no fallback.** Ask for a domain without a profile and
projection raises rather than quoting plant numbers at you.

The bird replication licenses `plan`'s *structural* warnings in any domain with a
label/group hierarchy. It does not license the numbers.

## Install

```bash
pip install -e .            # plan, card, and --embeddings workflows
pip install -e '.[encode]'  # adds torch/open_clip to turn images into vectors
```

## Provenance

Extracted from [plantid](https://github.com/semajyllek/plantid). Every claim on
this page is backed by a measurement there — see `EMBEDDED_FINDINGS.md`,
`OREGON_SAFETY_FINDINGS.md`, `BIRDS_FINDINGS.md` and `CONTAMINATION_FINDINGS.md`,
including two in-place retractions of earlier versions of these same claims.

The numbers here come from one domain and one encoder family. The *structural*
warnings generalise — that is what the bird replication establishes — but if you
run narrowcast somewhere new, its measurements are the ones to trust, not these.

MIT — see [LICENSE](LICENSE).
