# narrowcast

**Audit a classifier over a narrow label set: what it names, what it retreats to,
and what it declines.**

```bash
narrowcast audit --scores     scores.npz --out models/mine   # a model you already have
narrowcast audit --embeddings vecs.npz   --out models/mine   # vectors, head fitted here
narrowcast card  models/mine
```

Training a classifier on your own classes is commodity — a dozen tools do it.
None of them tell you *what you actually got*, and for narrow label sets the
metrics everyone publishes are actively misleading.

narrowcast does not choose an encoder, fetch a dataset, or read a pixel. You
bring the model; it tells you how the model will fail.

> **This used to be a builder.** It searched encoders under a size budget,
> projected outcomes from a shipped grid, and discovered candidates on the Hub.
> That part did not work — the configuration space is enormous and almost
> entirely data-dependent — and it is gone as of 0.2.0. The measurement is what
> held up. See [`docs/deep_dive.html`](docs/deep_dive.html), and `DISPOSITION.md`
> in [narrowcast-plantid](https://github.com/semajyllek/narrowcast-plantid) for
> the reasoning.

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
best. So no report here prints coverage without the label-level share beside it,
and the card gates on **measured** retreat rather than asserting where the
answers went.

Reproduced outside biology: on birds, a *Larus*/*Calidris* set scored **higher**
coverage than 13 distinct genera while label-level fell 0.958 → 0.718. The trap
is a property of hierarchical label sets, not of plants.

The full account — what governs the trap, when it fires, and the operating point
that decides whether it fires at all — is in
**[docs/deep_dive.html](docs/deep_dive.html)**, an 18-section technical reference
with every number traced to a findings doc.

## Two ways in, both self-contained

| flag | shape | what gets fitted |
|---|---|---|
| `--scores FILE` | npz: `proba`, `classes`, `label` [, `group`, `cluster`, `origin`] | the two thresholds |
| `--embeddings FILE` | npz: `descriptor`, `label` [, `group`, `cluster`, `origin`] | a logistic head, then the thresholds |

**`--scores` is the audit path.** `proba` is one row per observation and one
column per entry of `classes`, from whatever produced it — a logistic head, a
fine-tuned network, an ensemble, a vendor API. Rows need not sum to 1: a model
that abstains by leaving mass unassigned is still auditable, because the cascade
compares the largest label mass against the largest group mass and both survive a
positive rescale.

Any row whose `label` is not among `classes` is **out-of-list** by construction,
and is bucketed by its group: a relative of something on your list (`near_ood`,
reliably the weakest bucket) or unrelated (`distant_ood`). So the audit path needs
no separate negatives file.

**`--embeddings` still fits a head**, and takes `--background-embeddings` for
negatives. Without them there is no reject class: the model is closed-set, cannot
decline, and the card says so rather than implying a capability that was never
fitted.

Both paths converge on one function — `build.frame_from_posteriors` — so a model
narrowcast fitted and a model it merely audited are measured by identical code
and cannot drift apart.

**`cluster`** is the unit that must not straddle a train/test split — several
photographs of one subject, one specimen, one production run. Without it every
row is treated as independent and the card records that its intervals are
anticonservative.

**`group`** is the coarse rank the cascade retreats to, defaulting to the label's
first whitespace token. Right for Linnaean binomials, overridable everywhere else.

**`origin`** is which acquisition source or population a row came from — a
corpus, a device, a skin-type band, a speaker group. Optional, and it changes
nothing unless you supply it. Supply it when your rows come from more than one,
because the labels that have training rows from the origin you will actually
deploy against and the labels that do not **are not comparable**, and the ones
that do not are measurably worse off than if none had them. Pass
`--deployment-origin NAME` and the card measures it (`--embeddings` only — it
refits a head twice, and with `--scores` there is no head to refit):

## Origin composition — deployment origin `clinic_dark`

**3 of 28 labels have no training data from `clinic_dark`**, the origin this
model will be used against.

| labels | with the data | without it | measured effect |
|---|---|---|---|
| have it (25) | 0.679 | 0.529 | +0.149 |
| lack it (3)  | 0.647 | 0.683 | -0.036 |
```

Two heads on the same label set — one fitted on everything, one with every
deployment-origin training row removed — scored on the same held-out rows. It is
**measured rather than warned about** because its size is domain-dependent: ~0 on
plants once the label set is narrow, 10–20 points on dermatology and keyword
spotting at every label count tried. **It tracks the accuracy of the build, not
the number of labels, so it cannot be inferred from a small `K`.** Labels with no
deployment-origin rows *at all* cannot be scored; they are counted and named
rather than averaged away, because they are the most exposed and the least
measurable.

Where the data came from — which corpus, under what licence, reconciled against
whose taxonomy — is a domain decision, so it lives in your project, not here.

## Three commands

**`audit`** — fits label/group/decline thresholds by expected-utility
maximisation on a clustered calibration split, evaluates against the held-out
half, and writes a bundle plus a card. With `--embeddings` it fits a logistic head
first; with `--scores` your model *is* the head.

**`card`** — the report. Coverage, precision, label-level share, per-bucket
behaviour, cluster-bootstrapped intervals, measured headroom and the full
label/group/decline split, and a **gate** on labels you declared consequential.

**`predict`** — run a bundle *this tool fitted*, answering the way the card says
it answers. Takes `--embeddings` from the same encoder the bundle names; there is
no encoder here, so there is no way to hand it photographs.

```bash
narrowcast predict models/mine --embeddings ./new-vectors.npz
```

```
row 0    Sedum acre              0.969
row 1    Sedum (group only)      0.941
row 2    declined                0.612

972 rows — 316 named to a label (32.5%), 587 answered at group only, 69 declined
```

Three answers, not one. A bare argmax would report a label for all 972 rows and
throw away the only thing that makes a narrow-catalogue model honest — the option
to answer at the coarse rank, or not at all. **The summary line is the point**: a
run that answers 60% of its rows at group level is working exactly as fitted, and
a caller shown only the confident rows would never know.

The scores are recomputed from the saved weights and the decision uses the two
thresholds exactly as fitted, so **a prediction and the card cannot disagree** —
pinned by a test that checks both against the same arithmetic the measurement
used.

A bundle from `--scores` carries measurements but **no weights**, and `predict`
refuses it by name. We measured your model; we did not obtain a copy of it.

## Consequential labels

```bash
narrowcast audit --scores ./scores.npz --hazard "Conium maculatum" --out models/mine
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

## Install

```bash
pip install -e .
```

numpy, pandas and scikit-learn. **There is no `[encode]` extra** and torch is not
a dependency at any level — narrowcast never turns images into vectors, so
whatever produced your vectors or scores keeps ownership of that.

## Provenance

**The reference document is [`docs/deep_dive.html`](docs/deep_dive.html)** — 18
sections covering the decision rule and its calibration, the evaluation protocol,
the metric pathology this project exists to expose, and the experiments that
established or killed each claim. Open it in a browser; it is self-contained.
Three episodes where a wrong number was believed for a time are recorded in place
in §6, §8 and §14, struck through rather than deleted.

Extracted from [plantid](https://github.com/semajyllek/narrowcast-plantid). Every claim on
this page is backed by a measurement there — see `EMBEDDED_FINDINGS.md`,
`OREGON_SAFETY_FINDINGS.md`, `BIRDS_FINDINGS.md` and `CONTAMINATION_FINDINGS.md`,
including two in-place retractions of earlier versions of these same claims.

The numbers here come from one domain and one encoder family. The *structural*
warnings generalise — that is what the bird replication establishes — but if you
run narrowcast somewhere new, its measurements are the ones to trust, not these.

MIT — see [LICENSE](LICENSE).
