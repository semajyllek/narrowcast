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

## Constraint-driven: `fit`

Declare what you need and let it search:

```yaml
task: oregon-plants
data:        { manifest: ./data.parquet, background: { manifest: ./neg.parquet } }
objective:   { metric: label_share, minimum: 0.90 }
constraints: { max_size_mb: 50, encoders: [plantclef24, mobileclip2_s2] }
```

```bash
narrowcast fit --config task.yaml --out models/mine
```

Candidates are the encoders you name in `constraints.encoders`, or the built-in
registry filtered to your budget if you name none. **`fit` makes no network
calls and its candidate set does not change between runs.** Each runs through the
same path a single `build` uses, so the frontier and the card cannot disagree:

```
  encoder                       int4    label_share
  plantclef24                  43.3M         0.9246  ok    cov 0.790 prec 0.994
  mobileclip2_s2               17.9M         0.7166        cov 0.612 prec 0.973

  floor: label_share >= 0.9000, budget 50 MB
  selected plantclef24 — smallest that clears the floor
```

It picks the **smallest that clears the floor**, not the best: size is the
constraint you declared, and accuracy above your floor is not worth paying bytes
for.

**When nothing clears it, `fit` refuses** — prints the frontier, names the
closest candidate and the gap, writes no bundle, and exits non-zero so CI can
gate on it:

```
  REFUSED: nothing reached label_share >= 0.99.
  Closest was plantclef24 at 0.9379, short by 0.0521.
  No bundle written. Lower the floor, raise the size budget, or supply better data.
```

That refusal is the point. `minimum` is only a meaningful contract because the
card's numbers are trustworthy.

> `data.embeddings` pins the encoder that produced them, so `fit` refuses to
> sweep several candidates over one embeddings file — it would score identical
> numbers N times and present them as a comparison. Sweeping needs `images` or
> `manifest`.

## Choosing an encoder

**You choose it; the tool does not go looking.** Name one or more in
`constraints.encoders`, or omit the key and `fit` sweeps the built-in registry
within your budget. The registry is image-only, so any other modality must name
its encoder.

### Known-good encoders by domain

Sizes are int4 and were **counted from the loaded image tower**, not quoted from
a paper. Entries marked *precomputed* are used through `--embeddings`: narrowcast
never loads them, so the name is a label and the size is the upstream model's.

| domain | encoder | int4 | why |
|---|---|---|---|
| natural imagery, general | `mobileclip2_s0` | 5.7 MB | smallest that works at all |
| natural imagery, general | `mobileclip2_s2` | 17.9 MB | the small-budget default |
| **plants, fungi, animals** | `bioclip2` | 152 MB | strongest measured here; ties a server model at species rank |
| **plants specifically** | `plantclef24` | 43.3 MB | 1.5pp behind BioCLIP-2, *ahead* on hazard safety; 518px, so slower than its bytes suggest |
| audio, environmental | `ast-audioset` *(precomputed)* | — | general AudioSet model, not tuned per task |
| speech / keywords | `wav2vec2-base` *(precomputed)* | — | general speech model; groups by phonetics, not meaning |
| text | `all-MiniLM-L6-v2` *(precomputed)* | — | 22.7M sentence encoder |

Byte order is not speed order: `plantclef24` is a third of BioCLIP-2's parameters
and roughly twice its latency, because it runs at 5.3× the pixels. Storage and
compute are separate budgets and the registry ranks only one.

### Refreshing this list (maintenance)

```bash
narrowcast encoders --domain plant --domain flora --budget 50
```

Queries the Hugging Face Hub and verifies each candidate's size **client-side**
from `safetensors.total`. The Hub's own `num_parameters` filter cannot be trusted
— asking for `<100M` returns 303M models, because repos without an indexed count
pass through silently. Candidates whose size is not published are listed
separately rather than assumed small.

**This is a maintenance command, not a build step.** Nothing it prints can be fed
straight to `fit`: `encode.load_encoder` resolves variants against its own table,
so adopting a model means adding it there (how to load it) and to
`encoders.ENCODERS` (its measured size), then naming it in your config. `fit`
itself never touches the network — a candidate set that depends on what the Hub
returned today is not one you can reproduce tomorrow.

For many domains someone has already fine-tuned an encoder, and using theirs
beats compressing a general one. That is not speculation: the strongest
size/accuracy point in the parent project came from an off-the-shelf
domain-fine-tuned ViT-B that matched a model 3.5× its size and beat it on hazard
safety — found by accident.

**These are candidates, not recommendations.** Ranking is a name match plus
popularity, which is a weak proxy for fitness: a plant query returns mostly
*disease* classifiers when you asked about species. The search narrows the field;
`build` and `card` decide. That division is deliberate — the search does not need
to be clever when evaluating a candidate honestly is cheap.

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

Extracted from [plantid](https://github.com/semajyllek/narrowcast-plantid). Every claim on
this page is backed by a measurement there — see `EMBEDDED_FINDINGS.md`,
`OREGON_SAFETY_FINDINGS.md`, `BIRDS_FINDINGS.md` and `CONTAMINATION_FINDINGS.md`,
including two in-place retractions of earlier versions of these same claims.

The numbers here come from one domain and one encoder family. The *structural*
warnings generalise — that is what the bird replication establishes — but if you
run narrowcast somewhere new, its measurements are the ones to trust, not these.

MIT — see [LICENSE](LICENSE).
