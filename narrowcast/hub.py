"""Find candidate encoders on the Hugging Face Hub that fit a size budget.

Two things motivate this. First, the encoder shortlist a tool ships with goes
stale: the best result in the parent project came from `plantclef24`, an
off-the-shelf domain-fine-tuned ViT-B that matched a model 3.5x its size and beat
it on hazard safety -- and it was found by accident, from a registry entry that
had been dismissed for an unrelated reason. Second, for most domains somebody has
probably already fine-tuned an encoder, and using theirs beats compressing a
general one.

**The Hub's own parameter filter cannot be trusted.** Asking for `<100M` returns
303M models, because not every repo has an indexed parameter count and the filter
silently passes those through. Measured against hub 1.29.0:

    num_parameters="<100M"          -> 5 results, 1 over budget
    num_parameters=(0, 100_000_000) -> 1 result,  0 over budget  (strict, but sparse)
    num_parameters={"max": 1e8}     -> 5 results, 1 over budget

So the Hub is used for *candidate generation* only, and every candidate's size is
verified client-side from `safetensors.total`, which is authoritative when
present. Candidates without a published parameter count are returned separately
and flagged rather than dropped or assumed small -- a model whose size is unknown
is not a model that fits.

Nothing here downloads weights. It reads metadata and returns a shortlist for a
human, or for `plan`, to choose from.
"""

import math
import re
from dataclasses import dataclass

# Task tags that denote "produces an image representation", in rough order of how
# directly usable the output is as a frozen embedding.
PIPELINES = ("image-feature-extraction", "zero-shot-image-classification",
             "image-classification")


@dataclass(frozen=True)
class Candidate:
    repo: str
    params: int | None
    downloads: int
    likes: int
    pipeline: str | None
    library: str | None
    matched: tuple = ()       # which domain terms appear in the repo id, if any

    @property
    def domain_hit(self) -> bool:
        return bool(self.matched)

    def size_mb(self, bits: int = 4) -> float | None:
        return None if self.params is None else self.params * bits / 8 / 1e6

    def fits(self, budget_mb: float | None, bits: int = 4) -> bool | None:
        """True / False / None where None means 'size not published'."""
        if budget_mb is None:
            return True
        s = self.size_mb(bits)
        return None if s is None else s <= budget_mb


def _matched_terms(repo: str, terms) -> tuple:
    """Domain terms appearing in the repo id as whole words.

    Two failures shaped this rule, in opposite directions.

    Pure substring matching is harmful: searching "art" put `HFO-artifact` (EEG
    artifacts) and `vit-brats-artifact-classifier` (brain tumours) at the top of
    an artwork query, because the Hub's text search is substring-based.

    Pure whole-token matching is also wrong: it scored `gerald29/plantclef2024`
    as having no domain relevance for a plant query, because the id tokenises to
    `plantclef2024` and "plant" is not a token of it. That is the exact class of
    model this feature exists to surface -- a domain-fine-tuned encoder whose name
    concatenates the domain with a benchmark.

    So: whole token (or its plural), **or** a token beginning with the term when
    the term is at least four characters. "plant" prefixes "plantclef2024"; "art"
    at three characters is held to whole-token matching and no longer reaches
    "artifact".
    """
    tokens = {t for t in re.split(r"[^a-z0-9]+", repo.lower()) if t}
    hits = []
    for term in terms:
        tl = term.lower()
        if tl in tokens or f"{tl}s" in tokens:
            hits.append(term)
        elif len(tl) >= 4 and any(tok.startswith(tl) for tok in tokens):
            hits.append(term)
    return tuple(hits)


def _params(info) -> int | None:
    st = getattr(info, "safetensors", None)
    if st is None:
        return None
    total = getattr(st, "total", None)
    return int(total) if total else None


def search(domain_terms=(), budget_mb=None, bits=4, per_query=50, api=None):
    """Candidate encoders, size-verified client-side.

    `domain_terms` are free text ("plant", "bird", "car", "artwork"). Passing
    none returns strong general encoders, which is the right default when no
    domain-adapted model exists.

    Returns (fitting, oversize, unknown_size) -- three lists, because a candidate
    whose size the Hub does not publish is a different situation from one that is
    too big, and collapsing them would hide it.
    """
    from huggingface_hub import HfApi
    api = api or HfApi()

    seen, out = {}, []
    queries = [{"pipeline_tag": p} for p in PIPELINES]
    for term in domain_terms:
        queries += [{"search": term, "pipeline_tag": p} for p in PIPELINES]

    for q in queries:
        try:
            models = api.list_models(sort="downloads", limit=per_query,
                                     expand=["downloads", "likes", "safetensors",
                                             "pipeline_tag", "library_name"], **q)
        except Exception:
            continue                      # one bad query must not kill the search
        for m in models:
            if m.id in seen:
                seen[m.id] = seen[m.id] or bool(q.get("search"))
                continue
            seen[m.id] = bool(q.get("search"))
            out.append((m, q.get("search")))

    cands = [Candidate(repo=m.id, params=_params(m), downloads=m.downloads or 0,
                       likes=m.likes or 0, pipeline=getattr(m, "pipeline_tag", None),
                       library=getattr(m, "library_name", None),
                       matched=_matched_terms(m.id, domain_terms))
             for m, _ in out]

    fitting = [c for c in cands if c.fits(budget_mb, bits) is True]
    oversize = [c for c in cands if c.fits(budget_mb, bits) is False]
    unknown = [c for c in cands if c.fits(budget_mb, bits) is None]

    # Domain match is worth roughly two orders of magnitude of popularity -- enough
    # that a genuine fine-tune outranks a general backbone (the plantclef24
    # lesson), not so much that a 19-download repo of unknown provenance outranks
    # mobilenetv3 at 17M. Popularity is logged because download counts span seven
    # orders of magnitude and raw counts would otherwise decide everything.
    def rank(c):
        return -(2.0 * bool(c.matched) + math.log10(c.downloads + 1) / 3.0)

    return (sorted(fitting, key=rank), sorted(oversize, key=rank),
            sorted(unknown, key=rank))


def render(fitting, oversize, unknown, budget_mb=None, bits=4, top=12) -> str:
    L = []
    b = f"{budget_mb:.0f} MB int{bits}" if budget_mb else "no budget"
    L.append(f"  {len(fitting)} candidates fit ({b}), "
             f"{len(oversize)} too large, {len(unknown)} size not published\n")
    L.append(f"  {'repo':48s} {'int4':>8s} {'downloads':>11s}  matched")
    for c in fitting[:top]:
        s = c.size_mb(bits)
        L.append(f"  {c.repo[:48]:48s} {s:7.1f}M {c.downloads:11,}  "
                 f"{','.join(c.matched) or '-'}")
    if unknown:
        L.append("\n  size not published — cannot be said to fit, shown for triage:")
        for c in unknown[:4]:
            L.append(f"  {c.repo[:48]:48s} {'?':>8s} {c.downloads:11,}  "
                     f"{','.join(c.matched) or '-'}")
    L.append("\n  Candidates, not recommendations. `matched` says which of your terms")
    L.append("  appears in the repo id — it is a name match, not evidence of quality.")
    return "\n".join(L)
