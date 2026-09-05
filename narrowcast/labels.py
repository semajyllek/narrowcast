"""Reading a user's species list, and finding the relatives they left out.

Two things happen here, and the second is the one that earns the tool its keep.

**Congeners inside the set** drive how the model answers. When several chosen
species share a genus the cascade can fall back to a genus answer that is
technically correct and practically empty -- "it is a Sedum" on a list of six
Sedums -- so a high in-set congener fraction predicts a low species-level share.

**Congeners outside the set** drive how the model fails. A relative that is not
in the catalogue has no correct label available, and the measurements show
near-OOD is where rejection is weakest. Those are the species a user will be
confidently told are something else, and naming them is more useful than any
aggregate.

The reference pool for "what else is in this genus" is the local catalogue.
That is a floor, not a census: it holds 530 labels collapsing to 499 binomials
over 172 genera, so a genus whose relatives it never included will look safer
than it is. Widening it needs only a taxonomy, not images -- but note that at a
pool of thousands this check inverts: nearly every species would have relatives
outside the set, and "shares a genus" stops carrying information. The warning
would need to rank by embedding distance instead.
"""

import re
from collections import Counter, defaultdict
from pathlib import Path



def canonical_name(name: str) -> str:
    """Genus + epithet, keeping the hybrid marker and normalising `×` to `x`.

    One built-in label format, not a requirement. Supplied because biological
    binomials carry authorities ('Sedum acre L.') that must collapse to a single
    key, and because two spellings of a join key is a silent mismatch.
    """
    tokens = str(name).replace("×", " x ").split()
    if not tokens:
        return ""
    out, i = [tokens[0]], 1
    if i < len(tokens) and tokens[i].lower() == "x":
        out.append("x")
        i += 1
    if i < len(tokens):
        out.append(tokens[i])
    return " ".join(out)


_COMMENT = re.compile(r"#.*$")
# Validation only. Normalisation is `curation.canonical_name`'s job -- it is the
# join key the background pool and every existing evaluation already use, and a
# second implementation here disagreed with it on three catalogue names
# ('Pelargonium spp.', and the two hybrids, where it kept '×' against the
# repo's 'x'). Two spellings of a join key is a silent mismatch, not a style
# difference.
_BINOMIAL = re.compile(r"^[A-Z][a-z\-]+(?:\s+[×x])?\s+[a-z\-]+")


def canonical(name: str) -> str | None:
    """Canonical binomial, or None if it does not parse as one."""
    name = _COMMENT.sub("", str(name)).strip()
    if not name or not _BINOMIAL.match(name):
        return None
    return canonical_name(name)


def normalise(name: str, binomial: bool = True) -> str | None:
    """A usable label, or None for a blank/comment line.

    `binomial=True` applies the Linnaean join key, which is what plant work
    needs. Any other domain -- defect classes, SKUs, fungi with cultivar
    suffixes -- passes labels through untouched, because the tool has no
    business deciding what a well-formed label looks like outside biology.
    """
    raw = _COMMENT.sub("", str(name)).strip()
    if not raw:
        return None
    # In binomial mode a label that does not parse is an error, not something to
    # pass through: falling back to the raw string would turn a typo into a class.
    return canonical(raw) if binomial else raw


def read_list(path: str | Path, binomial: bool | None = None) -> list[str]:
    """One label per line; blank lines and `#` comments ignored.

    `binomial=None` auto-detects: if every line parses as `Genus species` the
    Linnaean join key is applied, otherwise labels pass through as written. That
    keeps plant lists normalised without rejecting a domain whose labels are not
    Latin binomials.

    With `binomial=True` an unparseable line raises rather than being dropped: a
    silently ignored label is a model that quietly cannot see a class the user
    asked for.
    """
    lines = [ln for ln in Path(path).read_text().splitlines()
             if _COMMENT.sub("", ln).strip()]
    if binomial is None:
        binomial = bool(lines) and all(canonical(ln) for ln in lines)
    out, bad = [], []
    for i, line in enumerate(lines, 1):
        c = normalise(line, binomial=binomial)
        (out.append(c) if c else bad.append((i, line.strip())))
    if bad:
        detail = "; ".join(f"line {i}: {t!r}" for i, t in bad[:5])
        raise ValueError(f"could not parse {len(bad)} line(s) -- {detail}")
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def group_of(label: str) -> str:
    """Coarse rank: first whitespace token unless told otherwise."""
    return str(label).split()[0] if str(label).split() else str(label)


def analyse(chosen: list[str], pool: list[str] | None = None) -> dict:
    """Composition of a species list, plus the relatives it leaves outside.

    `in_set_congener_frac` is the share of chosen species sharing a genus with
    another chosen species. It is the axis the projection interpolates on, and
    the measured arms sit at roughly 0.10-0.44 (unrelated draws) and 1.00
    (genus-dense draws).
    """
    pool = list(chosen) if pool is None else pool
    chosen = list(dict.fromkeys(chosen))
    gcount = Counter(group_of(s) for s in chosen)

    by_genus = defaultdict(list)
    for s in pool:
        by_genus[group_of(s)].append(s)

    inside = {g: n for g, n in gcount.items() if n >= 2}

    # Grouped by genus, not by species: on a list of eight Sedums, a per-species
    # listing repeats the same thirty relatives eight times and buries the one
    # fact that matters -- how much of each genus was left outside.
    outside = {}
    for g in gcount:
        rel = sorted(set(by_genus.get(g, [])) - set(chosen))
        if rel:
            outside[g] = rel
    n_exposed = sum(gcount[g] for g in outside)

    n = max(len(chosen), 1)
    return {
        "labels": chosen,
        "n_labels": len(chosen),
        "n_groups": len(gcount),
        "in_set_sibling_frac": sum(gcount[group_of(s)] >= 2 for s in chosen) / n,
        "crowded_groups": dict(sorted(inside.items(), key=lambda kv: -kv[1])),
        "outside_siblings": dict(sorted(outside.items(), key=lambda kv: -len(kv[1]))),
        "n_groups_with_outside": len(outside),
        "n_labels_exposed": n_exposed,
        "pool_size": len(pool),
        # underscore-prefixed keys are working state, stripped before serialising
        "_pool": pool,
    }
