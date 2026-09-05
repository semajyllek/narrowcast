"""`narrowcast plan | build | card`.

Usage:
    PYTHONPATH=. .venv/bin/python -m narrowcast.cli plan --labels my.txt --budget 20
    PYTHONPATH=. .venv/bin/python -m narrowcast.cli build --labels my.txt --out models/my
    PYTHONPATH=. .venv/bin/python -m narrowcast.cli card models/my
"""

import argparse
import sys
from pathlib import Path

from narrowcast import build as B
from narrowcast import card as C
from narrowcast import build as B  # noqa: F811
from narrowcast import config as CFG
from narrowcast import encoders, hub as HUB, labels as S, plan as P
from narrowcast import sources as SRC, sweep as SW


def _species_arg(args) -> list[str]:
    if args.labels:
        return S.read_list(args.labels)
    if args.name:
        out = []
        for n in args.name:
            c = S.canonical(n)
            if not c:
                raise ValueError(f"could not parse {n!r} as 'Genus labels'")
            out.append(c)
        return list(dict.fromkeys(out))
    raise ValueError("give --labels FILE or one or more --name 'Genus labels'")


def cmd_plan(args):
    chosen = _species_arg(args)
    pl = P.make_plan(chosen, budget_mb=args.budget, p_ood=args.ood_rate,
                     encoder=args.encoder)
    print()
    print(P.render(pl))
    print()
    print(f"  Next: narrowcast build --labels {args.labels or '<list>'} "
          f"--encoder {pl['encoder'].variant}")
    print()
    return 0


def _hazard_arg(args, chosen) -> list[str]:
    """Labels the user declares consequential. The tool cannot infer these."""
    out = list(args.hazard or [])
    if getattr(args, "hazard_file", None):
        out += S.read_list(args.hazard_file)
    out = [S.canonical(h) or h for h in out]
    unknown = sorted(set(out) - set(chosen))
    if unknown:
        raise ValueError(f"consequential labels not in your labels list: "
                         f"{', '.join(unknown)}")
    return sorted(set(out))


def cmd_build(args):
    enc = (encoders.BY_VARIANT[args.encoder] if args.encoder
           else encoders.choose(args.budget))
    external = args.images or args.manifest or args.embeddings

    if external:
        rows = SRC.load(args.images, args.manifest, args.embeddings)
        bg = (SRC.load(args.background_images, args.background_manifest,
                       args.background_embeddings)
              if (args.background_images or args.background_manifest
                  or args.background_embeddings) else None)
        chosen = rows.labels
        comp = S.analyse(chosen, pool=chosen)
        print(f"encoder {enc.label}, {len(chosen)} labels, {len(rows)} rows",
              file=sys.stderr)
        for n in rows.notes:
            print(f"  note: {n}", file=sys.stderr)
        ds = B.load_rows(rows, enc.variant, background=bg)
        source = args.images or args.manifest or args.embeddings
    else:
        chosen = _species_arg(args)
        comp = S.analyse(chosen)
        print(f"encoder {enc.label} ({enc.size_mb():.1f} MB int4), "
              f"{len(chosen)} labels", file=sys.stderr)
        ds = B.load_local(enc.variant, chosen)
        missing = set(chosen) - set(ds.y_train)
        if missing:
            print(f"warning: no training rows for {len(missing)} labels: "
                  f"{', '.join(sorted(missing)[:6])}", file=sys.stderr)
        source = "local-catalogue"
    if ds.counts["in_catalog"] == 0:
        raise SystemExit("no evaluation rows -- nothing to measure")
    print(f"  train {ds.counts['train']} rows | eval in-list {ds.counts['in_catalog']}, "
          f"relatives {ds.counts['near_ood']}, unrelated {ds.counts['distant_ood']}",
          file=sys.stderr)

    hazards = _hazard_arg(args, chosen)
    clf = B.fit_head(ds)
    frame = B.score_frame(clf, ds)
    metrics = B.fit_and_measure(frame, p_ood=args.ood_rate, hazards=hazards)

    out = B.save_bundle(Path(args.out), clf, chosen, enc.variant, metrics, comp,
                        ds.counts, source=str(source), hazards=hazards)
    card_path = C.write(out)
    print(f"\nbundle {out}\ncard   {card_path}", file=sys.stderr)
    print(f"\n  coverage {100*metrics['coverage']:.1f}%  "
          f"precision {100*metrics['precision']:.1f}%  "
          f"label-level {100*metrics['label_share']:.1f}%")
    return 0


def _candidates(cfg):
    """Encoders to try: the config's own list, else the built-in registry plus a
    Hub search, both filtered to the size budget."""
    if cfg.encoders:
        return [(e, encoders.BY_VARIANT[e].size_mb() if e in encoders.BY_VARIANT else None)
                for e in cfg.encoders]
    built_in = [(e.variant, e.size_mb()) for e in encoders.ENCODERS
                if cfg.max_size_mb is None or e.size_mb() <= cfg.max_size_mb]
    found = []
    if cfg.domain:
        fitting, _, _ = HUB.search(cfg.domain, budget_mb=cfg.max_size_mb)
        found = [(c.repo, c.size_mb()) for c in fitting]
    seen, out = set(), []
    for name, size in built_in + found:
        if name not in seen:
            seen.add(name)
            out.append((name, size))
    return out[: cfg.max_candidates]


def cmd_fit(args):
    """Search encoders under a size budget until one clears the metric floor."""
    cfg = CFG.load(args.config)
    rows = SRC.load(**cfg.data)
    background = SRC.load(**cfg.background) if cfg.background else None
    cands = _candidates(cfg)
    if not cands:
        raise SystemExit(f"no candidate encoders fit {cfg.max_size_mb} MB")
    # Precomputed vectors pin the encoder: `load_rows` returns them untouched and
    # never runs a model, so sweeping N encoders over one embeddings file scores
    # the same numbers N times and calls them a frontier. Refuse rather than
    # produce a comparison that cannot be real.
    if "embeddings" in cfg.data and len(cands) > 1:
        raise SystemExit(
            f"`data.embeddings` pins the encoder that produced it, so a sweep over "
            f"{len(cands)} candidates would score identical numbers {len(cands)} times.\n"
            f"Use `data.images` or `data.manifest` to sweep, or name exactly one "
            f"encoder in `constraints.encoders`.")

    print(f"\n  task {cfg.task}: {len(rows)} rows, {len(rows.labels)} labels")
    print(f"  trying {len(cands)} encoder(s) against {cfg.metric} >= {cfg.minimum}\n",
          flush=True)
    results = SW.run(cands, rows, background, cfg.metric, cfg.ood_rate, cfg.hazards,
                     on_result=lambda r: print(
                         f"    {r.encoder[:38]:38s} "
                         f"{'failed' if r.error else f'{r.value(cfg.metric):.4f}'}",
                         file=sys.stderr, flush=True))
    print()
    print(SW.render(results, cfg.metric, cfg.minimum, cfg.max_size_mb))

    pick = SW.choose(results, cfg.metric, cfg.minimum)
    if pick is None:
        best, gap = SW.shortfall(results, cfg.metric, cfg.minimum)
        print(f"\n  REFUSED: nothing reached {cfg.metric} >= {cfg.minimum}.")
        if best is not None:
            print(f"  Closest was {best.encoder} at {best.value(cfg.metric):.4f}, "
                  f"short by {gap:.4f}.")
        print("  No bundle written. Lower the floor, raise the size budget, or "
              "supply better data.\n")
        return 1

    print(f"\n  selected {pick.encoder} — smallest that clears the floor\n")
    ds = B.load_rows(rows, pick.encoder, background=background)
    clf = B.fit_head(ds)
    comp = S.analyse(rows.labels, pool=rows.labels)
    out = B.save_bundle(Path(args.out), clf, rows.labels, pick.encoder, pick.metrics,
                        comp, ds.counts, source=str(cfg.data), hazards=list(cfg.hazards))
    print(f"  bundle {out}\n  card   {C.write(out)}\n")
    return 0


def cmd_encoders(args):
    """Candidate encoders from the Hub, size-verified locally."""
    terms = tuple(args.domain or ())
    print(f"\n  searching the Hub"
          + (f" for: {', '.join(terms)}" if terms else " (no domain terms)"), flush=True)
    f, o, u = HUB.search(terms, budget_mb=args.budget)
    print()
    print(HUB.render(f, o, u, budget_mb=args.budget, top=args.top))
    print()
    return 0


def cmd_card(args):
    manifest = B.load_bundle(Path(args.bundle))
    print(C.render(manifest))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="narrowcast", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, out=False):
        p.add_argument("--labels", help="file with one label per line")
        p.add_argument("--images", metavar="DIR", help="DIR/<label>/*.jpg")
        p.add_argument("--manifest", metavar="FILE",
                       help="parquet/csv with columns label, path [, group, cluster]")
        p.add_argument("--embeddings", metavar="FILE",
                       help="npz with descriptor, label [, group, cluster]")
        p.add_argument("--name", action="append", help="a labels; repeatable")
        p.add_argument("--budget", type=float, metavar="MB",
                       help="size budget for the encoder, in MB")
        p.add_argument("--encoder", choices=sorted(encoders.BY_VARIANT),
                       help="override the budget-based choice")
        p.add_argument("--ood-rate", type=float, default=0.2, metavar="P",
                       help="assumed share of queries not on your list (default 0.2)")
        if out:
            p.add_argument("--out", required=True, help="bundle directory to write")
            p.add_argument("--hazard", action="append", metavar="LABEL",
                           help="a label where being mistaken for a harmless one "
                                "is the costly error; repeatable")
            p.add_argument("--hazard-file", metavar="FILE",
                           help="file of such labels, one per line")
            p.add_argument("--background-images", metavar="DIR")
            p.add_argument("--background-manifest", metavar="FILE")
            p.add_argument("--background-embeddings", metavar="FILE",
                           help="negatives, so the model can learn to decline")

    p_plan = sub.add_parser("plan", help="what this labels list will give you")
    common(p_plan)
    p_plan.set_defaults(fn=cmd_plan)

    p_build = sub.add_parser("build", help="fit, measure, and write a bundle")
    common(p_build, out=True)
    p_build.set_defaults(fn=cmd_build)

    p_enc = sub.add_parser("encoders", help="find candidate encoders that fit a size budget")
    p_enc.add_argument("--domain", action="append", metavar="TERM",
                       help="domain term, e.g. plant / car / painting; repeatable")
    p_enc.add_argument("--budget", type=float, metavar="MB",
                       help="size budget in MB (int4 assumed)")
    p_enc.add_argument("--top", type=int, default=12)
    p_enc.set_defaults(fn=cmd_encoders)

    p_fit = sub.add_parser("fit", help="search encoders under a size budget until "
                                       "one clears a metric floor")
    p_fit.add_argument("--config", required=True, help="task config (YAML or JSON)")
    p_fit.add_argument("--out", required=True, help="bundle directory to write")
    p_fit.set_defaults(fn=cmd_fit)

    p_card = sub.add_parser("card", help="print the card for a built bundle")
    p_card.add_argument("bundle")
    p_card.set_defaults(fn=cmd_card)

    args = ap.parse_args(argv)
    if args.cmd == "plan" and args.ood_rate not in P.projection.measured_p_ood():
        ap.error(f"--ood-rate for `plan` must be one of "
                 f"{P.projection.measured_p_ood()} (the rates measured); "
                 f"`build` accepts any value because it fits on your data")
    try:
        return args.fn(args)
    except (ValueError, FileNotFoundError) as e:
        ap.error(str(e))


if __name__ == "__main__":
    raise SystemExit(main())
