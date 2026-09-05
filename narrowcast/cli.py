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
from narrowcast import encoders, labels as S, plan as P, sources as SRC


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
