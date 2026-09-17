"""narrowcast — audit a classifier over a narrow label set.

    narrowcast audit --scores     scores.npz  --out models/mine
    narrowcast audit --embeddings vecs.npz    --out models/mine
    narrowcast card    models/mine
    narrowcast predict models/mine --embeddings new.npz

narrowcast does not choose an encoder, fetch a dataset, or read a pixel. You
bring a model's posteriors (`--scores`) or vectors it can fit a linear head over
(`--embeddings`), and it tells you what you actually have: the three-way split of
label / group / decline, at a prevalence you declare rather than one your
evaluation set happened to contain.

It used to try to build the model for you -- searching encoders under a size
budget, projecting outcomes from a shipped grid, discovering candidates on the
Hub. That is gone. See `docs/deep_dive.html` and `DISPOSITION.md` in
narrowcast-plantid for why the measurement was the part worth keeping.
"""

import argparse
import json
import sys
from pathlib import Path

from narrowcast import build as B
from narrowcast import cascade as CA
from narrowcast import card as C
from narrowcast import labels as S
from narrowcast import predict as PRED, sources as SRC


def _hazard_arg(args, chosen) -> list[str]:
    """Labels the user declares consequential. The tool cannot infer these."""
    out = list(args.hazard or [])
    if getattr(args, "hazard_file", None):
        out += S.read_list(args.hazard_file)
    out = [S.canonical(h) or h for h in out]
    unknown = [h for h in out if h not in set(chosen)]
    if unknown:
        raise SystemExit(
            "these --hazard labels are not in the label set: "
            + ", ".join(unknown)
            + "\nA hazard is a label you already have; naming one you do not "
              "measures nothing.")
    return out


def cmd_audit(args):
    rows = SRC.load(embeddings=args.embeddings, scores=args.scores)
    scored = args.scores is not None

    if scored:
        if args.background_embeddings:
            raise SystemExit(
                "--background-embeddings is for --embeddings only. With --scores "
                "the out-of-list rows are already in the file: any row whose "
                "label is not among `classes` is one, and it is bucketed by group.")
        chosen = sorted(set(rows.classes.tolist()))
        ds = B.load_scored(rows)
        source = args.scores
    else:
        bg = (SRC.from_embeddings(args.background_embeddings)
              if args.background_embeddings else None)
        chosen = rows.labels
        ds = B.load_rows(rows, args.encoder_name, background=bg)
        source = args.embeddings

    gmap = dict(zip(rows.label.tolist(), rows.group.tolist()))
    comp = S.analyse(chosen, pool=chosen, groups=gmap)

    print(f"{len(chosen)} labels, {len(rows)} rows"
          + (f", posteriors from {args.scores}" if scored else
             f", vectors from {args.embeddings}"), file=sys.stderr)
    for n in ds.counts.get("notes", []):
        print(f"  note: {n}", file=sys.stderr)
    if ds.counts["in_catalog"] == 0:
        raise SystemExit("no in-list evaluation rows -- nothing to measure")
    print(f"  eval in-list {ds.counts['in_catalog']}, "
          f"relatives {ds.counts['near_ood']}, unrelated {ds.counts['distant_ood']}",
          file=sys.stderr)

    hazards = _hazard_arg(args, chosen)

    if scored:
        # No head of ours: the posteriors are the model. `frame_from_posteriors`
        # is the same code `score_frame` runs, so an audited model and a built one
        # are measured identically and cannot drift apart.
        clf = None
        frame = B.frame_from_posteriors(rows.proba, rows.classes, ds)
    else:
        clf = B.fit_head(ds)
        frame = B.score_frame(clf, ds)

    utility = CA.PROFILES[args.profile]
    if args.profile != "standard":
        print(f"  note: utility profile {args.profile!r} -- wrong answers cost "
              f"{utility['wrong']} against the default {CA.UTILITY['wrong']}. "
              "Thresholds are fitted against these payoffs.", file=sys.stderr)
    absent = list(args.hazard_absent or [])
    if absent:
        present = [h for h in absent if h in set(chosen)]
        if present:
            raise SystemExit(
                "these --hazard-absent labels ARE on the list: "
                + ", ".join(present)
                + "\nUse --hazard for those; the two measure opposite things.")
    # You can only suppress what the model can emit, so unlike `--hazard-absent`
    # this one must name a label that IS on the list. Refusing is the convention:
    # a flag silently doing nothing is worse than a flag that will not start.
    never = list(args.never_answer or [])
    if never:
        unknown = [n for n in never if n not in set(chosen)]
        if unknown:
            raise SystemExit(
                "these --never-answer labels are not in the label set: "
                + ", ".join(sorted(unknown))
                + "\nYou can only suppress a label the model can emit. To name "
                  "an unlisted dangerous species, use --hazard-absent; to stop "
                  "this model answering with the harmless label that species "
                  "gets confused for, pass that label here.")
        if set(never) >= set(chosen):
            raise SystemExit(
                "--never-answer suppresses every label on the list, which leaves "
                "a model that can only decline. Nothing to measure.")
        print(f"  note: {len(never)} label(s) suppressed -- never answered, "
              "declined instead. Thresholds are fitted without the suppression "
              "and it is applied after, so the card reports what it cost.",
              file=sys.stderr)
    metrics = B.fit_and_measure(frame, p_ood=args.ood_rate, hazards=hazards,
                                groups=gmap, utility=utility,
                                hazards_absent=absent, never_answer=never,
                                labels=chosen)

    if args.deployment_origin:
        if scored:
            # `origin_cost` refits the head twice on different row subsets. With
            # --scores there is no head and no vectors to refit one from, so this
            # is not a thing that can be measured here -- say so instead of
            # reporting a silent null.
            print("  note: --deployment-origin needs vectors to refit a head on; "
                  "with --scores there is nothing to refit. Not measured.",
                  file=sys.stderr)
        else:
            metrics["origin_cost"] = B.origin_cost(ds, args.deployment_origin)
    elif ds.origin_eval is not None:
        origins = sorted(set(ds.origin_eval.tolist()) - {B.BG_ORIGIN})
        if not scored and len(origins) > 1:
            print(f"  note: rows carry {len(origins)} origins "
                  f"({', '.join(origins[:4])}); pass --deployment-origin to "
                  "measure what that costs", file=sys.stderr)

    out = B.save_bundle(Path(args.out), clf, chosen, args.encoder_name, metrics,
                        comp, ds.counts, source=str(source), hazards=hazards,
                        groups=gmap, utility=utility, never_answer=never)
    card_path = C.write(out)
    print(f"\nbundle {out}\ncard   {card_path}", file=sys.stderr)
    print(f"\n  coverage {100*metrics['coverage']:.1f}%  "
          f"precision {100*metrics['precision']:.1f}%  "
          f"label-level {100*metrics['label_share']:.1f}%")
    return 0


def cmd_predict(args):
    """Classify rows with a built model, through the same cascade the card measured."""
    b = PRED.Bundle(Path(args.bundle))
    for n in b.notes:
        print(f"  note: {n}", file=sys.stderr)
    X, rows = PRED.embed(b, embeddings=args.embeddings)
    results = b.predict(X)
    if args.json:
        out = [{**r, "path": (str(rows.path[i]) if rows.path is not None else None)}
               for i, r in enumerate(results)]
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.json}", file=sys.stderr)
    print(PRED.render(results, rows, limit=args.limit))
    return 0


def cmd_card(args):
    d = Path(args.bundle)
    manifest = json.loads((d / "manifest.json").read_text())
    print(C.render(manifest))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="narrowcast", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_audit = sub.add_parser(
        "audit", help="measure what a model over a narrow label set will do",
        description="Fit the two thresholds, measure the three-way split, and "
                    "write a bundle and a card.")
    src = p_audit.add_mutually_exclusive_group(required=True)
    src.add_argument("--scores", metavar="FILE",
                     help=".npz with proba, classes, label [, group, cluster, "
                          "origin] — posteriors from a model you already have")
    src.add_argument("--embeddings", metavar="FILE",
                     help=".npz with descriptor, label [, group, cluster, origin] "
                          "— a linear head is fitted over them")
    p_audit.add_argument("--out", required=True, help="bundle directory to write")
    p_audit.add_argument("--ood-rate", type=float, default=0.2, metavar="P",
                         help="assumed share of inputs that are of something not "
                              "on the list (default 0.2). This chooses the "
                              "operating point; it is not read off the data.")
    p_audit.add_argument("--encoder-name", metavar="NAME", default="unstated",
                         help="what produced the vectors or scores, recorded on "
                              "the card. No size is claimed for it.")
    p_audit.add_argument("--background-embeddings", metavar="FILE",
                         help="negatives, for --embeddings. Without them the "
                              "model is closed-set and cannot decline.")
    p_audit.add_argument("--deployment-origin", metavar="NAME",
                         help="the origin you will actually see, to measure what "
                              "it costs labels with no training rows from it "
                              "(--embeddings only)")
    p_audit.add_argument("--profile", default="standard", choices=sorted(CA.PROFILES),
                         help="declared utility profile. `wrong` is the stakes "
                              "dial: identify -2, standard -4, conserve -6, "
                              "forage -20. Payoffs are declared in source with "
                              "reasons; this selects among them, it does not tune "
                              "them.")
    p_audit.add_argument("--hazard", action="append", metavar="LABEL",
                         help="a label where being wrong is expensive; repeatable")
    p_audit.add_argument("--hazard-file", metavar="FILE",
                         help="file with one such label per line")
    p_audit.add_argument("--hazard-absent", action="append", metavar="LABEL",
                         help="a dangerous species deliberately NOT on the list, "
                              "measured for how often it receives the name of one "
                              "that is. This is the forager's case: nobody lists "
                              "poison hemlock among things they mean to eat, so "
                              "--hazard refuses it and the risk goes unmeasured. "
                              "Needs rows for it in the data. Repeatable.")
    p_audit.add_argument("--never-answer", action="append", metavar="LABEL",
                         help="a label this model will never emit: when the "
                              "cascade would name it, it declines instead. The "
                              "per-label dial the utility profile has no way to "
                              "express -- `--profile forage` makes the whole "
                              "model cautious, this makes it cautious about one "
                              "thing. Suppress the harmless LOOK-ALIKE, not the "
                              "hazard: the harm is hemlock being called wild "
                              "carrot, so the label to name here is the wild "
                              "carrot. Thresholds are fitted without it and it is "
                              "applied after, so the card prints what it cost. "
                              "Repeatable.")
    p_audit.set_defaults(func=cmd_audit)

    p_pred = sub.add_parser("predict", help="classify rows with a built bundle")
    p_pred.add_argument("bundle")
    p_pred.add_argument("--embeddings", metavar="FILE", required=True,
                        help="vectors from the same encoder the bundle names")
    p_pred.add_argument("--json", metavar="FILE", help="write full results as JSON")
    p_pred.add_argument("--limit", type=int, default=20,
                        help="rows to print; 0 for all")
    p_pred.set_defaults(func=cmd_predict)

    p_card = sub.add_parser("card", help="print the card for a built bundle")
    p_card.add_argument("bundle")
    p_card.set_defaults(func=cmd_card)

    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    sys.exit(main())
