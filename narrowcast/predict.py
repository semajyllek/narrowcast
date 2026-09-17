"""Run a model this tool built, and answer the way the card says it answers.

Without this, a bundle is a logistic regression and a promise. The user would
have to load `head.npz`, find the encoder, embed, apply the head, and then
**reimplement the cascade** from the two thresholds in the manifest -- and the
cascade is the product. A bare argmax throws away the thing that makes a
narrow-catalogue model honest: the option to answer at the group rank, or not at
all.

So the contract here is that a prediction and the card cannot disagree. The score
computation is the same arithmetic as `build.score_frame`, the decision is
`cascade.decide` with the thresholds exactly as fitted, and the group map is the
caller's own -- read from the bundle rather than re-derived, because the default
first-whitespace-token rule is a Latin-binomial convention that silently makes
every label its own group on any domain that does not use binomials.

Three answers, and the reason they are not collapsed into one:

    label     a specific label, when the model is confident enough to defend it
    group     the coarse rank only -- "some kind of Sedum"
    decline   nothing, because neither rank cleared its threshold

A caller that wants a bare argmax can read `label_conf` and ignore the rest. A
caller that wants the measured behaviour uses `answer`.
"""

import json
from pathlib import Path

import numpy as np

from narrowcast.cascade import DECLINE, GROUP, LABEL, decide, group_matrix, suppress

OTHER = "__OTHER__"


class Bundle:
    """A built model: head weights, thresholds, classes, and the group map."""

    def __init__(self, path):
        path = Path(path)
        self.path = path
        self.manifest = json.loads((path / "manifest.json").read_text())
        if not self.manifest.get("has_head", True):
            raise ValueError(
                f"{path} is an audit of a model this tool did not fit "
                f"(source: {self.manifest.get('source')!r}), so it carries "
                "measurements but no weights. Run `narrowcast card` on it; "
                "to predict, use the model you audited.")
        z = np.load(path / "head.npz", allow_pickle=True)
        self.coef, self.intercept = z["coef"], z["intercept"]
        self.classes = z["classes"].astype(str)
        sp = z["space"] if "space" in z.files else None
        self.space = None if sp is None or sp.size == 0 else np.asarray(sp, "float64")

        m = self.manifest["metrics"]
        self.t_group, self.t_label = float(m["t_group"]), float(m["t_label"])
        # The near-OOD gate, when one was fitted. Read from the manifest rather
        # than refitted: the card describes a model that answers under this
        # threshold, and a bundle that ignored it would be a different model.
        tn = m.get("t_novel")
        self.t_novel = None if tn is None else float(tn)
        self.encoder = self.manifest["encoder"]
        self.version = self.manifest.get("bundle_version", 1)
        self.groups = self.manifest.get("groups") or None
        # Labels this bundle will never emit. Read here rather than left to the
        # caller: the contract is that a prediction and the card cannot disagree,
        # and the card was written from a measurement that applied these.
        self.never_answer = set(self.manifest.get("never_answer") or [])
        # Filled by `predict`, which is the first point at which vectors exist.
        self.space_warning: list[str] = []

        # The reject class is a fitted label but never an answer: the user did not
        # ask about it, and `__OTHER__` winning the argmax is a decline in every
        # sense that matters. Masked out of both scores, as `score_frame` does.
        self.mask = self.classes != OTHER
        self.gmat, self.ugroups = group_matrix(self.classes, self.mask, self.groups)

    @property
    def notes(self):
        out = []
        if self.version < 2 or not self.groups:
            out.append(
                "bundle predates the stored group map (format 1), so the coarse "
                "rank is re-derived from each label's first whitespace token. "
                "That is right for Linnaean binomials and wrong everywhere else — "
                "rebuild to record the map the model was measured with")
        if self.never_answer:
            out.append(
                f"{len(self.never_answer)} label(s) suppressed at predict time and "
                f"never answered: " + ", ".join(sorted(self.never_answer))
                + " — the same suppression the card was measured under")
        if self.space is None:
            out.append(
                "bundle carries no embedding-space fingerprint, so vectors from a "
                "different encoder cannot be refused — rebuild to record one")
        if self.t_novel is not None and OTHER in set(self.classes.tolist()):
            out.append(
                f"a near-OOD gate is fitted at t_novel={self.t_novel:.3f}: a row "
                "keeping less than that share of its mass inside the label set is "
                "declined outright, whatever the other two thresholds say")
        if OTHER not in set(self.classes.tolist()):
            out.append(
                "no reject class was fitted, so this model cannot decline for "
                "being out-of-list — only for being unsure")
        return out

    def space_notes(self, X) -> list:
        """Refuse vectors that cannot have come from the encoder this was fitted on.

        `self.encoder` is a string the caller declared and catches nothing -- a
        Core ML export and its torch original carry the same model name and live
        in unrelated spaces. The stored direction catches it, and the failure it
        prevents is silent rather than loud.
        """
        from narrowcast.build import check_same_space
        if self.space is None:
            return []
        return check_same_space(self.space.reshape(1, -1),
                                np.asarray(X, dtype="float64"),
                                what="these vectors")

    def proba(self, X):
        """Softmax over the fitted head. Binary heads store one row of coefficients."""
        X = np.asarray(X, dtype="float64")
        X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
        z = X @ self.coef.T + self.intercept
        if z.shape[1] == 1:                      # sklearn's binary parameterisation
            z = np.hstack([-z, z])
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X):
        """-> list of dicts: answer, rank, and the confidences behind the decision.

        Identical arithmetic to `build.score_frame`: the per-label posterior is
        restricted to the labels the user chose, and the group score sums that
        restricted mass within each group. Those are nested by construction
        (`max_c P(c) <= max_g sum P(c)`), which is what makes two independent
        thresholds a well-ordered three-way decision rather than two guesses.
        """
        self.space_warning = self.space_notes(X)
        full = self.proba(X)
        cata = full[:, self.mask]
        # Same arithmetic as `build.frame_from_posteriors`, which is the only
        # other place this is computed. `proba` here is a softmax and sums to 1,
        # but the ratio form is kept so the two cannot drift.
        novelty = cata.sum(1) / np.clip(full.sum(1), 1e-12, None)
        gscore = cata @ self.gmat.T
        names = self.classes[self.mask]
        label_conf, group_conf = cata.max(1), gscore.max(1)
        pred_label = names[cata.argmax(1)]
        pred_group = self.ugroups[gscore.argmax(1)]
        lv = decide(label_conf, group_conf, self.t_group, self.t_label,
                    novelty if self.t_novel is not None else None, self.t_novel)
        if self.never_answer:
            members = {}
            for j, g in enumerate(self.ugroups):
                members[str(g)] = {str(n) for n in names[self.gmat[j] > 0]}
            lv = suppress(lv, pred_label, self.never_answer, pred_group, members)

        out = []
        for i, rank in enumerate(lv):
            label = str(pred_label[i])
            group = str(pred_group[i])
            out.append({
                "rank": rank,
                "answer": label if rank == LABEL else (group if rank == GROUP else None),
                "label": label, "group": group,
                "label_conf": float(label_conf[i]),
                "group_conf": float(group_conf[i]),
                "novelty": float(novelty[i]),
            })
        return out


def embed(bundle: Bundle, embeddings=None):
    """Vectors for the rows to classify.

    Narrowed with the rest of the tool: there is no encoder here any more, so the
    only way in is vectors produced by the same encoder the bundle names. Handing
    a directory of photographs to something that cannot open one would fail late
    and confusingly, so it is not accepted at all.
    """
    from narrowcast import sources

    if not embeddings:
        raise ValueError("give --embeddings FILE: vectors from the same encoder "
                         f"the bundle was built on ({bundle.encoder!r}). "
                         "narrowcast does not encode.")
    rows = sources.from_embeddings(embeddings)
    return np.asarray(rows.descriptor, dtype="float32"), rows


def render(results, rows, limit=0) -> str:
    """One line per row, plus the share of each answer kind.

    The summary is the point. A run that declines 70% of its input is working as
    fitted, and a caller who sees only the answered rows would never know.
    """
    L, n = [], len(results)
    shown = results if not limit else results[:limit]
    for i, r in enumerate(shown):
        who = (Path(str(rows.path[i])).name if rows.path is not None
               else f"row {i}")
        if r["rank"] == LABEL:
            L.append(f"{who}\t{r['answer']}\t{r['label_conf']:.3f}")
        elif r["rank"] == GROUP:
            L.append(f"{who}\t{r['answer']} (group only)\t{r['group_conf']:.3f}")
        else:
            L.append(f"{who}\tdeclined\t{r['group_conf']:.3f}")
    if limit and n > limit:
        L.append(f"… {n - limit} more")

    kinds = {k: sum(1 for r in results if r["rank"] == k)
             for k in (LABEL, GROUP, DECLINE)}
    L += ["", f"{n} rows — {kinds[LABEL]} named to a label "
              f"({100 * kinds[LABEL] / max(n, 1):.1f}%), "
              f"{kinds[GROUP]} answered at group only, "
              f"{kinds[DECLINE]} declined"]
    return "\n".join(L)
