"""narrowcast — audit a classifier over a narrow label set.

Three commands, and the ordering is the point:

    narrowcast audit    measure what your model will actually do
    narrowcast card     the honest report on what was measured
    narrowcast predict  run a bundle this tool fitted (--embeddings only)

You bring the model. `audit` takes its posteriors (`--scores`) or vectors it can
fit a linear head over (`--embeddings`), and reports the three-way split of
label / group / decline at a prevalence you declare.

The finding it exists for: a label set crowded with siblings of the same group
does not answer *wrongly* -- it answers **vacuously**, retreating to a group that
narrows nothing, while coverage and precision both go **up**. Measured on plants
(label-level 0.761 -> 0.476) and reproduced on birds (0.958 -> 0.718), text and
audio. So every report carries the share of answers made at the fine rank, and no
card prints coverage without it.

It used to try to build the model too -- choosing an encoder under a size budget,
projecting outcomes from a shipped grid, searching the Hub for candidates. That
part did not work and is gone; see `docs/deep_dive.html`.
"""

__version__ = "0.2.0"
