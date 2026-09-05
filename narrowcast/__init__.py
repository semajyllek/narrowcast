"""narrowcast — a small classifier over a narrow label set, and the truth about it.

Three commands, and the ordering is the point:

    narrowcast plan     what will this label set give me, and why    (no training)
    narrowcast build    fit, measure, write a bundle
    narrowcast card     the honest report on a built model

`plan` exists because the *composition* of a label set decides which failure mode
it has, and that failure mode is invisible in the metrics everyone publishes. A
set crowded with siblings of the same group does not answer wrongly -- it answers
*vacuously*, retreating to a group that narrows nothing, while coverage and
precision go **up**. Measured on plants (label-level 0.761 -> 0.476) and
reproduced on birds (0.958 -> 0.718).

So every report carries the share of answers made at the fine rank, and no card
prints coverage without it.
"""

__version__ = "0.1.0"
