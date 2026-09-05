"""Deployable encoders, their real sizes, and picking one for a byte budget.

Sizes are the **image tower only** and were counted from the loaded model, not
quoted from a paper: `_ImageTower.parameters()` proxies the whole CLIP model, so
the naive count includes a text tower that never ships and overstates BioCLIP-2
by 40%. int4 BioCLIP-2 at 152 MB reconciles with the shipped 160 MB artifact,
which is the check that these numbers are the right ones.

int4 is the default assumed precision because `EMBEDDED_FINDINGS.md` measures it
as indistinguishable from fp32 at every catalogue size tested.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Encoder:
    variant: str
    label: str
    params_m: float
    dim: int
    input_px: int = 224
    ms_per_image: float | None = None    # MPS, batch 8, M4 Max — indicative, not ANE

    def size_mb(self, bits: int = 4) -> float:
        return self.params_m * bits / 8.0


# Ordered smallest to largest by *bytes*, which `choose` relies on. Note that
# byte order is no longer speed order: PlantCLEF2024 is a third of BioCLIP-2's
# parameters and twice its latency, because it runs at 518px -- 5.3x the pixels.
# Storage and compute are separate budgets and this registry can only rank one.
ENCODERS = (
    Encoder("mobileclip2_s0", "MobileCLIP2-S0", 11.4, 512, 256),
    Encoder("mobileclip2_s2", "MobileCLIP2-S2", 35.8, 512, 256, 5.2),
    Encoder("plantclef24", "PlantCLEF2024 (ViT-B/14 @518)", 86.6, 768, 518, 38.6),
    Encoder("bioclip2", "BioCLIP-2 (ViT-L)", 304.0, 768, 224, 20.4),
)
BY_VARIANT = {e.variant: e for e in ENCODERS}


def choose(budget_mb: float | None, bits: int = 4) -> Encoder:
    """Largest encoder fitting the budget; the smallest if nothing fits.

    Largest-that-fits rather than smallest-that-works because accuracy rises
    with capacity across every measurement here, so unused budget is accuracy
    left on the table.

    This ranks storage only. It cannot see that PlantCLEF2024 is slower than the
    model three times its size, so a caller with a *latency* budget should pass
    `--encoder` explicitly rather than trust a byte budget.
    """
    if budget_mb is None:
        return BY_VARIANT["bioclip2"]
    fits = [e for e in ENCODERS if e.size_mb(bits) <= budget_mb]
    return fits[-1] if fits else ENCODERS[0]


def budget_note(budget_mb: float | None, bits: int = 4) -> str:
    """One line on what the budget does or does not buy."""
    if budget_mb is None:
        return "no budget given; assuming a phone or laptop, where 152 MB is free"
    chosen = choose(budget_mb, bits)
    if chosen is ENCODERS[0] and chosen.size_mb(bits) > budget_mb:
        return (f"nothing fits {budget_mb:.0f} MB; smallest available is "
                f"{chosen.label} at {chosen.size_mb(bits):.1f} MB int{bits}")
    bigger = [e for e in ENCODERS if e.size_mb(bits) > budget_mb]
    if not bigger:
        return f"{budget_mb:.0f} MB fits every encoder available"
    nxt = bigger[0]
    return (f"{nxt.label} would need {nxt.size_mb(bits):.1f} MB int{bits}, "
            f"over your {budget_mb:.0f} MB budget")
