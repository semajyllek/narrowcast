"""Build the worked-cases figure: three real evaluation observations.

Each panel is an observation from the held-out half of the 5,534-observation
iNaturalist evaluation set, with the scores the model actually produced and the
decision the fitted thresholds actually returned. Nothing is staged.

Only openly licensed photographs are embedded, and each carries its
photographer and licence. iNaturalist records a licence per photograph, and the
first candidate chosen for the decline panel was "all rights reserved" and was
replaced rather than used. Thumbnails are downscaled and re-encoded, which keeps
the page self-contained under a strict content-security policy.

Usage:  python docs/make_cases.py --plantid ../narrowcast-plantid
"""

import argparse
import base64
import io
import re
from pathlib import Path

from PIL import Image

# obs_id, truth, what the model did, and the licence of the photograph used.
# Scores are transcribed from the scored test half; see the module docstring.
CASES = [
    dict(obs=383750999, photo=0,
         state="Retreats to the genus",
         truth="Sedum anglicum", pred="Sedum atratum", answer="“a Sedum”",
         s=0.345, g=0.958,
         reading="Species confidence falls below t_species, so the fine answer is "
                 "withheld. Genus confidence is high, so an answer is still given. "
                 "It is correct, and with 37 Sedum species in the catalogue it "
                 "narrows very little.",
         credit="Stuart Milligan", lic="CC BY-NC"),
    dict(obs=383412840, photo=0,
         state="Names the wrong species, confidently",
         truth="Sedum brevifolium", pred="Sedum anglicum", answer="“Sedum anglicum”",
         s=0.676, g=0.952,
         reading="Species confidence clears t_species, so a species is named, and "
                 "it is the wrong one. This is the error the utility weights at "
                 "−4.0, and it is the reason the thresholds are fitted rather "
                 "than chosen.",
         credit="Helena Rodríguez", lic="CC BY-NC"),
    dict(obs=389451092, photo=0,
         state="Declines, correctly",
         truth="Zamia loddigesii", pred="Zamia furfuracea", answer="no answer",
         s=0.041, g=0.063,
         reading="Not in the catalogue. Both scores sit far below their "
                 "thresholds and nothing is returned. Under the declared utility "
                 "this earns +1.0 — the same as a correct species answer.",
         credit="Douglas Caro Lopez", lic="CC BY"),
]

W, PANEL, IMG = 720, 232, 168


def thumb(path: Path, px: int = 320, quality: int = 74) -> str:
    im = Image.open(path).convert("RGB")
    side = min(im.size)
    im = im.crop(((im.width - side) // 2, (im.height - side) // 2,
                  (im.width + side) // 2, (im.height + side) // 2)).resize((px, px), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def build(images: Path) -> str:
    cells = []
    for c in CASES:
        src = sorted(images.glob(f"{c['obs']}_*.jpg"))[c["photo"]]
        uri = thumb(src)
        cells.append(f'''
<div class="case">
  <img src="{uri}" alt="Photograph of {c['truth']}, {c['credit']}, {c['lic']}" width="{IMG}" height="{IMG}">
  <p class="state">{c['state']}</p>
  <table class="caserow">
    <tr><td>photographed</td><td><i>{c['truth']}</i></td></tr>
    <tr><td>top species</td><td><i>{c['pred']}</i></td></tr>
    <tr><td>species score <span class="num">s</span></td><td class="v">{c['s']:.3f}</td></tr>
    <tr><td>genus score <span class="num">γ</span></td><td class="v">{c['g']:.3f}</td></tr>
    <tr class="ans"><td>returned</td><td>{c['answer']}</td></tr>
  </table>
  <p class="reading">{c['reading']}</p>
  <p class="credit">© {c['credit']}, {c['lic']}, via iNaturalist · observation {c['obs']}</p>
</div>''')
    return '<div class="cases">' + "".join(cells) + "</div>"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plantid", type=Path, default=Path("../narrowcast-plantid"))
    ap.add_argument("--html", type=Path, default=Path("docs/deep_dive.html"))
    a = ap.parse_args()

    block = build(a.plantid / "data/processed/images_inat")
    html = a.html.read_text()
    pat = re.compile(r"(<!--FIG:cases-->).*?(<!--/FIG:cases-->)", re.S)
    if not pat.search(html):
        raise SystemExit("no <!--FIG:cases--> placeholder in the document")
    a.html.write_text(pat.sub(lambda m: m.group(1) + block + m.group(2), html))
    print(f"cases block: {len(block):,} bytes -> {a.html}")


if __name__ == "__main__":
    main()
