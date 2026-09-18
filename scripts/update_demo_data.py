"""
Copy the measured critical lines from results/phase_diagram.csv into the
interactive demo (docs/index.html), so the page always shows the latest data.

    python scripts/update_demo_data.py
"""

import json
import re

from _common import ROOT, RESULTS, parser

import pandas as pd

from firesim import theory


def main():
    parser(__doc__).parse_args()
    df = pd.read_csv(RESULTS / "phase_diagram.csv")
    lines = {"L": int(df.L.iloc[0])}
    for rule in "AB":
        d = df[df.rule == rule].sort_values(["p_burn", "density"], ascending=[True, False])
        lines[rule] = [[round(p, 4), round(D, 4)] for p, D in zip(d.p_burn, d.density)]
    lit = [[1.0, round(theory.SITE_PC, 4)], [theory.BOND_PC, 1.0]]
    lit += [[round(p, 4), round(D, 4)] for p, D in theory.tvdm_points()]
    lines["lit"] = sorted(lit)

    page = ROOT / "docs" / "index.html"
    html = page.read_text()
    new_block = ('<script id="critical-lines" type="application/json">\n'
                 + json.dumps(lines, separators=(",", ":")) + "\n</script>")
    html, n = re.subn(r'<script id="critical-lines" type="application/json">.*?</script>',
                      lambda _: new_block, html, count=1, flags=re.S)
    if n != 1:
        raise SystemExit("could not find the critical-lines block in docs/index.html")
    page.write_text(html)
    print(f"updated {page} with {len(lines['A'])} + {len(lines['B'])} critical points")


if __name__ == "__main__":
    main()
