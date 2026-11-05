"""Every CustomJS on the page compiles. BokehJS compiles callbacks lazily in the browser,
so a syntax error only shows as a frozen widget plus a console message; this catches it
before publishing. Needs `node` (on PATH or $NODE); skipped without it."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import scripts.analysis.dashboard as D
from bokeh.models import CustomJS

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "tests/data/dashboard"
NODE = os.environ.get("NODE") or shutil.which("node")
NAMES = {
    "category_scores.csv": "SAMPLE_category_scores_aggregate.csv",
    "wide_metrics_table.csv": "SAMPLE_wide_metrics_table.csv",
    "metadata_rows.csv": "SAMPLE_metadata_rows.csv",
    "generalization_by_tier.csv": "SAMPLE_generalization_by_tier.csv",
    "generalization_by_metric.csv": "SAMPLE_generalization_by_metric.csv",
}
COMPILE_JS = """
const items = JSON.parse(require('fs').readFileSync(process.argv[2], 'utf8'));
let bad = 0;
for (const it of items) {
  try { new Function(...it.args, 'cb_obj', 'cb_data', it.code); }
  catch (e) { bad++; console.log(it.where + ': ' + e.message); }
}
process.exit(bad ? 1 : 0);
"""


def _cohort_dir(tmp_path: Path, name: str, suffix: str = "") -> Path:
    d = tmp_path / name
    d.mkdir()
    for dst, src in NAMES.items():
        t = pd.read_csv(FIX / src)
        t["tokenizer"] = t["tokenizer"] + suffix
        t.to_csv(d / dst, index=False)
    return d


@pytest.mark.skipif(NODE is None, reason="node not available")
def test_every_custom_js_compiles(tmp_path):
    cohorts = [
        D.load_cohort("200 ms", _cohort_dir(tmp_path, "a")),
        D.load_cohort("1000 ms", _cohort_dir(tmp_path, "b", "_1s"), hide_decoding=True),
    ]
    page = D.build_page(cohorts)
    items = [
        {"where": f"CustomJS {m.id}", "args": sorted(m.args), "code": m.code}
        for m in page.references()
        if isinstance(m, CustomJS)
    ]
    assert len(items) >= 10
    src = tmp_path / "callbacks.json"
    src.write_text(json.dumps(items))
    runner = tmp_path / "compile.js"
    runner.write_text(COMPILE_JS)
    proc = subprocess.run([NODE, str(runner), str(src)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
