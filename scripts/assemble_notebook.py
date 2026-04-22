"""
Assemble the pilot notebook from src/cell_*.py files.

Usage:
    python scripts/assemble_notebook.py

Produces: notebooks/sentinel_audit.ipynb
"""
import nbformat as nbf
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
CELLS_DIR = ROOT / "src"
OUT_DIR = ROOT / "notebooks"
OUT_DIR.mkdir(exist_ok=True)
OUT = OUT_DIR / "sentinel_audit.ipynb"


def read_cell(name: str) -> str:
    path = CELLS_DIR / name
    if not path.exists():
        print(f"ERROR: missing {path}")
        sys.exit(1)
    return path.read_text()


def main():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell("""# Sentinel 3LoD — Post-Hoc VLM Grounding Audit

Three-agent adversarial pipeline auditing pre-existing reasoning chains from the
Felizzi et al. 2025 MMRL4H Italian MedVQA dataset. See `README.md` and
`data/prereg/PREREGISTRATION_v2.md` for context.

## Architecture

- **L1 Auditor** extracts claimed visual findings from the reasoning chain
- **L2 Verifier** checks each claim against the image
- **L3 Reconciler** emits G/P/U verdict + confidence + HITL gate

## Execution order

1. Cells 1–3: Config + client + Phase 0 probe
2. Cells 4–8: Prompts, parsers, loader, audit, auto-judge
3. Cell 9: Execution controls (flags default to False)
4. Cell 10: Analysis with bootstrap CIs
5. Cells 11–12: Stress tests + the targeted ECG cross-pattern test (the key contribution)"""))

    cells.append(nbf.v4.new_markdown_cell("## Phase 0 — Setup and probe"))
    for f in ["cell_01_config.py", "cell_02_client.py", "cell_03_phase0.py"]:
        cells.append(nbf.v4.new_code_cell(read_cell(f)))

    cells.append(nbf.v4.new_markdown_cell(
        "## Definitions — prompts, parsers, dataset loader"
    ))
    for f in ["cell_04_prompts.py", "cell_05_parsers.py", "cell_06_loader.py"]:
        cells.append(nbf.v4.new_code_cell(read_cell(f)))

    cells.append(nbf.v4.new_markdown_cell("## Audit pipeline + auto-judge"))
    for f in ["cell_07_audit.py", "cell_08_autojudge.py"]:
        cells.append(nbf.v4.new_code_cell(read_cell(f)))

    cells.append(nbf.v4.new_markdown_cell(
        "## Execution — flags default to False, opt in explicitly"
    ))
    cells.append(nbf.v4.new_code_cell(read_cell("cell_09_execute.py")))

    cells.append(nbf.v4.new_markdown_cell("## Analysis"))
    cells.append(nbf.v4.new_code_cell(read_cell("cell_10_analysis.py")))

    cells.append(nbf.v4.new_markdown_cell("""## Stress tests

Three diagnostic tests designed to pressure-test the audit primitive:

- **ST1**: cross-modality swap (expected: detected via modality_match=no)
- **ST2**: same-modality swap (expected if true grounding auditor: detected via per-finding)
- **ST3**: fabricated-finding chain on real image (expected: detected via per-finding)"""))
    cells.append(nbf.v4.new_code_cell(read_cell("cell_11_stress_tests.py")))

    cells.append(nbf.v4.new_markdown_cell("""## ST4 — Targeted ECG cross-pattern swap

The key diagnostic test. Pairs reasoning chains with images from different
diagnostic categories (same modality, different anatomy/rhythm). Measures
the semantic-plausibility verification failure rate."""))
    cells.append(nbf.v4.new_code_cell(read_cell("cell_12_stress_st4.py")))

    cells.append(nbf.v4.new_markdown_cell("""## Artifacts produced after a full run

- `pilot_outputs/phase0_results.json`
- `pilot_outputs/chains_unified.csv` (derived from Felizzi dataset; do not redistribute)
- `pilot_outputs/audit_checkpoint.csv`
- `pilot_outputs/autojudge_checkpoint.csv`
- `pilot_outputs/stress_checkpoint.csv`
- `pilot_outputs/stress_st4_checkpoint.csv`
- `pilot_outputs/audit_trail.jsonl` (contains reasoning chain excerpts — treat as derived from Felizzi dataset)"""))

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    nbf.write(nb, OUT)
    n_code = sum(1 for c in cells if c.cell_type == "code")
    n_md = sum(1 for c in cells if c.cell_type == "markdown")
    print(f"Wrote {OUT}")
    print(f"Cells: {len(cells)} ({n_code} code, {n_md} markdown)")


if __name__ == "__main__":
    main()
