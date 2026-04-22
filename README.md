# Sentinel 3LoD Audit Primitive

**Post-hoc adversarial VLM grounding audit on medical image question-answering.**

This repository contains the code, pre-registration, and results for a pilot
study of a three-agent adversarial pipeline applied to the Felizzi et al. 2025
MMRL4H Italian MedVQA dataset. The primitive audits pre-existing VLM
reasoning chains and emits a grounding verdict (G/P/U) per chain.

## Key finding

Same-family adversarial review exhibits a **semantic plausibility
verification failure mode**: at rate 60% (95% CI [30%, 90%], N=10) on
targeted ECG cross-pattern swaps, the L2 Verifier supports claimed findings
that are not present in the image, when the reasoning chain's narrative is
medically well-formed.

This failure mode is invisible in fake-image testing (H1 sensitivity = 100%
on placeholder images, N=240) and only surfaces under targeted
intra-modality adversarial pairing.

## Contents

```
.
├── notebooks/              Jupyter notebook (assembles src/cell_*.py)
├── src/                    Source cells, one per notebook cell
│   ├── cell_01_config.py       Configuration (Anthropic API)
│   ├── cell_02_client.py       Client, hash chain
│   ├── cell_03_phase0.py       Model/vision probe
│   ├── cell_04_prompts.py      L1 / L2 / L3 / auto-judge prompts
│   ├── cell_05_parsers.py      Output parsers
│   ├── cell_06_loader.py       Felizzi dataset unifier
│   ├── cell_07_audit.py        Audit runner (480 chains)
│   ├── cell_08_autojudge.py    Opus-class auto-judge (N=100 stratified)
│   ├── cell_09_execute.py      Execution controls
│   ├── cell_10_analysis.py     Bootstrap + κ analysis
│   ├── cell_11_stress_tests.py ST1 / ST2 / ST3
│   └── cell_12_stress_st4.py   ST4 (targeted ECG cross-pattern, the key test)
├── data/
│   ├── prereg/            Pre-registration v2 and transition from v1
│   └── results/           Aggregated CSVs from the pilot run
├── docs/                  Diagnostic notes
└── scripts/               Assembly and utility scripts
```

## Setup

### Requirements

- Python 3.11+
- Anthropic API key (https://console.anthropic.com/)
- The Felizzi et al. 2025 dataset (not included; clone separately)

```bash
pip install -r requirements.txt
```

### Dataset

This repo does not redistribute the Felizzi dataset. Clone it into your
working directory:

```bash
git clone https://github.com/felizzi/eurips2025-mmrl4h-italian-medvqa-visual-grounding.git
```

### API key

```bash
export ANTHROPIC_API_KEY='sk-ant-...'
```

### Assemble the notebook

```bash
python scripts/assemble_notebook.py
```

This produces `notebooks/sentinel_audit.ipynb` by concatenating the `src/cell_*.py`
files with markdown headers. The notebook is a thin assembler; all logic lives
in `src/`.

### Run

Open the notebook and run cells sequentially. The execution flags in Cell 9
default to `False` so "Run All" spends no API credits by accident:

| Flag                 | Action                             | Cost      | Time    |
|----------------------|------------------------------------|-----------|---------|
| `DO_AUDIT_SMOKE`     | 10 stratified chains               | ~$0.30    | ~3 min  |
| `DO_AUDIT_FULL`      | 480 chains                         | ~$30-50   | 2-4 h   |
| `DO_AUTOJUDGE`       | 100 stratified, Opus 4.5           | ~$30      | ~30 min |
| `DO_STRESS_TESTS`    | ST1 + ST2 + ST3                    | ~$0.60    | ~5 min  |
| `DO_ST4`             | ECG cross-pattern (critical)       | ~$2       | ~5 min  |

Total cost of a full replication: approximately $60-80 on Claude Sonnet 4.5
+ Opus 4.5. A minimum-viable pilot (smoke + ST4) is approximately $2.

## Results

Summary CSVs from the pilot run are in `data/results/`. These are the
aggregated metrics, not the raw reasoning chains (which are Felizzi et al.
property and must be loaded from the dataset).

| File                                           | Content                                    |
|------------------------------------------------|--------------------------------------------|
| `audit_summary.csv`                            | H1/H2/H3 + parse rates + cost              |
| `audit_H1_per_model.csv`                       | Fake-condition sensitivity per model       |
| `audit_H1_per_modality.csv`                    | Same, per imaging modality                 |
| `audit_D1_real_correct_by_model.csv`           | Verdict G/P/U split on real-correct        |
| `audit_D2_chain_length_by_verdict.csv`         | Chain length vs verdict                    |
| `audit_D3_modality_match.csv`                  | Modality match distribution                |
| `audit_D4_H3_confusion_matrix.csv`             | Sentinel vs auto-judge agreement           |
| `stress_ST1_cross_modality.csv`                | Per-pair ST1 results                       |
| `stress_ST2_same_modality.csv`                 | Per-pair ST2 results                       |
| `stress_ST3_fabricated.csv`                    | Per-pair ST3 results                       |
| `stress_ST4_ecg_cross_pattern.csv`             | Per-pair ST4 results (key finding)         |
| `stress_ST4_summary.csv`                       | ST4 rates + bootstrap CIs                  |

## Pre-registration

The pilot was pre-registered before running the main audit. See
`data/prereg/PREREGISTRATION_v2.md` for the full protocol, and
`data/prereg/TRANSITION_v1_to_v2.md` for the story behind the v1 → v2 pivot
(short version: v1 was abandoned after a smoke test revealed that the public
Felizzi dataset lacks the MCQ option text, so the task was reformulated
from MCQ-answering to post-hoc chain auditing).

Pre-registration hashes (commit these alongside the prereg file):

```bash
sha256sum data/prereg/PREREGISTRATION_v2.md
sha256sum src/cell_04_prompts.py  # prompt hashes are locked in this cell
```

## Reproducing with a different audit model

The key contribution of this pilot is the documented failure mode on
same-family adversarial review. Cross-family replication is a natural
follow-up:

1. Edit `src/cell_01_config.py` to set `SENTINEL_MODEL` to a different
   frontier VLM (e.g., through the OpenAI SDK — add an adapter in
   `src/cell_02_client.py`).
2. Keep the prompts in `src/cell_04_prompts.py` untouched — their hashes
   are part of the pre-registration.
3. Re-run ST4 via Cell 12 with `DO_ST4 = True`.
4. Compare the semantic-plausibility failure rate against the reported
   60% [30%, 90%].

A minimum-viable replication (ST4 only, N=10) costs approximately $2-5
depending on the VLM chosen.

## Citation

Preprint forthcoming. In the meantime, please cite:

```
@misc{orsi2026sentinel,
  author = {Orsi, Francesco},
  title = {Sentinel 3LoD: Post-hoc VLM grounding audit reveals
           semantic plausibility verification failure in
           same-family adversarial review},
  year = {2026},
  note = {Pilot study; paper in preparation, target ML4H@NeurIPS 2026},
  url = {https://github.com/[USERNAME]/sentinel-3lod-audit}
}
```

## Acknowledgments

This work builds on the Felizzi et al. 2025 MMRL4H Italian MedVQA dataset
(EurIPS 2025). The dataset's real/fake image pairing design and its
publicly released reasoning chains made the present audit framework possible.

## License

Code: MIT License (see LICENSE).

The Felizzi et al. dataset is governed by its own license. Refer to
https://github.com/felizzi/eurips2025-mmrl4h-italian-medvqa-visual-grounding
for terms of use.

## Contact

Francesco Orsi — independent researcher.
Issues and pull requests welcome.
