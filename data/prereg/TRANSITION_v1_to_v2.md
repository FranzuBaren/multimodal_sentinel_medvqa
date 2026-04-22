# Transition note: v1 → v2

**Date:** 2026-04-21

**What happened to v1:**

The v1 pre-registration (hash `912c680b513988e25801de7fe30c88021128bf3eb416ab8c3feb006256a4018c`) defined a three-arm inference-strategy comparison (single-pass vs self-consistency vs Sentinel 3LoD) on the felizzi et al. 2025 Italian MedVQA dataset. The planned protocol was:

- 3 arms, compute-matched at 3 VLM calls per question
- 60 questions × 2 conditions (real / fake image) × 3 arms = 360 main tasks
- Sonnet 4.5 via the Anthropic API
- Primary H1: Sentinel sensitivity > Self-Consistency sensitivity + 10pp on grounding failure detection

**Why v1 was archived:**

Phase 0 validation passed (vision works, temperature accepted but does not produce meaningful variation — logged as v1 Deviation #1). The dataset repository was then found to contain the question stems but not the MCQ option text (options A–E remain in an Excel file on the Felizzi side, not publicly released). A stratified smoke test on N=10 items confirmed the consequence: all three arms performed at the uniform random baseline (~20% accuracy), making inter-arm comparison dominated by random-guessing noise rather than the architectural differences v1's H1 was designed to test.

The smoke test produced $1.28 of spend and yielded one usable observation: Sentinel flagged 100% of fake-image cases via its L2 Challenger (`grounding_match=no`, `findings_supported=no`), while Self-Consistency flagged 0% and Single-pass flagged 60%. This signal is preserved and motivates the v2 research question.

**What v2 changes:**

v2 redefines the research question from "which inference strategy wins on MCQ accuracy" to "can Sentinel serve as a post-hoc audit primitive for pre-existing VLM reasoning chains." The shift addresses the dataset gap directly: v2 does not ask the VLM to solve MCQs; it audits reasoning chains Felizzi et al. have already generated and published.

Concretely:

- **Task:** shifted from MCQ answering to post-hoc grounding classification (G/P/U).
- **Input data:** 480 reasoning chains (4 models × 60 questions × 2 conditions × 1 repetition, all pre-existing in the felizzi results CSVs).
- **No new VLM generation calls.** API spend is entirely on the Sentinel audit pipeline and the Opus 4.7 auto-judge for medium ground truth.
- **Primary H1:** Sentinel classifies fake-image chains as Ungrounded with sensitivity ≥ 0.85, CI LB > 0.70.
- **Cost:** ~$30–50 for the main audit, ~$30 for the auto-judge. Well within the $100–500 v1 budget and much cheaper than v1 would have been had it proceeded.
- **Target venue:** ML4H 2026 full paper (unchanged from v1's revised target).

**Why this is not a fishing expedition:**

The v2 research question is empirically motivated by v1's smoke test observations, not by the outcome of a hypothesis test. The smoke test did not test v2's H1 (it couldn't — no auto-judge ran, no GT was established on 480 chains). The v2 pre-registration is therefore pre-results for all its pre-registered hypotheses. The smoke test functions as problem-formulation data, not hypothesis-test data. This is the same pattern as "pilot to motivate Phase II" in clinical trial design.

**What to cite and how, going forward:**

- In any writeup, v1 is referenced as "archived pre-registration, abandoned after gate failure on missing MCQ options." The v1 document remains committed in the repo for transparency.
- v2 supersedes v1 for all experimental claims. No v1 result is used as evidence for v2 conclusions except the N=10 smoke observation, which is cited as motivation.
- If any reviewer asks whether v2 is post-hoc, the answer is: yes, the research question is reformulated post-hoc relative to v1, but all hypothesis tests within v2 are pre-registered before the v2 main run.

**v1 artifacts retained:**

- `PREREGISTRATION.md` (v1, locked hash)
- `DEVIATIONS.md` (v1, 4 entries)
- `sentinel_pilot_v1.ipynb` and supporting cells
- `smoke_checkpoint_v2.csv` (the N=10 × 2 × 3 = 60-row smoke test output, renamed with clearer tag)

**v2 artifacts to be created:**

- `PREREGISTRATION_v2.md` (this commit)
- `DEVIATIONS_v2.md` (to be started at lock time)
- `sentinel_audit_v2.ipynb` (notebook adapted from v1, reusing Phase 0 and helpers)
- `audit_results_v2.csv` (main output: one row per chain with Sentinel verdict)
- `autojudge_results_v2.csv` (100 chain × Opus 4.7 labels)

---

Signed: Francesco Orsi, 2026-04-21.
