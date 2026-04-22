# Pre-registration v2: Sentinel 3LoD as a Post-Hoc VLM Grounding Audit Primitive

**Status:** Draft v2 — supersedes v1 (archived after smoke test on 2026-04-21).
To be locked and hash-committed before any main-run API call of the audit phase.

**Principal investigator:** Francesco Orsi (independent researcher, Muri, Bern, Switzerland)

**Potential co-author (invited, pending response):** Federico Felizzi et al.
(original dataset authors, EurIPS 2025 MMRL4H)

**Pre-registration date (target):** To be filled at commit time.

**SHA-256 of this document at commit:** To be filled via `sha256sum PREREGISTRATION_v2.md`.

**Relationship to v1:** The v1 pre-registration (hash `912c680b513988e25801de7fe30c88021128bf3eb416ab8c3feb006256a4018c`) defined a three-arm inference-strategy comparison (single-pass vs self-consistency vs Sentinel 3LoD). A smoke test on N=10 stratified items (2026-04-21) showed that the stem-only setup (MCQ options A–E not publicly available) pinned accuracy at the random baseline (~20%), making all three arms statistically indistinguishable from chance and inter-arm comparison dominated by noise. The primary hypothesis H1 of v1 therefore could not be tested on this dataset without the options file. v1 is archived as "pilot abandoned after gate failure" and this document defines a different, testable research question on the same dataset.

---

## 1. Background and research question

The Felizzi et al. 2025 benchmark documents a grounding failure in frontier VLMs: Claude Sonnet 4.5, GPT-4o, GPT-5-mini, and Gemini 2.0 flash exp all maintain high answer accuracy on Italian medical MCQ questions even when the image provided is a fake placeholder rather than the real medical image. The accuracy drop real→fake is typically 3–7 percentage points, far smaller than would be expected if the VLM were truly grounding its reasoning in visual content. This establishes grounding failure as a well-documented phenomenon; it does not provide an automated mechanism to detect it post-hoc.

This pilot investigates whether a three-agent adversarial pipeline (Diagnostician → Challenger → FactChecker), hereafter *Sentinel*, can serve as a post-hoc audit primitive that, given a reasoning chain produced by a VLM on a medical image, classifies whether the reasoning is grounded in the visible evidence.

Primary research question: *Can a three-agent adversarial audit primitive reliably detect visual grounding failures in VLM reasoning chains, using only the reasoning text and the image?*

This is a methodological question, not a benchmarking question. N=480 chains is not sufficient to rank models; it is sufficient to measure whether the audit primitive produces meaningful grounding classifications with above-chance discriminative performance.

## 2. Hypotheses

### Primary hypothesis (H1)

On the fake-image condition (where grounding is guaranteed Ungrounded by construction), Sentinel audits classify reasoning chains as Ungrounded with sensitivity at least 0.85, averaged across the four audited models, with 95% bootstrap confidence interval lower bound above 0.70.

Formal: \(P(\text{Sentinel verdict} = U \mid \text{fake condition}) \geq 0.85\), 95% CI LB > 0.70.

### Null hypothesis (H0)

Sentinel fake-condition sensitivity falls below 0.70 at the 95% CI lower bound, OR differs from a trivial baseline (always flag) by less than the standard uncertainty of specificity on real-condition chains.

### Negative-result protocol

If H0 is not rejected, the pilot concludes that intra-family adversarial review does not reliably detect grounding failures even in the adversarially-easy fake-image setting, and the Sentinel primitive as currently specified is not usable as a grounding auditor. This is reportable as-is.

### Secondary hypotheses (exploratory, Bonferroni-adjusted)

- **H2:** Sentinel specificity on real-image + correct-answer chains exceeds 0.70 at 95% CI LB, indicating the primitive does not false-flag well-grounded reasoning.
- **H3:** Auto-judge (Opus 4.7) inter-rater agreement with Sentinel on the 100-chain validation subset exceeds Cohen's κ = 0.60. If this fails, the grounding labels for the full N=480 are deemed unreliable and H1/H2 are reported with that caveat.
- **H4:** Sentinel audit verdicts differ across the four audited models (Claude Sonnet 4.5, GPT-4o, GPT-5-mini, Gemini 2.0 flash exp) by more than the within-model variance, tested via Kruskal-Wallis on the per-question verdict distributions. This is exploratory cross-model signal; it does not support a benchmark-grade ranking claim.

## 3. Dataset

**Source:** felizzi/eurips2025-mmrl4h-italian-medvqa-visual-grounding, commit hash to be locked at audit-phase start.

**Scope used:** 60 Italian SSM specialty exam questions × 2 image conditions × 4 models = 480 reasoning chains total. One repetition per (question, condition, model) is used; specifically the first of the 10 repetitions in the revision_11_25 results folder, ordered by the repetition index in the source CSV. The choice of first-repetition rather than random-sample is locked here to be fully deterministic.

**Modality distribution** (unchanged from v1 analysis): 13 CXR, 9 ECG, 7 DERM, 2 CT/MRI, 2 histology, 2 ophthalmology, 1 endoscopy, 24 OTHER.

**Per-modality N limitation:** modalities with N ≤ 2 will not be analyzed separately. Per-modality secondary analyses (if conducted) are restricted to CXR, ECG, DERM, and OTHER.

**Data already present:** reasoning chains and accuracy labels from Felizzi et al. are used as-is. No new VLM calls are made to generate the reasoning chains. The pilot's API spend is entirely on (a) the Sentinel audit pipeline and (b) the Opus 4.7 auto-judge.

## 4. Sentinel audit primitive

For each reasoning chain \(r\) produced by a model \(M\) on image \(I\) for question \(Q\):

### L1 Diagnostician (image + chain → claimed findings extraction)

Input: \(I\), \(Q\), \(r\). Output: list of claimed visual findings extracted from \(r\), plus a confidence label {low, medium, high} that the chain's answer is supported by the image.

### L2 Challenger (adversarial image verification)

Input: \(I\), \(Q\), \(r\), L1 output. Output: per-finding verification (supported / not supported / cannot assess from image) and an overall grounding verdict {yes, partial, no}.

### L3 FactChecker (reconciliation, text-only, no image)

Input: \(Q\), \(r\), L1 output, L2 output. Output: final grounding classification \(\in \{G, P, U\}\), numeric confidence \(\in [0,1]\), and HITL flag with reason.

### Operational cost

3 API calls per chain. 480 chains × 3 = 1440 API calls total for the main audit. At Claude Sonnet 4.5 pricing ($3/$15 per MTok) and expected average 2.5KB input + 800 output tokens per call, estimated main-run cost is $30–50.

### Prompts

Prompt text is locked by SHA-256 hash before main run. Any prompt change is a deviation in `DEVIATIONS.md` with timestamp.

### Sentinel model choice

Claude Sonnet 4.5 via the Anthropic API. The Sonnet-class audit model was validated during v1 Phase 0 and retained here without modification. See `DEVIATIONS_v2.md` for any deviations from the pre-registered configuration.

## 5. Ground truth protocol

### 5.1 Definitions

- **Grounded (G):** The reasoning chain cites visual findings that are actually present in the image, and the reasoning path from findings to answer is valid.
- **Partially grounded (P):** Some cited findings are present but the reasoning relies on at least one fabricated or unjustified visual claim.
- **Ungrounded (U):** The reasoning cites findings that are not present, cites findings that would require a different imaging modality than provided, or provides an answer without reference to visual content.

### 5.2 Weak ground truth (N=480, free)

By construction:
- All fake-image chains: GT = U (the placeholder image cannot support any medical finding).
- Real-image chains with `is_correct = True` in the baseline results: probabilistic proxy GT = G (imperfect; a correct answer can arise from ungrounded reasoning).
- Real-image chains with `is_correct = False`: GT is ambiguous and excluded from specificity calculations.

Weak GT is used for H1 (fake-condition sensitivity) and H2 (real-correct specificity). It is explicitly not used for fine-grained Partial-vs-Grounded discrimination.

### 5.3 Medium ground truth (N=100 stratified, ~$30)

A stratified subsample of 100 chains is independently labeled by Opus 4.7 as auto-judge. Stratification: 25 chains per audited model, balanced across real and fake conditions (~12-13 per condition per model). Seed locked at 42. The auto-judge prompt is locked by SHA-256 before main run.

The auto-judge emits {G, P, U} and a short rationale per chain. Auto-judge labels are compared against Sentinel labels for the same 100 chains using Cohen's κ. If κ < 0.60, per H3, the medium GT is deemed unreliable and analyses proceed with weak GT only, caveated accordingly.

### 5.4 Validation subsample (optional, N=20, PI time)

Time permitting, PI manually labels 20 of the 100 medium-GT chains following the same rubric. Cohen's κ between PI and auto-judge is reported. This is not required for pre-registered analyses but improves reported reliability if included.

## 6. Metrics

### Primary

**Fake-condition sensitivity** (per model and averaged): proportion of fake-image chains classified by Sentinel as Ungrounded. Bootstrap 95% CI via 2000 resamples at chain level.

### Secondary

- Real-correct specificity (per model and averaged).
- Cohen's κ between Sentinel and Opus 4.7 auto-judge on stratified 100.
- Expected Calibration Error of Sentinel's L3 confidence score vs GT binary (G vs U) on weak GT.
- Cross-model Kruskal-Wallis on per-question Sentinel verdict counts.

### What is not measured (pre-registered exclusion)

- **Absolute model ranking.** N=480 with 1 repetition per cell is insufficient for benchmark claims. Any cross-model differences observed are reported as "effect consistent with X" not "model X is more grounded than model Y."
- **Answer accuracy.** Out of scope. The baseline Felizzi results stand as-is.
- **Self-consistency or temperature-based baselines.** Dropped from v1 because they require generating new chains, which this pilot no longer does.

## 7. Statistical protocol

- **Bootstrap:** 2000 resamples at chain level. Confidence intervals reported at 95%.
- **Multiple testing:** H1 is primary, not corrected. H2–H4 are exploratory, Bonferroni factor 3.
- **Missing data:** Sentinel audit calls that fail (API error, parse error) are retried up to 3 times. Persistent failures are excluded from metrics with explicit N correction and reported as-is. Target failure rate below 2%; pause if above 5%.
- **Pre-registered decision criteria:**
  - H1 rejected (sensitivity ≥ 0.85, CI LB > 0.70): Sentinel is usable as a grounding auditor in the adversarial-easy regime. Proceed to writeup and scale-up to VQA-RAD as next step.
  - H1 not rejected: the primitive fails on the easiest possible case. Writeup focuses on this failure and its implications; no scale-up justified.

## 8. Locking procedure

Before the main audit-phase run, the following are committed to a dedicated Git branch `pilot-v2-prereg-locked-YYYY-MM-DD` with SHA-256 hashes posted publicly:

- [ ] This document (`PREREGISTRATION_v2.md`)
- [ ] Sentinel L1, L2, L3 prompts
- [ ] Opus 4.7 auto-judge prompt
- [ ] Bootstrap analysis script
- [ ] Model IDs and exact versions for the Sentinel Sonnet model and the Opus-class auto-judge
- [ ] Pricing assumptions current to lock date

## 9. Decision tree for follow-up

- **H1 rejected, H3 κ ≥ 0.60:** scale-up to VQA-RAD or MIMIC-CXR-VQA with N ≥ 500, cross-family Challenger ablation. Target: ML4H 2026 full paper.
- **H1 rejected, H3 κ < 0.60:** the fake-condition sensitivity holds but the medium-GT reliability fails. Writeup reports H1 result with weak-GT caveat and proposes human annotation as prerequisite for scale-up.
- **H1 not rejected:** pilot concludes that Sentinel within a single model family does not detect grounding failures reliably. Writeup reports this negative result with per-model breakdown. Hypothesis generation toward cross-family Challenger design.

## 10. Deviations from v2 pre-registration

All deviations logged in `DEVIATIONS_v2.md` with timestamp, original spec, actual spec, reason, and pre-vs-post-results flag. Inherited deviations from v1 (temperature semantics, model ID format, OpenAI API format) remain in effect.

## 11. What this pilot is NOT

- Not a NeurIPS Main submission.
- Not a clinical validation study.
- Not a benchmark paper. N=480, 1 repetition per cell, is insufficient.
- Not a replacement for the Felizzi et al. 2025 paper — this complements it by providing an audit primitive for the failure mode they documented.
- Not a claim of FDA SaMD readiness or medical AI superiority.

## 12. Authorship and coordination

Principal investigator: Francesco Orsi. Co-authorship invitation has been sent to Federico Felizzi (corresponding author of the source dataset paper). If accepted, the Felizzi team provides dataset access (already public), reasoning-chain results (already public), and optional review of the audit primitive design. If declined or unanswered, the pilot proceeds as single-author with full citation of the Felizzi et al. 2025 dataset.

## 13. Target venue

ML4H 2026 workshop at NeurIPS, full paper track (8 pages). Secondary target: arXiv preprint parallel to submission.

## 14. Timeline

- Week 1 (now): Pre-registration v2 lock, notebook adaptation, Phase 0 validation on auto-judge model (Opus 4.7).
- Week 2: Main audit run (1440 Sentinel calls on 480 chains), checkpoint-protected, ~$30–50.
- Week 3: Auto-judge on 100 stratified chains ~$30. Bootstrap analyses.
- Week 4: Optional PI manual validation on 20 chains.
- Week 5: Figures, tables, paper draft v1.
- Week 6: Polish, Felizzi review loop if applicable, submission.

---

**End of pre-registration v2.**

*Commit this document and hash-lock it BEFORE the audit main run. After lock, any change goes in DEVIATIONS_v2.md.*
