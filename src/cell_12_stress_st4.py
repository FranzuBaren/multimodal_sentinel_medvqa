# =============================================================================
# ST4 — TARGETED ECG CROSS-PATTERN SWAP
# =============================================================================
# Purpose: follow up on IT0063 false negative. Hypothesis: Sentinel L2 does
# semantic plausibility checking rather than pixel-level verification. Test
# this by swapping ECG images across questions with DIFFERENT diagnostic
# patterns (e.g., anterior STEMI chain + inferior STEMI image) and measuring
# the false-negative rate on a held-out set.
#
# If Sentinel verifies findings that the swapped image does NOT show, the
# IT0063 failure is a reproducible mode, not an anecdote.
#
# Design: all ECG questions in the dataset (N=9), annotated with their
# diagnostic category, swap images across diagnostic categories.

ST4_CHECKPOINT = OUT_DIR / "stress_st4_checkpoint.csv"

# Hand-annotated diagnostic categories (from inspection of baseline reasoning)
# Each ECG question maps to its anatomical/rhythmic diagnosis class.
# Swapping across classes guarantees the image shows a different pattern
# than what the reasoning chain describes.
ECG_DIAGNOSTIC_CATEGORIES = {
    "IT0006": "STEMI_anterior",    # anterior STEMI (V1-V6)
    "IT0063": "STEMI_inferior",    # inferior STEMI (II, III, aVF)
    "IT0064": "STEMI_inferior",    # related to IT0063 (same clinical scenario)
    "IT0065": "STEMI_inferior",    # related to IT0063 clinical scenario
    "IT0455": "Afib_arrhythmia",   # irregular rhythm, atrial fib
    "IT0713": "STEMI_anterior",    # anterior STEMI (V1-V6)
    "IT0919": "tachyarrhythmia",   # tachycardia
    "IT0972": "tachyarrhythmia",   # palpitations case
    "IT1053": "AV_block",          # high-degree AV block, bradycardia
}


def run_stress_4(client, chains_df, n_items=10) -> pd.DataFrame:
    """
    Target-paired swap: chain from pattern A, image from pattern B ≠ A.
    Reasoning chain describes one diagnostic pattern; image shows a different one.
    Sentinel L2 must detect the mismatch finding-by-finding, since modality is
    'ECG' for both.
    """
    print(f"\n{'=' * 70}\nST4: TARGETED ECG CROSS-PATTERN SWAP (N={n_items})\n{'=' * 70}")
    print(f"Hypothesis: Sentinel L2 semantically plausibility-checks rather than "
          f"pixel-verifies.")
    print(f"Expected if hypothesis HOLDS: many chains get verdict=G/P with all "
          f"findings SUPPORTED\n         even though image shows different "
          f"pattern.")
    print(f"Expected if hypothesis FAILS: most chains get verdict=U with "
          f"NOT_SUPPORTED per-finding.")

    if ST4_CHECKPOINT.exists():
        ST4_CHECKPOINT.unlink()
        print(f"  Cleared old {ST4_CHECKPOINT}")

    # Build candidate pairs: chain_qid ≠ image_qid, and category(chain) ≠ category(image)
    pool_qids = [qid for qid in ECG_DIAGNOSTIC_CATEGORIES.keys()
                 if (IMAGES_DIR / f"image_{qid}.png").exists()]

    pairs = []
    for chain_qid in pool_qids:
        chain_cat = ECG_DIAGNOSTIC_CATEGORIES[chain_qid]
        for img_qid in pool_qids:
            if img_qid == chain_qid:
                continue
            img_cat = ECG_DIAGNOSTIC_CATEGORIES[img_qid]
            if img_cat != chain_cat:
                pairs.append((chain_qid, chain_cat, img_qid, img_cat))

    print(f"  Total eligible (chain, image) cross-pattern pairs: {len(pairs)}")

    # Sample N deterministically — use STRESS_RNG from earlier cell
    sampled_pairs = STRESS_RNG.sample(pairs, min(n_items, len(pairs)))
    print(f"  Sampled {len(sampled_pairs)} pairs:")
    for (cq, cc, iq, ic) in sampled_pairs:
        print(f"    chain={cq}({cc}) ← image={iq}({ic})")

    results = []
    for chain_qid, chain_cat, img_qid, img_cat in sampled_pairs:
        # Get the chain from claude-sonnet-4-5 real condition
        mask = ((chains_df.question_id == chain_qid)
                & (chains_df.condition == "real")
                & (chains_df.model == "claude-sonnet-4-5"))
        chain_sub = chains_df[mask]
        if chain_sub.empty:
            print(f"  skip: no chain for {chain_qid}")
            continue
        chain_row = chain_sub.iloc[0].copy()

        img_path = IMAGES_DIR / f"image_{img_qid}.png"
        try:
            img_b64, media_type = image_path_to_b64(img_path, max_dim=1024)
        except Exception as e:
            print(f"  skip {chain_qid}: image load {e}")
            continue

        stressed = chain_row.copy()
        stressed["chain_id"] = f"ST4|{chain_qid}({chain_cat})|img_{img_qid}({img_cat})"
        stressed["condition"] = "stressed_ecg_cross_pattern"

        # Use dedicated st4 checkpoint
        _orig_ck = globals().get("STRESS_CHECKPOINT")
        globals()["STRESS_CHECKPOINT"] = ST4_CHECKPOINT

        try:
            r = run_stress_audit_one(
                client, stressed, img_b64, media_type,
                stress_id="ST4",
                stress_desc=f"chain {chain_qid}({chain_cat}) + image {img_qid}({img_cat})",
            )
            results.append(r)

            # Rich per-item diagnostic output
            flag_fn = ""
            if r["verdict"] == "G":
                flag_fn = "  ⚠ FALSE NEGATIVE (verdict=G despite wrong image)"
            elif r["verdict"] == "P" and r["n_not_supported"] == 0:
                flag_fn = "  ⚠ SUSPICIOUS (P but zero NOT_SUPPORTED)"

            print(f"  ST4 {chain_qid}({chain_cat}) + img_{img_qid}({img_cat}) → "
                  f"verdict={r['verdict']}, mm={r['modality_match']}, "
                  f"sup/not/cantass={r['n_supported']}/{r['n_not_supported']}/"
                  f"{r['n_cannot_assess']}{flag_fn}")
        except RuntimeError as e:
            print(f"  ABORT ST4: {e}")
            break
        finally:
            if _orig_ck is not None:
                globals()["STRESS_CHECKPOINT"] = _orig_ck

    return pd.DataFrame(results)


# --- Execute ---
DO_ST4 = False  # flip to True to run

if DO_ST4 and ANTHROPIC_API_KEY:
    df_st4 = run_stress_4(client, chains_df, n_items=10)

    if len(df_st4) > 0:
        print(f"\n{'=' * 70}\nST4 SUMMARY\n{'=' * 70}")
        print(f"N={len(df_st4)}")
        print(f"Verdicts: {df_st4['verdict'].value_counts().to_dict()}")
        print(f"modality_match: {df_st4['modality_match'].value_counts().to_dict()}")

        # Key metrics
        n_false_neg_pure = (df_st4['verdict'] == 'G').sum()
        n_partial_zero_notsupp = ((df_st4['verdict'] == 'P')
                                   & (df_st4['n_not_supported'] == 0)).sum()
        n_catched_via_findings = ((df_st4['verdict'].isin(['U', 'P']))
                                   & (df_st4['n_not_supported'] >= 1)).sum()
        n_catched_via_modality = ((df_st4['modality_match'].isin(['no', 'partial']))
                                   & (df_st4['verdict'].isin(['U', 'P']))).sum()

        print(f"\n--- Failure mode analysis ---")
        print(f"Pure false negatives (G despite wrong image): {n_false_neg_pure}/{len(df_st4)}")
        print(f"Suspicious partials (P but 0 NOT_SUPPORTED):  {n_partial_zero_notsupp}/{len(df_st4)}")
        print(f"Detected via finding mismatch (N≥1 NOT_SUPPORTED): {n_catched_via_findings}/{len(df_st4)}")
        print(f"Detected via modality_match signal:                {n_catched_via_modality}/{len(df_st4)}")

        # Bootstrap CI on pure false-negative rate
        from numpy.random import default_rng
        rng = default_rng(RANDOM_SEED)
        verdicts = df_st4['verdict'].tolist()
        boots = rng.choice(verdicts, size=(2000, len(verdicts)), replace=True)
        fn_rates = np.array([[v == 'G' for v in row] for row in boots]).mean(axis=1)
        lo, hi = np.percentile(fn_rates, [2.5, 97.5])
        print(f"\nPure false-negative rate: {n_false_neg_pure/len(df_st4):.2f} "
              f"[95% CI: {lo:.2f}, {hi:.2f}]")
        print(f"Cumulative cost: ${estimated_cost_usd():.2f}")

        # Semantic plausibility failure rate: verdicts G OR (P with 0 not_supported)
        sp_fail = n_false_neg_pure + n_partial_zero_notsupp
        print(f"\n--- Semantic plausibility failure mode (G or P-with-0-notsupp) ---")
        print(f"Rate: {sp_fail}/{len(df_st4)} = {sp_fail/len(df_st4)*100:.0f}%")
        sp_verdicts = [(v == 'G') or (v == 'P' and ns == 0)
                       for v, ns in zip(df_st4['verdict'], df_st4['n_not_supported'])]
        sp_boots = rng.choice(sp_verdicts, size=(2000, len(sp_verdicts)), replace=True).mean(axis=1)
        sp_lo, sp_hi = np.percentile(sp_boots, [2.5, 97.5])
        print(f"95% CI: [{sp_lo:.2f}, {sp_hi:.2f}]")
else:
    print("DO_ST4 = False. Set to True to run.")
    print(f"Cost estimate: ~$2, 5 minutes.")
