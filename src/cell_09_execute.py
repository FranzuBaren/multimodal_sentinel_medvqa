# =============================================================================
# CELL 9 — Execution controls (smoke + main run, both gated by flags)
# =============================================================================
# DEFAULT: both flags False. Notebook does NOT spend money on "Run All" unless
# you explicitly flip a flag.

DO_AUDIT_SMOKE = False       # 10 chains stratified (~$0.30, ~3 min)
DO_AUDIT_FULL = False        # 480 chains (~$30-50, ~2-4 hours)
DO_AUTOJUDGE = False         # 100 stratified chains via Opus (~$30, ~30 min)


if DO_AUDIT_SMOKE and ANTHROPIC_API_KEY and AUTOJUDGE_MODEL:
    print(">>> AUDIT SMOKE TEST (10 stratified chains) <<<")

    smoke_ck = OUT_DIR / "audit_smoke_checkpoint.csv"
    if smoke_ck.exists():
        smoke_ck.unlink()

    rng = random.Random(RANDOM_SEED)
    all_cids = chains_df["chain_id"].tolist()
    # Stratify: 2 per model × 2 conditions
    smoke_ids = []
    for model in sorted(chains_df["model"].unique()):
        real_sub = chains_df[(chains_df.model == model) & (chains_df.condition == "real")]["chain_id"].tolist()
        fake_sub = chains_df[(chains_df.model == model) & (chains_df.condition == "fake")]["chain_id"].tolist()
        if real_sub:
            smoke_ids.append(rng.choice(real_sub))
        if fake_sub:
            smoke_ids.append(rng.choice(fake_sub))
    smoke_ids = smoke_ids[:10]
    print(f"  Smoke IDs: {smoke_ids}")

    df_smoke = run_audit(client, SENTINEL_MODEL, chains_df,
                          checkpoint=smoke_ck, chain_id_subset=smoke_ids)

    print(f"\n  Smoke results ({len(df_smoke)} rows):")
    cols = ["chain_id", "condition", "weak_gt", "verdict", "confidence", "hitl"]
    print(df_smoke[cols].to_string(index=False))
    print(f"\n  Parse OK (L1/L2/L3): {df_smoke['l1_parse_ok'].mean()*100:.0f}% / "
          f"{df_smoke['l2_parse_ok'].mean()*100:.0f}% / {df_smoke['l3_parse_ok'].mean()*100:.0f}%")
    print(f"  Cost: ${estimated_cost_usd():.4f}")

    # Quick sanity check
    fake_rows = df_smoke[df_smoke.condition == "fake"]
    if len(fake_rows) > 0:
        u_rate = (fake_rows["verdict"] == "U").mean()
        print(f"  Fake-condition U-verdict rate: {u_rate*100:.0f}% (expect high; "
              f"H1 target ≥ 85%)")


if DO_AUDIT_FULL and ANTHROPIC_API_KEY:
    print(">>> AUDIT FULL RUN (480 chains) <<<")
    print(f"  Estimated: $30-50, 2-4 hours")
    print(f"  Hard cap: ${MAX_COST_USD_HARD_CAP}")
    print(f"  Checkpoint: {AUDIT_CHECKPOINT}")
    print()
    df_audit = run_audit(client, SENTINEL_MODEL, chains_df,
                          checkpoint=AUDIT_CHECKPOINT)
    print(f"\n  Completed: {len(df_audit)} chains")
    print(f"  Total cost: ${estimated_cost_usd():.2f}")
    df_audit.to_csv(AUDIT_RESULTS_CSV, index=False)


if DO_AUTOJUDGE and ANTHROPIC_API_KEY and AUTOJUDGE_MODEL:
    print(">>> AUTO-JUDGE ON 100 STRATIFIED CHAINS <<<")
    sampled_ids = stratified_autojudge_sample(chains_df, n_per_model=25,
                                               seed=RANDOM_SEED)
    print(f"  Sampled {len(sampled_ids)} chain IDs")
    df_aj = run_autojudge(client, AUTOJUDGE_MODEL, chains_df,
                           chain_id_subset=sampled_ids,
                           checkpoint=AUTOJUDGE_CHECKPOINT)
    print(f"\n  Auto-judge complete: {len(df_aj)} labels")
    df_aj.to_csv(AUTOJUDGE_RESULTS_CSV, index=False)


if not (DO_AUDIT_SMOKE or DO_AUDIT_FULL or DO_AUTOJUDGE):
    print("Run controls: all flags are False.")
    print()
    print("Suggested sequence:")
    print("  1. Set DO_AUDIT_SMOKE = True, re-run this cell. ~$0.30, 3 min.")
    print("  2. Inspect smoke output; check parse rates and fake-U rate.")
    print("  3. Set DO_AUDIT_SMOKE=False, DO_AUDIT_FULL=True. ~$30-50, 2-4h.")
    print("  4. Set DO_AUDIT_FULL=False, DO_AUTOJUDGE=True. ~$30, 30 min.")
    print("  5. Go to Cell 10 for analysis.")
