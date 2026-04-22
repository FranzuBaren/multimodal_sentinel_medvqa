# =============================================================================
# CELL 10 — Analysis: primary H1 + secondary + bootstrap CIs
# =============================================================================

def bootstrap_sensitivity(labels: list, target: str = "U",
                           n_boot: int = 2000, seed: int = 42) -> tuple:
    """Bootstrap mean and 95% CI for proportion of `labels` equal to `target`."""
    rng = np.random.default_rng(seed)
    labels = [l for l in labels if l is not None]
    if not labels:
        return float("nan"), (float("nan"), float("nan"))
    arr = np.array([1 if l == target else 0 for l in labels])
    n = len(arr)
    point = arr.mean()
    boots = rng.choice(arr, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, (lo, hi)


def analyze_audit(audit_df: pd.DataFrame) -> dict:
    """Compute primary + secondary metrics."""
    metrics = {
        "n_total": len(audit_df),
        "parse_rate_l1": audit_df["l1_parse_ok"].mean(),
        "parse_rate_l2": audit_df["l2_parse_ok"].mean(),
        "parse_rate_l3": audit_df["l3_parse_ok"].mean(),
    }

    # H1: fake-condition sensitivity (classify as U when GT = U)
    fake = audit_df[audit_df["condition"] == "fake"].copy()
    verdicts_fake = fake["verdict"].tolist()
    sens, (lo, hi) = bootstrap_sensitivity(verdicts_fake, target="U",
                                            n_boot=2000, seed=RANDOM_SEED)
    metrics["H1_fake_sensitivity_U"] = sens
    metrics["H1_fake_CI95"] = (lo, hi)
    metrics["H1_pass"] = (sens >= 0.85 and lo > 0.70)

    # H2: real-correct specificity (classify as G when GT = G)
    real_ok = audit_df[(audit_df["condition"] == "real") & (audit_df["weak_gt"] == "G")].copy()
    verdicts_real_ok = real_ok["verdict"].tolist()
    spec, (slo, shi) = bootstrap_sensitivity(verdicts_real_ok, target="G",
                                              n_boot=2000, seed=RANDOM_SEED)
    metrics["H2_real_correct_specificity_G"] = spec
    metrics["H2_real_correct_CI95"] = (slo, shi)
    metrics["H2_pass"] = slo > 0.70

    # Per-model breakdown (H4 exploratory)
    metrics["per_model_fake_U"] = {}
    for model in audit_df["model"].unique():
        sub = audit_df[(audit_df["model"] == model) & (audit_df["condition"] == "fake")]
        v = sub["verdict"].tolist()
        if v:
            p, (lo_, hi_) = bootstrap_sensitivity(v, target="U", n_boot=2000, seed=RANDOM_SEED)
            metrics["per_model_fake_U"][model] = (p, lo_, hi_, len(v))

    # Per-modality breakdown
    metrics["per_modality_fake_U"] = {}
    for mod in audit_df["modality"].unique():
        sub = audit_df[(audit_df["modality"] == mod) & (audit_df["condition"] == "fake")]
        v = sub["verdict"].tolist()
        if len(v) >= 7:  # pre-reg: only modalities with N >= 7
            p, (lo_, hi_) = bootstrap_sensitivity(v, target="U", n_boot=2000, seed=RANDOM_SEED)
            metrics["per_modality_fake_U"][mod] = (p, lo_, hi_, len(v))

    return metrics


def analyze_autojudge(audit_df: pd.DataFrame,
                       autojudge_df: pd.DataFrame) -> dict:
    """H3: Cohen's κ Sentinel vs auto-judge on shared chain_ids."""
    merged = audit_df.merge(
        autojudge_df[["chain_id", "autojudge_category"]],
        on="chain_id", how="inner",
    )
    kappa = cohens_kappa(merged["verdict"].tolist(),
                         merged["autojudge_category"].tolist())
    return {
        "n_paired": len(merged),
        "H3_kappa": kappa,
        "H3_pass": (kappa is not None and kappa >= 0.60),
    }


def print_analysis(audit_df: pd.DataFrame,
                    autojudge_df: Optional[pd.DataFrame] = None):
    m = analyze_audit(audit_df)

    print("=" * 70)
    print("PILOT v2 RESULTS")
    print("=" * 70)
    print(f"\nTotal audits: {m['n_total']}")
    print(f"Parse rate L1 / L2 / L3: "
          f"{m['parse_rate_l1']*100:.1f}% / "
          f"{m['parse_rate_l2']*100:.1f}% / "
          f"{m['parse_rate_l3']*100:.1f}%")
    print(f"Total cost: ${estimated_cost_usd():.2f}")

    print("\n--- H1 (primary): fake-condition sensitivity ---")
    print(f"  Target: ≥0.85 with CI lower bound > 0.70")
    sens = m["H1_fake_sensitivity_U"]
    lo, hi = m["H1_fake_CI95"]
    print(f"  Result: {sens:.3f} [95% CI: {lo:.3f}, {hi:.3f}]")
    print(f"  Verdict: {'✓ H1 REJECTED' if m['H1_pass'] else '✗ H0 NOT REJECTED (negative result)'}")

    print("\n--- H2 (secondary): real-correct specificity ---")
    print(f"  Target: CI lower bound > 0.70")
    spec = m["H2_real_correct_specificity_G"]
    slo, shi = m["H2_real_correct_CI95"]
    print(f"  Result: {spec:.3f} [95% CI: {slo:.3f}, {shi:.3f}]")
    print(f"  Verdict: {'✓' if m['H2_pass'] else '✗'}")

    print("\n--- Per-model fake U-verdict sensitivity ---")
    for model, (p, lo_, hi_, n) in m["per_model_fake_U"].items():
        print(f"  {model:25s}: {p:.3f} [{lo_:.3f}, {hi_:.3f}]  (N={n})")

    print("\n--- Per-modality fake U-verdict sensitivity (N ≥ 7) ---")
    for mod, (p, lo_, hi_, n) in m["per_modality_fake_U"].items():
        print(f"  {mod:10s}: {p:.3f} [{lo_:.3f}, {hi_:.3f}]  (N={n})")

    if autojudge_df is not None and len(autojudge_df):
        print("\n--- H3 (secondary): Sentinel vs auto-judge κ ---")
        h3 = analyze_autojudge(audit_df, autojudge_df)
        print(f"  Target: κ ≥ 0.60")
        print(f"  Paired N: {h3['n_paired']}")
        print(f"  Cohen's κ: {h3['H3_kappa']:.3f}")
        print(f"  Verdict: {'✓' if h3['H3_pass'] else '✗ grounding GT reliability weak, caveat results'}")

    # Decision tree
    print("\n" + "=" * 70)
    print("DECISION PER PRE-REG §9")
    print("=" * 70)
    if m["H1_pass"]:
        if autojudge_df is not None and analyze_autojudge(audit_df, autojudge_df)["H3_pass"]:
            print("→ H1 rejected, H3 κ ≥ 0.60: SCALE UP. Target VQA-RAD or MIMIC-CXR-VQA.")
        else:
            print("→ H1 rejected, H3 κ < 0.60: Report H1 with weak-GT caveat. "
                  "Human annotation prerequisite for scale-up.")
    else:
        print("→ H1 not rejected: Negative result. Sentinel (intra-family) does not "
              "reliably detect grounding failures even in adversarially-easy fake setting.")


# ---- Try to load and analyze whatever is on disk ----
if AUDIT_CHECKPOINT.exists():
    try:
        df_audit = pd.read_csv(AUDIT_CHECKPOINT, on_bad_lines='skip', engine='python')
        df_aj = None
        if AUTOJUDGE_CHECKPOINT.exists():
            df_aj = pd.read_csv(AUTOJUDGE_CHECKPOINT, on_bad_lines='skip', engine='python')
        print_analysis(df_audit, df_aj)
    except Exception as e:
        print(f"Analysis failed: {e}")
else:
    print("No audit checkpoint found. Run Cell 9 first.")
