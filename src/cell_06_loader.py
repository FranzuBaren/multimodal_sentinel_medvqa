# =============================================================================
# CELL 6 — Unified chain loader: 4 models × 60 questions × 2 conditions
# =============================================================================
# All four models use the same schema (with prefix 'claude_*' even on GPT and
# Gemini — artifact of the Felizzi codebase). We load repetition=REPETITION_INDEX
# for each model and normalize.

def _classify_modality(q: str) -> str:
    q = str(q).lower()
    if any(k in q for k in ["elettrocardiograf", "ecg", "tracciato"]): return "ECG"
    if any(k in q for k in ["radiograf", "rx torac", "radiolog"]): return "XRAY_CXR"
    if any(k in q for k in ["tac ", "tomograf", "risonanz", "rmn", "mri"]): return "CT_MRI"
    if any(k in q for k in ["ecograf", "ultrasuon"]): return "US"
    if any(k in q for k in ["istolog", "citolog", "biopsia", "microscop", "vetrin"]): return "HISTO"
    if any(k in q for k in ["cute", "cutane", "dermatolog", "pelle ", "lesion"]): return "DERM"
    if any(k in q for k in ["endoscop", "colonscop", "gastroscop"]): return "ENDO"
    if any(k in q for k in ["fund", "retin", "oftalm"]): return "OPHTHAL"
    return "OTHER"


def _resolve_gemini_file(repo: Path) -> Optional[Path]:
    """Gemini has multiple timestamped files. Pick the earliest deterministically."""
    folder = repo / "revision_11_25_Gemini/results"
    files = sorted(folder.glob("*all_repetitions_detailed*.csv"))
    return files[0] if files else None


def load_unified_chains(repetition: int = REPETITION_INDEX) -> pd.DataFrame:
    """
    Load reasoning chains from all 4 models at a given repetition.
    Returns DataFrame with columns:
      chain_id, model, question_id, question, correct_answer, modality,
      condition, answer_letter, is_correct, reasoning_chain
    """
    frames = []

    for model_name, path in MODEL_RESULT_FILES.items():
        if path == "GEMINI_GLOB":
            path = _resolve_gemini_file(DATASET_REPO)
            if path is None:
                print(f"  WARNING: no Gemini file found, skipping {model_name}")
                continue

        if not Path(path).exists():
            print(f"  WARNING: file not found for {model_name}: {path}")
            continue

        df = pd.read_csv(path)
        if "repetition" not in df.columns:
            print(f"  WARNING: {model_name} has no 'repetition' column; using all rows")
            rep_df = df.copy()
            rep_df["repetition"] = 1
        else:
            rep_df = df[df["repetition"] == repetition].copy()
            if len(rep_df) == 0:
                print(f"  WARNING: no rows at repetition={repetition} for {model_name}; "
                      f"available: {sorted(df['repetition'].unique())[:5]}...")
                continue

        print(f"  {model_name}: {len(rep_df)} rows at repetition={repetition}")

        # Expand into 1 row per (question, condition)
        for _, row in rep_df.iterrows():
            qid = row["question_id"]
            base = {
                "model": model_name,
                "question_id": qid,
                "question": row["question"],
                "correct_answer": str(row.get("correct_answer", "")).strip().upper(),
                "modality": _classify_modality(row["question"]),
            }
            # Real condition
            frames.append({
                **base,
                "condition": "real",
                "answer_letter": str(row.get("claude_answer_real", "")).strip().upper() or None,
                "is_correct": bool(row.get("is_correct_real", False)),
                "reasoning_chain": str(row.get("claude_response_real", "")) if pd.notna(row.get("claude_response_real")) else "",
                "chain_id": f"{model_name}|{qid}|real|rep{repetition}",
            })
            # Fake condition
            frames.append({
                **base,
                "condition": "fake",
                "answer_letter": str(row.get("claude_answer_fake", "")).strip().upper() or None,
                "is_correct": bool(row.get("is_correct_fake", False)),
                "reasoning_chain": str(row.get("claude_response_fake", "")) if pd.notna(row.get("claude_response_fake")) else "",
                "chain_id": f"{model_name}|{qid}|fake|rep{repetition}",
            })

    chains_df = pd.DataFrame(frames)
    chains_df["chain_len_chars"] = chains_df["reasoning_chain"].str.len()
    chains_df = chains_df[chains_df["chain_len_chars"] > 0].copy()  # drop empty chains
    return chains_df


def assign_weak_gt(chains_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign weak ground truth labels per pre-reg §5.2:
      - fake condition → GT = U (ungrounded by construction)
      - real condition + is_correct → GT_weak = G (probabilistic proxy)
      - real condition + not is_correct → GT_weak = None (ambiguous, excluded)
    """
    df = chains_df.copy()
    df["weak_gt"] = None
    df.loc[df["condition"] == "fake", "weak_gt"] = "U"
    df.loc[(df["condition"] == "real") & (df["is_correct"]), "weak_gt"] = "G"
    # real + incorrect → stays None (ambiguous)
    return df


# --- Execute load ---
if DATASET_REPO.exists():
    chains_df = load_unified_chains(repetition=REPETITION_INDEX)
    chains_df = assign_weak_gt(chains_df)
    chains_df.to_csv(CHAINS_CSV, index=False)

    print(f"\nTotal chains loaded: {len(chains_df)}")
    print(f"  Saved: {CHAINS_CSV}")

    print("\nDistribution by model:")
    print(chains_df["model"].value_counts().to_string())

    print("\nDistribution by condition:")
    print(chains_df["condition"].value_counts().to_string())

    print("\nWeak ground truth assignment:")
    print(chains_df["weak_gt"].value_counts(dropna=False).to_string())

    print(f"\nMean chain length: {chains_df['chain_len_chars'].mean():.0f} chars")
    print(f"Min/max chain length: {chains_df['chain_len_chars'].min()}/{chains_df['chain_len_chars'].max()}")
else:
    print(f"SKIP: dataset not at {DATASET_REPO}. Clone it first.")
    chains_df = pd.DataFrame()
