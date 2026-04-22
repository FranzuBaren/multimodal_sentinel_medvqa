# =============================================================================
# CELL 8 — Auto-judge on stratified 100-chain subsample
# =============================================================================

AUTOJUDGE_CHECKPOINT_COLUMNS = [
    "chain_id", "model", "question_id", "condition", "modality",
    "autojudge_category", "autojudge_rationale",
    "autojudge_parse_ok", "timestamp", "cumulative_cost_usd",
]


def append_autojudge_checkpoint(path: Path, row: dict) -> None:
    full_row = {k: row.get(k, "") for k in AUTOJUDGE_CHECKPOINT_COLUMNS}
    file_exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv_mod.DictWriter(
            f, fieldnames=AUTOJUDGE_CHECKPOINT_COLUMNS, quoting=csv_mod.QUOTE_MINIMAL,
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(full_row)


def stratified_autojudge_sample(chains_df: pd.DataFrame,
                                 n_per_model: int = 25,
                                 seed: int = RANDOM_SEED) -> list:
    """Return list of chain_ids: 25 per model, balanced real/fake."""
    rng = random.Random(seed)
    sampled = []
    for model in sorted(chains_df["model"].unique()):
        sub = chains_df[chains_df["model"] == model]
        real_ids = sub[sub["condition"] == "real"]["chain_id"].tolist()
        fake_ids = sub[sub["condition"] == "fake"]["chain_id"].tolist()
        k_real = min(n_per_model // 2, len(real_ids))
        k_fake = min(n_per_model - k_real, len(fake_ids))
        sampled.extend(rng.sample(real_ids, k_real))
        sampled.extend(rng.sample(fake_ids, k_fake))
    return sampled


def run_autojudge(client, autojudge_model: str,
                   chains_df: pd.DataFrame,
                   chain_id_subset: list,
                   checkpoint: Path) -> pd.DataFrame:
    """Run Opus 4.7 auto-judge on selected chain_ids."""
    completed = load_completed_chain_ids(checkpoint)
    todo_ids = [cid for cid in chain_id_subset if cid not in completed]
    print(f"  Auto-judge queue: {len(todo_ids)} chains (from {len(chain_id_subset)} selected)")

    image_cache: dict[str, tuple] = {}
    df = chains_df[chains_df["chain_id"].isin(todo_ids)].copy()

    pbar = tqdm(df.iterrows(), total=len(df), desc="Auto-judging", unit="chain")
    for _, row in pbar:
        qid = row["question_id"]
        cond = row["condition"]
        img_path = (IMAGES_DIR / f"image_{qid}.png") if cond == "real" else FAKE_IMAGE
        if not img_path.exists():
            pbar.write(f"  SKIP {row['chain_id']}: image missing")
            continue

        cache_key = str(img_path)
        if cache_key not in image_cache:
            try:
                image_cache[cache_key] = image_path_to_b64(img_path, max_dim=1024)
            except Exception as e:
                pbar.write(f"  SKIP {row['chain_id']}: image load {e}")
                continue
        img_b64, media_type = image_cache[cache_key]

        user_text = (
            f"QUESTION (Italian):\n{row['question']}\n\n"
            f"REASONING CHAIN:\n{row['reasoning_chain']}\n\n"
            f"Classify in the required format."
        )
        try:
            r = call_vlm(
                client, autojudge_model, SYSTEM_AUTOJUDGE, user_text,
                image_b64=img_b64, media_type=media_type, max_tokens=200,
            )
        except RuntimeError as e:
            pbar.write(f"  ABORT: {e}")
            break

        p = parse_autojudge(r["text"]) if r["text"] else {
            "category": None, "rationale": "", "parse_ok": False,
        }
        append_autojudge_checkpoint(checkpoint, {
            "chain_id": row["chain_id"],
            "model": row["model"],
            "question_id": qid,
            "condition": cond,
            "modality": row["modality"],
            "autojudge_category": p.get("category"),
            "autojudge_rationale": p.get("rationale", "")[:300],
            "autojudge_parse_ok": p.get("parse_ok", False),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cumulative_cost_usd": estimated_cost_usd(),
        })

        pbar.set_postfix({
            "cost": f"${estimated_cost_usd():.2f}", "calls": len(USAGE_LOG),
        })

    if checkpoint.exists():
        return pd.read_csv(checkpoint, on_bad_lines='skip', engine='python')
    return pd.DataFrame()


def cohens_kappa(a: list, b: list) -> float:
    """Cohen's κ for categorical labels."""
    from sklearn.metrics import cohen_kappa_score
    # Drop pairs with None
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 2:
        return float("nan")
    a_, b_ = zip(*pairs)
    return cohen_kappa_score(a_, b_)


print("Cell 8: auto-judge sampler + runner defined")
