# =============================================================================
# CELL 7 — Sentinel audit function + runner
# =============================================================================

# ---- Checkpoint schema ------------------------------------------------------
AUDIT_CHECKPOINT_COLUMNS = [
    "chain_id", "model", "question_id", "condition", "modality",
    "correct_answer", "answer_letter", "is_correct", "weak_gt",
    "verdict", "confidence", "hitl", "hitl_reason",
    "n_findings", "n_supported", "n_not_supported", "n_cannot_assess",
    "modality_match", "overall_grounding", "reasoning_quality",
    "l1_parse_ok", "l2_parse_ok", "l3_parse_ok",
    "timestamp", "cumulative_cost_usd", "hash_final",
]


def append_audit_checkpoint(path: Path, row: dict) -> None:
    full_row = {k: row.get(k, "") for k in AUDIT_CHECKPOINT_COLUMNS}
    file_exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv_mod.DictWriter(
            f, fieldnames=AUDIT_CHECKPOINT_COLUMNS, quoting=csv_mod.QUOTE_MINIMAL,
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(full_row)


def load_completed_chain_ids(path: Path) -> set:
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path)
        return set(df["chain_id"].astype(str))
    except Exception as e:
        print(f"  Checkpoint load warning: {e}")
        return set()


def append_audit_trail(entry: dict) -> None:
    with open(AUDIT_TRAIL, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


# ---- Single-chain audit -----------------------------------------------------

def sentinel_audit_one(
    client, audit_model: str, chain_row: pd.Series,
    image_b64: str, media_type: str,
) -> dict:
    """Run L1 → L2 → L3 on one reasoning chain. 3 API calls."""

    # User text: question + chain under review
    user_base = (
        f"QUESTION (Italian):\n{chain_row['question']}\n\n"
        f"CORRECT ANSWER: {chain_row['correct_answer']} "
        f"(do not use this to judge grounding; grounding is about reasoning process, not outcome)\n\n"
        f"REASONING CHAIN UNDER REVIEW:\n{chain_row['reasoning_chain']}"
    )

    # --- L1 ---
    r1 = call_vlm(
        client, audit_model, SYSTEM_L1_AUDITOR,
        user_base + "\n\nExtract claimed findings in the required format.",
        image_b64=image_b64, media_type=media_type, max_tokens=600,
    )
    p1 = parse_l1_auditor(r1["text"]) if r1["text"] else {
        "claimed_findings": [], "n_findings": 0, "reasoning_quality": "unknown",
        "notes": "", "parse_ok": False,
    }

    # --- L2 ---
    findings_text = "\n".join(f"- {f}" for f in p1.get("claimed_findings", []))
    l2_user = (
        f"{user_base}\n\n"
        f"--- Line 1 extraction ---\n"
        f"CLAIMED_FINDINGS:\n{findings_text or '- (none extracted)'}\n"
        f"REASONING_QUALITY: {p1.get('reasoning_quality', 'unknown')}\n"
        f"NOTES: {p1.get('notes', '')}\n\n"
        f"Provide your per-finding verification in the required format."
    )
    r2 = call_vlm(
        client, audit_model, SYSTEM_L2_VERIFIER, l2_user,
        image_b64=image_b64, media_type=media_type, max_tokens=700,
    )
    p2 = parse_l2_verifier(r2["text"]) if r2["text"] else {
        "per_finding": [], "n_findings_total": 0, "n_supported": 0,
        "n_not_supported": 0, "n_cannot_assess": 0,
        "modality_match": "unknown", "overall_grounding": "unknown",
        "parse_ok": False,
    }

    # --- L3 (text only, no image) ---
    l3_user = (
        f"QUESTION:\n{chain_row['question']}\n\n"
        f"REASONING CHAIN:\n{chain_row['reasoning_chain']}\n\n"
        f"--- Line 1 ---\nreasoning_quality={p1.get('reasoning_quality')}, "
        f"n_findings={p1.get('n_findings')}, "
        f"notes={p1.get('notes')}\n\n"
        f"--- Line 2 ---\nmodality_match={p2.get('modality_match')}, "
        f"overall_grounding={p2.get('overall_grounding')}, "
        f"supported/not/cannot={p2.get('n_supported')}/{p2.get('n_not_supported')}/"
        f"{p2.get('n_cannot_assess')}\n\n"
        f"Decide the final grounding verdict in the required format."
    )
    r3 = call_vlm(
        client, audit_model, SYSTEM_L3_RECONCILER, l3_user,
        image_b64=None, max_tokens=400,
    )
    p3 = parse_l3_reconciler(r3["text"]) if r3["text"] else {
        "verdict": None, "confidence": 0.5, "hitl": True,
        "hitl_reason": "parse_fail", "rationale": "", "parse_ok": False,
    }

    # --- Fallback if L3 parse failed ---
    if not p3["parse_ok"]:
        # Derive verdict from L2 signals
        if p2.get("overall_grounding") == "no" or p2.get("modality_match") == "no":
            fallback_verdict = "U"
        elif p2.get("overall_grounding") == "partial":
            fallback_verdict = "P"
        elif p2.get("overall_grounding") == "yes":
            fallback_verdict = "G"
        else:
            fallback_verdict = None
        p3 = {
            **p3,
            "verdict": fallback_verdict,
            "hitl": True,
            "hitl_reason": "l3_parse_failed_fallback",
        }

    # --- Hash chain ---
    genesis = sha256_of({"chain_id": chain_row["chain_id"]})
    h1 = extend_chain(genesis, {"agent": "l1", "raw": r1["text"]})
    h2 = extend_chain(h1, {"agent": "l2", "raw": r2["text"]})
    h3 = extend_chain(h2, {"agent": "l3", "raw": r3["text"]})

    return {
        "p1": p1, "p2": p2, "p3": p3,
        "l1_raw": r1["text"], "l2_raw": r2["text"], "l3_raw": r3["text"],
        "l1_error": r1["error"], "l2_error": r2["error"], "l3_error": r3["error"],
        "hash_chain": [genesis, h1, h2, h3],
    }


# ---- Runner -----------------------------------------------------------------

def run_audit(
    client, audit_model: str, chains_df: pd.DataFrame,
    checkpoint: Path,
    limit: Optional[int] = None,
    chain_id_subset: Optional[list] = None,
) -> pd.DataFrame:
    """Run audit on all chains in chains_df (or subset), with resume + checkpoint."""
    completed = load_completed_chain_ids(checkpoint)
    print(f"  Resuming with {len(completed)} chains already audited")

    df = chains_df.copy()
    if chain_id_subset is not None:
        df = df[df["chain_id"].isin(chain_id_subset)]
    if limit is not None:
        df = df.head(limit)

    # Pre-filter: skip completed
    df = df[~df["chain_id"].isin(completed)]
    print(f"  Work queue: {len(df)} chains")

    # Pre-load images (dedupe)
    image_cache: dict[str, tuple] = {}

    pbar = tqdm(df.iterrows(), total=len(df), desc="Auditing", unit="chain")
    for _, chain_row in pbar:
        qid = chain_row["question_id"]
        cond = chain_row["condition"]

        # Determine image path
        if cond == "real":
            img_path = IMAGES_DIR / f"image_{qid}.png"
        else:
            img_path = FAKE_IMAGE

        if not img_path.exists():
            pbar.write(f"  SKIP {chain_row['chain_id']}: image missing ({img_path})")
            continue

        # Load image (cached)
        cache_key = str(img_path)
        if cache_key not in image_cache:
            try:
                image_cache[cache_key] = image_path_to_b64(img_path, max_dim=1024)
            except Exception as e:
                pbar.write(f"  SKIP {chain_row['chain_id']}: image load {e}")
                continue
        img_b64, media_type = image_cache[cache_key]

        # Run audit
        try:
            result = sentinel_audit_one(
                client, audit_model, chain_row, img_b64, media_type,
            )
        except RuntimeError as e:
            pbar.write(f"  ABORT on {chain_row['chain_id']}: {e}")
            break

        # Flatten
        p1, p2, p3 = result["p1"], result["p2"], result["p3"]
        flat = {
            "chain_id": chain_row["chain_id"],
            "model": chain_row["model"],
            "question_id": qid,
            "condition": cond,
            "modality": chain_row["modality"],
            "correct_answer": chain_row["correct_answer"],
            "answer_letter": chain_row["answer_letter"],
            "is_correct": chain_row["is_correct"],
            "weak_gt": chain_row["weak_gt"],
            "verdict": p3.get("verdict"),
            "confidence": p3.get("confidence"),
            "hitl": p3.get("hitl"),
            "hitl_reason": p3.get("hitl_reason"),
            "n_findings": p1.get("n_findings"),
            "n_supported": p2.get("n_supported"),
            "n_not_supported": p2.get("n_not_supported"),
            "n_cannot_assess": p2.get("n_cannot_assess"),
            "modality_match": p2.get("modality_match"),
            "overall_grounding": p2.get("overall_grounding"),
            "reasoning_quality": p1.get("reasoning_quality"),
            "l1_parse_ok": p1.get("parse_ok", False),
            "l2_parse_ok": p2.get("parse_ok", False),
            "l3_parse_ok": p3.get("parse_ok", False),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cumulative_cost_usd": estimated_cost_usd(),
            "hash_final": result["hash_chain"][-1],
        }
        append_audit_checkpoint(checkpoint, flat)
        append_audit_trail({
            "chain_id": chain_row["chain_id"], "result": {
                "l1_raw": result["l1_raw"][:2000],
                "l2_raw": result["l2_raw"][:2000],
                "l3_raw": result["l3_raw"][:1000],
                "hash_chain": result["hash_chain"],
            },
        })

        pbar.set_postfix({
            "cost": f"${estimated_cost_usd():.2f}",
            "calls": len(USAGE_LOG),
        })

    # Consolidate
    if checkpoint.exists():
        return pd.read_csv(checkpoint, on_bad_lines='skip', engine='python')
    return pd.DataFrame()


print("Cell 7: audit function + runner defined")
