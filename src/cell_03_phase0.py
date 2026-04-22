# =============================================================================
# CELL 3 — Phase 0: probe Sentinel + auto-judge models
# =============================================================================

def phase0(client: anthropic.Anthropic) -> dict:
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sentinel_model": SENTINEL_MODEL,
        "sentinel_ok": False,
        "autojudge_candidates_tested": [],
        "autojudge_model": None,
        "autojudge_is_fallback": False,
        "notes": [],
    }

    # Test image: 256x256 red square
    img_array = np.zeros((256, 256, 3), dtype=np.uint8)
    img_array[:, :, 0] = 200
    test_img = Image.fromarray(img_array)
    buf = BytesIO()
    test_img.save(buf, format="JPEG", quality=85)
    test_b64 = base64.b64encode(buf.getvalue()).decode()

    # ---- 1. Verify Sentinel model ----
    print(f"\n  Verifying Sentinel model: {SENTINEL_MODEL}")
    r_sent = call_vlm(
        client, SENTINEL_MODEL,
        system="You reply with exactly one word.",
        user_text="Reply PING.", max_tokens=10, retries=1,
    )
    if r_sent["error"] is None and r_sent["text"]:
        results["sentinel_ok"] = True
        print(f"    Sentinel OK → {r_sent['text'][:40]!r}")
    else:
        print(f"    Sentinel FAIL → {r_sent['error']}")
        results["notes"].append(f"Sentinel model failed probe: {r_sent['error']}")

    # ---- 2. Vision probe on Sentinel ----
    print(f"  Vision probe on Sentinel")
    r_vis = call_vlm(
        client, SENTINEL_MODEL,
        system="You describe images concisely.",
        user_text="What is the dominant color in this image? One word.",
        image_b64=test_b64, media_type="image/jpeg",
        max_tokens=10, retries=1,
    )
    sentinel_vision_ok = r_vis["error"] is None and "red" in r_vis["text"].lower()
    print(f"    Vision OK: {sentinel_vision_ok} → {r_vis['text'][:40]!r}")
    if not sentinel_vision_ok:
        results["sentinel_ok"] = False
        results["notes"].append(f"Sentinel vision probe failed: {r_vis['error'] or r_vis['text'][:80]}")

    # ---- 3. Probe auto-judge candidates ----
    print(f"\n  Probing auto-judge candidates (Opus-class preferred for H3)")
    for cand in AUTOJUDGE_CANDIDATES:
        print(f"    {cand}")

        # Text probe
        r = call_vlm(
            client, cand,
            system="Reply with one word.",
            user_text="Reply PONG.", max_tokens=10, retries=1,
        )
        entry = {
            "model_id": cand,
            "text_probe_ok": r["error"] is None and bool(r["text"]),
            "error": r["error"],
        }

        if entry["text_probe_ok"]:
            # Vision probe
            r_vis = call_vlm(
                client, cand,
                system="You describe images concisely.",
                user_text="What is the dominant color? One word.",
                image_b64=test_b64, media_type="image/jpeg",
                max_tokens=10, retries=1,
            )
            entry["vision_probe_ok"] = (
                r_vis["error"] is None and "red" in r_vis["text"].lower()
            )
            entry["vision_response"] = r_vis["text"][:50]
            print(f"      text+vision OK: {entry['vision_probe_ok']}")
        else:
            entry["vision_probe_ok"] = False
            print(f"      FAIL: {(r['error'] or '')[:80]}")

        results["autojudge_candidates_tested"].append(entry)

        if (entry["text_probe_ok"] and entry["vision_probe_ok"]
            and results["autojudge_model"] is None):
            results["autojudge_model"] = cand
            if cand == SENTINEL_MODEL:
                results["autojudge_is_fallback"] = True
                results["notes"].append(
                    "Auto-judge falls back to Sentinel model (same family). "
                    "This weakens H3 inter-rater reliability interpretation "
                    "since both judges share architecture. "
                    "Log as deviation in DEVIATIONS_v2.md."
                )
            break

    return results


def phase0_gate(results: dict) -> bool:
    if not results.get("sentinel_ok"):
        print("GATE FAIL: Sentinel model probe failed.")
        return False
    if results.get("autojudge_model") is None:
        print("GATE FAIL: No auto-judge candidate available.")
        print("           Options: run weak-GT only, or request access to an Opus-class model.")
        return False
    print("GATE PASS.")
    print(f"  Sentinel:   {SENTINEL_MODEL}")
    print(f"  Auto-judge: {results['autojudge_model']}")
    if results.get("autojudge_is_fallback"):
        print("  WARN: auto-judge = Sentinel model (same family). H3 interpretation weakened.")
    return True


# --- Execute Phase 0 ---
if not ANTHROPIC_API_KEY:
    print("SKIP Phase 0: set ANTHROPIC_API_KEY first.")
elif not DATASET_REPO.exists():
    print(f"SKIP Phase 0: dataset not at {DATASET_REPO}. Clone it first (see README).")
else:
    client = make_client(ANTHROPIC_API_KEY)
    phase0_result = phase0(client)
    with open(PHASE0_OUT, "w") as f:
        json.dump(phase0_result, f, indent=2)
    print(f"\n  Saved: {PHASE0_OUT}")
    if phase0_gate(phase0_result):
        AUTOJUDGE_MODEL = phase0_result["autojudge_model"]
        print(f"\n  AUTOJUDGE_MODEL locked to: {AUTOJUDGE_MODEL}")
        print(f"  Phase 0 cost: ${estimated_cost_usd():.4f}")
    else:
        print("\n>>> Do NOT proceed until gate passes. <<<")
