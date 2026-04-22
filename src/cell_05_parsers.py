# =============================================================================
# CELL 5 — Parsers for audit outputs
# =============================================================================

def _find_field(text: str, field_name: str) -> Optional[str]:
    """Parse FIELD: value, tolerant of markdown bold."""
    pat = re.compile(
        rf"^\s*\*?\*?{re.escape(field_name)}\*?\*?\s*:\s*(.+?)\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    m = pat.search(text)
    return m.group(1).strip().strip("*").strip() if m else None


def _norm_yes_no(x: Optional[str]) -> str:
    if not x: return "unknown"
    x = x.lower().strip()
    if x.startswith("yes"): return "yes"
    if x.startswith("no"): return "no"
    if x.startswith("partial"): return "partial"
    return "unknown"


def parse_l1_auditor(text: str) -> dict:
    findings = []
    # Parse CLAIMED_FINDINGS: section as bullet list until next section or end
    m = re.search(
        r"CLAIMED_FINDINGS\s*:\s*\n(.*?)(?=\n\s*(?:REASONING_QUALITY|NOTES|$))",
        text, re.IGNORECASE | re.DOTALL,
    )
    if m:
        block = m.group(1)
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("-") or line.startswith("*"):
                f = line.lstrip("-*").strip()
                if f:
                    findings.append(f)

    rq = (_find_field(text, "REASONING_QUALITY") or "").lower().strip()
    if "high" in rq: rq = "high"
    elif "medium" in rq or "mid" in rq: rq = "medium"
    elif "low" in rq: rq = "low"
    else: rq = "unknown"

    return {
        "claimed_findings": findings,
        "n_findings": len(findings),
        "reasoning_quality": rq,
        "notes": _find_field(text, "NOTES") or "",
        "parse_ok": len(findings) > 0 and rq != "unknown",
    }


def parse_l2_verifier(text: str) -> dict:
    # Parse PER_FINDING block
    per_finding = []
    m = re.search(
        r"PER_FINDING\s*:\s*\n(.*?)(?=\n\s*(?:MODALITY_MATCH|OVERALL_GROUNDING|$))",
        text, re.IGNORECASE | re.DOTALL,
    )
    if m:
        block = m.group(1)
        for line in block.splitlines():
            line = line.strip()
            if not (line.startswith("-") or line.startswith("*")):
                continue
            # Format: "- <finding>: <VERDICT> — <justification>"
            body = line.lstrip("-*").strip()
            # Verdict is in uppercase at some position
            verdict = "UNKNOWN"
            for v in ["NOT_SUPPORTED", "SUPPORTED", "CANNOT_ASSESS"]:
                if v in body.upper():
                    verdict = v
                    break
            per_finding.append({"raw": body, "verdict": verdict})

    mm = _norm_yes_no(_find_field(text, "MODALITY_MATCH"))
    og = _norm_yes_no(_find_field(text, "OVERALL_GROUNDING"))

    # Counts
    n_supported = sum(1 for f in per_finding if f["verdict"] == "SUPPORTED")
    n_not_supported = sum(1 for f in per_finding if f["verdict"] == "NOT_SUPPORTED")
    n_cannot_assess = sum(1 for f in per_finding if f["verdict"] == "CANNOT_ASSESS")

    return {
        "per_finding": per_finding,
        "n_findings_total": len(per_finding),
        "n_supported": n_supported,
        "n_not_supported": n_not_supported,
        "n_cannot_assess": n_cannot_assess,
        "modality_match": mm,
        "overall_grounding": og,
        "parse_ok": og != "unknown" and len(per_finding) > 0,
    }


def parse_l3_reconciler(text: str) -> dict:
    verdict_raw = (_find_field(text, "GROUNDING_VERDICT") or "").upper().strip()
    verdict = None
    for v in ["G", "P", "U"]:
        if re.search(rf"\b{v}\b", verdict_raw):
            verdict = v
            break

    conf_raw = _find_field(text, "CONFIDENCE") or "0.5"
    try:
        conf = float(re.search(r"[0-9]*\.?[0-9]+", conf_raw).group(0))
        conf = max(0.0, min(1.0, conf))
    except (AttributeError, ValueError):
        conf = 0.5

    hitl_raw = (_find_field(text, "HITL") or "NO").upper()
    hitl = hitl_raw.startswith("YES")

    return {
        "verdict": verdict,
        "confidence": conf,
        "hitl": hitl,
        "hitl_reason": _find_field(text, "HITL_REASON") or "none",
        "rationale": _find_field(text, "RATIONALE") or "",
        "parse_ok": verdict is not None,
    }


def parse_autojudge(text: str) -> dict:
    cat_raw = (_find_field(text, "CATEGORY") or "").upper().strip()
    cat = None
    for v in ["G", "P", "U"]:
        if re.search(rf"\b{v}\b", cat_raw):
            cat = v
            break
    return {
        "category": cat,
        "rationale": _find_field(text, "RATIONALE") or "",
        "parse_ok": cat is not None,
    }


# ---- Self-tests -------------------------------------------------------------
_test_l1 = """CLAIMED_FINDINGS:
- ST elevation in V1-V6
- Poor R wave progression

REASONING_QUALITY: high
NOTES: chain engages with the ECG specifically."""
_r = parse_l1_auditor(_test_l1)
assert _r["n_findings"] == 2 and _r["reasoning_quality"] == "high" and _r["parse_ok"], _r

_test_l2 = """PER_FINDING:
- ST elevation in V1-V6: SUPPORTED — visible in precordial leads
- Poor R wave progression: CANNOT_ASSESS — leads unclear

MODALITY_MATCH: yes
OVERALL_GROUNDING: partial"""
_r = parse_l2_verifier(_test_l2)
assert _r["n_supported"] == 1 and _r["n_cannot_assess"] == 1
assert _r["modality_match"] == "yes" and _r["overall_grounding"] == "partial"
assert _r["parse_ok"]

_test_l3 = """GROUNDING_VERDICT: P
CONFIDENCE: 0.65
HITL: NO
HITL_REASON: none
RATIONALE: Mixed verification."""
_r = parse_l3_reconciler(_test_l3)
assert _r["verdict"] == "P" and 0.6 < _r["confidence"] < 0.7 and not _r["hitl"]
assert _r["parse_ok"]

_test_aj = "CATEGORY: U\nRATIONALE: Image is a red placeholder, not a medical image."
_r = parse_autojudge(_test_aj)
assert _r["category"] == "U" and _r["parse_ok"]

print("Cell 5: parsers defined and self-tested OK")
