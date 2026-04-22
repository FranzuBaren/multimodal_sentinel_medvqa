# =============================================================================
# CELL 1 — Configuration
# =============================================================================
# Sentinel 3LoD: a three-agent adversarial VLM grounding audit primitive.
# This notebook audits reasoning chains from the Felizzi et al. 2025
# MMRL4H Italian MedVQA dataset (480 chains: 4 models × 60 questions
# × 2 conditions × 1 repetition).
#
# Pre-registration: see PREREGISTRATION_v2.md. Hash-locked before main run.

import os, sys, json, base64, hashlib, re, random, time, csv as csv_mod
from io import BytesIO
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import httpx
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

try:
    import anthropic
except ImportError:
    raise ImportError(
        "This notebook requires the anthropic Python SDK. "
        "Install with: pip install anthropic"
    )


# ---- API credentials --------------------------------------------------------
# Set ANTHROPIC_API_KEY in your environment:
#   export ANTHROPIC_API_KEY='sk-ant-...'
# Or inline (not persisted in notebook):
#   import getpass; os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Key: ")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


# ---- Models -----------------------------------------------------------------
# Sentinel agents: Claude Sonnet 4.5 (public model ID). Replace with a newer
# Sonnet variant if you want to re-run with updated model. For cross-family
# replication, swap in another frontier VLM — see DEVIATIONS.md for guidance.
SENTINEL_MODEL = "claude-sonnet-4-5-20250929"

# Auto-judge for inter-rater agreement (H3).
# Prefer Opus-class for independent validation. Fall back to Sonnet with a
# documented caveat (same-family judge inflates κ).
AUTOJUDGE_CANDIDATES = [
    "claude-opus-4-5-20250101",   # adjust to the Opus variant you have access to
    "claude-sonnet-4-5-20250929",  # same-family fallback, document as deviation
]
AUTOJUDGE_MODEL: Optional[str] = None  # populated by Phase 0


# ---- Dataset ----------------------------------------------------------------
# Dataset is NOT included in this repo — it is Felizzi et al. property.
# Clone it separately into the notebook working directory:
#   git clone https://github.com/felizzi/eurips2025-mmrl4h-italian-medvqa-visual-grounding.git
DATASET_REPO = Path("./eurips2025-mmrl4h-italian-medvqa-visual-grounding")
IMAGES_DIR = DATASET_REPO / "data/images"
FAKE_IMAGE = DATASET_REPO / "data/Fake_Image_path/image.png"

MODEL_RESULT_FILES = {
    "claude-sonnet-4-5": DATASET_REPO / "revision_11_25/results/all_repetitions_detailed.csv",
    "gpt-4o": DATASET_REPO / "revision_11_25_OpenAI/results/openai_gpt-4o_all_repetitions_detailed.csv",
    "gpt-5-mini": DATASET_REPO / "revision_11_25_OpenAI/results/openai_gpt-5-mini_all_repetitions_detailed.csv",
    "gemini-2-0-flash-exp": "GEMINI_GLOB",  # resolved in loader cell
}


# ---- Output paths -----------------------------------------------------------
OUT_DIR = Path("./pilot_outputs")
OUT_DIR.mkdir(exist_ok=True)

PHASE0_OUT = OUT_DIR / "phase0_results.json"
CHAINS_CSV = OUT_DIR / "chains_unified.csv"
AUDIT_CHECKPOINT = OUT_DIR / "audit_checkpoint.csv"
AUTOJUDGE_CHECKPOINT = OUT_DIR / "autojudge_checkpoint.csv"
AUDIT_TRAIL = OUT_DIR / "audit_trail.jsonl"

# Pre-reg hash file — paste the sha256 of PREREGISTRATION_v2.md here before
# running the main audit, and verify against the committed value.
PREREG_HASH_FILE = OUT_DIR / "prereg_hash.txt"
EXPECTED_PREREG_HASH: Optional[str] = None  # paste when locking


# ---- Experimental parameters (pre-registered) ------------------------------
RANDOM_SEED = 42
REPETITION_INDEX = 1
N_STRATIFIED_AUTOJUDGE = 100

# ---- Safety caps ------------------------------------------------------------
MAX_COST_USD_HARD_CAP = 50.0
MAX_API_CALLS = 3000
RETRY_MAX = 3
RETRY_BACKOFF_BASE = 2.0

# ---- Pricing — Claude public pricing (USD per million tokens) --------------
# Update at lock time if prices have changed.
PRICE_PER_MTOK_INPUT = {
    "sonnet-4-5": 3.0, "sonnet-4": 3.0, "sonnet-3-7": 3.0, "sonnet-3-5": 3.0,
    "opus-4-5": 15.0, "opus-4": 15.0, "opus-3": 15.0,
    "haiku-4-5": 1.0, "haiku-3-5": 1.0,
    "unknown": 3.0,
}
PRICE_PER_MTOK_OUTPUT = {
    "sonnet-4-5": 15.0, "sonnet-4": 15.0, "sonnet-3-7": 15.0, "sonnet-3-5": 15.0,
    "opus-4-5": 75.0, "opus-4": 75.0, "opus-3": 75.0,
    "haiku-4-5": 5.0, "haiku-3-5": 5.0,
    "unknown": 15.0,
}


if not ANTHROPIC_API_KEY:
    print("=" * 70)
    print("ERROR: ANTHROPIC_API_KEY environment variable is empty.")
    print()
    print("Set your key via one of:")
    print("  export ANTHROPIC_API_KEY='sk-ant-...'")
    print("  import os, getpass")
    print("  os.environ['ANTHROPIC_API_KEY'] = getpass.getpass('Key: ')")
    print()
    print("Get a key at: https://console.anthropic.com/")
    print("=" * 70)
else:
    print(f"Notebook start:    {datetime.now(timezone.utc).isoformat()}")
    print(f"Output dir:        {OUT_DIR.resolve()}")
    print(f"Sentinel model:    {SENTINEL_MODEL}")
    print(f"Auto-judge probe:  {len(AUTOJUDGE_CANDIDATES)} candidates")
    print(f"Dataset repo:      {DATASET_REPO.resolve()} (exists: {DATASET_REPO.exists()})")
    if not DATASET_REPO.exists():
        print()
        print("WARNING: dataset not found. Clone it first:")
        print("  git clone https://github.com/felizzi/"
              "eurips2025-mmrl4h-italian-medvqa-visual-grounding.git")
    print(f"Pre-reg hash:      {EXPECTED_PREREG_HASH or '(not yet locked — see PREREGISTRATION_v2.md §8)'}")
