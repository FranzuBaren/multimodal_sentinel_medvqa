# =============================================================================
# CELL 2 — Anthropic client, call helpers, hash chain
# =============================================================================

def make_client(api_key: str, timeout_s: float = 60.0) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        api_key=api_key,
        timeout=httpx.Timeout(timeout_s, connect=10.0),
    )


USAGE_LOG: list[dict] = []


def _model_family_key(model_id: str) -> str:
    m = (model_id or "").lower()
    if "sonnet-4-5" in m: return "sonnet-4-5"
    if "sonnet-4" in m: return "sonnet-4"
    if "3-7-sonnet" in m: return "sonnet-3-7"
    if "3-5-sonnet" in m: return "sonnet-3-5"
    if "opus-4-5" in m: return "opus-4-5"
    if "opus-4" in m: return "opus-4"
    if "opus-3" in m: return "opus-3"
    if "haiku-4-5" in m: return "haiku-4-5"
    if "haiku-3-5" in m: return "haiku-3-5"
    return "unknown"


def estimated_cost_usd() -> float:
    total = 0.0
    for u in USAGE_LOG:
        k = u.get("family", "unknown")
        total += (u.get("input_tokens", 0) / 1e6) * PRICE_PER_MTOK_INPUT.get(k, 3.0)
        total += (u.get("output_tokens", 0) / 1e6) * PRICE_PER_MTOK_OUTPUT.get(k, 15.0)
    return total


def _build_content(user_text: str, image_b64: Optional[str] = None,
                    media_type: str = "image/jpeg") -> list:
    """Build Anthropic-native content list for a user message."""
    content = []
    if image_b64:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": image_b64,
            },
        })
    content.append({"type": "text", "text": user_text})
    return content


def call_vlm(
    client: anthropic.Anthropic, model_id: str, system: str, user_text: str,
    image_b64: Optional[str] = None, media_type: str = "image/jpeg",
    temperature: float = 0.0, max_tokens: int = 800,
    retries: int = RETRY_MAX,
) -> dict:
    """Call an Anthropic model. Returns dict with text, tokens, cost info."""
    last_error = None

    for attempt in range(retries):
        try:
            t0 = time.time()
            kwargs = {
                "model": model_id,
                "system": system,
                "messages": [{
                    "role": "user",
                    "content": _build_content(user_text, image_b64, media_type),
                }],
                "max_tokens": max_tokens,
            }
            if temperature is not None and temperature > 0:
                kwargs["temperature"] = temperature

            resp = client.messages.create(**kwargs)
            elapsed = time.time() - t0

            # Anthropic SDK returns content as list of blocks
            text = ""
            for block in resp.content:
                if hasattr(block, "text"):
                    text += block.text
                elif isinstance(block, dict) and block.get("type") == "text":
                    text += block.get("text", "")
            text = text.strip()

            in_tok = resp.usage.input_tokens if resp.usage else 0
            out_tok = resp.usage.output_tokens if resp.usage else 0

            USAGE_LOG.append({
                "model_id": model_id, "family": _model_family_key(model_id),
                "input_tokens": in_tok, "output_tokens": out_tok,
                "elapsed_s": elapsed, "timestamp": time.time(),
            })

            if estimated_cost_usd() > MAX_COST_USD_HARD_CAP:
                raise RuntimeError(
                    f"Cost hard cap exceeded: ${estimated_cost_usd():.2f} "
                    f"> ${MAX_COST_USD_HARD_CAP:.2f}"
                )

            return {
                "text": text, "input_tokens": in_tok, "output_tokens": out_tok,
                "elapsed_s": elapsed, "model_id": model_id, "error": None,
            }

        except RuntimeError:
            raise
        except anthropic.NotFoundError as e:
            return {
                "text": "", "input_tokens": 0, "output_tokens": 0,
                "elapsed_s": 0.0, "model_id": model_id,
                "error": f"model_unavailable: {str(e)[:200]}",
            }
        except anthropic.BadRequestError as e:
            return {
                "text": "", "input_tokens": 0, "output_tokens": 0,
                "elapsed_s": 0.0, "model_id": model_id,
                "error": f"bad_request: {str(e)[:200]}",
            }
        except (anthropic.APIConnectionError, anthropic.RateLimitError,
                anthropic.InternalServerError) as e:
            last_error = e
            if attempt < retries - 1:
                sleep_s = RETRY_BACKOFF_BASE ** attempt
                print(f"  [retry {attempt+1}/{retries}] {str(e)[:120]} — sleep {sleep_s:.1f}s")
                time.sleep(sleep_s)
                continue
        except Exception as e:
            last_error = e
            if attempt < retries - 1:
                sleep_s = RETRY_BACKOFF_BASE ** attempt
                time.sleep(sleep_s)
                continue
            break

    return {
        "text": "", "input_tokens": 0, "output_tokens": 0,
        "elapsed_s": 0.0, "model_id": model_id,
        "error": f"all_retries_failed: {str(last_error)[:200]}",
    }


def image_path_to_b64(path: Path, max_dim: int = 1024) -> tuple[str, str]:
    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        if "A" in img.mode:
            bg.paste(img, mask=img.split()[-1])
        else:
            bg.paste(img.convert("RGB"))
        img = bg
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"


# ---- Hash chain (PROV-O audit trail) ----------------------------------------

def sha256_of(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()


def extend_chain(prev_hash: str, block: dict) -> str:
    h = hashlib.sha256()
    h.update(prev_hash.encode())
    h.update(json.dumps(block, sort_keys=True, default=str).encode())
    return h.hexdigest()


print("Cell 2: Anthropic client + helpers + hash chain defined")
