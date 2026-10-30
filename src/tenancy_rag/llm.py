"""Thin LLM interface with two free backends and a disk cache.

  * "ollama"       local development (qwen2.5:3b, GTX 1650) and the Colab T4 demo (qwen2.5:7b), via the HTTP API
  * "transformers" Qwen2.5-Instruct in 4-bit NF4 via bitsandbytes - kept as an alternative; on Colab with
                   transformers 5.18 the model stayed on the CPU, so the demo uses Ollama

Every call is cached on disk keyed by (backend, model, messages, params), so evaluation runs and
ablations can be re-scored without re-generating (and without re-paying in time).
Deterministic decoding (temperature 0) is the default for evaluation.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from functools import lru_cache

import httpx

from tenancy_rag import config

OLLAMA_URL = "http://localhost:11434"
NUM_CTX = 6144  # prompts reach ~3.6k tokens (6 full articles); answer.fit_context() guards the budget


@dataclass
class LLMResult:
    text: str
    model: str
    backend: str
    latency_s: float
    cached: bool = False
    usage: dict = field(default_factory=dict)


def _cache_path(key: dict):
    h = hashlib.sha256(json.dumps(key, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return config.CACHE / "llm" / h[:2] / f"{h}.json"


@lru_cache(maxsize=1)
def _hf_model(model_name: str):
    """Load an instruct model with transformers. On CUDA with bitsandbytes: 4-bit NF4 (Qwen2.5-7B fits a
    Colab T4 in ~5.5 GB). Otherwise fp16 on tensor-core GPUs, fp32 elsewhere (GTX fp16 is slow)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    kwargs = {}
    if torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes  # noqa: F401
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                               bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
            # transformers 5 keeps the checkpoint's bf16 for the unquantised parts (embeddings, lm_head, norms);
            # the T4 has no bf16 hardware, so force fp16
            kwargs["dtype"] = torch.float16
        except ImportError as e:   # loud, not silent: an fp16 7B does not fit a T4 and gets offloaded to CPU on Colab
            print(f"WARNING: bitsandbytes unavailable ({e!r}); loading {model_name} unquantised")
            kwargs["dtype"] = torch.float32 if "GTX" in torch.cuda.get_device_name(0) else torch.float16
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if torch.cuda.is_available() and {p.device.type for p in model.parameters()} != {"cuda"}:
        # fail fast: Colab run 1 (transformers 5.18) left the 4-bit 7B on the CPU and an answer never finished
        raise RuntimeError(f"{model_name} is not fully on the GPU (devices: {sorted({str(p.device) for p in model.parameters()})}); "
                           "use the Ollama backend instead")
    return tok, model.eval()


def _hf_generate(model_name: str, messages: list[dict], temperature: float, max_tokens: int,
                 json_mode: bool = False) -> tuple[str, dict]:
    """json_mode: transformers has no constrained decoding (Ollama's format=json), so the reply is prefilled
    with "{" - the model must continue a JSON object (smoke test: without it Qwen2.5-0.5B wrote free text)."""
    import torch
    tok, model = _hf_model(model_name)
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False) + ("{" if json_mode else "")
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    ids = enc["input_ids"]
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_tokens, do_sample=temperature > 0,
                             temperature=temperature if temperature > 0 else None, top_p=None, top_k=None,
                             pad_token_id=tok.eos_token_id)
    new = out[0, ids.shape[1]:]
    return ("{" if json_mode else "") + tok.decode(new, skip_special_tokens=True), {"prompt_tokens": int(ids.shape[1]), "completion_tokens": int(new.shape[0])}


def chat(messages: list[dict], *, backend: str = "ollama", model: str | None = None, temperature: float = 0.0,
         max_tokens: int = 700, json_mode: bool = False, use_cache: bool = True) -> LLMResult:
    model = model or (config.GEN_MODEL_OLLAMA if backend == "ollama" else config.GEN_MODEL_HF)
    key = {"backend": backend, "model": model, "messages": messages, "temperature": temperature, "num_ctx": NUM_CTX,
           "max_tokens": max_tokens, "json_mode": json_mode}
    path = _cache_path(key)
    if use_cache and path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        return LLMResult(**{**d, "cached": True})

    t0 = time.perf_counter()
    if backend == "ollama":
        payload = {"model": model, "messages": messages, "stream": False,
                   # 4k context keeps the 3B model + KV cache inside the 4 GB GPU (default 32k spills to CPU)
                   "options": {"temperature": temperature, "num_predict": max_tokens, "seed": 0, "num_ctx": NUM_CTX}}
        if json_mode:
            payload["format"] = "json"
        for attempt in range(3):   # a model (re)load can fail transiently (seen: CUDA init error -> HTTP 500)
            r = httpx.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=600)
            if r.status_code < 500 or attempt == 2:
                break
            time.sleep(15)
        r.raise_for_status()
        d = r.json()
        text = d["message"]["content"]
        usage = {"prompt_tokens": d.get("prompt_eval_count"), "completion_tokens": d.get("eval_count")}
    elif backend == "transformers":
        text, usage = _hf_generate(model, messages, temperature, max_tokens, json_mode)
    else:
        raise ValueError(backend)
    res = LLMResult(text=text, model=model, backend=backend, latency_s=round(time.perf_counter() - t0, 3), usage=usage)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: v for k, v in res.__dict__.items() if k != "cached"}, ensure_ascii=False),
                    encoding="utf-8")
    return res
