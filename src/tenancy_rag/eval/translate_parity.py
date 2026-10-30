"""Build the multilingual parity set: the 50 gold questions machine-translated into Arabic and Hindi
with NLLB-200 (facebook/nllb-200-distilled-600M), plus English back-translations for a quality check.

Caveat: machine-translated questions are cleaner than real Arabic/Hindi tenant
messages, so parity measured here is an optimistic lower bound on the real language gap.
Output: data/eval/multilingual/gold_{ar,hi}.jsonl
"""
from __future__ import annotations

import json
from functools import lru_cache

from tenancy_rag import config

LANGS = {"ar": "arb_Arab", "hi": "hin_Deva"}
SRC = "eng_Latn"
OUT = config.EVAL / "multilingual"


@lru_cache(maxsize=1)
def _mt():
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(config.MT_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.MT_MODEL).to(device).eval()
    return tok, model, device


def translate(texts: list[str], src: str, tgt: str, batch_size: int = 8) -> list[str]:
    import torch
    tok, model, device = _mt()
    tok.src_lang = src
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(tgt), max_new_tokens=256, num_beams=4)
        out += tok.batch_decode(gen, skip_special_tokens=True)
    return out


def build() -> dict:
    gold = [json.loads(l) for l in (config.EVAL / "gold.jsonl").open(encoding="utf-8")]
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for code, nllb in LANGS.items():
        qs = translate([g["question"] for g in gold], SRC, nllb)
        back = translate(qs, nllb, SRC)
        with (OUT / f"gold_{code}.jsonl").open("w", encoding="utf-8") as f:
            for g, q, b in zip(gold, qs, back):
                f.write(json.dumps({**g, "question": q, "question_en": g["question"], "back_translation": b, "lang": code},
                                   ensure_ascii=False) + "\n")
        report[code] = len(qs)
    return report


if __name__ == "__main__":
    print(build())
