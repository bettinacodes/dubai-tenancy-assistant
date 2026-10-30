"""Build the intent-classification dataset: hand-written seeds + local-LLM paraphrases.

1. Seeds (data/intent/seeds.jsonl): ~18 per class, written by hand in plain / colloquial /
   non-native styles.
2. Augmentation: qwen2.5:3b (local, free) writes 3 paraphrases per seed - colloquial,
   non-native English, formal - keeping intent and facts (JSON mode, temperature 0.7, seeded).
3. Leakage filter: drop any item whose bge-m3 cosine to ANY gold/adversarial question is >= 0.85,
   so the classifier is never evaluated on near-copies of its training data.
4. Near-duplicate filter: drop items with cosine >= 0.95 to an item already kept.
5. Stratified split 70/15/15 (train/val/test) by *seed group*: all paraphrases of a seed go to the
   same split, so test items are never paraphrases of training items.
Output: data/intent/intent_dataset.jsonl (+ build_report.json).
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict

import numpy as np

from tenancy_rag import config
from tenancy_rag.llm import chat

DIR = config.DATA / "intent"
LABELS = ["factual", "calculation", "procedural", "out_of_scope", "advice_seeking", "crisis"]
PROMPT = ("Rewrite the user's message in 3 different ways that a tenant in Dubai might actually type. "
          "Keep exactly the same meaning, request type and any numbers/dates. "
          "1) casual/colloquial, 2) non-native English with simple grammar, 3) formal. "
          'Return JSON: {"paraphrases": ["...", "...", "..."]}')


def paraphrase(text: str) -> list[str]:
    res = chat([{"role": "system", "content": PROMPT}, {"role": "user", "content": text}],
               temperature=0.7, json_mode=True, max_tokens=300)
    try:
        out = json.loads(res.text).get("paraphrases", [])
    except json.JSONDecodeError:
        return []
    return [p.strip() for p in out if isinstance(p, str) and 8 <= len(p.strip()) <= 300][:3]


def generate() -> list[dict]:
    seeds = [json.loads(l) for l in (DIR / "seeds.jsonl").open(encoding="utf-8")]
    items = []
    for i, s in enumerate(seeds):
        group = f"s{i:03d}"
        items.append({"text": s["text"], "intent": s["intent"], "group": group, "origin": "seed"})
        for p in paraphrase(s["text"]):
            items.append({"text": p, "intent": s["intent"], "group": group, "origin": "llm_paraphrase"})
        print(f"[{i + 1}/{len(seeds)}]", flush=True)
    (DIR / "raw_candidates.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in items) + "\n", encoding="utf-8")
    return items


import re

_STYLE_PREFIX = re.compile(r"^\s*(?:\d\)\s*)?(?:casual|colloquial|non[- ]native[^:]*|formal|informal)[^:]{0,40}:\s*", re.I)
DOMAIN_WORDS = {"rdc", "ejari", "dewa", "aed", "jvc", "rera", "dld", "rest", "difc", "noc", "k", "lakh", "dubai", "uae",
                "landlord", "landlady", "ok", "pls", "plz", "u", "ur", "flat", "airbnb", "bro", "hey", "nah", "gonna", "wanna"}


def _numbers(text: str) -> list[str]:
    """Significant digits of each number; '250,000' and '250k' both -> '250'."""
    out = []
    for m in re.finditer(r"(\d[\d,]*(?:\.\d+)?)\s*(k|lakh)?", text.lower()):
        digits = m.group(1).replace(",", "").split(".")[0].rstrip("0") or "0"
        out.append(digits)
    return out


def quality_gate(item: dict, seed_text: str) -> tuple[str | None, str]:
    """Return (reason_to_drop or None, cleaned_text)."""
    from wordfreq import zipf_frequency
    text = _STYLE_PREFIX.sub("", item["text"]).strip().strip('"')
    words = re.findall(r"[a-zA-Z]+", text)
    if words:
        english = sum(1 for w in words if w.lower() in DOMAIN_WORDS or zipf_frequency(w, "en") >= 2.5)
        if english / len(words) < 0.75:
            return "garbled_non_english", text
    if item["intent"] == "calculation" and item["origin"] != "seed":
        have = "".join(_numbers(text))
        missing = [n for n in _numbers(seed_text) if n not in have]
        if missing:
            return f"numbers_lost:{missing}", text
    return None, text


def round2_items() -> list[dict]:
    """Round-2 human seeds (no LLM paraphrasing) targeting the contrast the round-1 model missed:
    'Can I / Is it allowed / Do I need' (what the LAW says) vs 'Should I / will I win' (ADVICE).
    Round 2 exists because round-1 paraphrases taught "Can I..." = advice."""
    f = DIR / "seeds_round2.jsonl"
    if not f.exists():
        return []
    return [{"text": s["text"], "intent": s["intent"], "group": f"r2_{i:03d}", "origin": "seed_round2"}
            for i, s in enumerate(map(json.loads, f.open(encoding="utf-8")))]


def filter_and_split(items: list[dict], seed: int = 0) -> dict:
    from tenancy_rag.index.embed import Embedder
    n_raw, raw_origin = len(items), Counter(x["origin"] for x in items)
    seed_text = {x["group"]: x["text"] for x in items if x["origin"] in ("seed", "seed_round2")}
    manual = {(r["text"]) : r["reason"] for r in json.loads((DIR / "manual_rejects.json").read_text(encoding="utf-8"))} \
        if (DIR / "manual_rejects.json").exists() else {}
    gated, dropped_quality = [], []
    for x in items:
        reason, text = quality_gate(x, seed_text[x["group"]])
        x = {**x, "text": text}
        reason = reason or manual.get(text)
        (dropped_quality.append({**x, "reason": reason}) if reason else gated.append(x))
    items = gated
    emb = Embedder()
    evalq = [json.loads(l)["question"] for f in ("gold.jsonl", "adversarial.jsonl") for l in (config.EVAL / f).open(encoding="utf-8")]
    E = emb.encode(evalq, "query")
    X = emb.encode([x["text"] for x in items], "query")
    leak_sim = (X @ E.T).max(axis=1)
    report = {"candidates": n_raw, "by_origin": raw_origin}
    kept, kept_vecs, dropped_leak, dropped_dup = [], [], [], []
    for x, v, ls in zip(items, X, leak_sim):
        if ls >= 0.85:
            dropped_leak.append({**x, "max_sim_to_eval": round(float(ls), 3)})
            continue
        if kept_vecs and float(np.max(np.stack(kept_vecs) @ v)) >= 0.95:
            dropped_dup.append(x)
            continue
        kept.append({**x, "max_sim_to_eval": round(float(ls), 3)})
        kept_vecs.append(v)
    # stratified group split; round-1 groups keep their original split so test sets stay comparable
    prev = DIR / "intent_dataset_round1.jsonl"
    split_of = {x["group"]: x["split"] for x in map(json.loads, prev.open(encoding="utf-8"))} if prev.exists() else {}
    rng = random.Random(seed)
    groups_by_label = defaultdict(list)
    for g in sorted({(x["intent"], x["group"]) for x in kept if x["group"] not in split_of}):
        groups_by_label[g[0]].append(g[1])
    for label, groups in groups_by_label.items():
        rng.shuffle(groups)
        n = len(groups)
        n_test, n_val = max(2, round(0.15 * n)), max(2, round(0.15 * n))
        for j, g in enumerate(groups):
            split_of[g] = "test" if j < n_test else "val" if j < n_test + n_val else "train"
    for x in kept:
        x["split"] = split_of.get(x["group"], "train")   # round-1 groups dropped earlier and re-kept -> train
    with (DIR / "intent_dataset.jsonl").open("w", encoding="utf-8") as f:
        for x in kept:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    report.update({"dropped_quality": len(dropped_quality),
                   "dropped_quality_reasons": Counter(d["reason"].split(":")[0] for d in dropped_quality),
                   "quality_examples": [{"text": d["text"], "reason": d["reason"]} for d in dropped_quality],
                   "kept": len(kept), "dropped_leakage": len(dropped_leak), "dropped_near_duplicate": len(dropped_dup),
                   "leakage_examples": dropped_leak[:10],
                   "by_label": Counter(x["intent"] for x in kept),
                   "by_split": {s: Counter(x["intent"] for x in kept if x["split"] == s) for s in ("train", "val", "test")}})
    (DIR / "build_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=dict), encoding="utf-8")
    return report


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["generate"]:
        generate()
    else:
        items = [json.loads(l) for l in (DIR / "raw_candidates.jsonl").open(encoding="utf-8")] + round2_items()
        print(json.dumps(filter_and_split(items), indent=1, default=dict)[:2500])
