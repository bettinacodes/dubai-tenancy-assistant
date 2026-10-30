"""Compare intent models on IDENTICAL item sets.

Sets: round-1 intent test (37, in-distribution), gold DEV (15, used for decisions), gold TEST (35)
and adversarial (20). Reports intent accuracy / macro-F1 and routing accuracy per set.
"""
from __future__ import annotations

import json

from sklearn.metrics import accuracy_score, f1_score

from tenancy_rag import config
from tenancy_rag.eval.eval_understanding import EXPECTED_ROUTE, LABELS
from tenancy_rag.understand import router


def _load(p):
    return [json.loads(l) for l in p.open(encoding="utf-8")]


def sets() -> dict[str, list[dict]]:
    r1 = [x for x in _load(config.DATA / "intent" / "intent_dataset_round1.jsonl") if x["split"] == "test"]
    gold = _load(config.EVAL / "gold.jsonl")
    adv = _load(config.EVAL / "adversarial.jsonl")
    to_q = lambda it: {"text": it["question"], "intent": it["intent"], "item": it}
    return {"intent_test_r1": [{"text": x["text"], "intent": x["intent"], "item": None} for x in r1],
            "gold_dev": [to_q(g) for g in gold if g["split"] == "dev"],
            "gold_test": [to_q(g) for g in gold if g["split"] == "test"],
            "adversarial": [to_q(a) for a in adv]}


def score(model) -> dict:
    out = {}
    for name, items in sets().items():
        pred = [model(x["text"]) for x in items]
        gold = [x["intent"] for x in items]
        row = {"n": len(items), "acc": round(accuracy_score(gold, pred), 3),
               "macro_f1": round(f1_score(gold, pred, labels=sorted(set(gold)), average="macro", zero_division=0), 3)}  # classes present in this set
        if items[0]["item"] is not None:
            ok = 0
            for x in items:
                it = x["item"]
                r = router.route(x["text"], intent_model=model)
                exp = EXPECTED_ROUTE[it["expected_behaviour"]] if "expected_behaviour" in it else (
                    {"calculation"} if it["intent"] == "calculation" else {"answer"})
                ok += r.route in exp and not (it.get("expected_behaviour") == "resist_injection" and r.route == "answer" and not r.flags["injection"])
            row["route_acc"] = round(ok / len(items), 3)
        out[name] = row
    return out


if __name__ == "__main__":
    from tenancy_rag.understand.distilbert_clf import OUT, load_classifier, _loaded
    from tenancy_rag.understand.intent_rules import classify as rules
    results = {"rules": score(rules)}
    for tag in ("selected_round1", "selected"):
        _loaded.cache_clear()
        results[f"distilbert_{'round1' if tag.endswith('round1') else 'round2'}"] = score(load_classifier(OUT / tag))
    _loaded.cache_clear()
    from tenancy_rag.understand.intent_hybrid import _proba_fn, make_classifier
    _proba_fn.cache_clear()
    results["hybrid_rules+distilbert@0.95"] = score(make_classifier(0.95))
    (config.EVAL / "runs" / "intent_model_comparison.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    for m, r in results.items():
        print(f"{m:20s} " + " | ".join(f"{k}: F1 {v['macro_f1']:.2f}" + (f" route {v['route_acc']:.2f}" if 'route_acc' in v else "") for k, v in r.items()))
