"""Fine-tuned intent classifier: distilbert-base-multilingual-cased, 6 intents.

Why this model: a small (134M) multilingual encoder that runs in milliseconds on CPU, so intent
routing needs no LLM call and no data leaves the device; multilingual so Arabic/Hindi queries can
be routed (measured on Day 9). Trained on data/intent/intent_dataset.jsonl (train split), model
selection on val macro-F1 per epoch, reported on the held-out test split. Several seeds are run
because with ~176 training examples a single run is not trustworthy.
Usage: python -m tenancy_rag.understand.distilbert_clf train [--seeds 0 1 2] [--lora]
"""
from __future__ import annotations

import json
import random
import time
from functools import lru_cache

import numpy as np

from tenancy_rag import config

LABELS = ["factual", "calculation", "procedural", "out_of_scope", "advice_seeking", "crisis"]
L2I = {l: i for i, l in enumerate(LABELS)}
OUT = config.ROOT / "models" / "intent-distilbert"
DATA = config.DATA / "intent" / "intent_dataset.jsonl"


def _data(split: str) -> list[dict]:
    return [x for x in map(json.loads, DATA.open(encoding="utf-8")) if x["split"] == split]


def _seed_all(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _evaluate(model, tok, items, device, batch_size=32) -> tuple[float, float, list[str]]:
    import torch
    from sklearn.metrics import accuracy_score, f1_score
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(items), batch_size):
            b = tok([x["text"] for x in items[i:i + batch_size]], truncation=True, max_length=128, padding=True, return_tensors="pt").to(device)
            preds += model(**b).logits.argmax(-1).tolist()
    pred = [LABELS[p] for p in preds]
    gold = [x["intent"] for x in items]
    return accuracy_score(gold, pred), f1_score(gold, pred, labels=LABELS, average="macro", zero_division=0), pred


def train(seed: int = 0, epochs: int = 12, lr: float = 5e-5, batch_size: int = 16, lora: bool = False, save: bool = True) -> dict:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    _seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(config.INTENT_BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(config.INTENT_BASE_MODEL, num_labels=len(LABELS),
                                                               id2label=dict(enumerate(LABELS)), label2id=L2I)
    if lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(task_type="SEQ_CLS", r=8, lora_alpha=16, lora_dropout=0.1,
                                                 target_modules=["q_lin", "v_lin"],
                                                 modules_to_save=["pre_classifier", "classifier"]))  # both heads are new: must train
        lr = max(lr, 5e-4)
    model.to(device)
    train_set, val_set = _data("train"), _data("val")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    steps = epochs * ((len(train_set) + batch_size - 1) // batch_size)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    best = {"val_macro_f1": -1}
    history = []
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        order = list(range(len(train_set)))
        random.shuffle(order)
        losses = []
        for i in range(0, len(order), batch_size):
            batch = [train_set[j] for j in order[i:i + batch_size]]
            enc = tok([x["text"] for x in batch], truncation=True, max_length=128, padding=True, return_tensors="pt").to(device)
            labels = torch.tensor([L2I[x["intent"]] for x in batch], device=device)
            loss = model(**enc, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            losses.append(loss.item())
        acc, f1, _ = _evaluate(model, tok, val_set, device)
        history.append({"epoch": ep, "train_loss": round(float(np.mean(losses)), 4), "val_acc": round(acc, 3), "val_macro_f1": round(f1, 3)})
        if f1 > best["val_macro_f1"]:
            best = {"epoch": ep, "val_acc": acc, "val_macro_f1": f1}
            if save:
                target = OUT / (f"seed{seed}" + ("-lora" if lora else ""))
                target.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(target)  # LoRA: adapter + trained heads; full: whole model
                tok.save_pretrained(target)
    return {"seed": seed, "lora": lora, "best": best, "history": history, "train_seconds": round(time.perf_counter() - t0, 1),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_params": sum(p.numel() for p in model.parameters())}


def test_checkpoint(path) -> dict:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(path)
    if (path / "adapter_config.json").exists():
        # AutoPeftModel rebuilds the base with the default 2 labels -> load the 6-label base first
        from peft import PeftModel
        base = AutoModelForSequenceClassification.from_pretrained(config.INTENT_BASE_MODEL, num_labels=len(LABELS),
                                                                  id2label=dict(enumerate(LABELS)), label2id=L2I)
        model = PeftModel.from_pretrained(base, path).to(device)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(path).to(device)
    acc, f1, _ = _evaluate(model, tok, _data("test"), device)
    return {"test_acc": round(acc, 3), "test_macro_f1": round(f1, 3)}


@lru_cache(maxsize=1)
def _loaded(path: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to("cpu").eval()  # CPU: milliseconds, frees the GPU
    torch.set_num_threads(max(1, torch.get_num_threads()))
    return tok, model


def load_classifier(path=None):
    """Return a callable text -> intent label (CPU inference)."""
    import torch
    path = str(path or (OUT / "selected"))
    tok, model = _loaded(path)

    def classify(text: str) -> str:
        with torch.no_grad():
            enc = tok([text], truncation=True, max_length=128, return_tensors="pt")
            return LABELS[int(model(**enc).logits.argmax(-1))]
    return classify


if __name__ == "__main__":
    import argparse
    import shutil
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lora", action="store_true")
    a = ap.parse_args()
    runs = []
    for s in a.seeds:
        r = train(seed=s, epochs=a.epochs, lora=a.lora)
        r.update(test_checkpoint(OUT / (f"seed{s}" + ("-lora" if a.lora else ""))))
        runs.append(r)
        print(f"seed {s}{' lora' if a.lora else ''}: best epoch {r['best']['epoch']} val F1 {r['best']['val_macro_f1']:.3f} "
              f"| test acc {r['test_acc']} F1 {r['test_macro_f1']} | {r['train_seconds']} s | trainable {r['trainable_params']:,}", flush=True)
    tag = "lora" if a.lora else "full"
    f1s = [r["test_macro_f1"] for r in runs]
    # std: sample standard deviation over seeds (ddof=1)
    summary = {"variant": tag, "runs": runs, "test_macro_f1_mean": round(float(np.mean(f1s)), 3), "test_macro_f1_std": round(float(np.std(f1s, ddof=1)), 3),
               "test_acc_mean": round(float(np.mean([r["test_acc"] for r in runs])), 3)}
    (config.EVAL / "runs" / f"intent_finetune__{tag}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    if not a.lora:   # the deployed model = seed with the best VALIDATION F1 (test is not used to choose)
        best = max(runs, key=lambda r: r["best"]["val_macro_f1"])
        sel = OUT / "selected"
        if sel.exists():
            shutil.rmtree(sel)
        shutil.copytree(OUT / f"seed{best['seed']}", sel)
        print(f"selected seed {best['seed']} (by val F1) -> {sel}")
    print(f"{tag}: test macro-F1 {summary['test_macro_f1_mean']} ± {summary['test_macro_f1_std']} (n seeds {len(runs)})")
