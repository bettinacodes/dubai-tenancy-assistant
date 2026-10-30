"""Generates notebooks/03_end_to_end_eval.ipynb (reproducible notebook source)."""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
md("""# 03 · End-to-end evaluation: grounding, safety, calculators, latency, multilingual parity — Musta'jir

**Purpose.** Measure the whole pipeline (route -> retrieve -> calculate -> generate -> verify -> fallback) against the
50 gold questions and 20 adversarial items, compare two generator sizes (qwen2.5 3B vs 7B), run **ablation 7 (no citation
validator)**, and measure **multilingual parity** (Arabic, Hindi).

**How the runs are staged** (`tenancy_rag.eval.run_e2e`): stage 1 routes, extracts slots, runs the calculators and retrieves
for every item (no LLM); stage 2 generates and verifies with a given model. Every LLM call is disk-cached, so this notebook
re-scores saved runs in seconds. Set `RUN_GENERATION = True` to regenerate (about 45 min for 3B and 1.5 h for 7B on a 4 GB
GTX 1650; faster on a Colab T4).

**Ablation 7 needs no extra run:** the first generation attempt of every item is stored *before* verification, so "what
would the user have seen without the validator" is scored from the same run."""),
md("## 1 · Setup"),
code("""import os, sys, subprocess
from pathlib import Path
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import userdata
    if not Path("dubai-tenancy-assistant").exists():
        try:   # a read-only token is only needed if the repository is private
            auth = userdata.get("GITHUB_TOKEN") + "@"
        except Exception:
            auth = ""
        subprocess.run(["git", "clone", f"https://{auth}github.com/bettinacodes/dubai-tenancy-assistant.git"], check=True)
    os.chdir("dubai-tenancy-assistant")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)
    sys.path.insert(0, "src")
else:
    root = Path.cwd()
    while not (root / "pyproject.toml").exists():
        root = root.parent
    os.chdir(root); sys.path.insert(0, str(root / "src"))
import json
import pandas as pd
import matplotlib.pyplot as plt
from tenancy_rag import config
from tenancy_rag.eval.score_e2e import score
from tenancy_rag.eval.score_baseline import score as score_baseline
RUNS = config.EVAL / "runs"
MODELS = ["qwen2.5:3b", "qwen2.5:7b"]
RUN_GENERATION = False
load = lambda p: [json.loads(l) for l in open(p, encoding="utf-8")]
gold = {g["id"]: g for g in load(config.EVAL / "gold.jsonl")}
adv = {a["id"]: a for a in load(config.EVAL / "adversarial.jsonl")}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#a3a29e"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True, "savefig.dpi": 200, "savefig.bbox": "tight"})
FIG = config.ROOT / "figures"; FIG.mkdir(parents=True, exist_ok=True)"""),
code("""if RUN_GENERATION:
    subprocess.run([sys.executable, "-m", "tenancy_rag.eval.run_e2e", "1", "en"], check=True)
    for m in MODELS:
        subprocess.run([sys.executable, "-m", "tenancy_rag.eval.run_e2e", "2", "en", m], check=True)
S = {m: score("en", m) for m in MODELS}"""),
md("""## 2 · Starting point: the same model without retrieval (Day 3 baseline, config 0)
Asked directly, both models invent legislation. Scored automatically by extracting every instrument / article they cite and
checking it against the corpus."""),
code("""base = {m: score_baseline(RUNS / f"config0_no_retrieval__{m.replace(':', '-')}.jsonl")["summary"] for m in MODELS}
pd.DataFrame({m: {"answers citing a law that does not exist": f"{b['answers_citing_unknown_instrument']}/{b['n_gold']}",
                  "binding provision correctly cited": b["primary_citation_correct"],
                  "calculation answered correctly": b["numeric_correct"],
                  "misnames an institution": b["answers_misnaming_an_institution"]} for m, b in base.items()})"""),
md("""## 3 · Main result: the full system, and ablation 7 (no validator)
* **verified** = generated answer whose every claim passed: the cited source exists in the retrieved set, it is in force,
  the claim's numbers occur in it, and an NLI model finds the claim entailed by a window of the cited text.
* **fallback** = verification failed twice, so the user sees a quotation of the law (or an options template for advice,
  or a 2007-vs-2008 comparison for "what changed") instead of an unverified paraphrase.
* **unsupported shown** = answers that display at least one claim failing verification. For the validated system it is 0 by
  construction; for ablation 7 it is what users would have seen."""),
code("""rows = {}
for m in MODELS:
    s = S[m]
    rows[(m, "validated")] = {"verified answers": s["verified_rate"], "fallback shown": s["fallback_rate"],
        "unsupported claim shown": s["validated"]["unsupported_shown"], "citation precision": s["validated"]["citation_precision"],
        "binding provision cited": s["validated"]["primary_cited"]}
    rows[(m, "no validator (abl. 7)")] = {"verified answers": None, "fallback shown": 0.0,
        "unsupported claim shown": s["no_validator"]["unsupported_shown"], "citation precision": s["no_validator"]["citation_precision"],
        "binding provision cited": s["no_validator"]["primary_cited"]}
main = pd.DataFrame(rows).T
main"""),
code("""pd.DataFrame({m: S[m]["modes"] for m in MODELS}).fillna(0).astype(int)"""),
code("""fig, ax = plt.subplots(figsize=(6.4, 2.6))
labels = [f"{m.split(':')[1].upper()}" for m in MODELS]
y = range(len(MODELS))
nv = [S[m]["no_validator"]["unsupported_shown"] for m in MODELS]
va = [S[m]["validated"]["unsupported_shown"] for m in MODELS]
ax.barh([i + 0.2 for i in y], nv, height=0.36, color=ORANGE, label="without validator (ablation 7)")
ax.barh([i - 0.2 for i in y], va, height=0.36, color=BLUE, label="with validator (deployed)")
for i, (a, b) in enumerate(zip(nv, va)):
    ax.text(a + 0.01, i + 0.2, f"{a:.0%}", va="center", color=INK, fontsize=9)
    ax.text(b + 0.01, i - 0.2, f"{b:.0%}", va="center", color=INK, fontsize=9)
ax.set_yticks(list(y), [f"qwen2.5 {l}" for l in labels]); ax.set_xlim(0, 1); ax.grid(axis="y", visible=False)
ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax.set_xlabel("share of gold answers showing at least one unsupported claim")
ax.legend(frameon=False, loc="lower right", fontsize=9)
fig.savefig(FIG / "fig7_2_validator_ablation.png"); plt.show()"""),
md("""**Reading.** Without verification, 60% (3B) and 42% (7B) of answers would show a claim that its cited text does not
support. The validator removes all of them, paid for with fallbacks - which the 7B model needs less often. Safety
(below) is identical across models because routing and templates, not the generator, decide it.

**Fixes made after the first scoring, each found by error analysis on these 50 questions** (so the later rows are
not held-out results; all three stages are reported):
1. *Fallback fix.* In 5 items per model the first attempt cited the binding article but a claim failed,
   and the extractive fallback quoted a *different* passage chosen by word overlap. It now prefers the binding source the
   model cited. Deterministic, so re-scored from the cached generations.
2. *Slot and history-detector fixes (9.8, 10.2).* Dates were assigned by order of mention, relative periods ("in 12
   months") were not parsed, and the "what changed?" detector fired on the word "before" (G20 and G35 were shown a
   2007-vs-2008 comparison instead of an answer). The 8 affected items per model were regenerated.
Run files for each stage: `data/eval/runs/pre_fallback_fix/`, `data/eval/runs/pre_day10_fixes/`, and the current runs."""),
code("""def stage_scores(folder):
    base = RUNS / folder if folder else RUNS
    return {m: json.loads((base / f"e2e_score__en__{m.replace(':', '-')}.json").read_text(encoding="utf-8")) for m in MODELS}
stages = {"1 first scoring": stage_scores("pre_fallback_fix"), "2 + fallback fix": stage_scores("pre_day10_fixes"),
          "3 + slot & history fixes (current)": S}
pd.DataFrame({(m, st): {"verified answers": d[m]["verified_rate"], "binding provision cited": d[m]["validated"]["primary_cited"],
                        "citation precision": d[m]["validated"]["citation_precision"],
                        "unsupported shown (validated)": d[m]["validated"]["unsupported_shown"],
                        "unsupported shown (no validator)": d[m]["no_validator"]["unsupported_shown"],
                        "calculators correct": f"{d[m]['calculators']['correct']}/{d[m]['calculators']['items']}"}
              for m in MODELS for st, d in stages.items()}).T"""),
md("""## 4 · Calculators
The three calculators are pure Python; the question is whether the pipeline extracts the right inputs and calls them.
First the end-to-end runs, then a direct re-check of routing + slots + tools on the current code (no LLM)."""),
code("""pd.DataFrame({m: S[m]["calculators"] for m in MODELS})"""),
code("""from tenancy_rag.understand.router import route
from tenancy_rag.pipeline import run_tools
chk = []
for g in gold.values():
    if not g.get("tool"):
        continue
    r = route(g["question"]); t = run_tools(r.slots, r.query)
    res, exp = (t or {}).get("result", {}), g["tool"]["expected"]
    ok = bool(t) and t["tool"] == g["tool"]["name"] and all(
        abs(float(res.get(k, -1e9)) - float(v)) < 0.01 if isinstance(v, (int, float)) and not isinstance(v, bool) else res.get(k) == v
        for k, v in exp.items() if k in res)
    chk.append({"id": g["id"], "expected tool": g["tool"]["name"], "fired": t["tool"] if t else None, "correct": ok})
chk = pd.DataFrame(chk); print(f"current code: {chk.correct.sum()}/{len(chk)} correct"); chk"""),
md("""## 5 · Safety: adversarial items and the abstention matrix
`must_refuse` items (other emirates, commercial leases, drafting, injection, crisis...) must never get a substantive answer."""),
code("""from tenancy_rag.eval.eval_understanding import EXPECTED_ROUTE
st2 = {m: {r["id"]: r for r in load(RUNS / f"e2e_stage2__en__{m.replace(':', '-')}.jsonl")} for m in MODELS}
a_rows = []
for a in adv.values():
    r = st2["qwen2.5:7b"][a["id"]]
    a_rows.append({"id": a["id"], "type": a["type"], "expected": a["expected_behaviour"], "must refuse": a["must_refuse"],
                   "route": r["route"], "shown": r["mode"], "ok": r["route"] in EXPECTED_ROUTE[a["expected_behaviour"]]})
A = pd.DataFrame(a_rows)
print({m: S[m]["adversarial"] for m in MODELS})
pd.crosstab(A.expected, A.route)"""),
code("""A[~A.ok]   # the one miss (shared by both models)"""),
md("""## 6 · Latency (local GTX 1650, 4 GB)
Seconds per answered gold question (stage 1 retrieval + stage 2 generation and verification). The 7B model does not fit in
4 GB of VRAM and partly runs on the CPU; the target (p50 < 15 s) applies to the Colab T4 and is measured on Day 10."""),
code("""fig, ax = plt.subplots(figsize=(6.4, 2.4))
data = []
for m in MODELS:
    rr = [r for r in st2[m].values() if r["id"] in gold and r["route"] in ("answer", "calculation", "advice")]
    data.append([r.get("stage1_s", 0) + r.get("stage2_s", 0) for r in rr])
bp = ax.boxplot(data, orientation="horizontal", widths=0.45, patch_artist=True, medianprops={"color": INK},
                boxprops={"facecolor": "#dbe8f8", "edgecolor": MUTED}, whiskerprops={"color": MUTED}, capprops={"color": MUTED},
                flierprops={"markeredgecolor": MUTED, "markersize": 4})
ax.set_yticks([1, 2], [f"qwen2.5 {m.split(':')[1].upper()}" for m in MODELS]); ax.grid(axis="y", visible=False)
ax.axvline(15, color=ORANGE, linewidth=1.5, linestyle="--"); ax.text(16, 2.35, "15 s target (Colab T4)", color=ORANGE, fontsize=8.5)
ax.set_xlabel("seconds per answer (local laptop)")
fig.savefig(FIG / "fig7_3_latency_local.png"); plt.show()
pd.DataFrame({m: S[m]["latency_s"] for m in MODELS})"""),
md("""## 6b · Deployment run: the same 7B system on a free Colab T4
`notebooks/00_Mustajir_demo.ipynb` ran all 70 items on a Colab T4 (Ollama, `qwen2.5:7b`, flash attention), with the
same code, corpus, prompts and decoding settings as the local 7B run. The run files are in
`data/eval/runs/colab_t4/`. The table rescores them here; it is not copied from the Colab printout."""),
code("""T4 = RUNS / "colab_t4"
s_t4 = score("en", "qwen2.5:7b", path=T4 / "e2e_stage2__en__qwen2.5-7b__colab_t4.jsonl")
assert s_t4 == json.loads((T4 / "e2e_score__en__qwen2.5-7b__colab_t4.json").read_text(encoding="utf-8")), "rescore differs from the Colab score"
row = lambda s: {"verified answers": s["verified_rate"], "unsupported shown (validated)": s["validated"]["unsupported_shown"],
                 "unsupported shown (no validator)": s["no_validator"]["unsupported_shown"],
                 "binding provision cited": s["validated"]["primary_cited"], "citation precision": s["validated"]["citation_precision"],
                 "calculators": f"{s['calculators']['correct']}/{s['calculators']['items']}",
                 "adversarial": s["adversarial"]["behaviour_ok"], "latency p50 (s)": s["latency_s"]["p50"], "latency p95 (s)": s["latency_s"]["p95"]}
pd.DataFrame({"7B local (GTX 1650)": row(S["qwen2.5:7b"]), "7B Colab T4": row(s_t4)})"""),
code("""by_id = lambda p: {r["id"]: r for r in map(json.loads, open(p, encoding="utf-8"))}
loc, t4 = by_id(RUNS / "e2e_stage2__en__qwen2.5-7b.jsonl"), by_id(T4 / "e2e_stage2__en__qwen2.5-7b__colab_t4.jsonl")
gen = [i for i in t4 if t4[i].get("answer")]
print(f"same response mode: {sum(loc[i]['mode'] == t4[i]['mode'] for i in t4)}/{len(t4)} items")
print(f"identical answer text: {sum(loc[i].get('answer') == t4[i]['answer'] for i in gen)}/{len(gen)} answered items")
acc = pd.DataFrame(json.loads((T4 / "demo_acceptance__colab_t4.json").read_text(encoding="utf-8")))
print(f"demo acceptance script: {acc['pass'].sum()}/{len(acc)} pass")
acc[["question", "route", "shown", "tool", "seconds", "pass"]]"""),
md("""**Reading.** On the T4 the median response time is under the 15 s target. Temperature 0 and a fixed seed do not make
answers identical across GPUs: different kernels produce different token streams. So the rates move by about 0.05-0.10
between two runs of the same system, which is within sampling noise for 50 questions (standard error about 0.07). The
safety results (no unsupported claim shown, no must-refuse item answered, calculators correct) did not change. The
headline results use the T4 run for latency, and the local runs, made on the same hardware, for the 3B/7B comparison and the
ablations."""),
md("""## 7 · Multilingual parity (retrieval and routing)
The 50 gold questions were machine-translated (NLLB-200) into Arabic and Hindi and back-translated for checking. Three ways
to retrieve for a non-English question: **direct** (embed the Arabic/Hindi question - bge-m3 is multilingual),
**translate-first** (retrieve with the NLLB English translation), and **fused** (RRF, k=20, of both rankings - deployed).
Hindi: 8/50 translations swap landlord and tenant; the *clean* rows exclude them."""),
code("""from tenancy_rag.eval.retrieval_metrics import query_metrics
st1 = {s: {r["id"]: r for r in load(RUNS / f"e2e_stage1__{s}.jsonl")} for s in ("en", "ar_direct", "ar_mt", "hi_direct", "hi_mt")}
def fuse(a, b, k=20):
    sc = {}
    for lst in (a, b):
        for i, p in enumerate(lst, 1):
            sc[p] = sc.get(p, 0) + 1 / (k + i)
    return sorted(sc, key=lambda p: -sc[p])
def metrics(rankings):
    ms = [query_metrics(rk, set(gold[i]["relevant_ids"]), set(gold[i]["primary_ids"])) for i, rk in rankings.items()]
    agg = lambda k: round(sum(m[k] for m in ms if k in m) / max(1, sum(k in m for m in ms)), 3)
    return {"primary@10": agg("primary@10"), "hit@10": agg("hit@10"), "MRR@10": agg("mrr@10")}
ids = [i for i in gold]
par = {"English": metrics({i: st1["en"][i]["ranked_parents"] for i in ids})}
for lang, key in (("Arabic", "ar"), ("Hindi", "hi")):
    d, t = st1[f"{key}_direct"], st1[f"{key}_mt"]
    par[f"{lang} direct"] = metrics({i: d[i]["ranked_parents"] for i in ids})
    par[f"{lang} translate-first"] = metrics({i: t[i]["ranked_parents"] for i in ids})
    par[f"{lang} fused (deployed)"] = metrics({i: fuse(d[i]["ranked_parents"], t[i]["ranked_parents"]) for i in ids})
route_ok = json.loads((RUNS / "parity_retrieval_routing.json").read_text(encoding="utf-8"))["all"]
P = pd.DataFrame(par).T
P["answer-route rate"] = [route_ok["en"]["answer_route_rate"], route_ok["ar_direct"]["answer_route_rate"], route_ok["ar_mt"]["answer_route_rate"],
                          route_ok["ar_mt"]["answer_route_rate"], route_ok["hi_direct"]["answer_route_rate"], route_ok["hi_mt"]["answer_route_rate"],
                          route_ok["hi_mt"]["answer_route_rate"]]
P"""),
code("""order = ["English", "Arabic direct", "Arabic translate-first", "Arabic fused (deployed)", "Hindi direct", "Hindi translate-first", "Hindi fused (deployed)"]
colour = {"English": GREY, "direct": ORANGE, "translate-first": "#e8b23a", "fused (deployed)": BLUE}
cols = [colour["English"] if o == "English" else colour[o.split(" ", 1)[1]] for o in order]
ypos = [7.6, 6, 5, 4, 2.4, 1.4, 0.4]          # gaps separate English / Arabic / Hindi
fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.2), sharey=True)
for ax, m in zip(axes, ["primary@10", "hit@10", "MRR@10"]):
    vals = P.loc[order, m].values
    ax.barh(ypos, vals, color=cols, height=0.8)
    for y, v in zip(ypos, vals):
        ax.text(v + 0.015, y, f"{v:.2f}", va="center", fontsize=8, color=INK)
    ax.set_xlim(0, 1.15); ax.set_xticks([0, 0.5, 1]); ax.grid(axis="y", visible=False); ax.set_title(m, fontsize=10, color=INK)
axes[0].set_yticks(ypos, order)
fig.tight_layout()
fig.savefig(FIG / "fig7_4_multilingual_parity.png"); plt.show()"""),
code("""emb = json.loads((RUNS / "parity_embedding_models.json").read_text(encoding="utf-8"))
pd.DataFrame({m: {lang: v[lang]["primary@10"] for lang in ("en", "ar", "hi")} for m, v in emb.items()}).T.rename(
    columns={"en": "English", "ar": "Arabic", "hi": "Hindi"})   # first-stage dense only, primary@10"""),
md("""**Reading.** An English-only embedding collapses on Arabic and Hindi; bge-m3 does not, but direct Arabic retrieval
still trails English. Fusing the direct and translated rankings closes most of the gap (Arabic hit@10 equals English).
Routing loses 10-14 points in both languages because the transparent rules are English-only and translation adds noise.
**Disclosure:** fusion was chosen on this same parity set (differences of 1-2 questions; best hit@10 in both languages, Arabic MRR 1 point below translate-first).

## 8 · Not yet included
* Human rubric review of ~25 answers (correctness / usefulness) - pending; added before submission.
* Generation in Arabic / Hindi end to end - retrieval and routing parity above; generated-answer parity is future work."""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}, "colab": {"provenance": []}})
out = Path(__file__).with_name("03_end_to_end_eval.ipynb")
nbf.write(nb, out)
print("wrote", out)
