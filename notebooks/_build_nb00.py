"""Generates notebooks/00_Mustajir_demo.ipynb - the main deliverable (reproducible notebook source)."""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
md("""# Musta'jir - a grounded assistant for Dubai residential tenancy law

**Working prototype**

Ask a question about renting a home in Dubai - in English, Arabic or Hindi - and get an answer drawn only from Dubai
tenancy legislation and official guidance, with every provision cited, rent and notice calculations done in code, and
every generated claim checked against the text it cites before it is shown.

**How to run (Google Colab, free tier):**
1. `Runtime -> Change runtime type -> T4 GPU`.
2. *Only while the repository is private:* add a Colab secret named `GITHUB_TOKEN` (key icon in the left bar) holding
   a read-only token for the repository, and allow this notebook to access it.
3. `Runtime -> Run all`. First run: about 10-15 minutes (package install, Ollama + model downloads, index build,
   intent-model training). The app opens at the bottom with a public share link.

No paid API is used: every model is open and runs inside this runtime, so questions never leave it.

| Step | What runs | Model |
|---|---|---|
| Understand | PII redaction, crisis / injection / scope rules, intent (rules + fine-tuned DistilBERT), slot regex | `distilbert-base-multilingual-cased` (fine-tuned here) |
| Translate | Arabic / Hindi question -> English for routing, slots and claim checking | `facebook/nllb-200-distilled-600M` |
| Retrieve | dense search, superseded / flagged filters, cross-encoder rerank, explicit-citation lookup | `BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3` |
| Calculate | rent-increase cap, notice period, RDC filing fee (pure Python, each constant sourced) | - |
| Generate | answer with a citation contract (JSON claims -> sources) | `Qwen2.5-7B-Instruct`, 4-bit (Ollama `qwen2.5:7b`, Q4_K_M) |
| Verify | source exists and is in force, numbers present, NLI entailment; retry once, then quote the law | `cross-encoder/nli-deberta-v3-base` |"""),
md("## 1 · Setup"),
code("""import os, sys, subprocess, time
from pathlib import Path
# Colab preinstalls JAX and TensorFlow; if any library starts them they grab most of the GPU (JAX: 75%). Colab run 2
# had ~11 GB of the T4 held outside PyTorch, so Ollama put 96% of the 7B on the CPU. Set before imports.
os.environ.update(XLA_PYTHON_CLIENT_PREALLOCATE="false", TF_FORCE_GPU_ALLOW_GROWTH="true", USE_TF="0", USE_JAX="0",
                  USE_FLAX="0", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import userdata
    if not Path("dubai-tenancy-assistant").exists():
        try:   # a read-only token is only needed while the repository is private
            auth = userdata.get("GITHUB_TOKEN") + "@"
        except Exception:
            auth = ""
        subprocess.run(["git", "clone", "-q", f"https://{auth}github.com/bettinacodes/dubai-tenancy-assistant.git"], check=True)
    os.chdir("dubai-tenancy-assistant")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", "."], check=True)
    # T4 (16 GB): everything on the GPU. (On the 4 GB development laptop the NLI model and query embedding run on CPU.)
    os.environ.update(TENANCY_NLI_DEVICE="cuda", TENANCY_EMBED_DEVICE="cuda")
    # Generator: Ollama (llama.cpp, 4-bit) - the same backend as the local evaluation runs. transformers 5.18 +
    # bitsandbytes left the 4-bit 7B on the CPU on Colab, so answers never finished.
    import urllib.request
    def ollama_up():
        try:
            return urllib.request.urlopen("http://localhost:11434/api/version", timeout=2).status == 200
        except OSError:
            return False
    if not ollama_up():
        if subprocess.run(["which", "ollama"], capture_output=True).returncode:
            subprocess.run("apt-get -qq install -y zstd pciutils lshw > /dev/null && curl -fsSL https://ollama.com/install.sh | sh",
                           shell=True, check=True)
        subprocess.Popen(["ollama", "serve"], stdout=open("/content/ollama.log", "w"), stderr=subprocess.STDOUT,
                         env={**os.environ, "OLLAMA_KEEP_ALIVE": "-1", "OLLAMA_NUM_PARALLEL": "1",
                              "OLLAMA_FLASH_ATTENTION": "1"})   # keep the model loaded; one slot, smaller KV memory
        for _ in range(60):
            if ollama_up():
                break
            time.sleep(1)
    subprocess.run(["ollama", "pull", "qwen2.5:7b"], check=True, stdout=subprocess.DEVNULL)   # ~4.7 GB
    # Load the 7B onto the GPU now, before any PyTorch model, with the same context size the pipeline requests
    # (a different num_ctx would make Ollama reload it). keep_alive=-1: it stays there for the whole session.
    import json, re, urllib.request
    num_ctx = int(re.search(r"NUM_CTX = ([0-9]+)", Path("src/tenancy_rag/llm.py").read_text()).group(1))
    req = urllib.request.Request("http://localhost:11434/api/generate", headers={"Content-Type": "application/json"},
                                 data=json.dumps({"model": "qwen2.5:7b", "prompt": "", "keep_alive": -1,
                                                  "options": {"num_ctx": num_ctx}}).encode())
    urllib.request.urlopen(req, timeout=600).read()
    print(subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout)   # PROCESSOR should read 100% GPU

else:
    root = Path.cwd()
    while not (root / "pyproject.toml").exists():
        root = root.parent
    os.chdir(root)
    os.environ.setdefault("TENANCY_EMBED_DEVICE", "cpu")
sys.path.insert(0, str(Path.cwd() / "src"))
import json
import torch
import pandas as pd
from tenancy_rag import config
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
      "| corpus as at", config.CORPUS_AS_AT)"""),
md("""## 2 · Build the search index and the intent model
The processed corpus (382 provisions / guidance records, 408 chunks) is in the repository; embeddings and the
fine-tuned intent classifier are rebuilt here (they are too large for git). Both steps are skipped when present."""),
code("""t0 = time.perf_counter()
subprocess.run([sys.executable, "-m", "tenancy_rag", "build-index"], check=True)
print(f"index ready in {time.perf_counter() - t0:.0f} s")
from tenancy_rag.understand.distilbert_clf import OUT
if not (OUT / "selected" / "config.json").exists():
    t0 = time.perf_counter()     # 3 seeds, the one with the best validation F1 is deployed (notebook 04)
    subprocess.run([sys.executable, "-m", "tenancy_rag.understand.distilbert_clf", "train", "--seeds", "0", "1", "2"], check=True)
    print(f"intent model trained in {time.perf_counter() - t0:.0f} s")"""),
md("""## 3 · Load the assistant
On Colab the generator is **Qwen2.5-7B-Instruct in 4-bit** served by Ollama (`qwen2.5:7b`, ~5.5 GB of VRAM). On the
development laptop it is `qwen2.5:3b`, also through Ollama. The first answer also loads the generator, the retrieval and
NLI models (about 1-2 minutes)."""),
code("""from tenancy_rag.pipeline import Assistant
BACKEND = "ollama"
MODEL = config.GEN_MODEL_OLLAMA_7B if IN_COLAB else config.GEN_MODEL_OLLAMA
assistant = Assistant(backend=BACKEND, model=MODEL)
t0 = time.perf_counter()
warm = assistant.answer("How much notice must a landlord give before changing the rent at renewal?")
print(f"{BACKEND} / {MODEL} - first answer (includes model loading) {time.perf_counter() - t0:.0f} s -> {warm['mode']}")
if IN_COLAB:   # the generator must stay 100% on the GPU
    print(subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout)
    print(subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"],
                         capture_output=True, text=True).stdout)"""),
md("""## 4 · Acceptance script
The behaviours the system must show (plan: definition of done). Each row is checked automatically; the same
questions are the examples in the app."""),
code("""from tenancy_rag.ui import EXAMPLES
EXPECT = [   # (expected routes, what must also be true)
    ({"calculation", "answer"}, lambda r: (r["tool"] or {}).get("tool") == "rent_increase_cap"),
    ({"answer", "calculation"}, lambda r: any("25" in c["locator"] for c in r["citations"]) or (r["tool"] or {}).get("tool") == "notice_period_check"),
    ({"out_of_scope"}, lambda r: not r["citations"]),
    ({"answer"}, lambda r: r["mode"] in ("generated", "generated_after_retry", "historical_diff", "extractive_fallback")),
    ({"answer"}, lambda r: any("43" in c["instrument"] for c in r["citations"]) or r["mode"] == "extractive_fallback"),
    ({"refuse_injection", "answer"}, lambda r: r["route"] == "refuse_injection" or r["flags"].get("injection")),
    ({"answer", "calculation"}, lambda r: r["language"] == "Arabic"),
    ({"crisis"}, lambda r: r["mode"] == "crisis_card" and r["language"] == "Hindi"),
    ({"crisis"}, lambda r: r["mode"] == "crisis_card"),
]
rows, responses = [], []
for q, (routes, check) in zip(EXAMPLES, EXPECT):
    r = assistant.answer(q); responses.append(r)
    rows.append({"question": q[:70], "route": r["route"], "shown": r["mode"], "cited": ", ".join(c["locator"] for c in r["citations"][:3]),
                 "tool": (r["tool"] or {}).get("tool"), "seconds": r["seconds"], "pass": r["route"] in routes and bool(check(r))})
acc = pd.DataFrame(rows)
print(f"acceptance: {acc['pass'].sum()}/{len(acc)} pass | median {acc.seconds.median():.1f} s per answer")
runtime = "colab_t4" if IN_COLAB else "local"
(config.EVAL / "runs" / f"demo_acceptance__{runtime}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
acc"""),
code("""from IPython.display import HTML, display
from tenancy_rag import ui
display(HTML(f"<style>{ui.CSS}</style>" + ui.answer_html(responses[0]) + "<br>" + ui.citations_html(responses[0]) + "<br>" + ui.tool_html(responses[0])))"""),
md("""## 5 · Latency benchmark on the T4 (all 70 evaluation items)
Runs the same staged evaluation as notebook 03 with the 7B model on this GPU, to measure the latency target
(p50 under ~15 s). About 20-30 minutes on a T4; skipped when not on Colab. Results go to
`data/eval/runs/e2e_stage2__en__qwen2.5-7b.jsonl` (in the Colab runtime) and are scored like the local runs."""),
code("""RUN_BENCHMARK = IN_COLAB
if RUN_BENCHMARK:
    from tenancy_rag.eval.run_e2e import stage1, stage2
    from tenancy_rag.eval.score_e2e import score
    t0 = time.perf_counter()
    stage1("en")
    stage2("en", MODEL, backend=BACKEND)          # the 7B stays loaded in Ollama (keep-alive)
    s = score("en", MODEL)
    (config.EVAL / "runs" / "e2e_score__en__qwen2.5-7b__colab_t4.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    print(f"benchmark finished in {(time.perf_counter() - t0) / 60:.0f} min")
    print(json.dumps({k: s[k] for k in ("verified_rate", "validated", "no_validator", "calculators", "adversarial", "latency_s")}, indent=1))"""),
md("""## 6 · The app
On Colab, `share=True` prints a public link (valid 72 h) for the live demo; locally the app opens inline."""),
code("""app = ui.build_app(assistant)
ui.launch(app, share=IN_COLAB, inline=True, height=1100, prevent_thread_lock=not IN_COLAB)"""),
md("""**Limits to keep in mind:** the corpus is a snapshot (see the "as at" date in every answer); English
texts are translations and the Arabic text prevails; answers are general information, not legal advice; commercial
leases, other emirates and DIFC are out of scope or caveated."""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"},
                                                "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}})
out = Path(__file__).with_name("00_Mustajir_demo.ipynb")
nbf.write(nb, out)
print("wrote", out)
