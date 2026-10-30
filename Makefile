# Linux / Colab convenience targets. On Windows run the python commands directly.
PY ?= python

.PHONY: fetch corpus index eval eval-retrieval eval-understanding eval-e2e notebooks test

fetch:        ## re-download official sources (cached copies are already in data/raw)
	$(PY) -m tenancy_rag fetch

corpus:       ## parse + chunk the cached sources
	$(PY) -m tenancy_rag build-corpus

test:
	$(PY) -m pytest -q

index:        ## embed chunks into Chroma (GPU if available)
	$(PY) -m tenancy_rag build-index

eval-retrieval:
	$(PY) -m tenancy_rag eval-retrieval

eval: eval-retrieval eval-understanding eval-e2e   ## every evaluation

eval-understanding:   ## intent + routing (rules, DistilBERT) on the intent test split and the 70 items
	$(PY) -m tenancy_rag.eval.eval_understanding rules
	$(PY) -m tenancy_rag.eval.eval_understanding distilbert
	$(PY) -m tenancy_rag.eval.compare_intent_models

eval-e2e:             ## staged end-to-end run (stage 1: route/tools/retrieval; stage 2: generate + validate); needs Ollama
	$(PY) -m tenancy_rag.eval.run_e2e 1 en
	$(PY) -m tenancy_rag.eval.run_e2e 2 en qwen2.5:3b
	$(PY) -m tenancy_rag.eval.run_e2e 2 en qwen2.5:7b
	for s in ar_direct ar_mt hi_direct hi_mt; do $(PY) -m tenancy_rag.eval.run_e2e 1 $$s; done

notebooks:            ## re-execute the analysis notebooks from the saved runs
	for n in 01_corpus_and_preprocessing 02_retrieval_eval 03_end_to_end_eval 04_intent_finetune; do 		jupyter nbconvert --to notebook --execute --inplace notebooks/$$n.ipynb; done

