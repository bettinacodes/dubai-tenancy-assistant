"""Command-line entry point: python -m tenancy_rag <command>

  fetch          download / cache all official sources (appends to manifest.jsonl)
  build-corpus   parse cached sources -> parents.jsonl, then chunk -> chunks.jsonl (+ fixed-512 baseline)
  build-index    embed chunks (bge-m3, cached) into persistent Chroma collections
  eval-retrieval run the retrieval configurations over the gold set; writes data/eval/runs/
  ask "question"  answer one question end to end (prints the full response as JSON)
"""
import sys


def main(argv: list[str]) -> None:
    cmd = argv[0] if argv else "help"
    if cmd == "fetch":
        from tenancy_rag.ingest.fetch import fetch_all
        fetch_all()
    elif cmd == "build-corpus":
        from tenancy_rag.ingest.build import build_parents
        from tenancy_rag.preprocess.chunk import build_chunks
        parents = build_parents()
        chunks, fixed = build_chunks()
        print(f"parents: {len(parents)}  chunks: {len(chunks)}  fixed-512 windows: {len(fixed)}")
    elif cmd == "build-index":
        from tenancy_rag.index.bm25_index import load_chunks
        from tenancy_rag.index.vector_store import DenseIndex
        for chunkset, mode in (("struct", "none"), ("struct", "code"), ("fixed512", "none")):
            DenseIndex(load_chunks(chunkset), chunkset=chunkset, prefix_mode=mode)
            print(f"dense index ready: {chunkset} / bge-m3 / prefix={mode}")
    elif cmd == "eval-retrieval":
        from tenancy_rag.eval import run_retrieval
        run_retrieval.main(argv[1:])
    elif cmd == "ask":
        import json
        from tenancy_rag.pipeline import Assistant
        print(json.dumps(Assistant().answer(" ".join(argv[1:])), indent=1, ensure_ascii=False, default=str))
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
