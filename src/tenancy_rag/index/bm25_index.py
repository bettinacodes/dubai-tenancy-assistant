"""Sparse retrieval: BM25 (bm25s) over chunks, with a legal-aware tokeniser.

The dense path keeps text untouched; BM25 gets its own tokeniser that is aggressive where it
helps lexical matching (lower-casing, stopwords, Snowball stemming) but never splits the tokens
that carry legal meaning:
  instrument refs  "26/2007", "No. (26) of 2007" -> "26/2007"
  percentages      "10%", "ten percent (10%)"    -> "10%"
  clause refs      "25(1)(a)"                    -> "25(1)(a)" and "25"
  money            "AED 20,000"                  -> "aed", "20000"
Negations ("not", "no") are kept: "may not" vs "may" matters in law.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

import bm25s

from tenancy_rag import config

STOPWORDS = set("""a an the and or of to in on at by for with from as is are was were be been being this that these
those it its into than then there their them they he she his her him we you your i me my our us do does did done
has have had having will would shall should can could may might must which who whom whose what when where why how
such any all each other same so also only own very just upon thereof therein hereby""".split()) - {"may", "must", "shall"}
# legal modal verbs are kept above: "may"/"must"/"shall" distinguish permission from obligation
NEGATIONS = {"not", "no", "nor", "without", "never"}
# Domain acronyms are never stemmed (Snowball turns "aed" into "a")
PROTECTED = {"aed", "rera", "rdc", "dld", "ejari", "dewa", "difc", "noc", "poa", "vat", "rest"}

_INSTR_LONG = re.compile(r"no\.?\s*\(?\s*(\d{1,3})\s*\)?\s*of\s*(\d{4})", re.I)
_TOKEN = re.compile(r"\d{1,3}/\d{4}|\d+(?:\.\d+)?%|\d{1,3}(?:\(\d{1,2}\))+(?:\([a-z]\))?|\d+(?:,\d{3})*(?:\.\d+)?|[a-z]+(?:'[a-z]+)?", re.I)


@lru_cache(maxsize=1)
def _stemmer():
    import Stemmer  # PyStemmer (Snowball)
    return Stemmer.Stemmer("english")


def legal_tokenize(text: str, stem: bool = True) -> list[str]:
    text = _INSTR_LONG.sub(lambda m: f" {m.group(1)}/{m.group(2)} ", text)
    out = []
    for tok in _TOKEN.findall(text.lower()):
        if tok[0].isdigit():
            if "(" in tok:                       # 25(1)(a) -> keep whole + article number
                out += [tok, tok.split("(")[0]]
            else:
                out.append(tok.replace(",", ""))
        elif tok in NEGATIONS or tok not in STOPWORDS:
            out.append(_stemmer().stemWord(tok) if stem and tok not in NEGATIONS | PROTECTED else tok)
    return out


class BM25Index:
    def __init__(self, chunks: list[dict], prefix_mode: str = "code", stem: bool = True, k1: float = 1.5, b: float = 0.75):
        from tenancy_rag.index.embed import chunk_text
        self.chunks, self.prefix_mode, self.stem = chunks, prefix_mode, stem
        docs = [legal_tokenize(chunk_text(c, prefix_mode), stem) for c in chunks]
        self.model = bm25s.BM25(k1=k1, b=b)
        self.model.index(docs, show_progress=False)

    def search(self, query: str, k: int = 30) -> list[dict]:
        q = legal_tokenize(query, self.stem)
        if not q:
            return []
        k = min(k, len(self.chunks))
        idx, scores = self.model.retrieve([q], k=k, show_progress=False)
        return [{**self.chunks[i], "score": float(s), "retriever": "bm25"}
                for i, s in zip(idx[0], scores[0]) if s > 0]


def load_chunks(name: str = "struct") -> list[dict]:
    path = {"struct": config.CHUNKS, "fixed512": config.PROCESSED / "chunks_fixed512.jsonl"}[name]
    return [json.loads(l) for l in path.open(encoding="utf-8")]
