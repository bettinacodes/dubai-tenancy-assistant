"""Conservative text normalisation.

No lowercasing, stemming or stopword removal: numerals, AED amounts, percentages, dates
and cross-references ("Article (25) Clause (1)") carry the legal meaning.
"""
import re
import unicodedata

_INVISIBLE = re.compile(r"[­​‌‍⁠﻿]")  # soft hyphen, zero-width chars
_LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}
_SPACES = re.compile(r"[ \t  -   　]+")
_HYPHEN_WRAP = re.compile(r"(\w)-\n(\w)")  # "obliga-\ntion" -> "obligation" (PDF line wraps)
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})
_WORD = re.compile(r"[A-Za-z]+")


def normalise(text: str) -> str:
    for lig, rep in _LIGATURES.items():
        text = text.replace(lig, rep)
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE.sub("", text)
    text = text.translate(_QUOTES)
    text = _HYPHEN_WRAP.sub(r"\1\2", text)
    text = _SPACES.sub(" ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # "Article(5)" / "Article ( 5 )" -> "Article (5)" so cross-references are uniform
    text = re.sub(r"Article\s*\(\s*(\d+)\s*\)", r"Article (\1)", text)
    return text.strip()


def build_vocab(texts) -> set[str]:
    """Lower-cased word vocabulary from clean (HTML-sourced) corpus text."""
    vocab: set[str] = set()
    for t in texts:
        vocab.update(w.lower() for w in _WORD.findall(t))
    return vocab


def repair_dropped_ligatures(text: str, vocab: set[str]) -> tuple[str, dict[str, str]]:
    """Some PDFs (e.g. the DLD Tenancy Guide) lose the second letter of an 'fi'/'fl' ligature
    on extraction: "first" -> "frst", "affixed" -> "affxed", "Official" -> "Offcial".
    For each word not in the vocabulary, try inserting 'i' or 'l' after each 'f' and accept a
    candidate that is (1) a word of the clean corpus, or else (2) a common English word
    (wordfreq Zipf >= 3) while the broken form is rare (Zipf < 2), which guards against
    "repairing" real words such as "fled". Returns (text, {broken: repaired})."""
    from wordfreq import zipf_frequency

    repairs: dict[str, str] = {}

    def fix(m: re.Match) -> str:
        w = m.group(0)
        if w.lower() in vocab or "f" not in w.lower():
            return w
        cands = [w[: i + 1] + letter + w[i + 1:] for i, ch in enumerate(w) if ch.lower() == "f" for letter in ("i", "l")]
        for cand in cands:                      # (1) evidence from the clean corpus
            if cand.lower() in vocab:
                repairs[w] = cand
                return cand
        if zipf_frequency(w, "en") < 2.0:       # (2) general English frequency, rare broken form only
            best = max(cands, key=lambda c: zipf_frequency(c, "en"), default=None)
            if best and zipf_frequency(best, "en") >= 3.0:
                repairs[w] = best
                return best
        return w

    return _WORD.sub(fix, text), repairs
