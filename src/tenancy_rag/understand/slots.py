"""Slot extraction for the deterministic tools (regex; no LLM needed for numbers and dates).

Slots: current_rent, market_avg, shortfall_pct, annual_rent, dates (notice_date, expiry_date),
served_via (notice method), eviction_reason, notice_period (relative: "vacate in 12 months"). Amounts accept "AED 80,000", "80k", "80000",
"1.2 lakh". Assignment of two amounts to rent vs average uses nearby cue words, falling back
to order of mention (first = current rent). Two dates are assigned by cue words ("contract ends on"
-> expiry), falling back to chronological order (notice precedes expiry). Every slot keeps the text span it came from so the
UI can show the working.
"""
from __future__ import annotations

import re
from datetime import date, timedelta

MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_AMOUNT = re.compile(r"(?:aed|dhs?|dirhams?)?\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(k|lakh|lac|thousand)?\b", re.I)
_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*(%|percent|per cent)", re.I)
_DATE_DMY = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(jan\w*|feb\w*|mar\w*|apr\w*|may|jun\w*|jul\w*|aug\w*|sep\w*|oct\w*|nov\w*|dec\w*)\s*,?\s*(\d{4})?\b", re.I)
_DATE_ISO = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")
CURRENT_CUES = r"(i pay|pay|my rent|current(ly)?|rent is|paying|annual rent|rent of)"
EXPIRY_CUES = r"\b(ends?|ending|expir\w*|renew\w*|until|end of (the |my )?(lease|contract|tenancy)|vacate by|move out by|leave by)\b"
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "six": 6, "nine": 9, "twelve": 12, "a": 1}
_REL_PERIOD = re.compile(r"\b(?:in|within|after|giving me|gave me|only)\s+(\d{1,3}|one|two|three|six|nine|twelve|a)\s+(months?|weeks?|days?)\b"
                         r"|\b(\d{1,3})[- ](months?|weeks?|days?)'?s?'? notice\b", re.I)
MARKET_CUES = r"(average|index|market|similar|comparable|go for|going rate|area|other flats|like mine)"
METHODS = {"whatsapp": "whatsapp", "sms": "sms", "text message": "sms", "e-?mail": "email", "by hand": "by_hand",
           "verbal|told me|said|phone call|called me": "verbal", "notary": "notary public", "registered (mail|post)": "registered mail",
           "courier": "courier"}
REASONS = {r"\b(sell|selling|sale|sold)\b": "sale",
           r"\b(own use|himself|herself|his (son|daughter|wife|family|parents?|brother|sister)|her (son|daughter|husband|family)|relative|moving in)\b": "owner_or_relative_use",
           r"\b(demolish\w*|rebuild\w*|reconstruct\w*)\b": "demolish_or_rebuild",
           r"\b(renovat\w*|restor\w*|major (repair|maintenance))\b": "major_restoration"}


def _to_number(num: str, unit: str | None) -> float:
    v = float(num.replace(",", ""))
    unit = (unit or "").lower()
    return v * 1000 if unit in ("k", "thousand") else v * 100_000 if unit in ("lakh", "lac") else v


def amounts(text: str) -> list[dict]:
    """Money-like amounts (>= 1,000 AED) with their position; percentages and years excluded."""
    out = []
    pct_spans = [m.span() for m in _PCT.finditer(text)]
    date_spans = [m.span() for m in _DATE_DMY.finditer(text)] + [m.span() for m in _DATE_ISO.finditer(text)]
    for m in _AMOUNT.finditer(text):
        if any(a <= m.start(1) < b for a, b in pct_spans + date_spans):
            continue
        v = _to_number(m.group(1), m.group(2))
        if 1000 <= v <= 50_000_000 and not (1990 <= v <= 2100 and not m.group(2)):
            out.append({"value": v, "span": m.group(0).strip(), "pos": m.start()})
    return out


def dates(text: str, today: date | None = None) -> list[dict]:
    today = today or date.today()
    out = []
    for m in _DATE_DMY.finditer(text):
        try:
            out.append({"date": date(int(m.group(3) or today.year), MONTHS[m.group(2)[:3].lower()], int(m.group(1))).isoformat(),
                        "span": m.group(0), "pos": m.start()})
        except ValueError:
            pass
    for m in _DATE_ISO.finditer(text):
        out.append({"date": f"{m.group(1)}-{m.group(2)}-{m.group(3)}", "span": m.group(0), "pos": m.start()})
    return sorted(out, key=lambda d: d["pos"])


def _plus_months(d: date, months: int) -> date:
    y, m = divmod(d.month - 1 + months, 12)
    for day in (d.day, 30, 29, 28):
        try:
            return date(d.year + y, m + 1, day)
        except ValueError:
            continue
    raise AssertionError


def relative_period(text: str) -> dict | None:
    """'vacate in 12 months', '30 days' notice' -> {"months"|"days": n, "span"}."""
    m = _REL_PERIOD.search(text)
    if not m:
        return None
    num, unit = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
    n = int(num) if num.isdigit() else NUMBER_WORDS[num.lower()]
    unit = unit.lower()
    if unit.startswith("month"):
        return {"months": n, "span": m.group(0)}
    return {"days": n * 7 if unit.startswith("week") else n, "span": m.group(0)}


def _cue_near(text: str, pos: int, cues: str, window: int = 45) -> bool:
    return bool(re.search(cues, text[max(0, pos - window): pos + window], re.I))


def extract(text: str, today: date | None = None) -> dict:
    s: dict = {}
    am = amounts(text)
    cur = [a for a in am if _cue_near(text, a["pos"], CURRENT_CUES) and not _cue_near(text, a["pos"], MARKET_CUES, 25)]
    mkt = [a for a in am if _cue_near(text, a["pos"], MARKET_CUES)]
    if len(am) >= 2:
        current = cur[0] if cur else am[0]
        market = next((a for a in mkt if a is not current), next(a for a in am if a is not current))
        s["current_rent"], s["market_avg"] = current, market
    elif len(am) == 1:
        s["annual_rent"] = am[0]
    # each percentage is attached to its nearest cue word ("25% below" vs "a 10% increase")
    below = r"\b(below|under|less than|cheaper|lower)\b"
    rise = r"\b(increase|hike|raise|more|bump|rise|up)\b"
    for m in _PCT.finditer(text):
        val = float(m.group(1))
        after, before = text[m.end(): m.end() + 30], text[max(0, m.start() - 25): m.start()]
        if re.search(below, after, re.I) and "shortfall_pct" not in s:
            s["shortfall_pct"] = val
        elif (re.search(rise, after, re.I) or re.search(rise, before, re.I)) and "requested_increase_pct" not in s:
            s["requested_increase_pct"] = val
    ds = dates(text, today)
    if len(ds) >= 2:
        first, last = ds[0], ds[-1]
        cued = [d for d in (first, last) if re.search(EXPIRY_CUES, text[max(0, d["pos"] - 30): d["pos"]], re.I)]
        if len(cued) == 1:      # "my contract ends on X ... told me on Y" -> X is the expiry
            expiry = cued[0]
            notice = last if expiry is first else first
        else:                   # notice is served before the expiry/eviction date
            notice, expiry = sorted((first, last), key=lambda d: d["date"])
        s["notice_date"], s["expiry_date"] = notice, expiry
    elif len(ds) == 1:
        s["date"] = ds[0]
    rel = relative_period(text)
    if rel and "expiry_date" not in s:
        # "must vacate in 12 months": the notice date is the date given, else today (assumption shown to the user)
        nd = date.fromisoformat(s["date"]["date"]) if "date" in s else (today or date.today())
        ed = _plus_months(nd, rel["months"]) if "months" in rel else nd + timedelta(days=rel["days"])
        s["notice_period"] = rel
        s["notice_date"] = s.pop("date", None) or {"date": nd.isoformat(), "span": "today (assumed)", "pos": -1, "assumed": True}
        s["expiry_date"] = {"date": ed.isoformat(), "span": rel["span"], "pos": -1, "derived": True}
    for pat, method in METHODS.items():
        if re.search(rf"\b(?:{pat})\b", text, re.I):
            s["served_via"] = method
            break
    for pat, reason in REASONS.items():
        if re.search(pat, text, re.I):
            s["eviction_reason"] = reason
            break
    return s
