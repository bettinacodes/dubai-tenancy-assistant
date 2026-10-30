"""Deterministic legal calculators. Numbers come from code, never from the language model.

Every function returns a ToolResult with: the inputs it used, the rule applied (quoted
paraphrase), each step of the working, the result, any caveats, and the corpus parent ids of
its legal sources, so the UI can "show the working" and the citation validator can check them.
Constants live in tools/constants.py, each verified against the cached primary text.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, timedelta

from tenancy_rag.tools import constants as K


@dataclass
class ToolResult:
    tool: str
    inputs: dict
    result: dict
    working: list[str]
    rule: str
    sources: list[dict]
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _aed(x: float) -> str:
    return f"AED {x:,.0f}"


# --------------------------------------------------------------------------- rent increase
def band_for_shortfall(shortfall_pct: float) -> tuple[float, str]:
    for (upper, pct), clause in zip(K.RENT_INCREASE_BANDS, "abcde"):
        if shortfall_pct <= upper:
            return pct, clause
    raise AssertionError("unreachable")


def rent_increase_cap(current_rent: float | None = None, market_avg: float | None = None,
                      shortfall_pct: float | None = None, requested_increase_pct: float | None = None) -> ToolResult:
    """Decree 43/2013 Art. 1: maximum renewal increase from the rent's shortfall below the average
    rental value of similar units (Art. 3: per the official Rent Index - the Smart Rental Index
    since 2 January 2025). Give either (current_rent, market_avg) or shortfall_pct."""
    working, caveats = [], []
    if current_rent is not None and market_avg is not None:
        if current_rent <= 0 or market_avg <= 0:
            raise ValueError("rents must be positive")
        shortfall_pct = (market_avg - current_rent) / market_avg * 100
        working.append(f"Shortfall = ({_aed(market_avg)} - {_aed(current_rent)}) / {_aed(market_avg)} = {shortfall_pct:.1f}% below the average")
    elif shortfall_pct is None:
        raise ValueError("need (current_rent and market_avg) or shortfall_pct")
    else:
        working.append(f"Shortfall given: {shortfall_pct:.1f}% below the average")
    if shortfall_pct <= 0:
        caveats.append("Your rent is at or above the average for similar units, so no increase is permitted under Art. 1(a).")
    pct, clause = band_for_shortfall(max(shortfall_pct, 0.0))
    working.append(f"Decree 43/2013 Art. 1({clause}) applies -> maximum increase {pct:g}%")
    res = {"shortfall_pct": round(shortfall_pct, 2), "clause": f"Art. 1({clause})", "max_increase_pct": pct}
    if current_rent is not None:
        inc = round(current_rent * pct / 100)
        res.update(max_increase_aed=inc, max_new_rent_aed=round(current_rent + inc))
        working.append(f"Maximum increase = {pct:g}% x {_aed(current_rent)} = {_aed(inc)} (new rent at most {_aed(current_rent + inc)})")
    if requested_increase_pct is not None:
        res["requested_increase_pct"] = requested_increase_pct
        res["requested_within_cap"] = requested_increase_pct <= pct
        working.append(f"Requested {requested_increase_pct:g}% {'is within' if requested_increase_pct <= pct else 'EXCEEDS'} the {pct:g}% cap")
    if 10 < shortfall_pct < 11 or 20 < shortfall_pct < 21 or 30 < shortfall_pct < 31:
        caveats.append("The decree's bands leave a gap (e.g. 10% to 11%); this tool treats the boundaries as continuous.")
    caveats.append("The 'average rental value of similar units' is the official Rent Index figure for your unit (DLD Smart Rental Index since 2 January 2025), not a figure you estimate.")
    caveats.append("Any change must also be notified at least 90 days before the contract expires (Law 26/2007 Art. 14, as amended), unless agreed otherwise.")
    return ToolResult("rent_increase_cap", {"current_rent": current_rent, "market_avg": market_avg,
                      "shortfall_pct": shortfall_pct if current_rent is None else None,
                      "requested_increase_pct": requested_increase_pct}, res, working,
                      "Decree 43/2013 Art. 1: no increase up to 10% below average; 5% for 11-20%; 10% for 21-30%; 15% for 31-40%; 20% for more than 40% below.",
                      [K.RENT_INCREASE_SOURCE, K.AVERAGE_RENT_SOURCE, K.AMEND_TERMS_SOURCE], caveats)


# --------------------------------------------------------------------------- notices
def _minus_months(d: date, months: int) -> date:
    y, m = divmod(d.month - 1 - months, 12)
    year, month = d.year + y, m + 1
    for day in (d.day, 30, 29, 28):
        try:
            return date(year, month, day)
        except ValueError:
            continue
    raise AssertionError


EXPIRY_GROUNDS = set(K.GROUNDS_ON_EXPIRY)
DURING_TERM_GROUNDS = set(K.GROUNDS_DURING_TERM)


def notice_period_check(reason: str, notice_date: str, expiry_date: str | None = None,
                        served_via: str | None = None) -> ToolResult:
    """reason: "amend_terms" (rent change / new terms, Art. 14), an Art. 25(2) ground
    (sale, owner_or_relative_use, demolish_or_rebuild, major_restoration: notice >= 12 months before
    the eviction date) or an Art. 25(1) ground with a 30-day cure period (rent_default, breach_of_obligation).
    `expiry_date` is the lease expiry / requested eviction date."""
    nd = date.fromisoformat(notice_date)
    working, caveats, sources = [], [], []
    result: dict = {"reason": reason}
    if reason == "amend_terms":
        ed = date.fromisoformat(expiry_date)
        days = (ed - nd).days
        latest = ed - timedelta(days=K.AMEND_TERMS_NOTICE_DAYS)
        period_ok = days >= K.AMEND_TERMS_NOTICE_DAYS
        working += [f"Notice given {nd.isoformat()}, contract expires {ed.isoformat()}: {days} days",
                    f"Art. 14 requires at least {K.AMEND_TERMS_NOTICE_DAYS} days -> latest valid notice date {latest.isoformat()}",
                    f"{days} days {'meets' if period_ok else 'does NOT meet'} the {K.AMEND_TERMS_NOTICE_DAYS}-day requirement"]
        result.update(days_given=days, days_required=K.AMEND_TERMS_NOTICE_DAYS, latest_valid_notice=latest.isoformat(), period_ok=period_ok)
        caveats.append("Art. 14 applies 'unless otherwise agreed by the parties' - check your contract for a different notice clause.")
        # Art. 14 does not prescribe a method; Art. 2 (am.) defines Notice.
        service_ok = None if served_via is None else (True if served_via in K.NOTICE_DEFINITION_METHODS else None)
        if served_via and service_ok is None:
            caveats.append(f"Art. 14 does not prescribe how the notice is sent. Art. 2 defines a Notice as written and sent by notary public, "
                           f"registered mail, by hand, or other technological means approved by law; whether '{served_via}' qualifies is not settled by the corpus.")
        rule = "Law 26/2007 Art. 14 (am.): either party wishing to change the terms must notify the other at least 90 days before expiry, unless otherwise agreed."
        sources += [K.AMEND_TERMS_SOURCE, K.NOTICE_DEFINITION_SOURCE]
    elif reason in EXPIRY_GROUNDS:
        ed = date.fromisoformat(expiry_date)
        earliest_eviction = None
        latest = _minus_months(ed, K.EVICTION_ON_EXPIRY_NOTICE_MONTHS)
        period_ok = nd <= latest
        working += [f"Ground: {reason} = Art. {K.GROUNDS_ON_EXPIRY[reason]} (only available when the lease expires)",
                    f"Requested eviction date {ed.isoformat()} -> notice must be served on or before {latest.isoformat()} (12 months earlier)",
                    f"Notice served {nd.isoformat()}: {'in time' if period_ok else 'TOO LATE'}"]
        service_ok = None if served_via is None else served_via in K.EVICTION_NOTICE_SERVICE
        if served_via is not None:
            working.append(f"Served via {served_via}: Art. 25 requires a notary public or registered mail -> {'OK' if service_ok else 'NOT valid service'}")
        result.update(latest_valid_notice=latest.isoformat(), period_ok=period_ok, ground=f"Art. {K.GROUNDS_ON_EXPIRY[reason]}")
        if reason == "owner_or_relative_use":
            caveats.append("The owner must also prove they own no other suitable property (Art. 25(2)(c)); after eviction the unit cannot be re-let to a third party for 2 years if residential (Art. 26).")
        if reason == "demolish_or_rebuild":
            caveats.append("The owner needs the required permits (Art. 25(2)(a)); the tenant has a right of first refusal to return, exercised within 30 days of notification (Art. 29).")
        if reason == "major_restoration":
            caveats.append("The condition must be verified by a technical report issued or certified by Dubai Municipality (Art. 25(2)(b)).")
        rule = "Law 26/2007 Art. 25(2) (am.): on expiry, eviction for these grounds needs at least 12 months' notice of the reasons, served through a notary public or by registered mail."
        sources.append(K.EVICTION_SOURCE)
    elif reason in DURING_TERM_GROUNDS:
        cure_until = nd + timedelta(days=K.EVICTION_DURING_TERM_CURE_DAYS)
        period_ok = True
        working += [f"Ground: {reason} = Art. {K.GROUNDS_DURING_TERM[reason]}",
                    f"Notice served {nd.isoformat()} -> the tenant has until {cure_until.isoformat()} ({K.EVICTION_DURING_TERM_CURE_DAYS} days) to pay or comply"
                    if reason in ("rent_default", "breach_of_obligation") else "This ground does not carry a 30-day cure period in Art. 25(1)."]
        service_ok = None if served_via is None else served_via in K.EVICTION_NOTICE_SERVICE
        if served_via is not None:
            working.append(f"Served via {served_via}: Art. 25 requires a notary public or registered mail -> {'OK' if service_ok else 'NOT valid service'}")
        result.update(cure_deadline=cure_until.isoformat() if reason in ("rent_default", "breach_of_obligation") else None,
                      ground=f"Art. {K.GROUNDS_DURING_TERM[reason]}")
        if reason == "rent_default":
            caveats.append("Art. 25(1)(a) applies 'unless otherwise agreed by the parties'. Eviction itself needs a ruling from the Rental Disputes Center.")
        rule = "Law 26/2007 Art. 25(1) (am.): during the term, eviction only on the listed grounds; for rent default or breach, after a notice (notary public or registered mail) and 30 days to remedy."
        sources.append(K.EVICTION_SOURCE)
    else:
        raise ValueError(f"unknown reason {reason!r}")
    result["service_ok"] = service_ok
    result["valid"] = bool(period_ok) and service_ok is not False
    return ToolResult("notice_period_check", {"reason": reason, "notice_date": notice_date, "expiry_date": expiry_date,
                      "served_via": served_via}, result, working, rule, sources, caveats)


# --------------------------------------------------------------------------- RDC fees
def rdc_fee_estimate(annual_rent: float | None = None, claim_amount: float | None = None,
                     case_type: str = "rent_case", poa: bool = False, trustee_centre: bool = False) -> ToolResult:
    """RDC filing fee per the RDC's official fee schedule (service guidance; legal basis D26/2013 Art. 23).
    case_type "rent_case" (eviction, renewal, rent claim, termination, return): 3.5% of annual rent, min 500, max 20,000;
    "monetary": 3.5% of the claimed amount, min 500, max 15,000."""
    if case_type == "rent_case":
        if not annual_rent or annual_rent <= 0:
            raise ValueError("annual_rent required for a rent case")
        base, lo, hi, what = annual_rent, K.RDC_RENT_CASE_MIN, K.RDC_RENT_CASE_MAX, "annual rent"
    elif case_type == "monetary":
        if not claim_amount or claim_amount <= 0:
            raise ValueError("claim_amount required for a monetary claim")
        base, lo, hi, what = claim_amount, K.RDC_MONETARY_CLAIM_MIN, K.RDC_MONETARY_CLAIM_MAX, "claimed amount"
    else:
        raise ValueError(case_type)
    raw = base * K.RDC_FEE_RATE
    fee = min(max(raw, lo), hi)
    working = [f"3.5% x {what} {_aed(base)} = {_aed(raw)}"]
    if raw < lo:
        working.append(f"Below the minimum -> filing fee {_aed(lo)}")
    elif raw > hi:
        working.append(f"Above the maximum -> filing fee capped at {_aed(hi)}")
    fixed = dict(K.RDC_FIXED_FEES)
    if poa:
        fixed["power_of_attorney"] = K.RDC_POA_FEE
    if trustee_centre:
        fixed["trustee_centre_partner_fee_excl_vat"] = K.RDC_TRUSTEE_CENTRE_FEE
    working.append("Fixed charges: " + ", ".join(f"{k.replace('_', ' ')} AED {v}" for k, v in fixed.items()))
    fee = round(fee + 1e-9, 2)  # money to the fils; +1e-9 guards float artefacts (500.4999...)
    total = round(fee + sum(fixed.values()), 2)
    working.append(f"Estimated total {_aed(total)}" + (" + VAT on the trustee-centre fee" if trustee_centre else ""))
    caveats = ["Figures from the Rental Disputes Center's official fee schedule (service guidance); the legal basis (D26/2013 Art. 23) "
               "delegates fees to an Executive Council resolution. Confirm the amount when filing.",
               f"If the dispute is settled through conciliation, half of the fee for basic claims is refunded (about {_aed(fee * K.RDC_SETTLEMENT_REFUND_SHARE)})."]
    return ToolResult("rdc_fee_estimate", {"annual_rent": annual_rent, "claim_amount": claim_amount, "case_type": case_type,
                      "poa": poa, "trustee_centre": trustee_centre},
                      {"filing_fee": fee, "fixed_fees": sum(fixed.values()), "fixed_breakdown": fixed, "estimated_total": total},
                      working, "RDC fee schedule: 3.5% of annual rent (rent cases, min AED 500, max AED 20,000) or of the claimed amount (monetary claims, min AED 500, max AED 15,000), plus fixed charges.",
                      [K.RDC_FEE_SOURCE, K.RDC_FEE_LEGAL_BASIS], caveats)
