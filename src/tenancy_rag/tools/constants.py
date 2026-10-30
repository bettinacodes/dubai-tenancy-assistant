"""Legally operative numbers used by the deterministic tools.

Rule: no number ships without a primary-source citation. Every constant below was checked
on 2026-10-09 against the cached copy of the cited instrument (see data/raw/manifest.jsonl).
Figures that could only be found in secondary sources are listed in UNVERIFIED and are NOT
used by any tool.
"""

DLP = "https://dlp.dubai.gov.ae/Legislation%20Reference"
D43_URL = f"{DLP}/2013/Decree%20No.%20(43)%20of%202013%20Determining%20Rent%20Increase%20for%20Real%20Property.html"
L33_URL = f"{DLP}/2009/Law%20No.%20(33)%20of%202008%20Amending%20Law%20No.%20(26)%20of%202007.html"
L26_URL = f"{DLP}/2007/Law%20No.%20(26)%20of%202007.html"

# Decree 43/2013 Art. 1: maximum renewal increase by how far the current rent is BELOW the
# average rental value of similar units (Art. 3: average per the official Rent Index; since
# 2 Jan 2025 that is DLD's Smart Rental Index).
# (upper bound of shortfall % inclusive, max increase %). The decree's text leaves gaps between
# bands (e.g. 10% vs 11%); we treat boundaries as continuous: a shortfall of 10.5% falls in
# the 5% band. This interpretation is stated wherever the tool's output is shown.
RENT_INCREASE_BANDS = [
    (10.0, 0.0),           # Art. 1(a): up to 10% below average -> no increase
    (20.0, 5.0),           # Art. 1(b): 11%-20% below -> up to 5%
    (30.0, 10.0),          # Art. 1(c): 21%-30% below -> up to 10%
    (40.0, 15.0),          # Art. 1(d): 31%-40% below -> up to 15%
    (float("inf"), 20.0),  # Art. 1(e): more than 40% below -> up to 20%
]
RENT_INCREASE_SOURCE = {"instrument": "Decree No. 43 of 2013", "article": "1", "url": D43_URL,
                        "parent_id": "dubai-decree-43-2013__art-1"}
AVERAGE_RENT_SOURCE = {"instrument": "Decree No. 43 of 2013", "article": "3", "url": D43_URL,
                       "parent_id": "dubai-decree-43-2013__art-3"}

# Law 26/2007 Art. 14 (as amended by Law 33/2008): notice to amend lease terms (incl. rent)
AMEND_TERMS_NOTICE_DAYS = 90
AMEND_TERMS_SOURCE = {"instrument": "Law No. 26 of 2007 (as amended by Law No. 33 of 2008)", "article": "14", "url": L33_URL,
                      "parent_id": "dubai-law-26-2007__art-14__as-amended-L33-2008"}
# Art. 2 (am.) defines "Notice": notary public, registered mail, by hand, or other technological means approved by law
NOTICE_DEFINITION_SOURCE = {"instrument": "Law No. 26 of 2007 (as amended by Law No. 33 of 2008)", "article": "2", "url": L33_URL,
                            "parent_id": "dubai-law-26-2007__art-2__as-amended-L33-2008"}
NOTICE_DEFINITION_METHODS = ("notary public", "registered mail", "by_hand")  # + "other technological means approved by law"

# Law 26/2007 Art. 25 (as amended by Law 33/2008)
EVICTION_DURING_TERM_CURE_DAYS = 30      # Art. 25(1)(a) rent default, 25(1)(h) breach of obligation
EVICTION_ON_EXPIRY_NOTICE_MONTHS = 12    # Art. 25(2): demolition, restoration, own use, sale
EVICTION_NOTICE_SERVICE = ("notary public", "registered mail")  # required for both 25(1) and 25(2)
EVICTION_SOURCE = {"instrument": "Law No. 26 of 2007 (as amended by Law No. 33 of 2008)", "article": "25", "url": L33_URL,
                   "parent_id": "dubai-law-26-2007__art-25__as-amended-L33-2008"}

# Grounds, keyed to Art. 25 paragraphs
GROUNDS_DURING_TERM = {
    "rent_default": "25(1)(a)", "unauthorised_sublet": "25(1)(b)", "illegal_use": "25(1)(c)",
    "commercial_unoccupied": "25(1)(d)", "damage_or_unsafe_change": "25(1)(e)",
    "use_for_other_purpose": "25(1)(f)", "collapse_risk": "25(1)(g)",
    "breach_of_obligation": "25(1)(h)", "government_demolition": "25(1)(i)",
}
GROUNDS_ON_EXPIRY = {
    "demolish_or_rebuild": "25(2)(a)", "major_restoration": "25(2)(b)",
    "owner_or_relative_use": "25(2)(c)", "sale": "25(2)(d)",
}

# Rental Disputes Center fee schedule. Legal basis: Decree 26/2013 Art. 23 delegates fees to an
# Executive Council resolution (not published on dlp.dubai.gov.ae). The figures below are taken
# from the RDC's own official FAQ (rdc.gov.ae, page dated 2026-06-18), cached as source "rdc-faq".
# Authority tier: service_guidance (official, but not legislation) - surfaced as such in answers.
RDC_FAQ_URL = "https://rdc.gov.ae/en/frequently-asked-questions/"
RDC_FEE_RATE = 0.035
RDC_RENT_CASE_MIN, RDC_RENT_CASE_MAX = 500, 20_000          # eviction, renewal, rent claim, termination, return
RDC_MONETARY_CLAIM_MIN, RDC_MONETARY_CLAIM_MAX = 500, 15_000  # base = claimed amount
RDC_FIXED_FEES = {"process_service": 100, "knowledge_fee": 10, "innovation_fee": 10}
RDC_POA_FEE = 25                       # only if filed through an attorney
RDC_TRUSTEE_CENTRE_FEE = 130           # plus VAT, only if filed at a Real Estate Services Trustee Centre
RDC_SETTLEMENT_REFUND_SHARE = 0.5      # half the fee for basic claims refunded on conciliation settlement
RDC_FEE_SOURCE = {"instrument": "Rental Disputes Center FAQ (official)", "article": "Fees",
                  "url": RDC_FAQ_URL, "authority_tier": "service_guidance", "parent_id": "rdc-faq__q-14"}
RDC_FEE_LEGAL_BASIS = {"instrument": "Decree No. 26 of 2013", "article": "23", "parent_id": "dubai-decree-26-2013__art-23"}

# Figures that could not be traced to an official source. None are used by any tool.
UNVERIFIED: dict[str, dict] = {}
