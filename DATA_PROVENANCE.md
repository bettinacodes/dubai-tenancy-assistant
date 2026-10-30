# Data provenance

**Corpus "as at" date: 2026-10-09.** This date is shown on every answer. The legal position after this date is not represented.

## Acquisition policy
- **Official sources only.** Sources come from the Dubai Legislation Portal (dlp.dubai.gov.ae) and the Dubai Land Department (dubailand.gov.ae). Each fetch is cached under `data/raw/<source>/<source>__<date>.<ext>` and logged in `data/raw/manifest.jsonl` with the URL, SHA-256, timestamp, HTTP status and robots.txt verdict.
- **Redaction.** Seven cached pages embed the publishers' own client-side Google API keys in their scripts. These were replaced with `REDACTED_API_KEY`; the manifest keeps the as-fetched hash (`sha256_as_fetched`) next to the hash of the stored file. No parsed text changes.
- **Polite fetching.** Requests carry an identifying User-Agent, wait 3 s between requests to the same host, and retry with exponential backoff. robots.txt is checked before every fetch. dlp.dubai.gov.ae allows legislation pages. dubailand.gov.ae publishes no robots.txt, which places no restriction.
- **If a site blocks automated access**, the file is downloaded by hand and logged with `acquisition: "manual"`. Blocks are never worked around.
- **Excluded on purpose:** law-firm blogs, property portals (Bayut, Property Finder, etc.), news media and paywalled databases (LexisNexis). They were used only to cross-check while building the gold set and are never indexed. Excluding them is a provenance decision: the assistant must not cite a source a tenant cannot check against the law itself.

## Sources (fetched 2026-10-09, all HTTP 200)

| id | Instrument | Authority tier | Format | Notes |
|---|---|---|---|---|
| dubai-law-26-2007 | Law No. 26 of 2007 (landlords and tenants) | legislation | HTML | 37/37 articles parsed. Footnote says the Arabic text prevails. This is the **original** 2007 text. |
| dubai-law-33-2008 | Law No. 33 of 2008 amending L26/2007 | legislation | HTML | Replaces L26 Arts 2, 3, 4, 9, 13, 14, 15, 25, 26, 29, 36. The originals stay indexed with `status: superseded`. |
| dubai-decree-43-2013 | Decree No. 43 of 2013 (rent increase) | decree | HTML | 4 articles. Source of the 0/5/10/15/20% bands. |
| dubai-decree-26-2013 | Decree No. 26 of 2013 (Rent Disputes Settlement Centre) | decree | HTML | 31 articles. Covers jurisdiction, mediation and the fee mechanism (Art. 23). |
| dld-decree-43-2013-pdf | DLD publication of Decree 43/2013 | regulatory_guidance | PDF | Cross-check of the bands |
| dld-smart-rental-index | DLD announcement: Smart Rental Index 2025 (2 Jan 2025) | regulatory_guidance | HTML | Contains errors (see below) |
| dld-tenancy-guide | DLD Tenancy Guide (EN) | regulatory_guidance | PDF, 42 pp | |
| dld-ejari-register | DLD service page: Register/Renew Ejari | service_guidance | HTML | |
| dld-faq | DLD FAQ | service_guidance | HTML | |
| rdc-faq | Rental Disputes Center FAQ, including the fee schedule (rdc.gov.ae; no robots.txt) | service_guidance | HTML | Source of the RDC fee figures |
| dld-real-estate-legislation | DLD compendium of real-estate legislation | regulatory_guidance | PDF, 149 pp | Kept for reference only. It duplicates the instruments above, so it is not indexed. |

## Verification of operative numbers
Each number used by a calculator tool was checked against the cached primary text (`src/tenancy_rag/tools/constants.py`):

| Rule | Value | Primary source |
|---|---|---|
| Max renewal increase by rent shortfall vs. average | ≤10%: 0 · 11–20%: 5% · 21–30%: 10% · 31–40%: 15% · >40%: 20% | D43/2013 Art. 1 |
| How the average rent is determined | Official Rent Index (now the Smart Rental Index) | D43/2013 Art. 3 |
| Notice to amend lease terms (incl. rent) | ≥ 90 days before expiry, unless agreed otherwise | L26 Art. 14 (am. L33/2008) |
| Cure period before eviction during the term | 30 days from notice | L26 Art. 25(1)(a),(h) (am.) |
| Eviction at expiry (demolition, restoration, own use, sale) | ≥ 12 months' notice | L26 Art. 25(2) (am.) |
| How notices must be served | Notary public or registered mail | L26 Art. 25 (am.) |

| RDC filing fee, rent cases | 3.5% of annual rent, min AED 500, max AED 20,000 | RDC FAQ (official service guidance) |
| RDC filing fee, monetary claims | 3.5% of claimed amount, min AED 500, max AED 15,000 | RDC FAQ |
| RDC fixed add-ons | Process service AED 100, knowledge fee AED 10, innovation fee AED 10, power of attorney AED 25 (if used), trustee centre AED 130 + VAT (if used) | RDC FAQ |
| Refund on conciliation | Half of the fee for basic claims | RDC FAQ; D26/2013 also provides for a half refund on amicable settlement |

**RDC fee provenance:** D26/2013 Art. 23 delegates fees to an Executive Council resolution, which is not published on dlp.dubai.gov.ae. On Day 1 the fee could be found only in secondary sources, so the fee tool was suspended. On Day 1 (decision 1) the RDC's own official FAQ (rdc.gov.ae, page dated 2026-06-18) was found and cached as `rdc-faq`. The fee figures therefore rest on official **service guidance**, not legislation, and answers show that tier.

## What is indexed (Day 2)
- **Indexed:** 9 sources, giving 431 parent records and 457 chunks (`data/processed/`). The parse statistics are in `parse_report.json`.
- **Cached but not indexed (duplicates):**
  - the DLD PDF copy of Decree 43/2013;
  - the DLD legislation compendium;
  - Tenancy Guide pp. 17–37, which reprint L26, L33 and D43.
- **Filtered:**
  - DLD FAQ: only the *Rental Services* and *Dispute Services* sections are kept (194 of 398 Q&As).
  - RDC FAQ: all 130 Q&As are kept.
- **Superseded but kept:**
  - the 11 original L26/2007 articles replaced by L33/2008;
  - Tenancy Guide pp. 40–42 (the RERA Rent Index, replaced by the Smart Rental Index on 2 Jan 2025).

## Known data-quality issues in official sources
- The DLD Smart Rental Index announcement cites "Decree No. (34) of 2013", which should be 43. It also says the 20% band applies to rents that "exceed the average by more than 40%", whereas D43/2013 Art. 1 says rents more than 40% **below** the average. The system ranks legislation above guidance, and the calculator reads the decree directly.
- **RDC FAQ Q85 (= DLD FAQ Q163)** says "according to the law, it is mandatory to collect a security deposit". L26/2007 Art. 20 says the landlord "may obtain" one and sets no maximum.
- **DLD FAQ Q82** (property sold) says a new owner keeps tenants only "if the new owner wants to", and may evict for own use by asking in writing. This contradicts Art. 28 (a fixed-term lease survives a transfer of ownership) and Art. 25(2)(c) as amended (own-use eviction only at expiry, with 12 months' notarised notice). It appears to be adapted from a non-Dubai source.
- All three guidance errors are registered in `data/known_conflicts.json`. The affected records carry `quality_flags`, are never used as gold evidence, and are down-weighted in favour of the binding text.
- The DLP pages are Word-exported HTML in windows-1252, so the parser detects the declared charset. Words split across `<span>` tags are re-joined.

## Licence and terms
- These are public government publications. They are cached for non-commercial research only and are not redistributed beyond what reproducing these results requires.
- The English texts are **not authoritative translations**. Every record carries `translation_authoritative: false`.
