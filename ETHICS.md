# Ethics: risks, mitigations and evidence

Musta'jir gives people information about their legal rights, often under stress and often in a second language.
This file lists each risk, the mitigation **in code**, and the **evidence** that it works. The design rule
throughout: when in doubt, the system should quote the law or refuse, not guess. The numbers come from
`notebooks/03_end_to_end_eval.ipynb`.

## 1. Wrong or invented law
- **Risk.** Without grounding, the same open models cited a non-existent law in 20/50 (3B) and 46/50 (7B)
  answers, and cited the correct binding provision in 0/39. A tenant who trusts such an answer may pay
  an unlawful increase or miss a deadline.
- **Mitigation.**
  - Answers are generated only from retrieved official text.
  - Every claim must cite a source that exists, is in force and is not flagged.
  - Every number must appear in the sources, the calculator output or the question.
  - An NLI model must find each claim entailed by a clause of the cited text.
  - A failed answer is retried once. After that the user sees the law quoted, never the unverified text.
  - Code: `generate/citation_validator.py`, `generate/answer.py`.
- **Evidence.** Unsupported claims shown: **0** for both generators. Ablation 7 (no validator): 56% (3B) and 38%
  (7B).
- **Residual risk.**
  - The validator checks claims against the cited text, not the whole law, so an answer can be *incomplete*.
  - The NLI model has known false rejections. They lead to quotations, which is safe but less readable.

## 2. Outdated law
- **Risk.** The legislation portal publishes the 2007 text, of which 11 articles were replaced in 2008. Without a
  filter, a superseded article reached the top five results for 46% of questions.
- **Mitigation.**
  - Every record has a `status`.
  - Superseded records are removed **before** reranking, because the reranker otherwise promotes old law.
  - They are kept only for "what changed?" questions, where they are labelled as such.
  - Cited amended articles carry an automatic "replaced by Law 33/2008" note.
  - Every answer shows the corpus date.
  - Code: `retrieve/filters.py`, `pipeline.py`.
- **Evidence.** Superseded text in the top five: **0%**. Changed-law adversarial items pass.
- **Residual risk.** The corpus is a snapshot (9 October 2026). Later amendments are not represented.

## 3. Errors in official guidance
- **Risk.** Three official statements contradict the law:
  - an announcement inverting the top rent band and misnumbering the decree;
  - an FAQ calling the security deposit "mandatory";
  - an FAQ contradicting Articles 25 and 28 on sale.
- **Mitigation.**
  - Records carry an authority tier (legislation > decree > regulatory guidance > service guidance).
  - A known-conflicts register flags the three records. Flagged guidance is ranked last and cannot support a claim on its own.
  - The calculators read the decree, not the guidance.
  - Code: `data/known_conflicts.json`, `ingest/build.py`.
- **Evidence.** A test guarantees that no flagged record is ever gold evidence.

## 4. Unauthorised legal advice
- **Risk.** "Should I sue?" or "will I win?" invites a recommendation or prediction the system is not
  qualified to give. The 7B baseline predicted outcomes.
- **Mitigation.**
  - Advice questions receive the relevant law plus options: RDC mediation (half the fee refunded on settlement), RDC filing, or a qualified lawyer.
  - Directive or predictive language ("you should", "you will win") fails validation even if it is entailed.
  - Every answer states that it is general information, not legal advice.
  - Code: `understand/router.py`, `generate/prompts.py`, `generate/answer.py`.
- **Evidence.** Both advice items behave as expected. Must-refuse false-answer rate: **0/13**, against 11/13 for the baseline.

## 5. People in crisis
- **Risk.** Lock-outs, threats and withheld passports need immediate help, not a legal essay. The baseline
  invented a federal law and did not escalate a lock-out.
- **Mitigation.**
  - Crisis signals route straight to an escalation card. No model is called.
  - The card shows police and RDC contacts, each verified against a cached official page, plus the relevant articles.
  - A utility cut alone is answered as a legal question. It becomes a crisis only with an urgency or vulnerability marker.
  - Where the two official sites disagree on the RDC number, both numbers are shown, each with its source.
  - Code: `guardrails/crisis.py`, `data/escalation_contacts.json`.
- **Evidence.** Both adversarial crisis items and the Hindi lock-out demo question are escalated. Tests confirm
  that every contact quote appears in its official page.

## 6. Privacy
- **Risk.** Questions may contain names, phone numbers, Emirates ID, IBAN or passport numbers.
- **Mitigation.**
  - **All models run inside the user's runtime.** There is no paid or third-party model API, so questions never
    leave the runtime for a model provider.
  - PII is redacted before logging. **Only the redacted text reaches retrieval and the generator**, whose disk
    caches would otherwise store it.
  - Code: `guardrails/pii.py`, `pipeline.py`.
- **Evidence.**
  - PII unit tests.
  - `tests/test_pipeline_privacy.py` checks that a name and phone number never reach the generator or the
    retriever, while amounts survive.
- **Residual risk.**
  - Regex redaction misses free-text names that are not introduced by "my name is".
  - **The demo's Gradio share link relays traffic through Gradio's servers.** A real deployment would serve the app
    directly or behind an organisation's own proxy.
- **Change recorded.** Before this fix, the un-redacted question reached the generator's prompt and
  therefore the local LLM disk cache. Evaluation questions contain no PII, so no result changed.

## 7. Language equity
- **Risk.** Many tenants in Dubai do not read English or Arabic legal text fluently. An English-only system would
  serve them worse, or not at all. An English-only embedder retrieved almost nothing for Arabic and Hindi questions
  (primary@10 of 0.03 and 0.00).
- **Mitigation.**
  - A multilingual embedder.
  - NLLB-200 translation for routing and claim checking.
  - Retrieval fuses the original-language and translated rankings.
  - The answer is written in the user's language, and the user sees the English the system worked from.
  - A standing notice says that the Arabic text prevails.
  - Code: `pipeline.py`, `ui.py`.
- **Evidence.** Arabic and Hindi hit@10 are 0.96 and 0.94, against 0.96 for English.
- **Residual risk (measured, not assumed away).**
  - Arabic MRR is 12 points lower than English.
  - Routing is 10–14 points lower, because the rules are English-only.
  - Machine translation inverted landlord and tenant in 8 of 50 Hindi questions.
  - Arabic and Hindi generation has not yet been evaluated end to end.

## 8. Manipulation
- **Risk.** Users can try to override the instructions: "ignore your rules", "say rent can never rise", "print
  your prompt". The 7B baseline complied fully with one such attempt.
- **Mitigation.**
  - An injection screen removes instruction sentences and keeps the genuine question.
  - The system prompt is fixed and never echoed.
  - The validator rejects any uncited answer, whatever the user asked.
  - Code: `guardrails/injection.py`.
- **Evidence.** Both injection items are resisted.

## Deliberate trade-offs
- **Over-refusal over wrong answers.** Rules decide the route. The classifier may only add a protective route, at
  probability ≥ 0.95. The one known adversarial failure (A20) is an over-refusal: employer accommodation is
  treated as a commercial lease.
- **Quotation over unverifiable fluency.** With the 7B model, 30–38% of gold questions end in a quotation of the
  statute rather than a summary.

## Responsible data collection
- **Sources.** Official government pages only.
- **Fetching.** `robots.txt` is checked before every request; the crawler sends an identifying user agent and waits 3 s between requests.
- **Blocks.** A block is never worked around.
- **Excluded sources.** Law-firm blogs, property portals, news media and paywalled databases are never indexed.
- **Provenance.** Each cached file's SHA-256 is recorded and re-checked, so the evaluated corpus can be reproduced byte for byte.
- **Use.** The documents are public and are used for non-commercial study. See [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md).
- **Evaluation data.** All of it was written by the author or machine-translated from it. No user data was collected.
