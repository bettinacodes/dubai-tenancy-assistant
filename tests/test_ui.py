"""UI rendering tests (pure functions; no Gradio server)."""
from tenancy_rag import ui


def _resp(**kw):
    base = {"mode": "generated", "answer": "Notice must be 90 days <b>before</b> expiry.", "language": "English",
            "citations": [{"label": "S1", "instrument": "Law No. 26 of 2007", "locator": "Art. 14", "url": "https://dlp.dubai.gov.ae/x",
                           "authority": "legislation", "status": "in_force"}],
            "tool": None, "caveats": ["Check your contract."], "notices": {"not_advice": "Not legal advice."},
            "escalation": {"title": "Contacts", "contacts": [{"label": "Police", "value": "999", "when": "danger", "url": "https://u.ae"}],
                           "verified_on": "2026-10-09"}}
    return {**base, **kw}


def test_answer_is_escaped_and_badged():
    a = ui.answer_html(_resp())
    assert "Verified summary" in a and "&lt;b&gt;before&lt;/b&gt;" in a and "<b>before</b>" not in a


def test_chips_show_tier_and_superseded_status():
    c = ui.citations_html(_resp(citations=[{**_resp()["citations"][0], "status": "superseded"}]))
    assert "Law" in c and "Art. 14" in c and "superseded" in c and 'href="https://dlp.dubai.gov.ae/x"' in c


def test_crisis_card_shows_contacts_inline_and_render_returns_six_parts():
    r = _resp(mode="crisis_card")
    assert "999" in ui.answer_html(r)
    assert len(ui.render(r)) == 6


def test_translation_is_shown_for_non_english():
    a = ui.answer_html(_resp(language="Arabic", query_english="Can the landlord raise the rent?"))
    assert "Understood as" in a and "translated from Arabic" in a and 'dir="auto"' in a
