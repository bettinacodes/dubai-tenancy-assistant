from tenancy_rag.pipeline import Assistant, detect_language


def test_script_based_language_id():
    assert detect_language("Can my landlord raise the rent?") == "English"
    assert detect_language("هل يمكن للمالك زيادة الإيجار؟") == "Arabic"
    assert detect_language("کیا مالک مکان کرایہ بڑھا سکتا ہے؟") == "Urdu"
    assert detect_language("क्या मकान मालिक किराया बढ़ा सकता है?") == "Hindi"


def test_fusion_rewards_agreement(monkeypatch):
    a = Assistant.__new__(Assistant)
    a.n_sources = 3
    rankings = {"orig": ["dubai-law-26-2007__art-20", "dld-faq__q-27", "dubai-law-26-2007__art-21"],
                "en": ["dubai-law-26-2007__art-21", "dubai-law-26-2007__art-20", "rdc-faq__q-50"]}
    monkeypatch.setattr(a, "_ranked", lambda q: rankings[q])
    pids = [s["parent_id"] for s in a._sources("orig", None, False, alt_query="en")]
    assert pids[0] == "dubai-law-26-2007__art-20" and "dubai-law-26-2007__art-21" in pids
