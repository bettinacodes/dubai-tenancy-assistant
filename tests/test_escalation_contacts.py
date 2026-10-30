"""Every escalation contact must be quoted verbatim from its cached official page."""
import json
import re
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from tenancy_rag import config

E = json.loads((config.DATA / "escalation_contacts.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("c", E["contacts"], ids=lambda c: c["id"])
def test_contact_quoted_from_cached_official_page(c):
    files = sorted((config.RAW / c["source"]).glob("*"))
    assert files, f"no cached page for {c['source']}"
    text = re.sub(r"\s+", " ", BeautifulSoup(files[-1].read_text(encoding="utf-8", errors="replace"), "html.parser").get_text(" "))
    assert re.sub(r"\s+", " ", c["quote"]) in text
    digits = re.sub(r"\D", "", c["value"].split("(")[0])
    assert digits[-6:] in re.sub(r"\D", "", c["quote"]), "value must match the quoted number"
