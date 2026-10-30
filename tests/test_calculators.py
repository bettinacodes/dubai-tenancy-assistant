"""Boundary tests for every legally operative rule in the calculators."""
import pytest

from tenancy_rag.tools.calculators import notice_period_check, rdc_fee_estimate, rent_increase_cap


@pytest.mark.parametrize("shortfall,expected", [
    (-5, 0), (0, 0), (10, 0), (10.5, 5), (11, 5), (20, 5), (20.01, 10), (30, 10), (31, 15), (40, 15), (40.01, 20), (75, 20)])
def test_rent_bands_every_boundary(shortfall, expected):
    assert rent_increase_cap(shortfall_pct=shortfall).result["max_increase_pct"] == expected


@pytest.mark.parametrize("cur,avg,pct,aed", [(80000, 100000, 5, 4000), (50000, 52000, 0, 0),
                                             (60000, 100000, 15, 9000), (55000, 100000, 20, 11000)])
def test_rent_cap_from_amounts(cur, avg, pct, aed):
    r = rent_increase_cap(cur, avg).result
    assert r["max_increase_pct"] == pct and r["max_increase_aed"] == aed and r["max_new_rent_aed"] == cur + aed


def test_requested_increase_compared_with_cap_and_working_shown():
    t = rent_increase_cap(80000, 100000, requested_increase_pct=15)
    assert t.result["requested_within_cap"] is False
    assert any("EXCEEDS" in w for w in t.working) and t.sources[0]["parent_id"] == "dubai-decree-43-2013__art-1"


def test_rent_above_average_and_bad_input():
    assert rent_increase_cap(120000, 100000).result["max_increase_pct"] == 0
    with pytest.raises(ValueError):
        rent_increase_cap(0, 100000)
    with pytest.raises(ValueError):
        rent_increase_cap()


@pytest.mark.parametrize("notice,expiry,ok", [("2026-09-02", "2026-12-01", True),    # exactly 90 days
                                              ("2026-09-03", "2026-12-01", False),   # 89 days
                                              ("2026-11-01", "2026-12-01", False)])
def test_ninety_day_rule(notice, expiry, ok):
    r = notice_period_check("amend_terms", notice, expiry).result
    assert r["period_ok"] is ok and r["valid"] is ok and r["latest_valid_notice"] == "2026-09-02"


def test_amend_terms_whatsapp_is_flagged_not_decided():
    t = notice_period_check("amend_terms", "2026-08-01", "2026-12-01", served_via="whatsapp")
    assert t.result["service_ok"] is None and t.result["valid"] is True
    assert any("not settled" in c for c in t.caveats)


@pytest.mark.parametrize("notice,ok", [("2026-03-31", True), ("2026-04-01", False)])
def test_twelve_month_rule_exact(notice, ok):
    r = notice_period_check("sale", notice, "2027-03-31", served_via="registered mail").result
    assert r["period_ok"] is ok and r["latest_valid_notice"] == "2026-03-31"


def test_eviction_service_must_be_notary_or_registered_mail():
    r = notice_period_check("owner_or_relative_use", "2026-01-01", "2027-06-01", served_via="whatsapp").result
    assert r["period_ok"] is True and r["service_ok"] is False and r["valid"] is False
    assert notice_period_check("sale", "2026-01-01", "2027-06-01", served_via="notary public").result["valid"] is True


def test_twelve_months_handles_month_ends():
    assert notice_period_check("sale", "2024-02-28", "2025-02-28").result["latest_valid_notice"] == "2024-02-28"
    assert notice_period_check("sale", "2024-02-29", "2025-02-28").result["period_ok"] is False


def test_rent_default_cure_period():
    r = notice_period_check("rent_default", "2026-05-03", served_via="registered mail").result
    assert r["cure_deadline"] == "2026-06-02" and r["ground"] == "Art. 25(1)(a)"


@pytest.mark.parametrize("rent,fee", [(90000, 3150), (10000, 500), (14000, 500), (14285, 500), (14286, 500.01), (14300, 500.5),
                                      (571429, 20000), (700000, 20000)])
def test_rdc_rent_case_fee_floor_and_cap(rent, fee):
    assert rdc_fee_estimate(rent).result["filing_fee"] == pytest.approx(fee)


def test_rdc_monetary_cap_and_extras():
    assert rdc_fee_estimate(claim_amount=1_000_000, case_type="monetary").result["filing_fee"] == 15000
    r = rdc_fee_estimate(90000, poa=True, trustee_centre=True).result
    assert r["fixed_fees"] == 100 + 10 + 10 + 25 + 130 and r["estimated_total"] == 3150 + 275
