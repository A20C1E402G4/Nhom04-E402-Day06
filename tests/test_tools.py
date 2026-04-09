"""Unit tests for VSSA Pydantic schemas and tool implementations."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from vssa_agent.schemas import CalculateLoanArgs, GetVehicleDataArgs
from vssa_agent.tools import (
    _calculate_loan_impl,
    _find_showrooms_impl,
    calculate_loan,
    find_showrooms,
    get_promotion,
    get_vehicle_data,
)


# ---------- Schemas ----------------------------------------------------------


def test_calculate_loan_rejects_out_of_range_percentage() -> None:
    with pytest.raises(ValidationError):
        CalculateLoanArgs(
            vehicle_price_vnd=850_000_000,
            loan_percentage=150,
            duration_months=96,
        )


def test_calculate_loan_rejects_invalid_duration() -> None:
    with pytest.raises(ValidationError):
        CalculateLoanArgs(
            vehicle_price_vnd=850_000_000,
            loan_percentage=70,
            duration_months=99,
        )


def test_calculate_loan_rejects_zero_price() -> None:
    with pytest.raises(ValidationError):
        CalculateLoanArgs(
            vehicle_price_vnd=0,
            loan_percentage=70,
            duration_months=96,
        )


def test_get_vehicle_data_rejects_unknown_model() -> None:
    with pytest.raises(ValidationError):
        GetVehicleDataArgs(model_id="VF99")  # type: ignore[arg-type]


# ---------- calculate_loan ---------------------------------------------------


def test_calculate_loan_matches_case_1_baseline() -> None:
    """VF 7 base @ 70% / 96mo / Vietcombank should yield ~255tr down + ~8.5tr/mo."""
    result = _calculate_loan_impl(
        vehicle_price_vnd=850_000_000,
        loan_percentage=70,
        duration_months=96,
        bank_id="vietcombank",
    )
    assert result["down_payment_vnd"] == 255_000_000
    assert result["principal_vnd"] == 595_000_000
    assert result["bank_name"] == "Vietcombank"
    assert result["interest_rate_percent_per_year"] == 8.5
    # Standard amortization should land around 8.4-8.6 million VND/month.
    assert 8_300_000 <= result["monthly_payment_vnd"] <= 8_700_000
    assert result["total_interest_vnd"] > 0


def test_calculate_loan_caps_at_bank_max_percentage() -> None:
    result = _calculate_loan_impl(
        vehicle_price_vnd=500_000_000,
        loan_percentage=95,  # vietcombank max is 80
        duration_months=60,
        bank_id="vietcombank",
    )
    assert result["loan_percentage"] == 80


def test_calculate_loan_tool_decorator_runs() -> None:
    out = calculate_loan.invoke(
        {
            "vehicle_price_vnd": 850_000_000,
            "loan_percentage": 70,
            "duration_months": 96,
            "bank_id": "vietcombank",
        }
    )
    assert out["monthly_payment_vnd"] > 0


# ---------- get_vehicle_data -------------------------------------------------


def test_get_vehicle_data_returns_vf7_record() -> None:
    record = get_vehicle_data.invoke({"model_id": "VF7"})
    assert record["model_id"] == "VF7"
    assert record["model_name"] == "VinFast VF 7"
    assert (
        record["versions"]["base"]["price_without_battery_vnd"] == 850_000_000
    )


# ---------- find_showrooms ---------------------------------------------------


def test_find_showrooms_returns_thao_dien_for_vf7() -> None:
    result = _find_showrooms_impl(model_id="VF7", district="Thủ Đức")
    assert result["found"] is True
    assert result["recommended_showroom"]["showroom_name"] == "VinFast Thảo Điền"
    assert result["recommended_showroom"]["next_slot"].startswith("2026-04-1")


def test_find_showrooms_tool_decorator_runs() -> None:
    out = find_showrooms.invoke({"model_id": "VF7", "district": ""})
    assert out["found"] is True


# ---------- get_promotion ----------------------------------------------------


def test_get_promotion_returns_vf7_offer() -> None:
    out = get_promotion.invoke({"model_id": "VF7"})
    promo_codes = [p["promo_code"] for p in out["promotions"]]
    assert "VF7_FREE_CHARGE_1YR" in promo_codes
