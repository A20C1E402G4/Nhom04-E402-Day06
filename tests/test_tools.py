"""Unit tests for VSSA Pydantic schemas and tool implementations."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from vssa_agent.schemas import (
    BookTestDriveArgs,
    CalculateLoanArgs,
    GetVehicleDataArgs,
)
from vssa_agent.tools import (
    _book_test_drive_impl,
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


# ---------- book_test_drive -------------------------------------------------


def test_book_test_drive_schema_rejects_short_phone() -> None:
    with pytest.raises(ValidationError):
        BookTestDriveArgs(
            name="Khách Demo",
            phone="123",
            model_id="VF7",
            showroom_id="SR_Q9",
            slot="2026-04-10T09:00:00",
        )


def test_book_test_drive_impl_rejects_unknown_showroom() -> None:
    result = _book_test_drive_impl(
        name="Khách Demo",
        phone="0901234567",
        model_id="VF7",
        showroom_id="SR_NOPE",
        slot="2026-04-10T09:00:00",
    )
    assert result["booked"] is False
    assert "không tồn tại" in result["error"]


def test_book_test_drive_impl_rejects_invalid_slot() -> None:
    result = _book_test_drive_impl(
        name="Khách Demo",
        phone="0901234567",
        model_id="VF7",
        showroom_id="SR_Q9",
        slot="1999-01-01T00:00:00",
    )
    assert result["booked"] is False
    assert "không có trong khung giờ" in result["error"]


def test_book_test_drive_impl_happy_path(tmp_path, monkeypatch) -> None:
    # Redirect telemetry writes into an isolated tmp log dir.
    from vssa_agent import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "LOGS_DIR", tmp_path)
    result = _book_test_drive_impl(
        name="Khách Demo",
        phone="0901234567",
        model_id="VF7",
        showroom_id="SR_Q9",
        slot="2026-04-10T09:00:00",
    )
    assert result["booked"] is True
    assert result["confirmation"]["showroom_name"] == "VinFast Thảo Điền"
    bookings_file = tmp_path / "bookings.jsonl"
    assert bookings_file.exists()
    assert "Khách Demo" in bookings_file.read_text(encoding="utf-8")
