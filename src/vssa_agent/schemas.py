"""Pydantic argument schemas for VSSA tools.

These schemas are bound to LangChain `@tool` functions in `tools.py` to
guarantee that the LLM cannot invoke business logic with hallucinated
parameters (Failure Mode #2 in spec.md).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

VehicleModelId = Literal["VF5", "VF7", "VF8"]
BankId = Literal["vietcombank", "techcombank"]

ALLOWED_DURATIONS_MONTHS: tuple[int, ...] = (12, 24, 36, 48, 60, 72, 84, 96)


class GetVehicleDataArgs(BaseModel):
    """Args for `get_vehicle_data`."""

    model_id: VehicleModelId = Field(
        ...,
        description="VinFast model identifier. Must be one of VF5, VF7, VF8.",
    )


class CalculateLoanArgs(BaseModel):
    """Args for `calculate_loan`. Hard caps prevent hallucinated values."""

    vehicle_price_vnd: int = Field(
        ..., gt=0, description="On-road vehicle price in VND (positive integer)."
    )
    loan_percentage: int = Field(
        ...,
        ge=0,
        le=100,
        description="Percentage of the vehicle price to finance (0-100).",
    )
    duration_months: int = Field(
        ..., description="Loan term in months. One of 12,24,36,48,60,72,84,96."
    )
    bank_id: BankId = Field(
        default="vietcombank", description="Banking partner identifier."
    )

    @field_validator("duration_months")
    @classmethod
    def _check_duration(cls, v: int) -> int:
        if v not in ALLOWED_DURATIONS_MONTHS:
            raise ValueError(
                f"duration_months must be one of {ALLOWED_DURATIONS_MONTHS}, got {v}"
            )
        return v


class FindShowroomsArgs(BaseModel):
    """Args for `find_showrooms`."""

    model_id: VehicleModelId = Field(
        ..., description="Model the customer wants to test drive."
    )
    district: str = Field(
        default="",
        description="Optional customer district hint (e.g. 'Quận 2'). May be empty.",
    )


class GetPromotionArgs(BaseModel):
    """Args for `get_promotion`."""

    model_id: VehicleModelId = Field(
        ..., description="Model to look up active promotions for."
    )


class BookTestDriveArgs(BaseModel):
    """Args for `book_test_drive` — persists a confirmed lead to bookings.jsonl."""

    name: str = Field(
        ..., min_length=1, max_length=80, description="Họ và tên khách hàng."
    )
    phone: str = Field(
        ...,
        min_length=8,
        max_length=15,
        description="Số điện thoại liên hệ (8-15 ký tự).",
    )
    model_id: VehicleModelId = Field(..., description="Mẫu xe khách muốn lái thử.")
    showroom_id: str = Field(
        ...,
        min_length=1,
        description="ID showroom lấy từ kết quả find_showrooms (vd: SR_Q9, SR_Q1).",
    )
    slot: str = Field(
        ...,
        description="Khung giờ ISO datetime lấy từ all_upcoming_slots của showroom.",
    )

    @field_validator("phone")
    @classmethod
    def _check_phone(cls, v: str) -> str:
        digits = "".join(c for c in v if c.isdigit())
        if len(digits) < 8:
            raise ValueError("phone must contain at least 8 digits")
        return v
