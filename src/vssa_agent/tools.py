"""LangChain tool implementations bound to Pydantic schemas.

All tools are pure Python (no network calls) so they can be invoked directly
by the Streamlit UI for the Correction Path slider as well as by the LangGraph
ToolNode for the Happy / Booking Paths.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from langchain_core.tools import tool

from .config import (
    load_finance,
    load_showrooms,
    load_vehicles,
    load_charging_stations,
)
from .schemas import (
    BookTestDriveArgs,
    CalculateLoanArgs,
    FindShowroomsArgs,
    GetPromotionArgs,
    GetVehicleDataArgs,
    FindChargingStationsArgs,
    GetAllVehiclesArgs,
)

try:  # pragma: no cover - telemetry is best-effort, never block business logic
    from telemetry.logger import log_booking
    from telemetry.metrics import incr_lead, persist as persist_metrics
except ImportError:  # pragma: no cover
    def log_booking(record: dict[str, Any]) -> None: ...
    def incr_lead() -> None: ...
    def persist_metrics() -> None: ...

DISCLAIMER_VI: str = (
    "Lưu ý: Các con số tính toán và giá trên đây chỉ mang tính tham khảo. "
    "Vui lòng xác nhận với Showroom để có giá và chương trình khuyến mãi chính xác nhất."
)


def _calculate_loan_impl(
    vehicle_price_vnd: int,
    loan_percentage: int,
    duration_months: int,
    bank_id: str = "vietcombank",
) -> dict[str, Any]:
    """Pure amortization calculator. Importable directly from the UI layer."""
    finance = load_finance()
    bank = finance["banking_partners"].get(bank_id)
    if bank is None:
        raise ValueError(f"Unknown bank_id: {bank_id}")

    if loan_percentage > bank["max_loan_percentage"]:
        loan_percentage = bank["max_loan_percentage"]
    if duration_months > bank["max_duration_months"]:
        duration_months = bank["max_duration_months"]

    principal = vehicle_price_vnd * loan_percentage // 100
    down_payment = vehicle_price_vnd - principal
    annual_rate = bank["interest_rate_percent_per_year"] / 100.0
    monthly_rate = annual_rate / 12.0

    if principal == 0:
        monthly_payment = 0
        total_interest = 0
    elif monthly_rate == 0:
        monthly_payment = principal // duration_months
        total_interest = 0
    else:
        factor = (1 + monthly_rate) ** duration_months
        monthly_payment_float = principal * monthly_rate * factor / (factor - 1)
        monthly_payment = int(round(monthly_payment_float))
        total_interest = monthly_payment * duration_months - principal

    return {
        "vehicle_price_vnd": vehicle_price_vnd,
        "loan_percentage": loan_percentage,
        "duration_months": duration_months,
        "down_payment_vnd": down_payment,
        "principal_vnd": principal,
        "monthly_payment_vnd": monthly_payment,
        "total_interest_vnd": int(total_interest),
        "bank_id": bank_id,
        "bank_name": bank["bank_name"],
        "interest_rate_percent_per_year": bank["interest_rate_percent_per_year"],
        "disclaimer": DISCLAIMER_VI,
    }


def _find_showrooms_impl(model_id: str, district: str = "") -> dict[str, Any]:
    """Return the first showroom that supports test-driving `model_id`."""
    showrooms = load_showrooms()
    today_iso = datetime.now().isoformat()

    matches: list[dict[str, Any]] = []
    for sr_id, sr in showrooms.items():
        if model_id not in sr["available_test_drive_models"]:
            continue
        future_slots = [s for s in sr["available_time_slots"] if s >= today_iso]
        if not future_slots:
            continue
        matches.append(
            {
                "showroom_id": sr_id,
                "showroom_name": sr["showroom_name"],
                "address": sr["address"],
                "district": sr.get("district", ""),
                "next_slot": sorted(future_slots)[0],
                "all_upcoming_slots": sorted(future_slots),
            }
        )

    # Naive proximity heuristic: prefer a showroom whose district contains the hint.
    if district:
        matches.sort(key=lambda m: 0 if district.lower() in m["district"].lower() else 1)

    if not matches:
        return {
            "found": False,
            "message": (
                "Hiện chưa có showroom hỗ trợ lái thử mẫu xe này trong hệ thống mock. "
                "Hãy trao đổi với saler để được tư vấn kỹ hơn."
            ),
        }

    best = matches[0]
    return {
        "found": True,
        "recommended_showroom": best,
        "alternatives": matches[1:],
        "disclaimer": DISCLAIMER_VI,
    }


@tool("get_vehicle_data", args_schema=GetVehicleDataArgs)
def get_vehicle_data(model_id: str) -> dict[str, Any]:
    """Look up VinFast vehicle specifications and pricing by model id."""
    vehicles = load_vehicles()
    record = vehicles.get(model_id)
    if record is None:
        return {"error": f"model_id {model_id} not found"}
    return {"model_id": model_id, **record}


@tool("calculate_loan", args_schema=CalculateLoanArgs)
def calculate_loan(
    vehicle_price_vnd: int,
    loan_percentage: int,
    duration_months: int,
    bank_id: str = "vietcombank",
) -> dict[str, Any]:
    """Compute monthly installment, down payment, and total interest for a vehicle loan."""
    return _calculate_loan_impl(
        vehicle_price_vnd=vehicle_price_vnd,
        loan_percentage=loan_percentage,
        duration_months=duration_months,
        bank_id=bank_id,
    )


@tool("find_showrooms", args_schema=FindShowroomsArgs)
def find_showrooms(model_id: str, district: str = "") -> dict[str, Any]:
    """Find a VinFast showroom that supports test-driving the given model."""
    return _find_showrooms_impl(model_id=model_id, district=district)


@tool("get_promotion", args_schema=GetPromotionArgs)
def get_promotion(model_id: str) -> dict[str, Any]:
    """Return active promotional offers for the given model."""
    finance = load_finance()
    matches = [
        p for p in finance["current_promotions"] if model_id in p["applicable_models"]
    ]
    return {"model_id": model_id, "promotions": matches}


def _book_test_drive_impl(
    name: str,
    phone: str,
    model_id: str,
    showroom_id: str,
    slot: str,
) -> dict[str, Any]:
    showrooms = load_showrooms()
    sr = showrooms.get(showroom_id)
    if sr is None:
        return {
            "booked": False,
            "error": (
                f"showroom_id {showroom_id} không tồn tại. Hãy gọi find_showrooms "
                "để lấy danh sách showroom hợp lệ."
            ),
        }
    if slot not in sr.get("available_time_slots", []):
        return {
            "booked": False,
            "error": (
                f"slot {slot} không có trong khung giờ trống của {showroom_id}. "
                "Hãy hỏi lại khách chọn một khung giờ khác."
            ),
        }
    if model_id not in sr.get("available_test_drive_models", []):
        return {
            "booked": False,
            "error": (
                f"Showroom {showroom_id} không hỗ trợ lái thử {model_id}."
            ),
        }

    record = {
        "name": name,
        "phone": phone,
        "model_id": model_id,
        "showroom_id": showroom_id,
        "showroom_name": sr.get("showroom_name", ""),
        "showroom_address": sr.get("address", ""),
        "slot": slot,
    }
    log_booking(record)
    incr_lead()
    persist_metrics()
    return {
        "booked": True,
        "confirmation": record,
    }


@tool("book_test_drive", args_schema=BookTestDriveArgs)
def book_test_drive(
    name: str,
    phone: str,
    model_id: str,
    showroom_id: str,
    slot: str,
) -> dict[str, Any]:
    """Persist a confirmed test-drive booking after the customer agreed."""
    return _book_test_drive_impl(
        name=name,
        phone=phone,
        model_id=model_id,
        showroom_id=showroom_id,
        slot=slot,
    )


@tool("find_charging_stations", args_schema=FindChargingStationsArgs)
def find_charging_stations(query: str = "") -> dict[str, Any]:
    """Find VinFast charging stations by name, address, or district."""
    data = load_charging_stations()
    stations = data.get("charging_stations", {})
    
    matches = []
    q = query.lower().strip()
    
    for sr_id, sr in stations.items():
        # Simple text search across name, address
        if (q in sr["location_name"].lower() or 
            q in sr["address"].lower()):
            matches.append({
                "station_id": sr_id,
                "location_name": sr["location_name"],
                "address": sr["address"],
                "coordinates": sr["coordinates"],
                "charger_types": sr["charger_types"],
                "status": sr["status"]
            })
            
    if not matches:
        return {
            "found": False,
            "message": f"Không tìm thấy trạm sạc nào tại '{query}'. Hãy thử tìm kiếm khu vực khác."
        }
        
    return {
        "found": True,
        "stations": matches,
        "disclaimer": "Lưu ý: Bn `` nA'i v>i khAch r_ng mAnh ch% l y thA'ng tin t dA liu cA3 s_n."
    }


@tool("get_all_vehicles", args_schema=GetAllVehiclesArgs)
def get_all_vehicles() -> dict[str, Any]:
    """Lấy danh sách đầy đủ tất cả các dòng xe VinFast và thông số chi tiết."""
    return load_vehicles()


TOOLS = [
    get_vehicle_data,
    calculate_loan,
    find_showrooms,
    get_promotion,
    book_test_drive,
    find_charging_stations,
    get_all_vehicles,
]
