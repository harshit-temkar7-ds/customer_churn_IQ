"""
main.py  — ChurnIQ Intelligence API  v3.0
==========================================
Key improvements over v2:
  1. predict_proba instead of predict  →  continuous risk scores
  2. Revised thresholds: LOW < 0.30 | MEDIUM 0.30–0.65 | HIGH > 0.65
  3. Business-rule correction layer  →  fiber / no-security boost
  4. Feature engineering mirrored from retrain_model.py
  5. Detailed probability breakdown in response
  6. Audit log entry returned per prediction
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.dirname(BASE_DIR)
STATIC_DIR   = os.path.join(ROOT_DIR, "static")
TEMPLATE_DIR = os.path.join(ROOT_DIR, "templates")
MODEL_PATH   = os.path.join(ROOT_DIR, "models", "churn_pipeline.pkl")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="ChurnIQ Intelligence API", version="3.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATE_DIR)

# ── Load model artifact ───────────────────────────────────────────────────────
_artifact = joblib.load(MODEL_PATH)

if isinstance(_artifact, dict):
    model                = _artifact["pipeline"]
    NUMERIC_FEATURES     = _artifact["numeric_features"]
    CATEGORICAL_FEATURES = _artifact["categorical_features"]
else:
    model = _artifact
    NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    CATEGORICAL_FEATURES = [
        "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod",
    ]

logger.info("Model loaded  →  %s", type(model).__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  REQUEST SCHEMA
# ═══════════════════════════════════════════════════════════════════════════════
class CustomerData(BaseModel):
    gender: str           = Field("Male")
    SeniorCitizen: int    = Field(0)
    Partner: str          = Field("No")
    Dependents: str       = Field("No")
    tenure: int           = Field(12, ge=0, le=72)
    PhoneService: str     = Field("Yes")
    MultipleLines: str    = Field("No")
    InternetService: str  = Field("DSL")
    OnlineSecurity: str   = Field("No")
    OnlineBackup: str     = Field("No")
    DeviceProtection: str = Field("No")
    TechSupport: str      = Field("No")
    StreamingTV: str      = Field("No")
    StreamingMovies: str  = Field("No")
    Contract: str         = Field("Month-to-month")
    PaperlessBilling: str = Field("Yes")
    PaymentMethod: str    = Field("Electronic check")
    MonthlyCharges: float = Field(70.0, ge=0)
    TotalCharges: float   = Field(840.0, ge=0)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  PROBABILITY THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════
THRESHOLD_LOW  = 0.30   # below this  → LOW
THRESHOLD_HIGH = 0.65   # above this  → HIGH


def classify_risk(probability: float) -> str:
    if probability >= THRESHOLD_HIGH:
        return "High"
    if probability >= THRESHOLD_LOW:
        return "Medium"
    return "Low"


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  FEATURE ENGINEERING  (mirrors retrain_model.py exactly)
# ═══════════════════════════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fiber_no_security"] = (
        (df["InternetService"] == "Fiber optic") & (df["OnlineSecurity"] == "No")
    ).astype(int)
    df["fiber_no_support"] = (
        (df["InternetService"] == "Fiber optic") & (df["TechSupport"] == "No")
    ).astype(int)
    protective = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
    df["missing_protection_count"] = sum(
        (df[col] == "No").astype(int) for col in protective
    )
    df["charge_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)
    df["long_contract_risky"] = (
        (df["Contract"].isin(["One year", "Two year"])) & (df["fiber_no_security"] == 1)
    ).astype(int)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  BUSINESS-RULE CORRECTION LAYER
# ═══════════════════════════════════════════════════════════════════════════════
def apply_business_rules(
    raw_proba: float,
    data: CustomerData,
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Nudge ML probability with domain knowledge.
    Each rule adds/subtracts a small delta (max ±0.08).
    All adjustments are bounded to [0.05, 0.97] and returned for auditability.
    """
    adjustments: List[Dict[str, Any]] = []
    delta = 0.0

    # Rule 1: Fiber + no online security (strongest medium-risk signal)
    if data.InternetService == "Fiber optic" and data.OnlineSecurity == "No":
        boost = 0.06
        delta += boost
        adjustments.append({
            "rule": "fiber_no_security",
            "description": "Fiber optic + no Online Security",
            "adjustment": f"+{boost:.0%}",
        })

    # Rule 2: Fiber + no tech support
    if data.InternetService == "Fiber optic" and data.TechSupport == "No":
        boost = 0.04
        delta += boost
        adjustments.append({
            "rule": "fiber_no_support",
            "description": "Fiber optic + no Tech Support",
            "adjustment": f"+{boost:.0%}",
        })

    # Rule 3: Long contract masking multiple risk indicators
    risk_indicators = sum([
        data.InternetService == "Fiber optic",
        data.OnlineSecurity  == "No",
        data.TechSupport     == "No",
        data.OnlineBackup    == "No",
        data.MonthlyCharges  > 70,
    ])
    if data.Contract in ("One year", "Two year") and risk_indicators >= 3:
        boost = 0.05
        delta += boost
        adjustments.append({
            "rule": "long_contract_multiple_risk_factors",
            "description": f"Long contract masks {risk_indicators} risk indicators",
            "adjustment": f"+{boost:.0%}",
        })

    # Rule 4: Electronic check payment
    if data.PaymentMethod == "Electronic check" and raw_proba > 0.20:
        boost = 0.03
        delta += boost
        adjustments.append({
            "rule": "electronic_check",
            "description": "Electronic check payment (high-churn segment)",
            "adjustment": f"+{boost:.0%}",
        })

    # Rule 5: Strong retention signals → slight downward correction
    if (
        data.Contract       == "Two year"
        and data.tenure      > 24
        and data.OnlineSecurity == "Yes"
        and data.TechSupport    == "Yes"
    ):
        cut = -0.05
        delta += cut
        adjustments.append({
            "rule": "strong_retention",
            "description": "2-yr contract + long tenure + full security",
            "adjustment": f"{cut:.0%}",
        })

    adjusted = float(np.clip(raw_proba + delta, 0.05, 0.97))
    return adjusted, adjustments


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  RECOMMENDATIONS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
def build_recommendations(data: CustomerData, risk: str) -> List[Dict[str, Any]]:
    recs = []
    if data.Contract == "Month-to-month":
        recs.append({"icon": "📋", "title": "Contract Upgrade",
            "detail": "Offer 15% discount to switch to a 1- or 2-year contract.", "impact": "High"})
    if data.OnlineSecurity == "No" and data.InternetService != "No":
        recs.append({"icon": "🛡️", "title": "Free Security Trial",
            "detail": "Provide complimentary Online Security add-on for 3 months. "
                      "Fiber customers without security churn at nearly 2× the base rate.",
            "impact": "High" if data.InternetService == "Fiber optic" else "Medium"})
    if data.TechSupport == "No" and data.InternetService != "No":
        recs.append({"icon": "🔧", "title": "Tech Support Enrollment",
            "detail": "Offer a free 60-day Tech Support trial to increase stickiness.", "impact": "Medium"})
    if data.PaymentMethod == "Electronic check":
        recs.append({"icon": "💳", "title": "Auto-Pay Incentive",
            "detail": "Grant $5/month bill credit for switching to automatic payment.", "impact": "Low"})
    if data.MonthlyCharges > 80:
        recs.append({"icon": "💰", "title": "Loyalty Discount",
            "detail": "Apply a 10% loyalty discount on monthly charges immediately.", "impact": "High"})
    if data.tenure < 6:
        recs.append({"icon": "🤝", "title": "Retention Specialist",
            "detail": "Assign a dedicated onboarding specialist for early lifecycle support.", "impact": "High"})
    if not recs:
        recs.append({"icon": "✅", "title": "Stable Customer",
            "detail": "Customer shows strong retention signals. Schedule a check-in in 90 days.", "impact": "Low"})
    return recs


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  ROUTES
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(data: CustomerData):
    try:
        df_raw = pd.DataFrame([data.dict()])
        has_engineered = any(
            f in NUMERIC_FEATURES for f in ["fiber_no_security", "charge_per_tenure"]
        )
        df_input = engineer_features(df_raw) if has_engineered else df_raw
        df_input = df_input[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

        raw_proba = float(model.predict_proba(df_input)[0][1])
        adjusted_proba, applied_rules = apply_business_rules(raw_proba, data)
        risk = classify_risk(adjusted_proba)
        recs = build_recommendations(data, risk)

        annual_risk  = adjusted_proba * data.MonthlyCharges * 12
        clv_estimate = data.MonthlyCharges * max(1, 24 - data.tenure)

        audit = {
            "timestamp_utc":         datetime.now(timezone.utc).isoformat(),
            "raw_model_probability":  round(raw_proba, 4),
            "adjusted_probability":   round(adjusted_proba, 4),
            "probability_delta":      round(adjusted_proba - raw_proba, 4),
            "applied_rules":          applied_rules,
            "thresholds_used": {"low_below": THRESHOLD_LOW, "high_above": THRESHOLD_HIGH},
        }

        logger.info(
            "Prediction  raw=%.3f  adj=%.3f  risk=%s  rules=%d",
            raw_proba, adjusted_proba, risk, len(applied_rules),
        )

        return {
            "churn_probability":      round(adjusted_proba, 4),
            "risk_level":             risk,
            "annual_revenue_at_risk": round(annual_risk, 2),
            "recommendations":        recs,
            "clv_estimate":           round(clv_estimate, 2),
            "audit":                  audit,
        }

    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/stats")
async def api_stats():
    return {
        "model":   "CalibratedGradientBoosting ChurnIQ Pipeline",
        "version": "3.0.0",
        "features": len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES),
        "thresholds": {
            "low":    f"< {THRESHOLD_LOW:.0%}",
            "medium": f"{THRESHOLD_LOW:.0%} – {THRESHOLD_HIGH:.0%}",
            "high":   f"> {THRESHOLD_HIGH:.0%}",
        },
        "status": "operational",
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}
