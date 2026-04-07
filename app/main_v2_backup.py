from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

# ─── Resolve absolute paths so the app works regardless of CWD ───────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.dirname(BASE_DIR)
STATIC_DIR   = os.path.join(ROOT_DIR, "static")
TEMPLATE_DIR = os.path.join(ROOT_DIR, "templates")
MODEL_PATH   = os.path.join(ROOT_DIR, "models", "churn_pipeline.pkl")

app = FastAPI(title="ChurnIQ Intelligence API", version="2.0.0")

# Mount static files & templates
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATE_DIR)

# Load model once at startup
model = joblib.load(MODEL_PATH)


# ─── Request Schema ────────────────────────────────────────────────────────────
class CustomerData(BaseModel):
    gender: str = "Male"
    SeniorCitizen: int = 0
    Partner: str = "No"
    Dependents: str = "No"
    tenure: int = 12
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "DSL"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 70.0
    TotalCharges: float = 840.0


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(data: CustomerData):
    df = pd.DataFrame([data.dict()])
    proba = float(model.predict_proba(df)[0][1])

    # Risk tier
    if proba > 0.6:
        risk = "High"
    elif proba > 0.4:
        risk = "Medium"
    else:
        risk = "Low"

    # Rule-based recommendations
    recs = []
    if data.Contract == "Month-to-month":
        recs.append({
            "icon": "📋",
            "title": "Contract Upgrade",
            "detail": "Offer 15% discount to switch to a 1-year or 2-year contract.",
            "impact": "High"
        })
    if data.OnlineSecurity == "No":
        recs.append({
            "icon": "🛡️",
            "title": "Free Security Trial",
            "detail": "Provide complimentary Online Security add-on for 3 months.",
            "impact": "Medium"
        })
    if data.TechSupport == "No":
        recs.append({
            "icon": "🔧",
            "title": "Tech Support Enrollment",
            "detail": "Offer a free 60-day Tech Support trial to increase stickiness.",
            "impact": "Medium"
        })
    if data.PaymentMethod == "Electronic check":
        recs.append({
            "icon": "💳",
            "title": "Auto-Pay Incentive",
            "detail": "Grant $5/month bill credit for switching to automatic payment.",
            "impact": "Low"
        })
    if data.MonthlyCharges > 80:
        recs.append({
            "icon": "💰",
            "title": "Loyalty Discount",
            "detail": "Apply a 10% loyalty discount on monthly charges immediately.",
            "impact": "High"
        })
    if data.tenure < 6:
        recs.append({
            "icon": "🤝",
            "title": "Retention Specialist",
            "detail": "Assign a dedicated onboarding specialist for early lifecycle support.",
            "impact": "High"
        })
    if data.InternetService == "DSL" and (data.StreamingTV == "No" or data.StreamingMovies == "No"):
        recs.append({
            "icon": "📡",
            "title": "Fiber Upgrade",
            "detail": "Offer fiber optic upgrade with a 30-day free trial of streaming bundle.",
            "impact": "Medium"
        })

    if not recs:
        recs.append({
            "icon": "✅",
            "title": "Stable Customer",
            "detail": "Customer shows strong retention signals. Schedule a check-in call in 90 days.",
            "impact": "Low"
        })

    annual_risk = proba * data.MonthlyCharges * 12

    return {
        "churn_probability": round(proba, 4),
        "risk_level": risk,
        "annual_revenue_at_risk": round(annual_risk, 2),
        "recommendations": recs,
        "clv_estimate": round(data.MonthlyCharges * max(1, 24 - data.tenure), 2)
    }


@app.get("/api/stats")
async def api_stats():
    """Dashboard summary stats — used by the frontend."""
    return {
        "model": "GradientBoosting Churn Pipeline",
        "version": "2.0.0",
        "features": 19,
        "status": "operational"
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}
