"""
test_medium_risk.py  — ChurnIQ  ·  Regression tests for medium-risk cases
==========================================================================
Validates the business-rule correction layer against the reported
misclassification cases M1 and M3 (and additional medium-risk scenarios).

Run: python test_medium_risk.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

import numpy as np

# ── Import the correction layer directly from main.py ─────────────────────────
# (We test the business rules without spinning up the full FastAPI server)
from main import apply_business_rules, classify_risk, CustomerData, THRESHOLD_LOW, THRESHOLD_HIGH

# ── Test case factory ─────────────────────────────────────────────────────────
def make_customer(**overrides) -> CustomerData:
    defaults = dict(
        gender="Male", SeniorCitizen=0, Partner="No", Dependents="No",
        tenure=12, PhoneService="Yes", MultipleLines="No",
        InternetService="DSL", OnlineSecurity="No", OnlineBackup="No",
        DeviceProtection="No", TechSupport="No", StreamingTV="No",
        StreamingMovies="No", Contract="Month-to-month",
        PaperlessBilling="Yes", PaymentMethod="Electronic check",
        MonthlyCharges=70.0, TotalCharges=840.0,
    )
    defaults.update(overrides)
    return CustomerData(**defaults)


# ── Simulate: what raw probability would the old model produce? ───────────────
# For cases M1/M3 the old model returned ~0.27 / ~0.20. We simulate that here
# and verify the rule layer corrects them into the medium-risk band.

def run_correction_test(
    name: str,
    simulated_raw_proba: float,
    customer: CustomerData,
    expected_risk: str,
    expected_min: float,
    expected_max: float,
):
    adjusted, rules = apply_business_rules(simulated_raw_proba, customer)
    risk = classify_risk(adjusted)

    passed = (expected_min <= adjusted <= expected_max) and (risk == expected_risk)
    status = "✅  PASS" if passed else "❌  FAIL"

    print(f"\n{status}  {name}")
    print(f"   Raw model probability : {simulated_raw_proba:.2%}")
    print(f"   Adjusted probability  : {adjusted:.2%}  (expected {expected_min:.0%}–{expected_max:.0%})")
    print(f"   Risk level            : {risk}  (expected {expected_risk})")
    print(f"   Rules applied         : {len(rules)}")
    for r in rules:
        print(f"     • {r['rule']}  {r['adjustment']}")

    return passed


# ── Define test cases ─────────────────────────────────────────────────────────
tests = [
    # Case M1 from the issue report
    dict(
        name="M1 — 1yr contract, tenure=18, fiber, no security",
        simulated_raw_proba=0.27,          # what old model returned (LOW)
        customer=make_customer(
            Contract="One year", tenure=18, MonthlyCharges=64,
            InternetService="Fiber optic", OnlineSecurity="No", TechSupport="On",
        ),
        expected_risk="Medium",
        expected_min=0.30,
        expected_max=0.65,
    ),
    # Case M3 from the issue report
    dict(
        name="M3 — 2yr contract, tenure=30, fiber, no security",
        simulated_raw_proba=0.20,          # what old model returned (LOW)
        customer=make_customer(
            Contract="Two year", tenure=30, MonthlyCharges=73,
            InternetService="Fiber optic", OnlineSecurity="No", TechSupport="On",
        ),
        expected_risk="Medium",
        expected_min=0.30,
        expected_max=0.65,
    ),
    # Additional medium-risk case: fiber + no security + no backup
    dict(
        name="M4 — Month-to-month, fiber, no security, no backup",
        simulated_raw_proba=0.40,
        customer=make_customer(
            Contract="Month-to-month", tenure=10, MonthlyCharges=75,
            InternetService="Fiber optic", OnlineSecurity="No",
            OnlineBackup="No", TechSupport="No",
        ),
        expected_risk="Medium",
        expected_min=0.30,
        expected_max=0.65,
    ),
    # Strong retention — should NOT be pushed to medium
    dict(
        name="LOW-1 — 2yr contract, long tenure, full security",
        simulated_raw_proba=0.22,
        customer=make_customer(
            Contract="Two year", tenure=36, MonthlyCharges=55,
            InternetService="DSL", OnlineSecurity="Yes", TechSupport="Yes",
            PaymentMethod="Bank transfer (automatic)",
        ),
        expected_risk="Low",
        expected_min=0.00,
        expected_max=0.29,
    ),
    # Clear high-risk — should remain HIGH
    dict(
        name="HIGH-1 — Month-to-month, fiber, no security, electronic check",
        simulated_raw_proba=0.72,
        customer=make_customer(
            Contract="Month-to-month", tenure=3, MonthlyCharges=90,
            InternetService="Fiber optic", OnlineSecurity="No",
            PaymentMethod="Electronic check",
        ),
        expected_risk="High",
        expected_min=0.65,
        expected_max=1.00,
    ),
]

# ── Run ───────────────────────────────────────────────────────────────────────
print("=" * 60)
print("ChurnIQ  ·  Medium-Risk Regression Test Suite")
print(f"Thresholds: LOW < {THRESHOLD_LOW:.0%}  |  MEDIUM {THRESHOLD_LOW:.0%}–{THRESHOLD_HIGH:.0%}  |  HIGH > {THRESHOLD_HIGH:.0%}")
print("=" * 60)

results = [run_correction_test(**t) for t in tests]

print("\n" + "=" * 60)
passed = sum(results)
total  = len(results)
print(f"Results: {passed}/{total} tests passed")

if passed == total:
    print("🎉  All tests passed — business-rule layer is working correctly.")
else:
    print("⚠️   Some tests failed — review rule deltas in main.py.")
    sys.exit(1)
