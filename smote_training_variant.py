"""
smote_training_variant.py  — ChurnIQ  ·  Class-balancing with SMOTE
====================================================================
Use this variant if you want to try SMOTE oversampling in addition
to (or instead of) class_weight="balanced".

Prerequisites:
    pip install imbalanced-learn

SMOTE vs class_weight — when to choose which
--------------------------------------------
class_weight="balanced"
    • Works within sklearn Pipeline seamlessly
    • No data augmentation — just reweights the loss
    • Safe default; always try this first

SMOTE (Synthetic Minority Oversampling)
    • Generates synthetic minority-class samples
    • Can improve recall on the minority (churn) class
    • Must be applied AFTER preprocessing (on numeric data)
    • Cannot be placed inside a standard sklearn Pipeline —
      use imblearn's Pipeline instead
    • Risk: can overfit if folds are not stratified properly
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# Optional SMOTE import — graceful fallback
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline  # imblearn-aware pipeline
    SMOTE_AVAILABLE = True
    print("✅  imbalanced-learn available — SMOTE will be used")
except ImportError:
    from sklearn.pipeline import Pipeline
    SMOTE_AVAILABLE = False
    print("⚠️  imbalanced-learn not installed — falling back to class_weight='balanced'")
    print("    Install with: pip install imbalanced-learn")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Telco-Customer-Churn.csv")

# ── Load & prep (same as retrain_model.py) ─────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df.drop(columns=["customerID"], errors="ignore", inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)
df["Churn"] = (df["Churn"] == "Yes").astype(int)

# Feature engineering
df["fiber_no_security"]       = ((df["InternetService"] == "Fiber optic") & (df["OnlineSecurity"] == "No")).astype(int)
df["fiber_no_support"]        = ((df["InternetService"] == "Fiber optic") & (df["TechSupport"] == "No")).astype(int)
df["missing_protection_count"]= sum((df[c] == "No").astype(int) for c in ["OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport"])
df["charge_per_tenure"]       = df["MonthlyCharges"] / (df["tenure"] + 1)
df["long_contract_risky"]     = ((df["Contract"].isin(["One year","Two year"])) & (df["fiber_no_security"] == 1)).astype(int)

NUMERIC_FEATURES = [
    "SeniorCitizen","tenure","MonthlyCharges","TotalCharges",
    "fiber_no_security","fiber_no_support","missing_protection_count",
    "charge_per_tenure","long_contract_risky",
]
CATEGORICAL_FEATURES = [
    "gender","Partner","Dependents","PhoneService","MultipleLines",
    "InternetService","OnlineSecurity","OnlineBackup","DeviceProtection",
    "TechSupport","StreamingTV","StreamingMovies","Contract",
    "PaperlessBilling","PaymentMethod",
]

X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ── Preprocessor ───────────────────────────────────────────────────────────────
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC_FEATURES),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
])

# ── Estimator ──────────────────────────────────────────────────────────────────
# When SMOTE handles the class balance, we do NOT also set class_weight="balanced"
# — that would double-correct and bias toward the minority class.
base_gb = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    min_samples_leaf=20,
    max_features="sqrt",
    random_state=42,
)

calibrated_gb = CalibratedClassifierCV(base_gb, method="isotonic", cv=5)

# ── Pipeline (with or without SMOTE) ──────────────────────────────────────────
if SMOTE_AVAILABLE:
    # SMOTE must come AFTER preprocessing so it works on numeric arrays.
    # imblearn's Pipeline supports this natively.
    smote = SMOTE(
        sampling_strategy=0.5,  # upsample minority to 50% of majority count
        k_neighbors=5,
        random_state=42,
    )
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("smote",         smote),         # ← only applied during fit, not predict
        ("classifier",    calibrated_gb),
    ])
    print("\nPipeline: Preprocessor → SMOTE → CalibratedGradientBoosting")
else:
    # Fallback: class_weight inside a plain sklearn pipeline
    base_gb_balanced = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.8, min_samples_leaf=20, max_features="sqrt",
        random_state=42,
        # GradientBoosting doesn't have class_weight; use sample_weight at fit time
    )
    from sklearn.pipeline import Pipeline as SklearnPipeline
    from sklearn.utils.class_weight import compute_sample_weight

    pipeline = SklearnPipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier",   CalibratedClassifierCV(base_gb_balanced, method="isotonic", cv=5)),
    ])
    print("\nPipeline: Preprocessor → CalibratedGradientBoosting (class_weight via sample_weight)")

# ── Train ──────────────────────────────────────────────────────────────────────
print("Training...")
if not SMOTE_AVAILABLE:
    sample_weights = compute_sample_weight("balanced", y_train)
    pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weights)
else:
    pipeline.fit(X_train, y_train)

# ── Evaluate ───────────────────────────────────────────────────────────────────
y_pred  = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("\n── Classification Report ──────────────────────────────")
print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

medium_mask = (y_proba >= 0.30) & (y_proba <= 0.65)
if medium_mask.sum() > 0:
    print(f"\nMedium-risk band (0.30–0.65): {medium_mask.sum()} samples")
    print(f"  Actual churn rate in band  : {y_test[medium_mask].mean():.2%}")

print("\n✅  Training complete. Run retrain_model.py to save as the active model.")
