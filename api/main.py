# fastapi service for churn predictions
# loads the trained xgb model + shap explainer
# exposes POST /predict that returns probability + segment + top reasons

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="bank churn predictor")

# load everything once at startup
MODEL_DIR = Path(__file__).parent.parent / "models"

with open(MODEL_DIR / "xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

with open(MODEL_DIR / "feature_columns.pkl", "rb") as f:
    feature_cols = pickle.load(f)

with open(MODEL_DIR / "shap_explainer.pkl", "rb") as f:
    explainer = pickle.load(f)


# the request body shape - matches the raw features the model expects
# pydantic validates incoming json against this schema
class CustomerInput(BaseModel):
    customer_age: int
    gender: str
    dependent_count: int
    education_level: str
    marital_status: str
    income_category: str
    card_category: str
    months_on_book: int
    total_relationship_count: int
    months_inactive_12_mon: int
    contacts_count_12_mon: int
    credit_limit: float
    total_revolving_bal: float
    avg_open_to_buy: float
    total_amt_chng_q4_q1: float
    total_trans_amt: float
    total_trans_ct: int
    total_ct_chng_q4_q1: float
    avg_utilization_ratio: float


def engineer_features(customer: dict) -> pd.DataFrame:
    """recreate the sql feature engineering in python for one customer"""
    df = pd.DataFrame([customer])

    # age bucket
    age = df["customer_age"].iloc[0]
    df["age_bucket"] = (
        "young" if age < 30
        else "mid_career" if age < 45
        else "mature" if age < 60
        else "senior"
    )

    # tenure bucket
    months = df["months_on_book"].iloc[0]
    df["tenure_bucket"] = (
        "new" if months < 24
        else "mid" if months < 48
        else "longterm"
    )

    df["is_inactive"] = 1 if df["months_inactive_12_mon"].iloc[0] >= 3 else 0
    df["contact_escalated"] = 1 if df["contacts_count_12_mon"].iloc[0] >= 4 else 0

    util = df["avg_utilization_ratio"].iloc[0]
    df["utilization_bucket"] = (
        "unused" if util < 0.05
        else "light" if util < 0.30
        else "moderate" if util < 0.70
        else "high"
    )

    ratio = df["total_ct_chng_q4_q1"].iloc[0]
    df["trans_count_trend"] = (
        "sharp_decline" if ratio < 0.5
        else "moderate_decline" if ratio < 0.8
        else "stable" if ratio < 1.2
        else "growing"
    )

    trans_ct = df["total_trans_ct"].iloc[0]
    df["avg_spend_per_trans"] = round(df["total_trans_amt"].iloc[0] / trans_ct, 2) if trans_ct > 0 else 0

    months_book = df["months_on_book"].iloc[0]
    df["products_per_year"] = round(df["total_relationship_count"].iloc[0] / (months_book / 12.0), 3) if months_book > 0 else 0

    # one-hot encode the same way pandas did during training
    encoded = pd.get_dummies(df, drop_first=True)

    # add any missing columns the model expects, fill with 0
    for col in feature_cols:
        if col not in encoded.columns:
            encoded[col] = 0

    # match exact column order
    encoded = encoded[feature_cols]

    return encoded


def get_segment(prob: float, revenue: float) -> dict:
    """assign segment + action based on risk and value"""
    is_high_risk = prob >= 0.08

    # use rough revenue cutoffs based on the dataset percentiles
    if revenue >= 400:
        value = "high_value"
    elif revenue >= 100:
        value = "mid_value"
    else:
        value = "low_value"

    if is_high_risk and value == "high_value":
        action = "priority_call_premium_offer"
    elif is_high_risk and value == "mid_value":
        action = "email_retention_offer"
    elif is_high_risk and value == "low_value":
        action = "basic_email"
    elif not is_high_risk and value == "high_value":
        action = "loyalty_rewards_monitor"
    else:
        action = "no_action"

    return {
        "risk_level": "high_risk" if is_high_risk else "low_risk",
        "value_tier": value,
        "recommended_action": action,
    }


@app.get("/")
def root():
    return {"status": "ok", "service": "bank churn predictor"}


@app.post("/predict")
def predict(customer: CustomerInput):
    try:
        # turn the input into model-ready features
        features = engineer_features(customer.model_dump())

        # predict
        prob = float(model.predict_proba(features)[0][1])

        # rough revenue estimate (same formula as in sql)
        revenue = customer.total_trans_amt * 0.01 + customer.total_revolving_bal * 0.18

        # segment
        segment = get_segment(prob, revenue)

        # shap explanation - top 3 features driving this prediction
        shap_vals = explainer.shap_values(features)[0]
        feature_impacts = sorted(
            zip(feature_cols, shap_vals),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:3]
        top_factors = [
            {"feature": name, "contribution": round(float(val), 3)}
            for name, val in feature_impacts
        ]

        return {
            "churn_probability": round(prob, 4),
            "estimated_revenue": round(revenue, 2),
            **segment,
            "top_factors": top_factors,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))