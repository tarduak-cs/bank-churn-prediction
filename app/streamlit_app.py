# streamlit ui that calls the fastapi prediction service
# slider/dropdown inputs -> POST to /predict -> show prediction + segment + top factors

import streamlit as st
import requests
import pandas as pd
import os

API_URL = os.getenv("API_URL", "http://localhost:8000") + "/predict"

st.set_page_config(page_title="bank churn predictor", layout="wide")

st.title("bank customer churn predictor")
st.write(
    "predicts whether a credit card customer will close their account, "
    "and recommends a retention action based on their value to the bank."
)

st.divider()

# ----- inputs -----
left, right = st.columns(2)

with left:
    st.subheader("customer profile")
    customer_age = st.slider("age", 18, 80, 42)
    gender = st.selectbox("gender", ["M", "F"])
    dependent_count = st.slider("dependents", 0, 5, 2)
    education_level = st.selectbox(
        "education level",
        ["High School", "Graduate", "Uneducated", "Unknown", "College", "Post-Graduate", "Doctorate"]
    )
    marital_status = st.selectbox("marital status", ["Single", "Married", "Divorced", "Unknown"])
    income_category = st.selectbox(
        "income",
        ["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +", "Unknown"]
    )
    card_category = st.selectbox("card tier", ["Blue", "Silver", "Gold", "Platinum"])

with right:
    st.subheader("account behavior")
    months_on_book = st.slider("tenure (months)", 12, 60, 36)
    total_relationship_count = st.slider("# of products held", 1, 6, 3)
    months_inactive_12_mon = st.slider("months inactive (last year)", 0, 6, 1)
    contacts_count_12_mon = st.slider("customer service contacts (last year)", 0, 6, 2)
    credit_limit = st.number_input("credit limit ($)", 1500, 35000, 8000)
    total_revolving_bal = st.number_input("revolving balance ($)", 0, 2700, 1000)
    avg_open_to_buy = credit_limit - total_revolving_bal
    avg_utilization_ratio = round(total_revolving_bal / credit_limit, 3) if credit_limit > 0 else 0

    st.subheader("transactions")
    total_trans_amt = st.number_input("total spend (12 months, $)", 500, 20000, 4400)
    total_trans_ct = st.slider("transaction count (12 months)", 10, 140, 64)
    total_amt_chng_q4_q1 = st.slider("Q4/Q1 spend ratio", 0.0, 3.5, 1.0, 0.05)
    total_ct_chng_q4_q1 = st.slider("Q4/Q1 transaction count ratio", 0.0, 3.5, 1.0, 0.05)

st.divider()

# ----- predict button -----
if st.button("predict churn", type="primary", use_container_width=True):
    payload = {
        "customer_age": customer_age,
        "gender": gender,
        "dependent_count": dependent_count,
        "education_level": education_level,
        "marital_status": marital_status,
        "income_category": income_category,
        "card_category": card_category,
        "months_on_book": months_on_book,
        "total_relationship_count": total_relationship_count,
        "months_inactive_12_mon": months_inactive_12_mon,
        "contacts_count_12_mon": contacts_count_12_mon,
        "credit_limit": float(credit_limit),
        "total_revolving_bal": float(total_revolving_bal),
        "avg_open_to_buy": float(avg_open_to_buy),
        "total_amt_chng_q4_q1": float(total_amt_chng_q4_q1),
        "total_trans_amt": float(total_trans_amt),
        "total_trans_ct": total_trans_ct,
        "total_ct_chng_q4_q1": float(total_ct_chng_q4_q1),
        "avg_utilization_ratio": float(avg_utilization_ratio),
    }

    try:
        r = requests.post(API_URL, json=payload, timeout=10)
        r.raise_for_status()
        result = r.json()
    except Exception as e:
        st.error(f"API call failed: {e}")
        st.stop()

    # ----- results -----
    prob = result["churn_probability"]
    risk = result["risk_level"]
    value = result["value_tier"]
    action = result["recommended_action"]
    revenue = result["estimated_revenue"]
    factors = result["top_factors"]

    # big metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("churn probability", f"{prob:.1%}")
    m2.metric("risk level", risk.replace("_", " "))
    m3.metric("value tier", value.replace("_", " "))
    m4.metric("estimated revenue", f"${revenue:,.0f}")

    st.divider()

    # action callout
    if risk == "high_risk":
        st.error(f"recommended action: **{action.replace('_', ' ')}**")
    elif action == "loyalty_rewards_monitor":
        st.info(f"recommended action: **{action.replace('_', ' ')}**")
    else:
        st.success("low risk - no action needed")

    # top factors
    st.subheader("top factors driving this prediction")
    factors_df = pd.DataFrame(factors)
    factors_df["direction"] = factors_df["contribution"].apply(
        lambda x: "increases churn risk" if x > 0 else "decreases churn risk"
    )
    factors_df["contribution"] = factors_df["contribution"].abs()
    st.bar_chart(factors_df.set_index("feature")["contribution"])
    st.dataframe(factors_df, use_container_width=True, hide_index=True)