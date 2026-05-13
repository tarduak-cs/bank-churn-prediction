-- builds customers_features view from customers_raw
-- adds derived cols for the model + segmentation

CREATE OR REPLACE VIEW customers_features AS

WITH base AS (
    -- target -> 1/0, rest passes through
    SELECT
        CASE WHEN attrition_flag = 'Attrited Customer' THEN 1 ELSE 0 END AS churn,
        customer_age,
        gender,
        dependent_count,
        education_level,
        marital_status,
        income_category,
        card_category,
        months_on_book,
        total_relationship_count,
        months_inactive_12_mon,
        contacts_count_12_mon,
        credit_limit,
        total_revolving_bal,
        avg_open_to_buy,
        avg_utilization_ratio,
        total_trans_amt,
        total_trans_ct,
        total_amt_chng_q4_q1,
        total_ct_chng_q4_q1
    FROM customers_raw
),

engineered AS (
    SELECT
        *,

        CASE
            WHEN customer_age < 30 THEN 'young'
            WHEN customer_age < 45 THEN 'mid_career'
            WHEN customer_age < 60 THEN 'mature'
            ELSE 'senior'
        END AS age_bucket,

        CASE
            WHEN months_on_book < 24 THEN 'new'
            WHEN months_on_book < 48 THEN 'mid'
            ELSE 'longterm'
        END AS tenure_bucket,

        -- 3+ inactive months = probably done
        CASE WHEN months_inactive_12_mon >= 3 THEN 1 ELSE 0 END AS is_inactive,

        -- lots of service contacts usually = unresolved complaints
        CASE WHEN contacts_count_12_mon >= 4 THEN 1 ELSE 0 END AS contact_escalated,

        -- low util = card unused, high util = financial stress, different stories
        CASE
            WHEN avg_utilization_ratio < 0.05 THEN 'unused'
            WHEN avg_utilization_ratio < 0.30 THEN 'light'
            WHEN avg_utilization_ratio < 0.70 THEN 'moderate'
            ELSE 'high'
        END AS utilization_bucket,

        -- q4/q1 < 0.5 means txn count roughly halved
        CASE
            WHEN total_ct_chng_q4_q1 < 0.5 THEN 'sharp_decline'
            WHEN total_ct_chng_q4_q1 < 0.8 THEN 'moderate_decline'
            WHEN total_ct_chng_q4_q1 < 1.2 THEN 'stable'
            ELSE 'growing'
        END AS trans_count_trend,

        ROUND(
            (total_trans_amt::numeric / NULLIF(total_trans_ct, 0))::numeric,
            2
        ) AS avg_spend_per_trans,

        ROUND(
            (total_relationship_count::numeric / NULLIF(months_on_book / 12.0, 0))::numeric,
            3
        ) AS products_per_year,

        -- below cols are for segmentation, model won't see them
        -- rough revenue: 1% interchange + 18% apr on revolving balance
        ROUND(
            (total_trans_amt * 0.01 + total_revolving_bal * 0.18)::numeric,
            2
        ) AS est_annual_revenue,

        CASE
            WHEN card_category IN ('Platinum', 'Gold') THEN 1
            ELSE 0
        END AS is_premium_card

    FROM base
),

ranked AS (
    SELECT
        *,
        ROUND(
            PERCENT_RANK() OVER (ORDER BY total_trans_amt)::numeric,
            3
        ) AS revenue_percentile,

        ROUND(
            PERCENT_RANK() OVER (ORDER BY credit_limit)::numeric,
            3
        ) AS credit_limit_percentile,

        CASE
            WHEN PERCENT_RANK() OVER (ORDER BY est_annual_revenue) >= 0.75 THEN 'high_value'
            WHEN PERCENT_RANK() OVER (ORDER BY est_annual_revenue) >= 0.25 THEN 'mid_value'
            ELSE 'low_value'
        END AS value_tier

    FROM engineered
)

SELECT
    churn,
    customer_age, age_bucket,
    gender, dependent_count, education_level, marital_status, income_category, card_category,
    months_on_book, tenure_bucket,
    total_relationship_count, products_per_year,
    months_inactive_12_mon, is_inactive,
    contacts_count_12_mon, contact_escalated,
    credit_limit, total_revolving_bal, avg_open_to_buy,
    avg_utilization_ratio, utilization_bucket,
    total_trans_amt, total_trans_ct, avg_spend_per_trans,
    total_amt_chng_q4_q1, total_ct_chng_q4_q1, trans_count_trend,
    est_annual_revenue, revenue_percentile, credit_limit_percentile,
    value_tier, is_premium_card
FROM ranked;