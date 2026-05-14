# Credit Card Customer Churn Prediction

Predicting which credit card customers will close their accounts, using the Kaggle BankChurners dataset. The goal isn't just prediction - it's figuring out who's worth retaining and what to do about each customer.

## Live demo

- **Try the app:** [tarduak-bank-churn.streamlit.app](https://tarduak-bank-churn.streamlit.app)
- **API docs:** [bank-churn-prediction-bxlx.onrender.com/docs](https://bank-churn-prediction-bxlx.onrender.com/docs)

The Streamlit app calls a FastAPI service hosted on Render. First request may take 30-60 seconds — Render's free tier puts the API to sleep after periods of inactivity, so the cold start adds a wake-up delay. Subsequent requests are fast.

## Why this project

Most churn portfolio projects stop at "I trained XGBoost and got 95% accuracy." I wanted to go further:
- Build features in SQL, not just pandas
- Compare three models honestly, not just pick the winner
- Use SHAP to explain why the model thinks what it thinks
- Pick a decision threshold based on actual dollars, not the default 0.5
- Group customers by both risk and value so the bank knows what to actually do

## Dataset

[BankChurners on Kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers). 10,127 customers, 20 features, ~16% churn rate. Synthetic benchmark - not real bank data - so the patterns are cleaner than they'd be in production.

The dataset ships with two pre-computed "Naive Bayes" prediction columns that will leak the answer if you don't drop them. I dropped them.

## Stack

- Postgres (free Neon instance) for data + SQL feature engineering
- Python: pandas, sqlalchemy, scikit-learn, xgboost, shap, matplotlib, seaborn
- Jupyter for the analysis notebook

## What I did

1. Loaded the CSV into Postgres
2. Wrote a SQL view (`customers_features`) that does feature engineering - bucketing, behavioral flags, percentile ranks via window functions
3. EDA in pandas to look for patterns
4. Trained Logistic Regression → Random Forest → XGBoost, comparing them properly with stratified train/test
5. SHAP analysis to figure out which features actually matter
6. Threshold optimization based on retention offer cost vs lifetime value
7. Customer segmentation matrix for retention strategy

## Results

| Model | ROC-AUC | Churn Recall | Churn Precision |
|---|---|---|---|
| Logistic Regression | 0.94 | 0.86 | 0.59 |
| Random Forest | 0.98 | 0.89 | 0.74 |
| **XGBoost** | **0.99** | **0.93** | **0.90** |

XGBoost wins on every metric. 93% of actual churners caught, 90% of flagged customers really were churners.

The 0.99 AUC is high enough to be suspicious - I checked for leakage by dropping the Naive Bayes columns and the customer ID, and separating segmentation features from prediction features. The number is real, it just reflects that this dataset has unusually clean signal compared to what real banks deal with.

## What drives churn

SHAP says the top features are all behavioral, not demographic:
- Total transactions and total spend (engagement level)
- Revolving balance carried (financial commitment to the card)
- Q4 vs Q1 ratios (whether the customer is slowing down)
- Months inactive

Age, gender, marital status, income - all near the bottom. The model isn't really predicting future churn. It's detecting customers who already started disengaging months ago.

## Threshold optimization

Assumed retention offer costs $25, average customer lifetime value is $800, retention offers work 40% of the time. Under those assumptions, saving a real churner is worth about 12x what a wasted offer costs.

The default 0.5 threshold gave ~$88,500 expected profit on the test set. Lowering to 0.08 gave ~$91,750 - about $3,200 better, or $1.58 per customer. Caught 98% of churners at that threshold instead of 93%.

## Customer segments

After flagging high-risk customers, I grouped them by estimated annual revenue:

| Risk | Value | Count | Action |
|---|---|---|---|
| High | High | 73 | Priority call |
| High | Mid | 110 | Personalized email |
| High | Low | 230 | Basic email |
| Low | High | 424 | Loyalty perks, monitor |
| Low | Mid/Low | 1,189 | Nothing |

The point: don't spend the same retention budget on every flagged customer. A $450/year customer is worth a phone call. A $40/year customer isn't.

## Honest limitations

- Synthetic dataset, not real bank data
- Only 12-month aggregate features, no transaction-level history
- The $25/$800/40% numbers are guesses, not measured
- No A/B test to verify the retention actions actually work
- A real deployment would need compliance review for fair-lending requirements

## Files

```
bank-churn-prediction/
├── data/
│   └── bank_churners.csv          raw kaggle data (gitignored)
├── sql/
│   ├── 01_create_schema.sql       drops views for clean re-runs
│   └── 02_feature_engineering.sql builds customers_features view
├── src/
│   └── load_to_postgres.py        csv -> postgres loader
├── notebooks/
│   └── 01_eda.ipynb               eda, modeling, shap, segmentation
├── .env                           db credentials (gitignored)
├── .gitignore
└── README.md
```
## To run it yourself

1. Get a free Postgres database on neon.tech
2. Put the connection string in a `.env` file as `DATABASE_URL=...`
3. Create a venv and install dependencies:
pip install pandas numpy sqlalchemy psycopg2-binary scikit-learn xgboost shap matplotlib seaborn jupyter python-dotenv

4. Download `BankChurners.csv` from Kaggle, save it to `data/bank_churners.csv`
5. `python src/load_to_postgres.py`
6. Run the two SQL files in your Neon SQL editor
7. Open the notebook and run all cells

## Things I'd add next

- Streamlit demo so the model is actually clickable
- FastAPI endpoint
- Proper hyperparameter tuning (I used defaults)