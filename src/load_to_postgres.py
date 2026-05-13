# load csv -> postgres
# kaggle bank churners dataset

from dotenv import load_dotenv
load_dotenv()

import os
import pandas as pd
from sqlalchemy import create_engine, text

db_url = os.getenv("DATABASE_URL")
if not db_url:
    raise ValueError("DATABASE_URL missing from .env")

engine = create_engine(db_url)

df = pd.read_csv("data/bank_churners.csv")
print(f"loaded {len(df):,} rows, {df.shape[1]} cols")

# the kaggle file ships with 2 naive bayes prediction columns at the end
# those are basically the answer key - using them = data leakage
leak_cols = [c for c in df.columns if "Naive_Bayes" in c]
if leak_cols:
    df = df.drop(columns=leak_cols)
    print(f"dropped leak cols: {leak_cols}")

# clientnum is a unique id, drop it so the model can't memorize ids
df = df.drop(columns=["CLIENTNUM"], errors="ignore")

print(df.dtypes)
print(df["Attrition_Flag"].value_counts())
print(f"churn rate: {(df['Attrition_Flag'] == 'Attrited Customer').mean():.2%}")

# postgres convention
df.columns = [c.lower() for c in df.columns]

# load into postgres
df.to_sql("customers_raw", engine, if_exists="replace", index=False, chunksize=1000)
print("loaded -> customers_raw")

# quick sanity check
with engine.connect() as conn:
    n = conn.execute(text("SELECT COUNT(*) FROM customers_raw")).scalar()
    print(f"row count: {n:,}")

    result = conn.execute(text("""
        SELECT attrition_flag, COUNT(*) AS n
        FROM customers_raw
        GROUP BY attrition_flag
    """))
    for row in result:
        print(f"  {row[0]}: {row[1]:,}")

    # check the behavioral signal looks right (churners should have lower trans counts)
    result = conn.execute(text("""
        SELECT
            attrition_flag,
            ROUND(AVG(total_trans_ct)::numeric, 1) AS avg_trans_ct,
            ROUND(AVG(months_inactive_12_mon)::numeric, 2) AS avg_inactive
        FROM customers_raw
        GROUP BY attrition_flag
    """))
    for row in result:
        print(f"  {row[0]}: trans_ct={row[1]}, inactive={row[2]}")