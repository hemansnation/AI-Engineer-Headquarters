
import pandas as pd
import os

# Path to your train and test CSV files
path = "Artifacts/2025-07-01-23-54-35/data_ingestion/ingested"

for file in ["train.csv", "test.csv"]:
    fpath = os.path.join(path, file)
    df = pd.read_csv(fpath)

    # Rename "gender" to "sex" if needed
    if "gender" in df.columns:
        df.rename(columns={"gender": "sex"}, inplace=True)

    # Ensure column names match schema.yaml
    expected_columns = ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']
    if all(col in df.columns for col in expected_columns):
        df = df[expected_columns]  # Keep only required columns, in order
        df.to_csv(fpath, index=False)
        print(f"✅ {file} fixed.")
    else:
        print(f"⚠️ {file} is still missing required columns.")
