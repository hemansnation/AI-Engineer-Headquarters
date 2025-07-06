import pandas as pd
import os

# ✅ FIXED: Use raw string (r"...") OR forward slashes
path = r"Artifacts\2025-07-02-00-03-42\data_ingestion\ingested"
# OR:
# path = "Artifacts/2025-07-01-23-46-06/data_ingestion/ingested"

for file in ["train.csv", "test.csv"]:
    fpath = os.path.join(path, file)
    df = pd.read_csv(fpath)
    print(f"\n📄 {file} columns: {df.columns.tolist()}")
