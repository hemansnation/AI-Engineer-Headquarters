import sqlite3
import os

db_path = os.path.join("data_config", "insurance_data.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Show table names
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print("Tables:", cursor.fetchall())

# Peek at the first few rows
print("\nSample data from 'insurance_data':")
rows = cursor.execute("SELECT * FROM insurance_data LIMIT 5;")
for row in rows:
    print(row)

conn.close()
