import sqlite3
import os

db_path = os.path.join("data_config", "insurance_data.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create the table
cursor.execute("""
CREATE TABLE IF NOT EXISTS insurance_data (
    age INTEGER,
    gender TEXT,
    bmi REAL,
    children INTEGER,
    smoker TEXT,
    region TEXT,
    charges REAL
)
""")

# Insert sample data
cursor.executemany("""
INSERT INTO insurance_data (age, gender, bmi, children, smoker, region, charges)
VALUES (?, ?, ?, ?, ?, ?, ?)
""", [
    (19, "female", 27.9, 0, "yes", "southwest", 16884.92),
    (18, "male", 33.77, 1, "no", "southeast", 1725.55),
    (28, "male", 33.0, 3, "no", "southeast", 4449.46),
    (33, "male", 22.705, 0, "no", "northwest", 21984.47),
    (32, "male", 28.88, 0, "no", "northwest", 3866.85)
])

conn.commit()
conn.close()

print("✅ insurance_data table created and populated.")
