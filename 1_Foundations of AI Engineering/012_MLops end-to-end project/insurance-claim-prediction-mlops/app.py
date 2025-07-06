from flask import Flask, render_template, request
import pandas as pd
import os
import numpy as np
from claim.utils import load_object
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load test data and preprocess
df = pd.read_csv("test_df.csv").drop(columns=["is_claim"], axis=1)
columns = df.columns.tolist()

# Detect data types for each column
def get_input_type(col):
    if df[col].nunique() < 10:
        return "select", df[col].dropna().unique().tolist()
    elif pd.api.types.is_numeric_dtype(df[col]):
        return "float", None
    elif "date" in col.lower():
        return "date", None
    else:
        return "text", None

@app.route("/", methods=["GET"])
def index():
    fields = []
    for col in columns:
        input_type, options = get_input_type(col)
        fields.append({
            "name": col,
            "type": input_type,
            "options": options
        })

    return render_template("base.html", fields=fields)

@app.route("/submit", methods=["POST"])
def submit():
    submitted = pd.DataFrame({field: [request.form.get(field)] for field in df.columns})

    # Manual mappings
    binary_map = {'Yes': 1, 'No': 0}
    submitted["is_esc"] = submitted["is_esc"].map(binary_map)
    submitted["is_adjustable_steering"] = submitted["is_adjustable_steering"].map(binary_map)
    submitted["is_parking_sensors"] = submitted["is_parking_sensors"].map(binary_map)
    submitted["is_parking_camera"] = submitted["is_parking_camera"].map(binary_map)
    submitted["transmission_type"] = submitted["transmission_type"].map({'Manual': 1, 'Automatic': 0})
    submitted["is_brake_assist"] = submitted["is_brake_assist"].map(binary_map)
    submitted["is_central_locking"] = submitted["is_central_locking"].map(binary_map)
    submitted["is_power_steering"] = submitted["is_power_steering"].map(binary_map)

    submitted["segment"] = submitted["segment"].map({'A': 1, 'B2': 2, 'C2': 3, 'Others': 4})
    submitted["model"] = submitted["model"].map({'M1': 1, 'M4': 2, 'M6': 3, 'Others': 4})
    submitted["fuel_type"] = submitted["fuel_type"].map({'CNG': 1, 'Petrol': 2, 'Diesel': 3})
    submitted = submitted.drop(["segment", "model", "fuel_type"], axis=1)

    # Load preprocessor and model
    preprocessor = load_object("final_models/preprocessor.pkl")
    model = load_object("final_models/model.pkl")

    data = preprocessor.transform(submitted)

    prediction = model.predict(data)[0]
    submitted["is_claim"] = "Confirmed" if prediction == 1 else "Not Confirmed"

    return f"<h2>Prediction Result:</h2><pre>{submitted['is_claim'][0]}</pre>"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
