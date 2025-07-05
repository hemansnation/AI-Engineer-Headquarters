# 🏥 Insurance Claim Prediction - MLOps Project

This is an end-to-end machine learning project to **predict insurance charges** using MLOps best practices.  
It uses a modular, config-driven design with **Python, YAML, scikit-learn, and SQLite**.

---

## 📊 Objective

Predict **medical insurance charges** using features such as:

- Age  
- Gender  
- BMI  
- Number of children  
- Smoking status  
- Region  

---

## 📁 Project Structure

```
insurance-claim-prediction-mlops/
│
├── claim/                         # Core ML pipeline components
│   ├── components/               # Data ingestion, validation, transformation, training, evaluation
│   ├── config/                   # Parses YAML configs into Python objects
│   ├── entity/                   # Data classes for config and artifacts
│   ├── utils/                    # Logging, database utilities, file handling
│
├── data_config/                  # YAML configuration files for each pipeline stage
│   ├── data_ingestion.yaml
│   ├── data_validation.yaml
│   ├── model_trainer.yaml
│   └── model_evaluation.yaml
│
├── artifacts/                    # Stores generated artifacts like train/test files, models, logs
│
├── schema.yaml                   # Defines expected schema for the dataset
├── insurance_data.db             # SQLite database used for storing and querying raw data
│
├── main.py                       # Main pipeline script
├── run_pipeline.py               # Simplified entry point
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Containerization setup (optional)
├── README.md                     # Project documentation
└── .gitignore                    # Files to exclude from Git tracking
```

---

## ⚙️ YAML Configuration

The project uses YAML files under the `data_config/` directory to control parameters for each step:

- `data_ingestion.yaml`: source path, output locations, split ratio, database info  
- `data_validation.yaml`: schema file path  
- `model_trainer.yaml`: model type, hyperparameters  
- `model_evaluation.yaml`: metrics to track and threshold values  

Schema definition is stored in `schema.yaml`.

---

## 🗃️ SQLite Integration

- The project **always uses a local SQLite database** (`insurance_data.db`) as part of the data ingestion process.
- Raw data is inserted into a database table and later queried to create the **train/test datasets**.
- The database file path and table name are defined in the `data_ingestion.yaml` config file.
- This approach ensures reproducibility, storage, and traceability of raw data in a structured format.

---

## 🚀 How to Run

1. **Clone the repository** and navigate to the project directory:
   ```bash
   git clone <repo-url>
   cd insurance-claim-prediction-mlops
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # For Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the full ML pipeline**:
   ```bash
   python run_pipeline.py
   ```

Artifacts (train/test data, model, logs) will be saved under timestamped folders inside `artifacts/`.

