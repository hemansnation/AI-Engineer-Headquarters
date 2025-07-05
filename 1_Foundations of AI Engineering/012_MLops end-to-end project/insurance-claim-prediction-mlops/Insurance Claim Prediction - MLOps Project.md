# 🏥 Insurance Claim Prediction - MLOps Project

- This is an end-to-end machine learning project to **predict insurance charges** using MLOps best practices.
- It follows a modular, config-driven design using **Python, YAML, and scikit-learn**.
- The pipeline includes stages like **data ingestion, validation, transformation, model training, evaluation**, and prediction.

---

## 📂 What's Inside

- The project folder is `insurance-claim-prediction-mlops`.
- The main code is inside the `claim/` folder, organized as:
  - `components/`: handles each ML stage (e.g., ingestion, training)
  - `config/`: reads YAML configuration files
  - `entity/`: defines config and artifact classes
  - `utils/`: helper functions (logging, file saving, etc.)
- YAML configuration files are located in `data_config/`.
- Outputs (e.g., `train.csv`, `test.csv`, model files, logs) are saved in the `artifacts/` folder.

---

## 🚀 How to Run

1. **Clone the repository** and move into the project folder:
   ```bash
   git clone <repo-url>
   cd insurance-claim-prediction-mlops
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   ```

3. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the pipeline**:
   ```bash
   python run_pipeline.py
   ```

---

## 📊 What It Predicts

The model predicts **medical insurance charges** based on:

- Age  
- Gender  
- BMI  
- Number of children  
- Smoking status  
- Region  

Logs for each step and final output files will be available in the `Artifacts/` folder.





