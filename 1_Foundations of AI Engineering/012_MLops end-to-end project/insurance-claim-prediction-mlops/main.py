from model_trainer import ModelTrainer
from model_evaluation import ModelEvaluator
from model_prediction import ModelPredictor


def main():
    # Training
    trainer = ModelTrainer()
    model, score = trainer.train_model()

    # Save model
    import joblib, os
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")

    # Evaluation
    evaluator = ModelEvaluator(model_path="models/model.pkl")
    evaluator.evaluate_model()

    # Prediction
    predictor = ModelPredictor(model_path="models/model.pkl")
    predictions = predictor.predict()
    print("✅ Predictions Sample:", predictions[:5])


if __name__ == "__main__":
    main()
