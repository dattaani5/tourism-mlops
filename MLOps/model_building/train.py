# for data manipulation
import os
import pandas as pd
# for building the preprocessing and modeling pipeline
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# for model serialization and experiment tracking
import joblib
import mlflow

# Local MLflow server started by the ngrok/MLflow cell (override with an env var if needed)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))   # FILLED
mlflow.set_experiment("Tourism_Package_Prediction")                                  # FILLED

# Train/test splits created by prep.py (adjust the folder if prep.py saves them elsewhere)
Xtrain = pd.read_csv("MLOps/data/Xtrain.csv")             # FILLED
Xtest = pd.read_csv("MLOps/data/Xtest.csv")               # FILLED
ytrain = pd.read_csv("MLOps/data/ytrain.csv").squeeze()   # FILLED
ytest = pd.read_csv("MLOps/data/ytest.csv").squeeze()     # FILLED

numeric_features = [   # FILLED
    "Age", "CityTier", "NumberOfPersonVisiting", "PreferredPropertyStar",
    "NumberOfTrips", "Passport", "OwnCar", "NumberOfChildrenVisiting",
    "MonthlyIncome", "PitchSatisfactionScore", "NumberOfFollowups", "DurationOfPitch"
]

categorical_features = [   # FILLED
    "TypeofContact", "Occupation", "Gender", "MaritalStatus",
    "Designation", "ProductPitched"
]

# Set the class weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)
# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 100],        # FILLED
    'xgbclassifier__max_depth': [3, 5],              # FILLED
    'xgbclassifier__colsample_bytree': [0.5, 0.7],   # FILLED
    'xgbclassifier__colsample_bylevel': [0.5, 0.7],  # FILLED
    'xgbclassifier__learning_rate': [0.05, 0.1],     # FILLED
    'xgbclassifier__reg_lambda': [0.5, 1],           # FILLED
}
# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)   # FILLED

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning with GridSearchCV
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log every parameter combination tried during the search as a nested run
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_test_score", results["mean_test_score"][i])
            mlflow.log_metric("std_test_score", results["std_test_score"][i])

    # Log the best hyperparameters in the main run
    mlflow.log_params(grid_search.best_params_)

    # Store the best model
    best_model = grid_search.best_estimator_

    # Set classification threshold
    classification_threshold = 0.45    # FILLED

    # Make predictions on the training and test data
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    # Evaluation
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log metrics
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    # Save the model next to app.py and log it as an MLflow artifact
    os.makedirs("MLOps/deployment", exist_ok=True)           # FILLED (creates the folder)
    model_path = "MLOps/deployment/model.joblib"             # FILLED
    joblib.dump(best_model, model_path)                      # FILLED
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved to {model_path}")
