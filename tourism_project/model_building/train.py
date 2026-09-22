import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://localhost:5000")      #  Setting the MLflow tracking URI
mlflow.set_experiment("Tourism_Project_ experiment")  # Setting the MLflow experiment name

# Xtrain/Xtest/ytrain/ytest are downloaded from the previous job's artifact
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()

# One-hot encode 'Type' and scale numeric features
numeric_features = ["Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome"]

categorical_features = ["TypeofContact", "Occupation", "Gender", "ProductPitched", "MaritalStatus",
    "Designation"]

# Setting the class weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Defining the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)
# Defining base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Defining hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [100, 200],        # Number of boosting trees. More trees can improve performance but increase training time.
    'xgbclassifier__max_depth': [3, 5],           # Maximum depth of each tree. Higher values increase model complexity and risk of overfitting.
    'xgbclassifier__colsample_bytree': [0.8, 1.0],    # Fraction of features sampled when building each tree.
    'xgbclassifier__colsample_bylevel': [0.8, 1.0],   # Fraction of features sampled at each tree level.
    'xgbclassifier__learning_rate': [0.05, 0.1],       # Step size used during boosting. Smaller values may improve generalization but require more trees.
    'xgbclassifier__reg_lambda': [1, 5],          # L2 regularization strength. Higher values help reduce overfitting.
}
# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)   # complete the code to build the model pipeline by chaining preprocessor and xgb_model

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning with GridSearchCV
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring="recall", n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)


    # Logging all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        param_set = results["params"][i]
        mean_score = results["mean_test_score"][i]
        std_score = results["std_test_score"][i]

        # Logging each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Logging the best hyperparameters in the main run
    mlflow.log_params(grid_search.best_params_)
    print("Best params:", grid_search.best_params_)

    # Storing the best model
    best_model = grid_search.best_estimator_

    # Setting classification threshold
    classification_threshold = 0.45

    # Make predictions on the training and test data
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    # Evaluation
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)
    print(classification_report(ytest, y_pred_test))

    # Logging metrics
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

    # Save the model next to app.py so the Streamlit app can load it directly,
    # and log it as an MLflow artifact for traceability
    model_path = "/content/tourism_project/deployment/best_tourism_package_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved to {model_path}")
