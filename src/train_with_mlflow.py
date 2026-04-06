import matplotlib
matplotlib.use('Agg')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


EXPERIMENT_NAME = "box-office-predictor"

# model configs — same as train_model.py so results are reproducible
MODELS = {
    "linear_regression": LinearRegression(),
    "ridge_regression": Ridge(alpha=1.0),
    "random_forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    ),
    "xgboost": XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
    ),
}

# hyperparams to log per model
HPARAMS = {
    "linear_regression": {},
    "ridge_regression": {"alpha": 1.0},
    "random_forest": {
        "n_estimators": 300,
        "max_depth": 20,
        "min_samples_split": 3,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    },
    "xgboost": {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
    },
}


def load_data():
    X = pd.read_csv("data/processed/X_features.csv")
    y = pd.read_csv("data/processed/y_target.csv").squeeze()
    return X, y


def save_importance_plot(model, feature_names, model_name, out_path):
    # save feature importance chart
    if not hasattr(model, "feature_importances_") and not hasattr(model, "coef_"):
        return None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = np.abs(model.coef_)

    top_n = 20
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=imp_df, x="importance", y="feature", palette="viridis")
    plt.title(f"top {top_n} features — {model_name}")
    plt.xlabel("importance")
    plt.ylabel("feature")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def run_experiment():
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"train/test split: {len(X_train)} / {len(X_test)}\n")

    best_r2 = -np.inf
    best_model_name = None
    best_model_obj = None

    os.makedirs("models", exist_ok=True)

    for name, model in MODELS.items():
        print(f"training {name}...")

        with mlflow.start_run(run_name=name):
            # log hyperparams
            mlflow.log_params(HPARAMS[name])
            mlflow.log_param("model_type", name)
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            mlflow.log_param("n_features", X.shape[1])

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # metrics in log space
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            # metrics in dollar space
            y_test_actual = np.expm1(y_test)
            y_pred_actual = np.expm1(y_pred)
            rmse_dollars = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
            mae_dollars = mean_absolute_error(y_test_actual, y_pred_actual)

            mlflow.log_metrics({
                "r2": r2,
                "rmse_log": rmse,
                "mae_log": mae,
                "rmse_dollars": rmse_dollars,
                "mae_dollars": mae_dollars,
            })

            print(f"  r2={r2:.4f}  rmse={rmse:.4f}  mae={mae:.4f}")

            # feature importance plot as artifact
            plot_path = f"models/{name}_importance.png"
            result = save_importance_plot(model, list(X.columns), name, plot_path)
            if result:
                mlflow.log_artifact(result)

            # log model artifact
            mlflow.sklearn.log_model(model, artifact_path="model")

            if r2 > best_r2:
                best_r2 = r2
                best_model_name = name
                best_model_obj = model

    # save the winner
    print(f"\nbest model: {best_model_name} (r2={best_r2:.4f})")
    joblib.dump(best_model_obj, "models/best_model.joblib")

    # one final run just to tag which model won
    with mlflow.start_run(run_name="best_model_summary"):
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_metric("best_r2", best_r2)
        mlflow.log_artifact("models/best_model.joblib")

    print("done — check mlflow ui at http://localhost:5000")


if __name__ == "__main__":
    run_experiment()
