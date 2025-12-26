"""
train_model.py
Train and evaluate regression models for box office prediction.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - must be before pyplot import

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor


def load_modeling_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the preprocessed feature matrix and target."""
    X = pd.read_csv("data/processed/X_features.csv")
    y = pd.read_csv("data/processed/y_target.csv").squeeze()
    return X, y


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Calculate evaluation metrics for a model."""
    y_pred = model.predict(X_test)
    
    metrics = {
        "model": model_name,
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }
    
    # Convert back from log scale for interpretability
    y_test_actual = np.expm1(y_test)
    y_pred_actual = np.expm1(y_pred)
    metrics["rmse_dollars"] = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    metrics["mae_dollars"] = mean_absolute_error(y_test_actual, y_pred_actual)
    
    return metrics


def plot_feature_importance(model, feature_names: list, model_name: str, top_n: int = 20):
    """Plot top N feature importances."""
    
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        print(f"Cannot extract feature importances for {model_name}")
        return
    
    # Create dataframe and sort
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df, x="importance", y="feature", palette="viridis")
    plt.title(f"Top {top_n} Feature Importances - {model_name}")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(f"models/{model_name.lower().replace(' ', '_')}_importance.png", dpi=150)
    plt.close()
    print(f"Saved feature importance plot for {model_name}")


def plot_predictions(y_test, y_pred, model_name: str):
    """Plot actual vs predicted values."""
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.5, edgecolors="none")
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction")
    
    plt.xlabel("Actual Log Revenue")
    plt.ylabel("Predicted Log Revenue")
    plt.title(f"Actual vs Predicted - {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"models/{model_name.lower().replace(' ', '_')}_predictions.png", dpi=150)
    plt.close()


def train_and_compare_models(X_train, X_test, y_train, y_test, feature_names: list) -> pd.DataFrame:
    """Train multiple models and compare their performance."""
    
    # Scale features for linear models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models to compare
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=42,
            n_jobs=-1
        ),
    }
    
    results = []
    best_model = None
    best_r2 = -np.inf
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Use scaled data for linear models
        if "Regression" in name:
            model.fit(X_train_scaled, y_train)
            metrics = evaluate_model(model, X_test_scaled, y_test, name)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test, name)
            y_pred = model.predict(X_test)
        
        results.append(metrics)
        
        print(f"  RMSE (log): {metrics['rmse']:.4f}")
        print(f"  MAE (log): {metrics['mae']:.4f}")
        print(f"  R² Score: {metrics['r2']:.4f}")
        print(f"  RMSE ($): ${metrics['rmse_dollars']:,.0f}")
        
        # Track best model
        if metrics["r2"] > best_r2:
            best_r2 = metrics["r2"]
            best_model = (name, model)
        
        # Plot predictions
        plot_predictions(y_test, y_pred, name)
        
        # Plot feature importance (for tree-based models)
        if "Regression" not in name:
            plot_feature_importance(model, feature_names, name)
    
    # Save best model
    print(f"\nBest model: {best_model[0]} with R² = {best_r2:.4f}")
    joblib.dump(best_model[1], "models/best_model.joblib")
    joblib.dump(best_model[1], f"models/best_model_{best_model[0].lower().replace(' ', '_')}.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    
    return pd.DataFrame(results)


def hyperparameter_tuning(X_train, y_train) -> XGBRegressor:
    """Perform grid search on XGBoost for best hyperparameters."""
    
    print("\nPerforming hyperparameter tuning on XGBoost...")
    
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.8, 1.0],
    }
    
    xgb = XGBRegressor(random_state=42, n_jobs=-1)
    
    grid_search = GridSearchCV(
        xgb, 
        param_grid, 
        cv=5, 
        scoring="r2",
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV R² score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def main():
    """Main training pipeline."""
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Load data
    print("Loading data...")
    X, y = load_modeling_data()
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Compare models
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    
    results_df = train_and_compare_models(
        X_train, X_test, y_train, y_test, 
        feature_names=list(X.columns)
    )
    
    # Save results
    results_df.to_csv("models/model_comparison.csv", index=False)
    
    # Print final comparison table
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(results_df[["model", "rmse", "mae", "r2"]].to_string(index=False))
    
    # Optional: Hyperparameter tuning
    # Uncomment below if you want to run grid search (takes longer)
    # best_xgb = hyperparameter_tuning(X_train, y_train)
    # joblib.dump(best_xgb, "models/xgboost_tuned.pkl")


if __name__ == "__main__":
    main()
