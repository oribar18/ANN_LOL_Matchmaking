import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib

from data_processing.games_data_processing import calculate_lineup_features, process_games_data, calculate_game_score

matplotlib.use('TkAgg')

# Define high-impact features based on the importance plots
HIGH_IMPACT_FEATURES = [
    'max_creeps_diff', 'max_kda_diff', 'maximal_kda_diff',
    'mean_creeps_diff', 'max_mmr_diff', 'maximal_mmr_diff',
    'mean_kda_diff', 'max_gold_diff'
]

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Train and evaluate a model."""
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate metrics
    rmse = mean_squared_error(y_test, predictions, squared=False)
    r_squared = r2_score(y_test, predictions)

    print(f"{model_name} - RMSE: {rmse:.4f}, R-squared: {r_squared:.4f}")

    return predictions, rmse, r_squared, model

def plot_results(y_test, predictions, model_name):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title(f"{model_name} - Actual vs Predicted")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.show()

def plot_feature_importance(features, importances, model_name):
    """Plot feature importances."""
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], alpha=0.7)
    plt.title(f'{model_name} Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.show()

def main():
    # Load and process data
    games_data = pd.read_csv('../data/games_data_raw_filtered.csv')
    games_data = process_games_data(games_data)
    scored_df = calculate_game_score(games_data)

    games_players = pd.read_csv("../data/games_players_data_filtered.csv")
    X_df = calculate_lineup_features(scored_df, games_players)

    # Ensure features are selected
    X = X_df[HIGH_IMPACT_FEATURES]
    y = scored_df['game_score']

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear Regression
    lr_model = LinearRegression()
    lr_predictions, lr_rmse, lr_r2, lr_model = evaluate_model(lr_model, X_train, X_test, y_train, y_test, "Linear Regression")
    plot_results(y_test, lr_predictions, "Linear Regression")

    # Random Forest
    rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
    rf_predictions, rf_rmse, rf_r2, rf_model = evaluate_model(rf_model, X_train, X_test, y_train, y_test, "Random Forest")
    plot_results(y_test, rf_predictions, "Random Forest")
    plot_feature_importance(HIGH_IMPACT_FEATURES, rf_model.feature_importances_, "Random Forest")

    # XGBoost
    xgb_model = XGBRegressor(random_state=42, n_estimators=100)
    xgb_predictions, xgb_rmse, xgb_r2, xgb_model = evaluate_model(xgb_model, X_train, X_test, y_train, y_test, "XGBoost")
    plot_results(y_test, xgb_predictions, "XGBoost")
    plot_feature_importance(HIGH_IMPACT_FEATURES, xgb_model.feature_importances_, "XGBoost")

    # Summary of results
    results = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
        "RMSE": [lr_rmse, rf_rmse, xgb_rmse],
        "R-squared": [lr_r2, rf_r2, xgb_r2]
    })

    print("\nModel Comparison:")
    print(results)

if __name__ == '__main__':
    main()
