import pandas as pd
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


def plot_results(y_test, predictions, model_name):
    """
    Plot actual vs predicted values for a given model.

    Args:
        y_test (array-like): True target values.
        predictions (array-like): Predicted target values by the model.
        model_name (str): Name of the model being plotted.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title(f"{model_name} - Actual vs Predicted")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.show()


def train_and_evaluate_model(X, y, model, model_name):
    """
    Train a machine learning model and evaluate its performance.

    Args:
        X (DataFrame): Feature set for training and testing.
        y (Series): Target variable for training and testing.
        model (object): The machine learning model to train.
        model_name (str): Name of the model being trained.

    Returns:
        tuple: Predictions, true test values, and the trained model.
    """
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate metrics
    rmse = mean_squared_error(y_test, predictions, squared=False)
    r_squared = model.score(X_test, y_test)

    print(f"{model_name} - RMSE: {rmse:.4f}, R-squared: {r_squared:.4f}")

    return predictions, y_test, model


def plot_feature_importance(features, importances, model_name):
    """
    Plot feature importance for a trained model.

    Args:
        features (list): List of feature names.
        importances (list): Corresponding feature importance values.
        model_name (str): Name of the model.
    """
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], alpha=0.7)
    plt.title(f'{model_name} Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.savefig(f"{model_name}_feature_importance.png")
    plt.show()


def main():
    """
    Main function to load data, train models, and evaluate results.
    """
    games_data = pd.read_csv('../data/games_data_raw_filtered.csv')
    games_data = process_games_data(games_data)
    scored_df = calculate_game_score(games_data)

    games_players = pd.read_csv("../data/games_players_data_filtered.csv")
    X_df = calculate_lineup_features(scored_df, games_players)

    # Ensure features are selected
    X = X_df[HIGH_IMPACT_FEATURES]
    y = scored_df['game_score']

    # Train Random Forest
    rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
    rf_predictions, y_test_rf, rf_model = train_and_evaluate_model(X, y, rf_model, "Random Forest")

    # Plot Random Forest results
    plot_results(y_test_rf, rf_predictions, "Random Forest")

    # Train XGBoost
    xgb_model = XGBRegressor(random_state=42, n_estimators=100)
    xgb_predictions, y_test_xgb, xgb_model = train_and_evaluate_model(X, y, xgb_model, "XGBoost")

    # Plot XGBoost results
    plot_results(y_test_xgb, xgb_predictions, "XGBoost")


if __name__ == '__main__':
    main()
