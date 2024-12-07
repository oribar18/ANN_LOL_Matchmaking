from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
# columns = ['var_mmr', 'var_win_rate', 'var_games_played', 'var_kda', 'var_creeps', 'var_gold', 'normed_var',
#            'maximal_mmr_diff', 'maximal_kda_diff', 'maximal_win_rate_diff', 'maximal_games_played_diff',
#            'maximal_creeps_diff', 'maximal_gold_diff', 'max_mmr_diff', 'max_kda_diff', 'max_win_rate_diff',
#            'max_games_played_diff', 'max_creeps_diff', 'max_gold_diff', 'mean_mmr_diff', 'mean_kda_diff',
#            'mean_win_rate_diff', 'mean_games_played_diff', 'mean_creeps_diff', 'mean_gold_diff']

COLUMNS = ['var_mmr', 'var_win_rate', 'var_kda', 'var_creeps', 'normed_var', 'maximal_mmr_diff', 'maximal_kda_diff',
           'maximal_win_rate_diff', 'maximal_creeps_diff',
           'maximal_gold_diff', 'max_mmr_diff', 'max_kda_diff', 'max_win_rate_diff',
           'max_creeps_diff', 'max_gold_diff', 'mean_mmr_diff', 'mean_kda_diff',
           'mean_win_rate_diff', 'mean_creeps_diff', 'mean_gold_diff']

def backward_select_features_by_aic(X_df, Y_df):
    """
    Select features using backward selection based on AIC and build a linear regression model.

    Parameters:
    - X_df: pd.DataFrame, explanatory features (columns) for samples (rows).
    - Y_df: pd.DataFrame, one-column DataFrame of the explained variable values.

    Returns:
    - chosen_features: List of feature names selected by AIC.
    - aic_score: The AIC score of the best feature subset.
    - model: The LinearRegression model trained on the chosen features.
    """
    # Ensure Y_df is a Series for simplicity
    Y = Y_df.squeeze()

    # Start with all features
    selected_features = list(X_df.columns)
    current_aic = float('inf')
    n_samples = X_df.shape[0]

    while True:
        best_candidate_aic = float('inf')
        worst_candidate_feature = None

        # Try removing each feature one by one
        for feature in selected_features:
            candidate_features = [f for f in selected_features if f != feature]
            X_subset = X_df[candidate_features]

            if X_subset.empty:
                continue

            # Fit the model
            model = LinearRegression()
            model.fit(X_subset, Y)

            # Predict and calculate RSS
            predictions = model.predict(X_subset)
            rss = mean_squared_error(Y, predictions) * n_samples

            # Calculate AIC
            n_features = len(candidate_features)
            aic = n_samples * np.log(rss / n_samples) + 0.5 * n_features

            # Update the best candidate to remove
            if aic < best_candidate_aic:
                best_candidate_aic = aic
                worst_candidate_feature = feature

        # Decide whether to remove a feature
        if best_candidate_aic < current_aic:
            current_aic = best_candidate_aic
            selected_features.remove(worst_candidate_feature)
        else:
            # Stop if no improvement in AIC
            break

    # Final model with selected features
    X_final = X_df[selected_features]
    final_model = LinearRegression()
    final_model.fit(X_final, Y)

    # Return the results
    return selected_features, current_aic, final_model


def plot_residuals(X_df, Y_df, chosen_features, model):
    # Predictions
    X_selected = X_df[chosen_features]
    predictions = model.predict(X_selected)
    residuals = Y_df.squeeze() - predictions

    # Plot residuals
    plt.figure(figsize=(8, 6))
    plt.scatter(predictions, residuals, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=1)
    plt.title('Residual Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.show()


def plot_actual_vs_predicted(X_df, Y_df, chosen_features, model):
    # Predictions
    X_selected = X_df[chosen_features]
    predictions = model.predict(X_selected)

    # Plot actual vs predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(Y_df, predictions, alpha=0.7)
    plt.plot([Y_df.min(), Y_df.max()], [Y_df.min(), Y_df.max()], 'r--')  # Diagonal line
    plt.title('Actual vs Predicted')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.show()


def plot_feature_importance(chosen_features, model):
    coefficients = model.coef_

    plt.figure(figsize=(8, 6))
    plt.barh(chosen_features, coefficients, alpha=0.7)
    plt.title('Feature Importance')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Features')
    plt.show()