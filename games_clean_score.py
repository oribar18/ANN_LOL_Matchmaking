import pandas as pd
import numpy as np
import re
from data_classes import Player
from utils import calculate_kda, calculate_mmr
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import utils
# MMR_SCALE = 3000.0
# GAMES_SCALE = 1000.0
# WINRATE_SCALE = 100.0
# KILLS_SCALE = 100.0
# DEATHS_SCALE = 100.0
# ASSISTS_SCALE = 100.0
# CREEPS_SCALE = 10.0
# GOLD_SCALE = 1000.0
# KDA_SCALE = 10.0

# Blue odd, Red even
BLUE_TEAM_SUFFIXES = ['', ' 3', ' 5', ' 7', ' 9']
RED_TEAM_SUFFIXES = [' 2', ' 4', ' 6', ' 8', ' 10']
STAT_COLUMNS = ['kills', 'deaths', 'assists', 'creeps', 'gold']


def process_games_data(df):
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")

    try:
        df['game_duration_mins'] = df['gameDuration'].apply(parse_game_duration)
        # Filter out games with less than 15 minutes
        df = df[df['game_duration_mins'] >= 15].copy()
        df = process_cs_gold_columns(df)
        df = calculate_team_stats(df)
        return df
    except Exception as e:
        raise Exception(f"Error processing games data: {str(e)}")


def parse_game_duration(duration_str):
    """
    Parse game duration string into total minutes.

    Args:
        duration_str (str): Game duration in format like '(24:27)'

    Returns:
        float: Total game duration in minutes
    """
    # Remove parentheses if present
    duration_str = duration_str.strip('()')

    try:
        # Split the time string by ':'
        minutes, seconds = map(int, duration_str.split(':'))

        # Convert to total minutes, rounding seconds
        total_minutes = minutes + (seconds / 60)

        return total_minutes
    except (ValueError, TypeError):
        # Return NaN if parsing fails
        return np.nan


def extract_cs_gold(value):
    """Extract CS and gold values from strings like '202 CS - 8.7k gold'"""
    if pd.isna(value):
        return pd.NA, pd.NA

    # Extract numbers using regex
    cs_match = re.search(r'(\d+)\s*CS', value)
    gold_match = re.search(r'(\d+\.?\d*)k?\s*gold', value)

    cs = int(cs_match.group(1)) if cs_match else pd.NA

    if gold_match:
        gold_str = gold_match.group(1)
        gold = float(gold_str) * 1000
    else:
        gold = pd.NA

    return cs, gold


def process_cs_gold_columns(df):
    # Process each pair of columns for 10 players
    for i in range(1, 11):
        # Define column suffix (empty for first player, ' 2' for second, etc.)
        suffix = '' if i == 1 else f' {i}'

        # Extract CS and gold from the CS column
        cs_series = df[f'cs{suffix}'].apply(lambda x: extract_cs_gold(x)[0])
        gold_series = df[f'cs{suffix}'].apply(lambda x: extract_cs_gold(x)[1])

        # Create new columns
        df[f'creeps{suffix}'] = cs_series
        df[f'gold{suffix}'] = gold_series
    return df


def calculate_team_stats(df):
    for stat in STAT_COLUMNS:
        df[f'blue_{stat}'] = sum(df[f'{stat}{suffix}'] for suffix in BLUE_TEAM_SUFFIXES)
        df[f'red_{stat}'] = sum(df[f'{stat}{suffix}'] for suffix in RED_TEAM_SUFFIXES)
    return df


def calculate_game_score(df):
    """
    Calculate a matchmaking quality score based on game balance metrics.

    Score components:
    1. Kill Difference: Lower difference indicates more balanced teams
    2. Gold Difference: Closer gold totals suggest more even match
    3. Game Duration: Moderate game length (around 30 mins) is ideal
    4. Assists Difference: More balanced total assists
    5. Creep Score Difference: More similar creep scores

    Args:
        df (pd.DataFrame): DataFrame containing match data

    Returns:
        pd.DataFrame: Original DataFrame with added 'game_score' column
    """
    # Create a copy to avoid modifying original DataFrame
    scored_df = df.copy()

    # Calculate differences
    scored_df['kill_diff'] = abs(scored_df['blue_kills'] - scored_df['red_kills'])
    scored_df['gold_diff'] = abs(scored_df['blue_gold'] - scored_df['red_gold'])
    scored_df['assists_diff'] = abs(scored_df['blue_assists'] - scored_df['red_assists'])
    scored_df['creeps_diff'] = abs(scored_df['blue_creeps'] - scored_df['red_creeps'])

    scored_df['kill_diff_norm'] = scored_df['kill_diff'] / scored_df['game_duration_mins']
    scored_df['gold_diff_norm'] = scored_df['gold_diff'] / scored_df['game_duration_mins']
    scored_df['assists_diff_norm'] = scored_df['assists_diff'] / scored_df['game_duration_mins']
    scored_df['creeps_diff_norm'] = scored_df['creeps_diff'] / scored_df['game_duration_mins']

    # Normalize metrics (lower is better)
    max_kill_diff = scored_df['kill_diff_norm'].max()
    max_gold_diff = scored_df['gold_diff_norm'].max()
    max_assists_diff = scored_df['assists_diff_norm'].max()
    max_creeps_diff = scored_df['creeps_diff_norm'].max()

    # Calculate normalized scores (0-1 range, where 1 is best)
    scored_df['kill_diff_score'] = 1 - (scored_df['kill_diff_norm'] / max_kill_diff)
    scored_df['gold_diff_score'] = 1 - (scored_df['gold_diff_norm'] / max_gold_diff)
    scored_df['assists_diff_score'] = 1 - (scored_df['assists_diff_norm'] / max_assists_diff)
    scored_df['creeps_diff_score'] = 1 - (scored_df['creeps_diff_norm'] / max_creeps_diff)

    # Ideal game duration around 30 minutes, with max score at 30 and decreasing as you move away
    scored_df['duration_score'] = np.exp(-((scored_df['game_duration_mins'] - 30) ** 2) / 200)

    # Adjust weights dynamically based on duration_score
    def dynamic_weights(duration_score):
        if duration_score < 0.65:  # If the duration score is low
            return [0.15, 0.15, 0.1, 0.1, 0.5]  # Higher weight for duration score
        else:  # If the duration score is high
            return [0.375, 0.375, 0.1, 0.1, 0.05]  # Reduced weight for duration score

    # Apply dynamic weights to calculate matchmaking score
    def calculate_game_score_with_dynamic_weights(row):
        weights = dynamic_weights(row['duration_score'])
        kill_weight, gold_weight, assists_weight, creeps_weight, duration_weight = weights

        return (
                       kill_weight * row['kill_diff_score'] +
                       gold_weight * row['gold_diff_score'] +
                       assists_weight * row['assists_diff_score'] +
                       creeps_weight * row['creeps_diff_score'] +
                       duration_weight * row['duration_score']
               ) * 100  # Scale to 0-100

    # Calculate the matchmaking score dynamically for each row
    scored_df['game_score'] = scored_df.apply(calculate_game_score_with_dynamic_weights, axis=1)

    scored_df['game_score'] = scored_df['game_score'].round(2)

    return scored_df


def create_matrix_for_game(game_row, games_players):
    team_1_players = ['name', 'name 3', 'name 5', 'name 7', 'name 9']
    team_2_players = ['name 2', 'name 4', 'name 6', 'name 8', 'name 10']
    all_players = team_1_players + team_2_players
    players = [game_row[player] for player in all_players]
    game_matrix = []
    game_matrix_dicts = []
    for player in players:
        for j in range(len(games_players)):
            if player == games_players.iloc[j]['username']:
                player = games_players.iloc[j]
                player = Player(
                    id=player.username,
                    mmr=None,
                    win_rate=player.winrate,
                    games_played=player.games_won,  # NOT PLAYED
                    role=player.most_played_role,
                    rank=player.rank,
                    division=player.division,
                    lp=player.lp,
                    kills=player.kills,
                    death=player.death,
                    assists=player.assists,
                    avg_creeps_per_min=player.avg_creeps_per_min,
                    avg_gold_per_min=player.avg_gold_per_min,
                    calculated_kda=None
                )
                player.mmr = calculate_mmr(player.__dict__)
                player.calculated_kda = calculate_kda(player.__dict__)
                game_matrix.append(
                    [max(player.mmr / utils.MMR_SCALE - 0.8, 0.0) * 5.0,
                     min(max(player.win_rate / utils.WINRATE_SCALE - 0.5, 0.0) * 3.0, 1.0),
                     player.games_played / utils.GAMES_SCALE,
                     player.calculated_kda / utils.KDA_SCALE,
                     player.avg_creeps_per_min / utils.CREEPS_SCALE,
                     player.avg_gold_per_min / utils.GOLD_SCALE])
                # game_matrix.append(
                #     [player.mmr / matchmaking.MMR_SCALE,
                #      player.win_rate / matchmaking.WINRATE_SCALE,
                #      player.games_played / matchmaking.GAMES_SCALE,
                #      player.calculated_kda / matchmaking.KDA_SCALE,
                #      player.avg_creeps_per_min / matchmaking.CREEPS_SCALE,
                #      player.avg_gold_per_min / matchmaking.GOLD_SCALE])
                player_dict_filtered = {'mmr': player.mmr / utils.MMR_SCALE,
                                        'win_rate': player.win_rate / utils.WINRATE_SCALE,
                                        'games_played': player.games_played / utils.GAMES_SCALE,
                                        'role': player.role,
                                        'calculated_kda': player.calculated_kda / utils.KDA_SCALE,
                                        'avg_creeps_per_min': player.avg_creeps_per_min / utils.CREEPS_SCALE,
                                        'avg_gold_per_min': player.avg_gold_per_min / utils.GOLD_SCALE}
                game_matrix_dicts.append(player_dict_filtered)
    game_matrix = np.array(game_matrix)
    return game_matrix, game_matrix_dicts

def calculate_lineup_variance(game_matrix):
    variance_vec = np.var(game_matrix, axis=0)
    variance_norm = np.linalg.norm(variance_vec)
    return tuple(variance_vec), variance_norm


def calculate_maximal_diff(game_dicts, feature, role_equal):
    team_1_locs = range(5)
    team_2_locs = range(5, 10)
    max_diff = 0
    for j in team_1_locs:
        for k in team_2_locs:
            if (not role_equal) or game_dicts[j]['role'] == game_dicts[k]['role']:
                diff = abs(game_dicts[j][feature] - game_dicts[k][feature])
                if diff > max_diff:
                    max_diff = diff
    return max_diff


def calculate_max_diff(game_dicts, feature):
    team_1_locs = range(5)
    team_2_locs = range(5, 10)
    max_team_1 = max([game_dicts[j][feature] for j in team_1_locs])
    max_team_2 = max([game_dicts[j][feature] for j in team_2_locs])
    return abs(max_team_1 - max_team_2)


def calculate_mean_diff(game_dicts, feature):
    team_1_locs = range(5)
    team_2_locs = range(5, 10)
    mean_team_1 = sum([game_dicts[j][feature] for j in team_1_locs]) / len(team_1_locs)
    mean_team_2 = sum([game_dicts[j][feature] for j in team_2_locs]) / len(team_2_locs)
    return abs(mean_team_1 - mean_team_2)


# game_matrix.append(
#     [max(player.mmr / matchmaking.MMR_SCALE - 0.8, 0.0) * 5.0,
#      min(max(player.win_rate / matchmaking.WINRATE_SCALE - 0.5, 0.0) * 3.0, 1.0),
#      player.games_played / matchmaking.GAMES_SCALE,
#      player.calculated_kda / matchmaking.KDA_SCALE,
#      player.avg_creeps_per_min / matchmaking.CREEPS_SCALE,
#      player.avg_gold_per_min / matchmaking.GOLD_SCALE])

def calculate_lineup_features(scored_df, games_players):
    team_1_locs = range(5)
    team_2_locs = range(5, 10)
    # columns = ['var_mmr', 'var_win_rate', 'var_games_played', 'var_kda', 'var_creeps', 'var_gold', 'normed_var',
    #            'maximal_mmr_diff', 'maximal_kda_diff', 'maximal_win_rate_diff', 'maximal_games_played_diff',
    #            'maximal_creeps_diff', 'maximal_gold_diff', 'max_mmr_diff', 'max_kda_diff', 'max_win_rate_diff',
    #            'max_games_played_diff', 'max_creeps_diff', 'max_gold_diff', 'mean_mmr_diff', 'mean_kda_diff',
    #            'mean_win_rate_diff', 'mean_games_played_diff', 'mean_creeps_diff', 'mean_gold_diff']

    columns = ['var_mmr', 'var_win_rate', 'var_kda', 'var_creeps', 'normed_var', 'maximal_mmr_diff', 'maximal_kda_diff',
               'maximal_win_rate_diff','maximal_creeps_diff',
               'maximal_gold_diff', 'max_mmr_diff', 'max_kda_diff', 'max_win_rate_diff',
               'max_creeps_diff', 'max_gold_diff', 'mean_mmr_diff', 'mean_kda_diff',
               'mean_win_rate_diff', 'mean_creeps_diff', 'mean_gold_diff']
    X_df = pd.DataFrame(columns=columns)
    for i in range(len(scored_df)):
        row = scored_df.iloc[i]
        game_matrix, game_matrix_dicts = create_matrix_for_game(row, games_players)
        variance_vec, variance_norm = calculate_lineup_variance(game_matrix)




        var_mmr, var_win_rate, var_games_played, var_kda, var_creeps, var_gold = variance_vec
        normed_var = variance_norm

        maximal_mmr_diff = calculate_maximal_diff(game_matrix_dicts, feature='mmr', role_equal=False)
        maximal_kda_diff = calculate_maximal_diff(game_matrix_dicts, feature='calculated_kda', role_equal=True)
        maximal_win_rate_diff = calculate_maximal_diff(game_matrix_dicts, feature='win_rate', role_equal=False)
        maximal_games_played_diff = calculate_maximal_diff(game_matrix_dicts, feature='games_played', role_equal=False)
        maximal_creeps_diff = calculate_maximal_diff(game_matrix_dicts, feature='avg_creeps_per_min', role_equal=True)
        maximal_gold_diff = calculate_maximal_diff(game_matrix_dicts, feature='avg_gold_per_min', role_equal=True)

        max_mmr_diff = calculate_max_diff(game_matrix_dicts, feature='mmr')
        max_kda_diff = calculate_max_diff(game_matrix_dicts, feature='calculated_kda')
        max_win_rate_diff = calculate_max_diff(game_matrix_dicts, feature='win_rate')
        max_games_played_diff = calculate_max_diff(game_matrix_dicts, feature='games_played')
        max_creeps_diff = calculate_max_diff(game_matrix_dicts, feature='avg_creeps_per_min')
        max_gold_diff = calculate_max_diff(game_matrix_dicts, feature='avg_gold_per_min')

        mean_mmr_diff = calculate_mean_diff(game_matrix_dicts, feature='mmr')
        mean_kda_diff = calculate_mean_diff(game_matrix_dicts, feature='calculated_kda')
        mean_win_rate_diff = calculate_mean_diff(game_matrix_dicts, feature='win_rate')
        mean_games_played_diff = calculate_mean_diff(game_matrix_dicts, feature='games_played')
        mean_creeps_diff = calculate_mean_diff(game_matrix_dicts, feature='avg_creeps_per_min')
        mean_gold_diff = calculate_mean_diff(game_matrix_dicts, feature='avg_gold_per_min')

        # X_df.loc[i] = [var_mmr, var_win_rate, var_games_played, var_kda, var_creeps, var_gold, normed_var,
        #                maximal_mmr_diff, maximal_kda_diff, maximal_win_rate_diff, maximal_games_played_diff,
        #                maximal_creeps_diff, maximal_gold_diff, max_mmr_diff, max_kda_diff, max_win_rate_diff,
        #                max_games_played_diff, max_creeps_diff, max_gold_diff, mean_mmr_diff, mean_kda_diff,
        #                mean_win_rate_diff, mean_games_played_diff, mean_creeps_diff, mean_gold_diff]

        X_df.loc[i] = [var_mmr, var_win_rate, var_kda, var_creeps, normed_var,
                       maximal_mmr_diff, maximal_kda_diff, maximal_win_rate_diff,
                       maximal_creeps_diff, maximal_gold_diff, max_mmr_diff, max_kda_diff, max_win_rate_diff,
                       max_creeps_diff, max_gold_diff, mean_mmr_diff, mean_kda_diff,
                       mean_win_rate_diff, mean_creeps_diff, mean_gold_diff]

    return X_df


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


# import statsmodels.api as sm
#
#
# def show_model_statistics(X_df, Y_df, chosen_features, model):
#     # Prepare the data
#     X_selected = X_df[chosen_features]
#     X_selected_with_const = sm.add_constant(X_selected)  # Add constant for intercept
#
#     # Fit the statsmodels OLS (Ordinary Least Squares) model
#     sm_model = sm.OLS(Y_df, X_selected_with_const).fit()
#
#     # Display model summary
#     print(sm_model.summary())


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


def main():
    games_data = pd.read_csv('Games_data_raw_filtered.csv')
    games_data = process_games_data(games_data)
    scored_df = calculate_game_score(games_data)

    # scored_df.to_csv('output_with_game_score.csv', index=False)

    games_players = pd.read_csv("games_players_data_filtered.csv")
    # scored_df = calculate_lineup_variance(scored_df, games_players)
    X_df = calculate_lineup_features(scored_df, games_players)
    Y_df = scored_df['game_score']

    chosen_features, AIC_score, model = backward_select_features_by_aic(X_df, Y_df)


    print(f"chosen_features: {chosen_features}")
    print(f"AIC_score: {AIC_score}")

    # # Call the function with selected features and model
    # show_model_statistics(X_df, Y_df, chosen_features, model)

    plot_residuals(X_df, Y_df, chosen_features, model)

    plot_actual_vs_predicted(X_df, Y_df, chosen_features, model)

    plot_feature_importance(chosen_features, model)

    print(f"weighs: {model.coef_}")

    # print(scored_df[['kill_diff', 'gold_diff', 'gameDuration', 'duration_score', 'kill_diff_score', 'gold_diff_score',  'game_score', 'lineup_score']].head())
    # # scored_df[['kill_diff', 'gold_diff', 'gameDuration', 'duration_score', 'kill_diff_score', 'gold_diff_score',  'game_score']].to_csv('scored_df.csv')
    # correlation = scored_df['game_score'].corr(scored_df['lineup_score'])
    # print(f"correlation: {correlation}")
    #
    # plt.scatter(scored_df['game_score'], scored_df['lineup_score'])
    # plt.title('Relationship between game_score and lineup_score')
    # plt.xlabel('game_score')
    # plt.ylabel('lineup_score')
    # plt.grid(True)
    # plt.show()


if __name__ == '__main__':
    main()
