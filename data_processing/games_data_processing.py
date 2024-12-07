from utils.utils import parse_game_duration, extract_cs_gold, calculate_team_stats
import numpy as np
import lineup_analysis
from lineup_analysis import calculate_maximal_diff, calculate_max_diff, calculate_mean_diff
import pandas as pd

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

    scored_df = lineup_analysis.calculate_kda_variance(scored_df)
    # Ensure intra_team_penalty is normalized to be within 0-1 range
    max_penalty = scored_df['intra_team_penalty'].max()
    if max_penalty > 0:
        scored_df['normalized_penalty'] = scored_df['intra_team_penalty'] / max_penalty
    else:
        scored_df['normalized_penalty'] = 0  # If there's no variance, no penalty

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

        base_score = (
                kill_weight * row['kill_diff_score'] +
                gold_weight * row['gold_diff_score'] +
                assists_weight * row['assists_diff_score'] +
                creeps_weight * row['creeps_diff_score'] +
                duration_weight * row['duration_score']
        )
        return base_score * 100

    # Calculate the matchmaking score dynamically for each row
    scored_df['game_score'] = scored_df.apply(calculate_game_score_with_dynamic_weights, axis=1)

    # Subtract penalty in the 0-1 range
    # scored_df['game_score'] -= scored_df['normalized_penalty'] * PENALTY_WEIGHT * 100

    # Scale to 0-100 after penalty adjustment
    scored_df['game_score'] = scored_df['game_score'].clip(0, 100).round(2)

    return scored_df



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
        game_matrix, game_matrix_dicts = lineup_analysis.create_matrix_for_game(row, games_players)
        variance_vec, variance_norm = lineup_analysis.calculate_lineup_variance(game_matrix)




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


