import pandas as pd
import numpy as np
import re

# Blue odd, Red even
BLUE_TEAM_SUFFIXES = ['', ' 3', ' 5', ' 7', ' 9']
RED_TEAM_SUFFIXES = [' 2', ' 4', ' 6', ' 8', ' 10']
STAT_COLUMNS = ['kills', 'deaths', 'assists', 'creeps', 'gold']


def process_games_data(df):
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")

    try:
        df['game_duration_mins'] = df['gameDuration'].apply(parse_game_duration)
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


def calculate_matchmaking_score(df):
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
        pd.DataFrame: Original DataFrame with added 'matchmaking_score' column
    """
    # Create a copy to avoid modifying original DataFrame
    scored_df = df.copy()

    # Calculate differences
    scored_df['kill_diff'] = abs(scored_df['blue_kills'] - scored_df['red_kills'])
    scored_df['gold_diff'] = abs(scored_df['blue_gold'] - scored_df['red_gold'])
    scored_df['assists_diff'] = abs(scored_df['blue_assists'] - scored_df['red_assists'])
    scored_df['creeps_diff'] = abs(scored_df['blue_creeps'] - scored_df['red_creeps'])

    # Normalize metrics (lower is better)
    max_kill_diff = scored_df['kill_diff'].max()
    max_gold_diff = scored_df['gold_diff'].max()
    max_assists_diff = scored_df['assists_diff'].max()
    max_creeps_diff = scored_df['creeps_diff'].max()

    # Calculate normalized scores (0-1 range, where 1 is best)
    scored_df['kill_diff_score'] = 1 - (scored_df['kill_diff'] / max_kill_diff)
    scored_df['gold_diff_score'] = 1 - (scored_df['gold_diff'] / max_gold_diff)
    scored_df['assists_diff_score'] = 1 - (scored_df['assists_diff'] / max_assists_diff)
    scored_df['creeps_diff_score'] = 1 - (scored_df['creeps_diff'] / max_creeps_diff)

    # Ideal game duration around 30 minutes, with max score at 30 and decreasing as you move away
    scored_df['duration_score'] = 1 - abs(scored_df['game_duration_mins'] - 30) / 30

    # Combine scores with weighted average
    scored_df['matchmaking_score'] = (
                                             0.3 * scored_df['kill_diff_score'] +
                                             0.3 * scored_df['gold_diff_score'] +
                                             0.1 * scored_df['assists_diff_score'] +
                                             0.1 * scored_df['creeps_diff_score'] +
                                             0.2 * scored_df['duration_score']
                                     ) * 100  # Scale to 0-100

    scored_df['matchmaking_score'] = scored_df['matchmaking_score'].round(2)

    return scored_df

def main():
    games_data = pd.read_csv('Games_data_raw.csv')
    games_data = process_games_data(games_data)
    scored_df = calculate_matchmaking_score(games_data)
    # scored_df.to_csv('output_with_matchmaking_score.csv', index=False)
    print(scored_df[['kill_diff','gold_diff', 'gameDuration','matchmaking_score']].head())


if __name__ == '__main__':
    main()
