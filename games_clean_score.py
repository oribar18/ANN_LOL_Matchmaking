import pandas as pd
import numpy as np
import re
import matchmaking
from matchmaking import Player
from matchmaking import calculate_kda
from matchmaking import calculate_mmr
import matplotlib.pyplot as plt

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
    def calculate_matchmaking_score_with_dynamic_weights(row):
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
    scored_df['matchmaking_score'] = scored_df.apply(calculate_matchmaking_score_with_dynamic_weights, axis=1)

    scored_df['matchmaking_score'] = scored_df['matchmaking_score'].round(2)

    return scored_df


def calculate_lineup_score(scored_df, games_players):
    scored_df['lineup_score'] = None
    for i in range(len(scored_df)):
        players = [scored_df.iloc[i]['name'], scored_df.iloc[i]['name 2'], scored_df.iloc[i]['name 3'], scored_df.iloc[i]['name 4'],
                   scored_df.iloc[i]['name 5'], scored_df.iloc[i]['name 6'], scored_df.iloc[i]['name 7'], scored_df.iloc[i]['name 8'],
                   scored_df.iloc[i]['name 9'], scored_df.iloc[i]['name 10']]
        game_matrix = []
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
                        [player.mmr / matchmaking.MMR_SCALE, player.win_rate / matchmaking.WINRATE_SCALE, player.games_played / matchmaking.GAMES_SCALE,
                         player.calculated_kda / matchmaking.KDA_SCALE, player.avg_creeps_per_min / matchmaking.CREEPS_SCALE,
                         player.avg_gold_per_min / matchmaking.GOLD_SCALE])
        game_matrix = np.array(game_matrix)
        print(game_matrix)
        # calculate variance between the different players stats
        variance = np.var(game_matrix, axis=0)
        print(variance)
        # calculate the norm of the variance
        norm = np.linalg.norm(variance)
        scored_df.loc[i, 'lineup_score'] = 1 / norm

    return scored_df


def main():
    games_data = pd.read_csv('Games_data_raw_filtered.csv')
    games_data = process_games_data(games_data)
    scored_df = calculate_matchmaking_score(games_data)

    # scored_df.to_csv('output_with_matchmaking_score.csv', index=False)

    games_players = pd.read_csv("games_players_data_filtered.csv")
    scored_df = calculate_lineup_score(scored_df, games_players)

    print(scored_df[['kill_diff', 'gold_diff', 'gameDuration', 'duration_score', 'kill_diff_score', 'gold_diff_score',  'matchmaking_score', 'lineup_score']].head())
    # scored_df[['kill_diff', 'gold_diff', 'gameDuration', 'duration_score', 'kill_diff_score', 'gold_diff_score',  'matchmaking_score']].to_csv('scored_df.csv')
    correlation = scored_df['matchmaking_score'].corr(scored_df['lineup_score'])
    print(f"correlation: {correlation}")

    plt.scatter(scored_df['matchmaking_score'], scored_df['lineup_score'])
    plt.title('Relationship between matchmaking_score and lineup_score')
    plt.xlabel('matchmaking_score')
    plt.ylabel('lineup_score')
    plt.grid(True)
    plt.show()

    scored_l_35 = scored_df[scored_df['lineup_score'] < 35.0]
    scored_ge_35 = scored_df[scored_df['lineup_score'] >= 35.0]


    correlation = scored_l_35['matchmaking_score'].corr(scored_l_35['lineup_score'])
    print(f"correlation l_35: {correlation}")
    print(f"mean l_35: {scored_l_35['matchmaking_score'].mean()}")

    plt.scatter(scored_l_35['matchmaking_score'], scored_l_35['lineup_score'])
    plt.title('when lineup_score < 35')
    plt.xlabel('matchmaking_score')
    plt.ylabel('lineup_score')
    plt.grid(True)
    plt.show()


    correlation = scored_ge_35['matchmaking_score'].corr(scored_ge_35['lineup_score'])
    print(f"correlation ge_35: {correlation}")
    print(f"mean ge_35: {scored_ge_35['matchmaking_score'].mean()}")

    plt.scatter(scored_ge_35['matchmaking_score'], scored_ge_35['lineup_score'])
    plt.title('when lineup_score >= 35')
    plt.xlabel('matchmaking_score')
    plt.ylabel('lineup_score')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
