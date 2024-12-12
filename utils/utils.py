"""
Utility functions for data processing, KDA calculations, MMR calculations,
and other operations required for matchmaking and game analysis.
"""

import numpy as np
import pandas as pd
import re

# Constants
MMR_SCALE = 3000.0
WINRATE_SCALE = 100.0
GAMES_SCALE = 1000.0
KILLS_SCALE = 100.0
DEATHS_SCALE = 100.0
ASSISTS_SCALE = 100.0
CREEPS_SCALE = 10.0
GOLD_SCALE = 1000.0
KDA_SCALE = 10.0
PENALTY_WEIGHT = 0.15
STAT_COLUMNS = ['kills', 'deaths', 'assists', 'creeps', 'gold']
# Blue odd, Red even
BLUE_TEAM_SUFFIXES = ['', ' 3', ' 5', ' 7', ' 9']
RED_TEAM_SUFFIXES = [' 2', ' 4', ' 6', ' 8', ' 10']




def calculate_mmr(player):
    """
    Calculate the Matchmaking Rating (MMR) for a player.

    Args:
        player (dict): A dictionary containing player data, including:
                       'rank', 'division', 'lp' (League Points), and 'win_rate'.

    Returns:
        float: The calculated MMR value based on rank, division, LP, and win rate.
    """
    base_mmr = {
        'Iron': 200, 'Bronze': 800, 'Silver': 1000, 'Gold': 1200,
        'Platinum': 1500, 'Diamond': 1800, 'Master': 2200,
        'GrandMaster': 2400, 'Challenger': 2600
    }
    mmr = base_mmr.get(player['division'], 0)

    rank_bonus = {1: 160, 2: 120, 3: 80, 4: 40}
    if player['division'] == 'Iron':
        rank_bonus = {1: 400, 2: 300, 3: 200, 4: 100}
    elif player['division'] in ['Gold', 'Platinum', 'Diamond']:
        rank_bonus = {1: 200, 2: 150, 3: 100, 4: 50}
    mmr += rank_bonus.get(player['rank'], 0)

    if player['win_rate'] >= 50:
        mmr += mmr * 0.2 * (player['win_rate'] * 0.01)
    else:
        mmr -= mmr * 0.2 * ((50 - player['win_rate']) * 0.01)

    return round(mmr, 2)


def calculate_kda(player):
    """
    Calculate the Kill-Death-Assist (KDA) ratio for a player.

    Args:
        player (dict): A dictionary containing player stats with keys:
                      'kills', 'death', 'assists', and optionally 'role'.

    Returns:
        float: The KDA ratio calculated as (kills + assists) / max(death, 1).
               If 'death' is zero, it avoids division by zero.
    """
    adj_deaths = max(1.0, player['death'])

    role_multipliers = {
        'Top': (1.2, 1.0),
        'Jungler': (1.0, 1.5),
        'Mid': (1.5, 1.0),
        'AD Carry': (2.0, 1.0),
        'Support': (1.0, 2.0)
    }

    kills_mult, assists_mult = role_multipliers.get(player['role'], (1.0, 1.0))
    kda = ((kills_mult * player['kills']) + (assists_mult * player['assists'])) / adj_deaths

    return round(kda, 2)


def normalize_features(features):
    """
    Normalize selected features in a feature vector to predefined scales.

    Args:
        features (list): A list of feature values to normalize.

    Returns:
        np.ndarray: A normalized feature vector.
    """
    scales = [MMR_SCALE, WINRATE_SCALE, GAMES_SCALE, CREEPS_SCALE,
              GOLD_SCALE, KILLS_SCALE, DEATHS_SCALE, ASSISTS_SCALE, KDA_SCALE]
    normalized_features_np = np.array([f / s for f, s in zip(features, scales) if f is not None])
    if len(features) == 6:
        normalized_features_np[5] = normalized_features_np[5] * scales[5] / scales[8]

    return normalized_features_np


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
    """
    Extract CS (Creep Score) and gold values from a string representation.

    Args:
        value (str): A string like '202 CS - 8.7k gold'.

    Returns:
        tuple: A tuple containing CS (int) and gold (float), or (pd.NA, pd.NA) if parsing fails.
    """
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


def calculate_team_stats(df):
    """
    Calculate aggregate team stats for blue and red teams.

    Args:
        df (pd.DataFrame): A DataFrame containing player stats with columns
                           for individual stats (e.g., kills, deaths, assists).

    Returns:
        pd.DataFrame: The updated DataFrame with aggregated team stats for both teams.
    """
    for stat in STAT_COLUMNS:
        df[f'blue_{stat}'] = sum(df[f'{stat}{suffix}'] for suffix in BLUE_TEAM_SUFFIXES)
        df[f'red_{stat}'] = sum(df[f'{stat}{suffix}'] for suffix in RED_TEAM_SUFFIXES)
    return df




