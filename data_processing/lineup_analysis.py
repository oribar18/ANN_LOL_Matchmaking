from pydoc import plain

import numpy as np
import pandas as pd
from data_processing.data_classes import Player
from data_processing.players_processing import process_player
from utils import utils
from utils.utils import calculate_kda, calculate_mmr


def retrieve_players_names(game_row=None, match=None):
    if game_row is not None:
        team_1_players = ['name', 'name 3', 'name 5', 'name 7', 'name 9']
        team_2_players = ['name 2', 'name 4', 'name 6', 'name 8', 'name 10']
        all_players = team_1_players + team_2_players
        players_names = [game_row[player] for player in all_players]
    elif match:
        players_names = []
        for player in match.team1:
            players_names.append(player.id)
        for player in match.team2:
            players_names.append(player.id)
    else:
        raise Exception('You must provide either game_row or names_list')

    return players_names


def retrieve_players_rows(games_players, players_names):
    game_players = pd.DataFrame(columns=games_players.columns)

    for player in players_names:
        rows_matching_condition = games_players[games_players['username'] == player]
        if not rows_matching_condition.empty:
            first_row = rows_matching_condition.iloc[0]
            game_players = pd.concat([game_players, pd.DataFrame([first_row])], ignore_index=True)
    return game_players

def create_matrix_from_lineup(game_players):
    game_matrix = []
    game_matrix_dicts = []
    for player in game_players.itertuples():
        player = process_player(player)
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


def create_matrix_for_game(players, game_row=None, match=None):
    players_names = retrieve_players_names(game_row=game_row, match=match)
    game_players = retrieve_players_rows(players, players_names)
    game_matrix, game_matrix_dicts = create_matrix_from_lineup(game_players)
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


def create_suffix_to_role_mapping():
    """
    Create a dictionary mapping suffixes to player roles.

    Returns:
        dict: A dictionary where keys are suffixes and values are roles.
    """
    suffix_to_role = {
        '': 'Top',
        ' 6': 'Top',
        ' 2': 'Jungler',
        ' 7': 'Jungler',
        ' 3': 'Mid',
        ' 8': 'Mid',
        ' 4': 'AD Carry',
        ' 9': 'AD Carry',
        ' 5': 'Support',
        ' 10': 'Support'
    }
    return suffix_to_role


def calculate_kda_variance(df):
    # Calculate KDA for each player
    for i in range(1, 11):
        suffix = '' if i == 1 else f' {i}'
        suffix_to_role = create_suffix_to_role_mapping()
        role = suffix_to_role.get(suffix, 'Unknown')
        df[f'kda{suffix}'] = df.apply(lambda row: calculate_kda({
            'kills': row[f'kills{suffix}'],
            'death': row[f'deaths{suffix}'],
            'assists': row[f'assists{suffix}'],
            'role': role
        }), axis=1)

    # Calculate intra-team variance
    df['blue_kda_variance'] = df[[f'kda{suffix}' for suffix in utils.BLUE_TEAM_SUFFIXES]].var(axis=1)
    df['red_kda_variance'] = df[[f'kda{suffix}' for suffix in utils.RED_TEAM_SUFFIXES]].var(axis=1)
    df['intra_team_penalty'] = df['blue_kda_variance'] + df['red_kda_variance']
    return df