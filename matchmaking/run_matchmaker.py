import time
import numpy as np
import pandas as pd
from data_processing.data_classes import Player
from matchmaking.cktree_matchmaker import CktreeMatchmaker
from utils.utils import calculate_mmr, calculate_kda
from data_processing.players_processing import process_player


def test_matchmaker(active_features_option='mmr, win_rate, games_played', players=None):
    matchmaker = CktreeMatchmaker()
    active_players = players.copy(deep=True)

    for player in active_players.itertuples():
        player_obj = process_player(player)
        active_features_dict = {
            'mmr, win_rate, games_played': [player_obj.mmr, player_obj.win_rate, player_obj.games_played],
            'mmr': [player_obj.mmr],
            'mmr, win_rate, games_played, avg_creeps_per_min, avg_gold_per_min, kills, death, assists':
                [player_obj.mmr, player_obj.win_rate, player_obj.games_played,
                 player_obj.avg_creeps_per_min, player_obj.avg_gold_per_min,
                 player_obj.kills, player_obj.death, player_obj.assists],
            'mmr, win_rate, games_played, avg_creeps_per_min, avg_gold_per_min, calculated_kda':
                [player_obj.mmr, player_obj.win_rate, player_obj.games_played,
                 player_obj.avg_creeps_per_min, player_obj.avg_gold_per_min,
                 player_obj.calculated_kda],
        }
        player_obj.features = np.array(active_features_dict[active_features_option])
        matchmaker.add_player(player_obj)

    start_time = time.time()
    final_match = matchmaker.find_match()
    end_time = time.time()

    if final_match:
        print(f"Formed teams in {end_time - start_time:.3f} seconds:")
        print("Team 1:")
        for player in final_match.team1:
            print(f"Role: {player.role}, Name: {player.id}, MMR: {player.mmr:.0f}, "
                  f"WR: {player.win_rate:.3f}, Games: {player.games_played}")

        team_mmr = np.mean([p.mmr for p in final_match.team1])
        mmr_spread = np.std([p.mmr for p in final_match.team1])
        print(f"\nTeam MMR: {team_mmr:.0f} ± {mmr_spread:.0f}")
        print()

        print("Team 2:")
        for player in final_match.team2:
            print(f"Role: {player.role}, Name: {player.id}, MMR: {player.mmr:.0f}, "
                  f"WR: {player.win_rate:.3f}, Games: {player.games_played}")

        team_mmr = np.mean([p.mmr for p in final_match.team2])
        mmr_spread = np.std([p.mmr for p in final_match.team2])
        print(f"\nTeam MMR: {team_mmr:.0f} ± {mmr_spread:.0f}")
        print(f"\nMMR Difference: {final_match.mmr_difference:.0f}\n")
        print("*" * 100)

        return final_match
    else:
        print("Could not find balanced teams")


def run():
    # Load players data
    players = pd.read_csv('../data/league_of_graphs_players_filtered.csv')

    # Configuration
    num_matches = 250
    active_features_options = [
        'mmr, win_rate, games_played',
        'mmr',
        'mmr, win_rate, games_played, avg_creeps_per_min, avg_gold_per_min, kills, death, assists',
        'mmr, win_rate, games_played, avg_creeps_per_min, avg_gold_per_min, calculated_kda'
    ]

    # Run matches for each feature set
    matches = {}
    for option in active_features_options:
        players_to_match = players.copy(deep=True)
        matches[option] = []
        print(f"Testing with {option} features:")
        for _ in range(num_matches):
            match = test_matchmaker(option, players_to_match)
            if match:
                matches[option].append(match)
                for player in match.team1:
                    players_to_match = players_to_match[players_to_match['username'] != player.id]
                for player in match.team2:
                    players_to_match = players_to_match[players_to_match['username'] != player.id]
        print("*" * 100)

    return matches


def main():
    matches = run()


if __name__ == "__main__":
    main()