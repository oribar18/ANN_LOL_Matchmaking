import time
import numpy as np
import pandas as pd
from data_processing.data_classes import Player
from matchmaking.cktree_matchmaker import CktreeMatchmaker
from utils import calculate_mmr, calculate_kda


def test_matchmaker(active_features_option='mmr, win_rate, games_played', players=None):
    matchmaker = CktreeMatchmaker()
    active_payers = players.sample(n=180)

    for player in active_payers.itertuples():
        player_obj = Player(
            id=player.username,
            mmr=None,
            win_rate=player.winrate,
            games_played=player.games_won,
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
        player_obj.mmr = calculate_mmr(player_obj.__dict__)
        player_obj.calculated_kda = calculate_kda(player_obj.__dict__)

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


def main():
    # Load players data
    players = pd.read_csv('../data/league_of_graphs_players_filtered.csv')

    # Configuration
    num_matches = 10
    active_features_options = [
        'mmr, win_rate, games_played',
        'mmr',
        'mmr, win_rate, games_played, avg_creeps_per_min, avg_gold_per_min, kills, death, assists',
        'mmr, win_rate, games_played, avg_creeps_per_min, avg_gold_per_min, calculated_kda'
    ]

    # Run matches for each feature set
    matches = []
    for option in active_features_options:
        print(f"Testing with {option} features:")
        for _ in range(num_matches):
            match = test_matchmaker(option, players)
            if match:
                matches.append(match)
        print("*" * 100)


if __name__ == "__main__":
    main()