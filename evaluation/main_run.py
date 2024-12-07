from matchmaking.run_matchmaker import run as matchmaker
from matchmaking.games_clean_score import run as create_model
import pandas as pd
from evaluation import stats_snd_graphs, test_matched_games

def main():
    players = pd.read_csv('../data/league_of_graphs_players_filtered.csv')
    matches = matchmaker()
    model, chosen_features = create_model()
    matching_scores = test_matched_games(players, matches, model, chosen_features)
    stats_snd_graphs(matching_scores)


if __name__ == '__main__':
    main()