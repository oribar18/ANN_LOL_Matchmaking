from matchmaking.run_matchmaker import run as matchmaker
from matchmaking.games_clean_score import run as create_model
from data_processing.lineup_analysis import create_matrix_for_game
import pandas as pd
from matchmaking.model import COLUMNS
from data_processing.games_data_processing import calculate_lineup_features

players = pd.read_csv('../data/league_of_graphs_players_filtered.csv')
matches = matchmaker()
model, chosen_features = create_model()
matching_scores = {}
for feature in matches.keys():
    matching_scores[feature] = []
    X_df = pd.DataFrame(columns=COLUMNS)
    for match in matches[feature]:
        game_matrix, game_matrix_dicts = create_matrix_for_game(players, match=match)
        X_df.loc[X_df.shape[0]] = calculate_lineup_features(game_matrix, game_matrix_dicts)
    score = model.predict(X_df[chosen_features])
    print("*"*100)
    print("*"*100)
    print(f"score: {score}")
    print("*"*100)
    print("*"*100)


#match.team1[0].id