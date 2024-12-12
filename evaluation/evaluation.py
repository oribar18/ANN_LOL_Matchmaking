import matplotlib.pyplot as plt
from data_processing.lineup_analysis import create_matrix_for_game
from matchmaking.model import COLUMNS
from data_processing.games_data_processing import calculate_lineup_features
from statistics import mean
from statistics import stdev
import pandas as pd


def test_matched_games(players, matches, model, chosen_features):
    """
    Evaluate matched games by calculating scores for each feature set using the provided model.

    Args:
        players (pd.DataFrame): The dataframe containing player data.
        matches (dict): Dictionary where keys are feature subsets, and values are lists of matches.
        model: The trained regression model for score prediction.
        chosen_features (list): The selected features used by the model.

    Returns:
        dict: A dictionary with mean, standard deviation, and scores for each feature set.
    """
    matching_scores = {}
    for features in matches.keys():
        matching_scores[features] = {}
        X_df = pd.DataFrame(columns=COLUMNS)
        for match in matches[features]:
            game_matrix, game_matrix_dicts = create_matrix_for_game(players, match=match)
            X_df.loc[X_df.shape[0]] = calculate_lineup_features(game_matrix, game_matrix_dicts)
        matching_scores[features]['scores'] = model.predict(X_df[chosen_features])
        matching_scores[features]['mean'] = mean(matching_scores[features]['scores'])
        matching_scores[features]['std'] = stdev(matching_scores[features]['scores'])

    return matching_scores


def stats_snd_graphs(matching_scores):
    """
    Display statistics and visualize scores for each feature set.

    Args:
        matching_scores (dict): Dictionary containing mean, standard deviation, and scores for each feature set.
    """
    for features in matching_scores.keys():
        print(f"Features: {features}")
        print(f"\tMean: {matching_scores[features]['mean']}")
        print(f"\tStd: {matching_scores[features]['std']}")
        print("*"*30)

    # plot graph for all scores by features

    for features in matching_scores.keys():
        x = range(len(matching_scores[features]['scores']))  # X-axis: indices of the list
        plt.plot(x, matching_scores[features]['scores'], marker='o', linestyle='None', markersize=3, label=features)  # Line plot with markers

    # Add legend and labels
    plt.legend(fontsize=8)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Points and Connecting Lines for Each List')

    # Show the plot
    plt.show()