import pandas as pd
from data_processing.games_data_processing import process_games_data, calculate_game_score, calculate_lineup_features, calculate_lineup_features_for_real_games
from matchmaking.model import backward_select_features_by_aic, plot_residuals, plot_feature_importance, plot_actual_vs_predicted

def run():
    games_data = pd.read_csv('../data/games_data_raw_filtered.csv')
    games_data = process_games_data(games_data)
    scored_df = calculate_game_score(games_data)

    games_players = pd.read_csv("../data/games_players_data_filtered.csv")

    X_df = calculate_lineup_features_for_real_games(scored_df, games_players)
    Y_df = scored_df['game_score']

    chosen_features, AIC_score, model = backward_select_features_by_aic(X_df, Y_df)


    print(f"chosen_features: {chosen_features}")
    print(f"AIC_score: {AIC_score}")

    # # Call the function with selected features and model
    # show_model_statistics(X_df, Y_df, chosen_features, model)

    plot_residuals(X_df, Y_df, chosen_features, model)

    plot_actual_vs_predicted(X_df, Y_df, chosen_features, model)

    plot_feature_importance(chosen_features, model)

    print(f"weighs: {model.coef_}")

    return model, chosen_features


def main():
    model, chosen_features = run()


if __name__ == '__main__':
    main()
