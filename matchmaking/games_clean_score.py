import pandas as pd
from games_data_processing import process_games_data,calculate_game_score, calculate_lineup_features
from model import backward_select_features_by_aic, plot_residuals, plot_feature_importance, plot_actual_vs_predicted

def main():
    games_data = pd.read_csv('data/games_data_raw_filtered.csv')
    games_data = process_games_data(games_data)
    scored_df = calculate_game_score(games_data)

    # scored_df.to_csv('output_with_game_score.csv', index=False)

    games_players = pd.read_csv("data/games_players_data_filtered.csv")
    # scored_df = calculate_lineup_variance(scored_df, games_players)
    X_df = calculate_lineup_features(scored_df, games_players)
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


    # print(scored_df[['kill_diff', 'gold_diff', 'gameDuration', 'duration_score', 'kill_diff_score', 'gold_diff_score',  'game_score', 'lineup_score']].head())
    # # scored_df[['kill_diff', 'gold_diff', 'gameDuration', 'duration_score', 'kill_diff_score', 'gold_diff_score',  'game_score']].to_csv('scored_df.csv')
    # correlation = scored_df['game_score'].corr(scored_df['lineup_score'])
    # print(f"correlation: {correlation}")
    #
    # plt.scatter(scored_df['game_score'], scored_df['lineup_score'])
    # plt.title('Relationship between game_score and lineup_score')
    # plt.xlabel('game_score')
    # plt.ylabel('lineup_score')
    # plt.grid(True)
    # plt.show()


if __name__ == '__main__':
    main()
