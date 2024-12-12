from data_processing.data_classes import Player
from utils.utils import calculate_kda, calculate_mmr

def process_player(player):
    """
    Process a player dictionary to create a Player object with computed attributes.

    Args:
        player (dict): A dictionary containing player attributes such as username, winrate,
                       games_won, most_played_role, rank, division, lp, kills, deaths,
                       assists, avg_creeps_per_min, and avg_gold_per_min.

    Returns:
        Player: An instance of the Player class with computed MMR and KDA attributes.
    """
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
    return player_obj