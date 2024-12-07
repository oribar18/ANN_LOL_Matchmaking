from data_processing.data_classes import Player
from data_processing.data_classes import Match
from utils.utils import calculate_kda, calculate_mmr

def process_player(player):
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