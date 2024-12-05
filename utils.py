import numpy as np

# Constants
MMR_SCALE = 3000.0
WINRATE_SCALE = 100.0
GAMES_SCALE = 1000.0
KILLS_SCALE = 100.0
DEATHS_SCALE = 100.0
ASSISTS_SCALE = 100.0
CREEPS_SCALE = 10.0
GOLD_SCALE = 1000.0
KDA_SCALE = 10.0


def calculate_mmr(player):
    base_mmr = {
        'Iron': 200, 'Bronze': 800, 'Silver': 1000, 'Gold': 1200,
        'Platinum': 1500, 'Diamond': 1800, 'Master': 2200,
        'GrandMaster': 2400, 'Challenger': 2600
    }
    mmr = base_mmr.get(player['division'], 0)

    rank_bonus = {1: 160, 2: 120, 3: 80, 4: 40}
    if player['division'] == 'Iron':
        rank_bonus = {1: 400, 2: 300, 3: 200, 4: 100}
    elif player['division'] in ['Gold', 'Platinum', 'Diamond']:
        rank_bonus = {1: 200, 2: 150, 3: 100, 4: 50}
    mmr += rank_bonus.get(player['rank'], 0)

    if player['win_rate'] >= 50:
        mmr += mmr * 0.2 * (player['win_rate'] * 0.01)
    else:
        mmr -= mmr * 0.2 * ((50 - player['win_rate']) * 0.01)

    return round(mmr, 2)


def calculate_kda(player):
    adj_deaths = max(1.0, player['death'])

    role_multipliers = {
        'Top': (1.2, 1.0),
        'Jungler': (1.0, 1.5),
        'Mid': (1.5, 1.0),
        'AD Carry': (2.0, 1.0),
        'Support': (1.0, 2.0)
    }

    kills_mult, assists_mult = role_multipliers.get(player['role'], (1.0, 1.0))
    kda = ((kills_mult * player['kills']) + (assists_mult * player['assists'])) / adj_deaths

    return round(kda, 2)


def normalize_features(features):
    scales = [MMR_SCALE, WINRATE_SCALE, GAMES_SCALE, CREEPS_SCALE,
              GOLD_SCALE, KILLS_SCALE, DEATHS_SCALE, ASSISTS_SCALE, KDA_SCALE]
    normalized_features_np = np.array([f / s for f, s in zip(features, scales) if f is not None])
    if len(features) == 6:
        normalized_features_np[5] = normalized_features_np[5] * scales[5] / scales[8]

    return normalized_features_np
