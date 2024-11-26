# === IMPORTS ===
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple
from scipy.spatial import cKDTree
import time

# === CONSTANTS ===
MMR_SCALE = 3000.0
GAMES_SCALE = 1000.0
WINRATE_SCALE = 100.0
KILLS_SCALE = 100.0
DEATHS_SCALE = 100.0
ASSISTS_SCALE = 100.0
CREEPS_SCALE = 10.0
GOLD_SCALE = 1000.0
KDA_SCALE = 10.0


# === HELPER FUNCTIONS ===
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

    # TODO: Use LP in mmr ?
    return round(mmr, 2)


def calculate_kda(player):
    adj_deaths = max(1.0, player['death'])

    if player['role'] == 'Top':
        kda = ((1.2 * player['kills']) + player['assists']) / adj_deaths
    elif player['role'] == 'Jungler':
        kda = (player['kills'] + (1.5 * player['assists'])) / adj_deaths
    elif player['role'] == 'Mid':
        kda = ((1.5 * player['kills']) + player['assists']) / adj_deaths
    elif player['role'] == 'AD Carry':
        kda = ((2.0 * player['kills']) + player['assists']) / adj_deaths
    elif player['role'] == 'Support':
        kda = (player['kills'] + (2.0 * player['assists'])) / adj_deaths
    else:
        # Default formula (standard KDA calculation)
        kda = (player['kills'] + player['assists']) / adj_deaths

    return round(kda, 2)


def normalize_features(features):
    scales = [MMR_SCALE, WINRATE_SCALE, GAMES_SCALE, CREEPS_SCALE,
              GOLD_SCALE, KILLS_SCALE, DEATHS_SCALE, ASSISTS_SCALE, KDA_SCALE]
    normalized_features_np = np.array([f / s for f, s in zip(features, scales) if f is not None])
    if len(features) == 6:
        normalized_features_np[5] = normalized_features_np[5] * scales[5] / scales[8]
    return normalized_features_np


# === DATA CLASSES ===
@dataclass
class Player:
    id: str
    mmr: float
    win_rate: float
    games_played: int
    role: str
    division: str
    rank: Optional[int] = None
    lp: Optional[int] = None
    kills: Optional[float] = None
    death: Optional[float] = None
    assists: Optional[float] = None
    avg_creeps_per_min: Optional[float] = None
    avg_gold_per_min: Optional[float] = None
    calculated_kda: Optional[float] = None
    features: Optional[np.ndarray] = None


@dataclass
class Match:
    team1: List[Player]
    team2: List[Player]
    mmr_difference: float


# === CLASSES ===
class RoleQueue:
    def __init__(self, role: str):
        self.role = role
        self.players: List[Player] = []
        self.tree: Optional[cKDTree] = None
        self.features: Optional[np.ndarray] = None
        self.needs_rebuild = True

    def add_player(self, player: Player):
        """Add a player and mark tree for rebuilding."""
        self.players.append(player)
        self.needs_rebuild = True

    def remove_player(self, player: Player):
        """Remove a player and mark tree for rebuilding."""
        self.players.remove(player)
        self.needs_rebuild = True

    def rebuild_tree(self):
        """Rebuild the KD-tree with current players."""
        if not self.players or len(self.players) < 2:  # Need at least 2 players for role mirror
            self.tree = None
            self.features = None
            return

        # Create feature array for all players
        self.features = np.array([
            normalize_features(player.features) for player in self.players
        ])
        # features_transformed = np.array([
        #     self.features[:, 0] ,
        #     self.features[:, 1],
        #     self.features[:, 2]
        # ]).T
        num_features = self.features.shape[1]
        features_transformed_list = []
        for i in range(num_features):
            features_transformed_list.append(self.features[:, i])
        features_transformed = np.array(features_transformed_list).T

        self.tree = cKDTree(features_transformed)
        self.needs_rebuild = False

    def find_closest_players(self,
                             target_features: np.ndarray,
                             k: int = 5,
                             mmr_threshold: float = 300,
                             player_id: str = None) -> List[Tuple[float, Player]]:
        # This function is our ANN search
        """Find k closest players using the KD-tree."""
        if self.needs_rebuild:
            self.rebuild_tree()

        if not self.tree or len(self.players) < 2:
            return []

        # Query the tree for k * 2 neighbors (to have options after MMR filtering)
        distances, indices = self.tree.query(normalize_features(target_features), k=min(k * 2, len(self.players)))

        # Filter and return results
        results = []
        for dist, idx in zip(distances, indices):
            candidate = self.players[idx]
            # Check if same player
            if player_id is not None and candidate.id == player_id:
                continue
            # Check MMR threshold
            if abs(candidate.mmr - target_features[0]) <= mmr_threshold:
                results.append((dist, candidate))

            if len(results) >= k:
                break

        return results


class ImprovedMatchmaker:
    def __init__(self):
        self.role_queues = {
            role: RoleQueue(role)
            for role in ["Jungler", "Support", "Mid", "Top", "AD Carry"]
        }

    def add_player(self, player: Player):
        """Add player to appropriate role queue."""
        self.role_queues[player.role].add_player(player)

    def _calculate_team_stats(self, team: List[Player]) -> dict:
        return {
            'avg_mmr': np.mean([p.mmr for p in team]),
            'avg_wr': np.mean([p.win_rate for p in team]),
            'mmr_spread': np.std([p.mmr for p in team])
        }

    def _is_balanced_match(self, team1: List[Player], team2: List[Player],
                           max_mmr_diff: float = 100) -> bool:
        """Check if two teams are balanced."""
        team1_stats = self._calculate_team_stats(team1)
        team2_stats = self._calculate_team_stats(team2)
        mmr_difference = abs(team1_stats['avg_mmr'] - team2_stats['avg_mmr'])
        return (mmr_difference <= max_mmr_diff and
                team1_stats['mmr_spread'] < 400 and
                team2_stats['mmr_spread'] < 400)

    def find_match(self, team_size: int = 5) -> Optional[List[Player]]:
        """Find a balanced team using efficient KD-tree searches."""
        # Need at least 2 players in each role
        if any(len(self.role_queues[role].players) < 2 for role in
               ["Jungler", "Support", "Mid", "Top", "AD Carry"]):
            return None

        # Sort roles by queue length for efficiency
        sorted_roles = sorted(
            self.role_queues.keys(),
            key=lambda r: len(self.role_queues[r].players),
            reverse=True
        )

        # Try each player in the shortest queue as a starting point
        shortest_queue = self.role_queues[sorted_roles[-1]]
        for starter in shortest_queue.players:
            match = self._build_teams_from_starter(starter, sorted_roles)
            if match:
                # Remove matched players from queues
                # for player in match.team1 + match.team2:
                #     self.role_queues[player.role].remove_player(player)
                return match

        return None

    def _build_teams_from_starter(self, starter: Player, role_order: List[str]) -> Optional[List[Player]]:
        """Try to build a balanced team starting with given player."""
        team1 = [starter]
        team2 = []
        used_roles = {starter.role}
        # Adding a mirror starter for team 2
        role_queue = self.role_queues[starter.role]
        matches = role_queue.find_closest_players(
            starter.features,
            k=3,  # Get top 3 candidates
            mmr_threshold=300,
            player_id=starter.id
        )
        if matches:
            # Take the closest match
            team2.append(matches[0][1])
        else:
            return None  # Cannot complete team

        for role in role_order:
            if role in used_roles:
                continue

            # Update target features based on current team
            feature_matrix = np.array([p.features for p in team1])
            team_features1 = np.mean(feature_matrix, axis=0)
            # feature_matrix2 = np.array([p.features for p in team2])
            # team_features2 = np.mean(feature_matrix, axis=0)

            # Find closest players in this role using KD-tree
            role_queue = self.role_queues[role]
            matches = role_queue.find_closest_players(
                team_features1,
                k=3,  # Get top 3 candidates
                mmr_threshold=300
            )

            if matches:
                # Take the closest match
                new_player = matches[0][1]
                team1.append(matches[0][1])
            else:
                return None  # Cannot complete team

            # find a soothing player for team 2
            matches = role_queue.find_closest_players(
                new_player.features,
                k=3,  # Get top 3 candidates
                mmr_threshold=300,
                player_id=new_player.id
            )
            if matches:
                # Take the closest match
                team2.append(matches[0][1])
                used_roles.add(role)
            else:
                return None  # Cannot complete team

        # Verify final match balance
        if self._is_balanced_match(team1, team2):
            mmr_diff = abs(np.mean([p.mmr for p in team1]) -
                           np.mean([p.mmr for p in team2]))
            return Match(team1, team2, mmr_diff)

        return None


def test_matchmaker(active_features_option='mmr, win_rate, games_played', players=None):
    matchmaker = ImprovedMatchmaker()
    # Add n players
    active_payers = players.sample(n=180)
    for player in active_payers.itertuples():
        player = Player(
            id=player.username,
            mmr= None ,
            win_rate=player.winrate,
            games_played=player.games_won, #NOT PLAYED
            role=player.most_played_role,
            rank=player.rank,
            division=player.division,
            lp=player.lp,
            kills=player.kills,
            death=player.death,
            assists=player.assists,
            avg_creeps_per_min=player.avg_creeps_per_min,
            avg_gold_per_min=player.avg_gold_per_min,
            calculated_kda = None
        )
        player.mmr = calculate_mmr(player.__dict__)
        player.calculated_kda = calculate_kda(player.__dict__)
        active_features_dict = {'mmr, win_rate, games_played': [player.mmr, player.win_rate, player.games_played],
                                'mmr': [player.mmr],
                                'mmr, win_rate, games_played, avg_creeps_per_min, avg_gold_per_min, kills, death, assists': [player.mmr, player.win_rate, player.games_played, player.avg_creeps_per_min, player.avg_gold_per_min, player.kills, player.death, player.assists],
                                'mmr, win_rate, games_played, avg_creeps_per_min, avg_gold_per_min, calculated_kda': [player.mmr, player.win_rate, player.games_played, player.avg_creeps_per_min, player.avg_gold_per_min, player.calculated_kda],
                                }
        player.features = np.array(active_features_dict[active_features_option])
        # player.features = np.array([player.mmr, player.win_rate, player.games_played])
        matchmaker.add_player(player)


    # Time the matching process
    start_time = time.time()
    final_match = matchmaker.find_match()
    end_time = time.time()

    if final_match:
        print(f"Formed teams in {end_time - start_time:.3f} seconds:")
        print("Team 1:")
        for player in final_match.team1:
            print(f"Role: {player.role}, Name: {player.id}, MMR: {player.mmr:.0f}, "
                  f"WR: {player.win_rate:.3f}, Games: {player.games_played}")

        # Calculate team1 stats
        team_mmr = np.mean([p.mmr for p in final_match.team1])
        mmr_spread = np.std([p.mmr for p in final_match.team1])
        print(f"\nTeam MMR: {team_mmr:.0f} ± {mmr_spread:.0f}")
        print()

        print("Team 2:")
        for player in final_match.team2:
            print(f"Role: {player.role}, Name: {player.id}, MMR: {player.mmr:.0f}, "
                  f"WR: {player.win_rate:.3f}, Games: {player.games_played}")

        # Calculate team2 stats
        team_mmr = np.mean([p.mmr for p in final_match.team2])
        mmr_spread = np.std([p.mmr for p in final_match.team2])
        print(f"\nTeam MMR: {team_mmr:.0f} ± {mmr_spread:.0f}")

        print(f"\nMMR Difference: {final_match.mmr_difference:.0f}\n")
        print("*"*100)

        return final_match


    else:
        print("Could not find balanced teams")


# === MAIN EXECUTION ===
if __name__ == "__main__":
    # More variables
    players = pd.read_csv('league_of_graphs_players_filtered.csv')
    num_matches = 10
    active_features_options = ['mmr, win_rate, games_played',
                               'mmr',
                               'mmr, win_rate, games_played, avg_creeps_per_min, avg_gold_per_min, kills, death, '
                               'assists',
                               'mmr, win_rate, games_played, avg_creeps_per_min, avg_gold_per_min, calculated_kda']

    matches = []
    for option in active_features_options:
        print(f"Testing with {option} features:")
        for i in range(num_matches):
            matches.append(test_matchmaker(option, players))
        print("*" * 100)
        print("*" * 100)
        print("*" * 100)
