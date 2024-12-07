from typing import List, Optional, Tuple
import numpy as np
from scipy.spatial import cKDTree
from data_classes import Player
from utils import normalize_features


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
        if not self.players or len(self.players) < 2:
            self.tree = None
            self.features = None
            return

        # Create feature array for all players
        self.features = np.array([
            normalize_features(player.features) for player in self.players
        ])
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
        """Find k closest players using the KD-tree."""
        # This function is our ANN search
        if self.needs_rebuild:
            self.rebuild_tree()

        # Need at least 2 players for role mirror
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