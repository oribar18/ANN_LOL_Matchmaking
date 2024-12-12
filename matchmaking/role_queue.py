from typing import List, Optional, Tuple
import numpy as np
from scipy.spatial import cKDTree
from data_processing.data_classes import Player
from utils.utils import normalize_features


class RoleQueue:
    """
       Represents a queue for a specific role in matchmaking, maintaining a KD-Tree for efficient search.

       Attributes:
           role (str): The role associated with the queue.
           players (list): List of player feature vectors.
           kd_tree (KDTree): KD-Tree for efficient nearest neighbor searches.
       """
    def __init__(self, role: str):
        """
        Initializes a RoleQueue with a given role.

        Args:
            role (str): The role for this queue.
        """
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
        """
        Rebuilds the KD-Tree using the current list of players in the queue.
        This improves the efficiency of nearest neighbor searches.
        """
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
        """
        Finds the k closest players using the KD-Tree.

        Args:
            target_features (np.ndarray): The feature vector of the target player.
            k (int, optional): The number of closest players to find. Default is 5.
            mmr_threshold (float, optional): Maximum MMR difference allowed. Default is 300.
            player_id (Optional[str], optional): ID of the player to exclude from results. Default is None.

        Returns:
            List[Tuple[float, Player]]: A list of tuples containing distances and player objects of the k closest players.
        """
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
