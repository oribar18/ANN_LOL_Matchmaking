from typing import List, Optional
import numpy as np
from data_processing.data_classes import Player, Match
from matchmaking.role_queue import RoleQueue


class CktreeMatchmaker:
    """
    A class for matchmaking players into balanced teams using role-based queues and KD-tree searches.
    """
    def __init__(self):
        """
        Initialize role queues for each player role.
        """
        self.role_queues = {
            role: RoleQueue(role)
            for role in ["Jungler", "Support", "Mid", "Top", "AD Carry"]
        }

    def add_player(self, player: Player):
        """Add player to appropriate role queue."""
        self.role_queues[player.role].add_player(player)

    def _calculate_team_stats(self, team: List[Player]) -> dict:
        """
        Calculate statistics for a given team.

        Args:
            team (List[Player]): List of players in a team.

        Returns:
            dict: A dictionary containing average MMR, average win rate, and MMR spread.
        """
        return {
            'avg_mmr': np.mean([p.mmr for p in team]),
            'avg_wr': np.mean([p.win_rate for p in team]),
            'mmr_spread': np.std([p.mmr for p in team])
        }

    def _is_balanced_match(self, team1: List[Player], team2: List[Player],
                           max_mmr_diff: float = 100) -> bool:
        """
        Check if two teams are balanced based on average MMR and MMR spread.

        Args:
            team1 (List[Player]): First team of players.
            team2 (List[Player]): Second team of players.
            max_mmr_diff (float): Maximum allowed MMR difference between teams.

        Returns:
            bool: True if the teams are balanced, False otherwise.
        """
        team1_stats = self._calculate_team_stats(team1)
        team2_stats = self._calculate_team_stats(team2)
        mmr_difference = abs(team1_stats['avg_mmr'] - team2_stats['avg_mmr'])
        return (mmr_difference <= max_mmr_diff and
                team1_stats['mmr_spread'] < 400 and
                team2_stats['mmr_spread'] < 400)

    def find_match(self) -> Optional[Match]:
        """
        Find a balanced match using KD-tree searches across all roles.

        Returns:
            Optional[Match]: A match object if a balanced match is found, None otherwise.
        """
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
                return match

        return None

    def _build_teams_from_starter(self, starter: Player, role_order: List[str]) -> Optional[Match]:
        """
        Attempt to build two balanced teams starting with a given player.

        Args:
            starter (Player): The player to start team formation.
            role_order (List[str]): Order of roles to fill for the teams.

        Returns:
            Optional[Match]: A Match object if successful, None otherwise.
        """
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
            return None

        for role in role_order:
            if role in used_roles:
                continue

            # Update target features based on current team
            feature_matrix = np.array([p.features for p in team1])
            team_features1 = np.mean(feature_matrix, axis=0)

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
                team1.append(new_player)
            else:
                return None

            # find a soothing player for team 2
            matches = role_queue.find_closest_players(
                new_player.features,
                k=3,
                mmr_threshold=300,
                player_id=new_player.id
            )
            if matches:
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
