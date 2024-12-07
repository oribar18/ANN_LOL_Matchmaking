from dataclasses import dataclass
from typing import List, Optional
import numpy as np

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
