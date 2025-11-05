# src/elo.py
import math
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class EloModel:
    k: float = 18.0
    hfa: float = 55.0           # ~ 1.5 to 2 points ~ 50-70 Elo; tune
    mov_scale: float = 2.2      # margin of victory dampening
    decay: float = 0.995        # weekly decay to emphasize recency
    ratings: dict = field(default_factory=lambda: defaultdict(lambda: 1500.0))

    def _expected(self, ra, rb, neutral=False):
        adj = 0.0 if neutral else self.hfa
        return 1.0 / (1 + 10 ** (-(ra + adj - rb) / 400))

    def update_game(self, home, away, home_pts, away_pts, neutral=False):
        pa = self._expected(self.ratings[home], self.ratings[away], neutral)
        sa = 1.0 if home_pts > away_pts else (0.5 if home_pts == away_pts else 0.0)
        margin = abs(home_pts - away_pts)
        mult = math.log(max(margin, 1) + 1) * (self.mov_scale)
        delta = self.k * mult * (sa - pa)
        self.ratings[home] += delta
        self.ratings[away] -= delta

    def decay_all(self):
        for t in list(self.ratings.keys()):
            self.ratings[t] = 1500 + (self.ratings[t] - 1500) * self.decay

    def win_prob(self, home, away, neutral=False):
        return self._expected(self.ratings[home], self.ratings[away], neutral)
