# src/bracket.py
import numpy as np
import pandas as pd

def build_cfp_field(rankings_df, champs_set, n_at_large=7):
    """
    rankings_df: columns [rank, team]; rank ascending = better
    champs_set: set of conference champion team names (week's projection or actual)
    Returns ordered seeds 1..12 following 5 champs + 7 at-large (top-4 seeds = best 4 champs)
    """
    champs = [t for t in rankings_df['team'] if t in champs_set]
    champs_sorted = [t for t in rankings_df['team'] if t in champs_set][:5]
    at_large = [t for t in rankings_df['team'] if t not in champs_set][:n_at_large]
    field = champs_sorted + at_large
    # Seed: top 4 of champs get byes; the rest in standard bracket positions
    return field

def game_prob(home, away, neutral, prob_func):
    # prob_func(home, away, neutral)->home win prob
    return prob_func(home, away, neutral)

def simulate_bracket(field, prob_func, n_sims=10000):
    """
    field: list of 12 seeds (1 best), seeds 1-4 have byes.
    prob_func: function(teamA, teamB, neutral) -> P(teamA beats teamB)
    Returns advancement odds per team.
    """
    teams = list(field)
    counts = {t: {'QF':0,'SF':0,'F':0,'CHAMP':0} for t in teams}

    # A full seeding layout gets verbose; sketch:
    # First round: 5 vs 12, 6 vs 11, 7 vs 10, 8 vs 9 (higher seeds host)
    # QF: top-4 seeds host winners; SF & Final at neutral sites.

    # Implement the tree you prefer; below is an outline:
    def play(a, b, neutral=False):
        p = prob_func(a, b, neutral)
        return a if np.random.rand() < p else b

    for _ in range(n_sims):
        s = teams[:]  # 1..12
        # R1
        r1 = [
            play(s[4], s[11], neutral=False),
            play(s[5], s[10], neutral=False),
            play(s[6], s[9],  neutral=False),
            play(s[7], s[8],  neutral=False),
        ]
        # QF vs seeds 1..4 (hosted)
        qf_pairs = [(s[0], r1[3]), (s[3], r1[0]), (s[1], r1[2]), (s[2], r1[1])]
        qf_w = [play(a,b, neutral=False) for a,b in qf_pairs]
        for t in qf_w: counts[t]['QF'] += 1

        # SF (neutral)
        sf_pairs = [(qf_w[0], qf_w[1]), (qf_w[2], qf_w[3])]
        sf_w = [play(a,b, neutral=True) for a,b in sf_pairs]
        for t in sf_w: counts[t]['SF'] += 1

        # F (neutral)
        champ = play(sf_w[0], sf_w[1], neutral=True)
        for t in [sf_w[0], sf_w[1]]: counts[t]['F'] += 1
        counts[champ]['CHAMP'] += 1

    out = pd.DataFrame.from_dict(counts, orient='index')
    out = out.div(n_sims).reset_index().rename(columns={'index':'team'})
    return out
