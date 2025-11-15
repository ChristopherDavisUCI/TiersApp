# optimals_utils.py
import numpy as np
import pandas as pd

def simulate_optimals(df: pd.DataFrame, sims: np.ndarray, top_k: int = 10) -> pd.DataFrame:
    """
    For each simulation, pick the highest-scoring player in Tiers 1–7
    and the top_k players in Tier 8, forming top_k teams per simulation.

    Parameters
    ----------
    df : pd.DataFrame
        Must include columns 'Player' and 'Tier'.
    sims : np.ndarray
        Shape (n_players, n_sims), simulated scores from run_player_simulations.
    top_k : int
        Number of Tier-8 variants to select per simulation.

    Returns
    -------
    pd.DataFrame
        Columns ['Sim', 'Rank', 'T1', ..., 'T8', 'TotalScore'].
    """
    tiers = sorted(df["Tier"].unique())
    n_sims = sims.shape[1]

    # Precompute player indices per tier
    tier_groups = {t: df[df["Tier"] == t].index.values for t in tiers}

    results = []

    for sim_idx in range(n_sims):
        scores = sims[:, sim_idx]

        # Pick best from tiers 1–7
        fixed_players = {}
        fixed_score = 0
        for t in tiers[:-1]:
            idxs = tier_groups[t]
            best_idx = idxs[np.argmax(scores[idxs])]
            fixed_players[t] = df.loc[best_idx, "Player"]
            fixed_score += scores[best_idx]

        # Top_k from tier 8
        t8_idxs = tier_groups[tiers[-1]]
        top8_order = np.argsort(scores[t8_idxs])[::-1][:top_k]
        for rank, rel_idx in enumerate(top8_order, start=1):
            player_idx = t8_idxs[rel_idx]
            total_score = fixed_score + scores[player_idx]
            row = {
                "Sim": sim_idx + 1,
                "Rank": rank,
                **{f"T{t}": fixed_players[t] for t in tiers[:-1]},
                f"T{tiers[-1]}": df.loc[player_idx, "Player"],
                "TotalScore": total_score,
            }
            results.append(row)

    return pd.DataFrame(results)
