# field_util.py
import numpy as np
import pandas as pd

def generate_weighted_field(df: pd.DataFrame, n_teams: int = 1000) -> pd.DataFrame:
    """
    Generate a field of teams (one player per tier) using ownership weights.

    Parameters
    ----------
    df : pd.DataFrame
        Must include columns 'Tier', 'Player', and 'Own-Calc'.
    n_teams : int
        Number of teams to generate.

    Returns
    -------
    pd.DataFrame
        A (n_teams × n_tiers) DataFrame where each row is a simulated team
        and each column corresponds to a Tier (e.g., T1 ... T8).
    """
    required = {"Tier", "Player", "Own-Calc"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")

    df = df.copy()
    tiers = sorted(df["Tier"].unique())
    teams = []

    for _ in range(n_teams):
        team = {}
        for tier in tiers:
            group = df[df["Tier"] == tier]
            weights = group["Own-Calc"].fillna(0)
            if weights.sum() == 0:
                # fallback to uniform probabilities
                probs = np.ones(len(group)) / len(group)
            else:
                probs = weights / weights.sum()
            choice = np.random.choice(group["Player"], p=probs)
            team[f"T{tier}"] = choice
        teams.append(team)

    return pd.DataFrame(teams)
