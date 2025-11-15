# simulation_utils.py
import numpy as np
import pandas as pd

def run_player_simulations(df: pd.DataFrame, num_sims: int) -> np.ndarray:
    """
    Generate simulated scores for each player.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain numeric columns 'Proj' and 'Std'.
    num_sims : int
        Number of simulations to run per player.

    Returns
    -------
    np.ndarray
        Array of shape (num_players, num_sims) containing simulated scores.
    """
    if "Proj" not in df.columns or "Std" not in df.columns:
        raise ValueError("DataFrame must contain 'Proj' and 'Std' columns.")
    if num_sims < 1:
        raise ValueError("num_sims must be at least 1.")

    # Each row = player; each column = simulation outcome
    sims = np.random.normal(
        loc=df["Proj"].values[:, None],
        scale=df["Std"].values[:, None],
        size=(len(df), num_sims),
    )
    return sims
