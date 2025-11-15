# ev_utils.py
import numpy as np
import pandas as pd
from typing import Dict, Tuple

# --------- Core helpers ---------

def _build_player_index(df: pd.DataFrame) -> Dict[str, int]:
    """
    Map Player name -> row index in df.
    Assumes Player names are unique in df.
    """
    return {p: i for i, p in enumerate(df["Player"].tolist())}

def _teams_to_row_indices(teams_df: pd.DataFrame, player_index: Dict[str, int]) -> np.ndarray:
    """
    Convert team columns (T1 ... Tn) of player names to a matrix of row indices into df.
    Shape: (n_teams, n_tiers)
    """
    tier_cols = [c for c in teams_df.columns if c.startswith("T") and c[1:].isdigit()]
    tier_cols_sorted = sorted(tier_cols, key=lambda x: int(x[1:]))  # T1, T2, ... in order
    idx = np.empty((len(teams_df), len(tier_cols_sorted)), dtype=int)

    for tcol_i, tcol in enumerate(tier_cols_sorted):
        idx[:, tcol_i] = [player_index[name] for name in teams_df[tcol].values]
    return idx

def compute_team_scores(
    teams_df: pd.DataFrame,
    sims: np.ndarray,
    player_df: pd.DataFrame,
) -> np.ndarray:
    """
    Sum simulated scores across the 8 players that form each team.

    teams_df: columns T1 ... T8 (player names)
    sims:     shape (n_players, n_sims) aligned with player_df row order
    returns:  team_scores, shape (n_teams, n_sims)
    """
    player_index = _build_player_index(player_df)
    team_row_idx = _teams_to_row_indices(teams_df, player_index)  # (n_teams, n_tiers)
    # Gather sims rows and sum across tiers
    # sims[team_row_idx] -> (n_teams, n_tiers, n_sims)
    team_scores = sims[team_row_idx, :].sum(axis=1)
    return team_scores  # (n_teams, n_sims)


# --------- Places & EV against the FIELD (fast, per-sim searchsorted) ---------

def _places_against_field(team_scores: np.ndarray, field_scores: np.ndarray) -> np.ndarray:
    """
    Compute places of team_scores relative to field_scores per simulation.

    team_scores:  (N, S)
    field_scores: (F, S)
    returns:      places (N, S), 1 = best
    """
    N, S = team_scores.shape
    places = np.empty((N, S), dtype=np.int32)

    # For each sim, sort field descending; place = 1 + #field strictly greater
    # Use searchsorted on negatives to get "descending" behavior in O(log F)
    for j in range(S):
        field_col = field_scores[:, j]
        field_sorted_desc = -np.sort(-field_col)  # descending
        # Convert to ascending by negation for searchsorted
        asc = -field_sorted_desc
        # For all teams in this sim:
        # place = np.searchsorted(asc, -team_score, side='left') + 1
        places[:, j] = np.searchsorted(asc, -team_scores[:, j], side="left") + 1
    return places


def compute_ev_against_field(
    optimals_scores: np.ndarray,
    field_scores: np.ndarray,
    payouts_df: pd.DataFrame
) -> np.ndarray:
    """
    EV for each 'optimals' team using places relative to the FIELD.
    Returns: ev (N_opt,)
    """
    # Map place -> payout (0 if missing)
    payout_map = payouts_df.set_index("place")["payout"].to_dict()

    # Places of optimals vs field per sim: (N_opt, S)
    places = _places_against_field(optimals_scores, field_scores)

    # Convert places to payouts per sim
    # Vectorized via loop over sims for mapping
    N, S = places.shape
    payouts = np.zeros((N, S), dtype=float)
    # Build a fast vectorized mapper by array indexing trick:
    # Get max place present in payouts file to build small array; else fallback to dict per sim
    max_place_in_table = int(payouts_df["place"].max()) if len(payouts_df) else 0

    for j in range(S):
        pcol = places[:, j]
        # Fast path if places are within a modest range
        if max_place_in_table > 0:
            # Build lookup array (size max_place_in_table+1) with zeros default
            lut = np.zeros(max_place_in_table + 1, dtype=float)
            for pl, pay in payout_map.items():
                if pl <= max_place_in_table:
                    lut[pl] = pay
            # For places > max_place_in_table, payout = 0 automatically
            mask = pcol <= max_place_in_table
            out = np.zeros_like(pcol, dtype=float)
            out[mask] = lut[pcol[mask]]
            payouts[:, j] = out
        else:
            # Degenerate case
            payouts[:, j] = 0.0

    # EV = mean over sims
    ev = payouts.mean(axis=1)
    return ev  # (N_opt,)


# --------- Bundle EV (only the TOP place counts across locked + candidate) ---------

def compute_bundle_ev(
    locked_scores: np.ndarray,      # (L, S) possibly L=0
    candidate_scores: np.ndarray,   # (C, S)
    field_scores: np.ndarray,       # (F, S)
    payouts_df: pd.DataFrame
) -> np.ndarray:
    """
    For each candidate team, the bundle (locked + candidate) gets payout of ONLY the best place
    achieved by any team in the bundle in each simulation. Places are relative to the FIELD.

    Returns: bundle_ev for each candidate, shape (C,)
    """
    payout_map = payouts_df.set_index("place")["payout"].to_dict()
    max_place_in_table = int(payouts_df["place"].max()) if len(payouts_df) else 0

    # Precompute field sorted (descending) per sim for fast place lookups
    # We'll do the same searchsorted trick on negatives
    F, S = field_scores.shape
    field_sorted_desc = np.sort(-field_scores, axis=0)  # ascending on negatives => columns are descending scores

    # Helper: place_of_scores vs field (vectorized per sim)
    def places_vs_field(scores: np.ndarray) -> np.ndarray:
        # scores: (K, S)
        # For each sim j: place = searchsorted(field_sorted_desc[:, j], -scores[:, j], 'left') + 1
        K = scores.shape[0]
        places = np.empty((K, S), dtype=np.int32)
        for j in range(S):
            asc_neg = field_sorted_desc[:, j]  # ascending negatives (i.e., descending originals)
            places[:, j] = np.searchsorted(asc_neg, -scores[:, j], side="left") + 1
        return places  # (K, S)

    # Locked places per sim (if no locked teams, treat as very large place)
    if locked_scores.size == 0:
        # No locks: min_locked_place = +infty → new team’s place dominates
        min_locked_place = None
    else:
        locked_places = places_vs_field(locked_scores)  # (L, S)
        # Per sim, the best (min) place among locks
        min_locked_place = locked_places.min(axis=0)    # (S,)

    # Candidate places
    cand_places = places_vs_field(candidate_scores)  # (C, S)

    # For each candidate, per sim: best_place = min(cand_place, min_locked_place)
    if min_locked_place is None:
        best_places = cand_places
    else:
        best_places = np.minimum(cand_places, min_locked_place[None, :])  # (C, S)

    # Map places -> payouts per sim, then average
    C = best_places.shape[0]
    payouts = np.zeros_like(best_places, dtype=float)
    for j in range(S):
        col = best_places[:, j]
        if max_place_in_table > 0:
            lut = np.zeros(max_place_in_table + 1, dtype=float)
            for pl, pay in payout_map.items():
                if pl <= max_place_in_table:
                    lut[pl] = pay
            mask = col <= max_place_in_table
            out = np.zeros_like(col, dtype=float)
            out[mask] = lut[col[mask]]
            payouts[:, j] = out
        else:
            payouts[:, j] = 0.0

    return payouts.mean(axis=1)  # (C,)


def compute_bundle_ev_marginal(
    locked_scores: np.ndarray,    # (L, S) possibly L=0
    candidate_scores: np.ndarray, # (C, S)
    field_scores: np.ndarray,     # (F, S)
    payouts_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (bundle_ev_abs, bundle_ev_marginal) for each candidate.

    - bundle_ev_abs: absolute bundle EV = E[max(payout_locked, payout_candidate)]
    - bundle_ev_marginal: incremental EV over locked bundle = E[max(...) - payout_locked]

    Places are always computed relative to FIELD (opponents).
    """
    # Build place->payout LUT
    payout_map = payouts_df.set_index("place")["payout"].to_dict()
    max_place = int(payouts_df["place"].max()) if len(payouts_df) else 0
    lut = np.zeros(max_place + 1, dtype=float)
    for pl, pay in payout_map.items():
        if pl <= max_place:
            lut[pl] = float(pay)

    # Prepare sorted field per sim (descending) via negatives (ascending)
    F, S = field_scores.shape
    field_sorted_neg = np.sort(-field_scores, axis=0)  # ascending negatives

    def places_vs_field(scores: np.ndarray) -> np.ndarray:
        # scores: (K, S)
        K = scores.shape[0]
        places = np.empty((K, S), dtype=np.int32)
        for j in range(S):
            # place = #field with strictly greater score + 1
            places[:, j] = np.searchsorted(field_sorted_neg[:, j], -scores[:, j], side="left") + 1
        return places

    # Locked payout per sim (vector length S)
    if locked_scores.size == 0:
        payout_locked = np.zeros((1, S), dtype=float)  # effectively all zeros
        min_locked_place = None
    else:
        locked_places = places_vs_field(locked_scores)           # (L, S)
        best_locked_places = locked_places.min(axis=0)           # (S,)
        # Map to payout (places > max_place -> 0)
        payout_locked = np.zeros(S, dtype=float)
        mask = best_locked_places <= max_place
        payout_locked[mask] = lut[best_locked_places[mask]]

    # Candidate payouts
    cand_places = places_vs_field(candidate_scores)              # (C, S)
    C = cand_places.shape[0]
    payout_cand = np.zeros_like(cand_places, dtype=float)        # (C, S)
    if max_place > 0:
        for j in range(S):
            col = cand_places[:, j]
            m = col <= max_place
            if m.any():
                payout_cand[m, j] = lut[col[m]]

    # Absolute bundle EV: E[ max(payout_locked, payout_candidate) ]
    if locked_scores.size == 0:
        bundle_abs = payout_cand.mean(axis=1)
        bundle_marginal = bundle_abs.copy()  # when no locks, marginal == absolute
    else:
        # Broadcast payout_locked (S,) against (C,S)
        better = np.maximum(payout_cand, payout_locked[None, :])
        bundle_abs = better.mean(axis=1)
        # Marginal EV: E[max(...) - payout_locked]
        incr = better - payout_locked[None, :]
        bundle_marginal = incr.mean(axis=1)

    return bundle_abs, bundle_marginal
