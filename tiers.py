import streamlit as st
import pandas as pd
import numpy as np
import os
from simulation_utils import run_player_simulations  # modular import
from field_utils import generate_weighted_field

from ev_utils import (
    compute_team_scores,
    compute_ev_against_field,
    compute_bundle_ev,
)
from optimals_utils import simulate_optimals



st.set_page_config(page_title="Editable Projections and Ownership", layout="wide")
st.title("Editable Projections and Ownership Grid")

st.write("Most of this was written by ChatGPT.")

DATA_PATH = os.path.join("data", "splash_tiers_week11.csv")

# --- Function to calculate Own-Calc ---
def compute_own_calc(df):
    df = df.copy()
    df["Own-Calc"] = np.nan
    for tier, group in df.groupby("Tier"):
        total_entered = group["Own"].sum(skipna=True)
        missing_mask = group["Own"].isna()
        n_missing = missing_mask.sum()
        if n_missing > 0:
            remaining = max(0, 100 - total_entered)
            fill_value = round(remaining / n_missing, 2)
            df.loc[group.index[missing_mask], "Own-Calc"] = fill_value
        df.loc[group.index[~missing_mask], "Own-Calc"] = group.loc[~missing_mask, "Own"]
    df["Own-Calc"] = df["Own-Calc"].round(2)
    return df

# --- Load data once ---
if "df" not in st.session_state:
    if not os.path.exists(DATA_PATH):
        st.error(f"File not found: {DATA_PATH}")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    if "Projected Points" in df.columns and "Proj" not in df.columns:
        df = df.rename(columns={"Projected Points": "Proj"})
    if "Std" not in df.columns:
        df["Std"] = df["Proj"] / 2.0
    if "Own" not in df.columns:
        df["Own"] = np.nan
    if "Own-Calc" not in df.columns:
        df["Own-Calc"] = np.nan
    df = compute_own_calc(df)
    st.session_state.df = df

# --- Editable grid ---
st.subheader("Edit Projections and Ownership")

edited_df = st.data_editor(
    st.session_state.df,
    column_config={
        "Proj": st.column_config.NumberColumn("Proj", format="%.2f"),
        "Std": st.column_config.NumberColumn("Std", format="%.2f"),
        "Own": st.column_config.NumberColumn("Own", format="%.2f"),
        "Own-Calc": st.column_config.NumberColumn("Own-Calc", format="%.2f"),
    },
    disabled=["Player", "Tier", "Team", "Opponent", "Own-Calc"],
    use_container_width=True,
    num_rows="dynamic",
    key="main_editor",
)

# --- Calculate ownership button ---
if st.button("Calculate Ownership"):
    st.session_state.df = compute_own_calc(edited_df)
    st.success("Ownership recalculated successfully!")
    st.rerun()

# --- Download CSV (immediately after Calculate) ---
csv = st.session_state.df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download updated CSV",
    csv,
    "updated_projections.csv",
    "text/csv",
    help="Download your modified projections after recalculation",
)

# --- Simulation controls ---
st.subheader("Simulations")

num_sims = st.number_input(
    "Number of simulations to run",
    min_value=1,
    max_value=1000,
    value=100,
    step=10,
    help="Each player will have scores drawn from a Normal(mean=Proj, sd=Std)",
)

if st.button("Run Simulations"):
    df = st.session_state.df
    sims = run_player_simulations(df, num_sims)
    st.session_state.sims = sims
    st.success(f"Simulations completed ({num_sims} per player).")

# --- Generate Field controls ---
st.subheader("Generate Field")

num_teams = st.number_input(
    "Number of teams to generate",
    min_value=1,
    max_value=10000,
    value=1000,
    step=100,
    help="Each team consists of one player per tier, drawn with probabilities from Own-Calc.",
)

if st.button("Generate Field"):
    df = st.session_state.df
    field_df = generate_weighted_field(df, n_teams=num_teams)
    st.session_state.field = field_df
    st.success(f"Generated {num_teams} teams successfully!")

    # Show preview of first few teams
    st.dataframe(field_df.head(), use_container_width=True)

    # Download option for generated field
    csv_field = field_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download generated field",
        csv_field,
        "generated_field.csv",
        "text/csv",
        help="Download the randomly generated teams.",
    )

# --- Simulate Optimals ---
st.subheader("Simulate Optimals")

st.write("For a fixed sim, the only variation here is in Tier 8.")

top_k = st.number_input(
    "Number of best teams per simulation",
    min_value=1,
    max_value=20,
    value=10,
    step=1,
)

if st.button("Simulate Optimals"):

    if "sims" not in st.session_state:
        st.error("You must run simulations first.")
    else:
        df = st.session_state.df
        sims = st.session_state.sims
        optimals_df = simulate_optimals(df, sims, top_k=top_k)
        st.session_state.optimals = optimals_df
        st.success(f"Generated {len(optimals_df):,} optimal teams "
                   f"({top_k} per simulation × {sims.shape[1]} simulations).")

        st.dataframe(optimals_df.head(), use_container_width=True)

        # Download
        csv_opt = optimals_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download simulated optimals",
            csv_opt,
            "simulated_optimals.csv",
            "text/csv",
            help="Download the top simulated teams for each simulation.",
        )

st.subheader("Compute EVs (vs Field)")

top_n = st.number_input(
    "Show top N optimals by EV",
    min_value=1, max_value=100, value=20, step=1,
)

if st.button("Compute EVs"):
    # Preconditions
    if "sims" not in st.session_state:
        st.error("Please run simulations first.")
    elif "field" not in st.session_state:
        st.error("Please generate a field first.")
    else:
        df_players = st.session_state.df           # player master DF
        sims = st.session_state.sims               # (n_players, n_sims)
        field_df = st.session_state.field          # teams (T1..T8)

        # Build/refresh optimals if not already present
        if "optimals" not in st.session_state:
            st.session_state.optimals = simulate_optimals(df_players, sims, top_k=10)
        optimals_df = st.session_state.optimals

        # Scores
        field_scores    = compute_team_scores(field_df,    sims, df_players)     # (F, S)
        optimals_scores = compute_team_scores(optimals_df, sims, df_players)     # (O, S)

        # EV vs field
        import os, pandas as pd
        payouts_df = pd.read_csv(os.path.join("data", "payouts_11.csv"))
        ev = compute_ev_against_field(optimals_scores, field_scores, payouts_df)  # (O,)

        # Store & display
        st.session_state.field_scores    = field_scores
        st.session_state.optimals_scores = optimals_scores
        st.session_state.optimals_ev     = ev

        # Combine into a table
        out = optimals_df.copy()
        out["EV"] = ev
        out = out.sort_values("EV", ascending=False).head(top_n)
        st.session_state.optimals_top = out

        st.success(f"Computed EV for {len(optimals_df):,} optimals against the field.")
        st.dataframe(out, use_container_width=True)

st.subheader("Lock Teams")

if "optimals_top" in st.session_state:
    top_tbl = st.session_state.optimals_top
    player_cols = [c for c in top_tbl.columns if c.startswith("T") and c[1:].isdigit()]

    # Always generate team_id safely
    top_tbl = top_tbl.assign(
        team_id=top_tbl[player_cols]
            .fillna("")
            .astype(str)
            .agg(" | ".join, axis=1)
    )

    # --- Display currently locked teams (if any) ---
    if "locked_df" in st.session_state and not st.session_state.locked_df.empty:
        st.markdown("### 🔒 Currently Locked Teams")
        st.dataframe(
            st.session_state.locked_df[["EV"] + player_cols]
            if "EV" in st.session_state.locked_df.columns
            else st.session_state.locked_df[player_cols],
            use_container_width=True,
        )

        # Download button for locked teams
        csv_locked = st.session_state.locked_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download locked teams",
            csv_locked,
            "locked_teams.csv",
            "text/csv",
            help="Download the teams you have currently locked.",
            key="download_locked_teams",   # unique key added
        )

        # Clear all button
        if st.button("Clear all locked teams", key="clear_locked"):
            st.session_state.locked_team_ids = []
            st.session_state.locked_df = pd.DataFrame()
            st.success("All locked teams cleared.")

    # --- Multiselect for new locks ---
    st.markdown("### ➕ Add New Locks from Current Top 20")
    choices = top_tbl["team_id"].tolist()
    locked_ids = set(st.session_state.get("locked_team_ids", []))
    # Filter out already locked teams from the multiselect list
    new_choices = [c for c in choices if c not in locked_ids]

    if new_choices:
        selected_to_lock = st.multiselect(
            "Select new teams to lock:",
            new_choices,
            help="Choose additional teams to lock from the current top candidates.",
        )

        if st.button("Lock Selected"):
            if "locked_team_ids" not in st.session_state:
                st.session_state.locked_team_ids = []
            # Add new locks (dedup)
            for tid in selected_to_lock:
                if tid not in st.session_state.locked_team_ids:
                    st.session_state.locked_team_ids.append(tid)

            st.success(f"Locked {len(selected_to_lock)} new team(s). Total locked: {len(st.session_state.locked_team_ids)}.")

            # Update locked_df
            locked_df = top_tbl[top_tbl["team_id"].isin(st.session_state.locked_team_ids)].copy()
            st.session_state.locked_df = locked_df[player_cols + (["EV"] if "EV" in locked_df.columns else [])]
            st.rerun()
    else:
        st.info("All current top candidates are already locked.")
else:
    st.info("Compute EVs first to see and lock top teams.")


# --- Download locked teams ---
if "locked_df" in st.session_state and not st.session_state.locked_df.empty:
    csv_locked = st.session_state.locked_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download locked teams",
        csv_locked,
        "locked_teams.csv",
        "text/csv",
        help="Download the teams you have currently locked.",
    )



st.subheader("Find next top teams")

st.write("Finds best teams, taking into account your current locked lineups.")

if st.button("Recompute top teams"):
    
    # Checks
    req = ["field_scores", "optimals_scores", "optimals", "df", "sims"]
    if not all(k in st.session_state for k in req):
        st.error("Please run simulations, generate field, and compute EVs first.")
    else:
        import os, pandas as pd

        df_players     = st.session_state.df
        sims           = st.session_state.sims
        field_scores   = st.session_state.field_scores        # (F, S)
        optimals_df    = st.session_state.optimals            # teams T1..T8
        optimals_scores= st.session_state.optimals_scores     # (O, S)

        payouts_df = pd.read_csv(os.path.join("data", "payouts_11.csv"))

        # Build team_id for all optimals
        player_cols = [c for c in optimals_df.columns if c.startswith("T")]
        team_ids_all = (
            optimals_df[player_cols]
            .fillna("")            # replace NaN
            .astype(str)           # ensure all strings
            .agg(" | ".join, axis=1)
        )

        # Locked subset
        locked_ids = set(st.session_state.get("locked_team_ids", []))
        locked_mask = team_ids_all.isin(locked_ids)
        locked_scores = optimals_scores[locked_mask.values, :] if locked_mask.any() else np.zeros((0, sims.shape[1]))

        # Candidates = optimals not locked
        cand_mask = ~locked_mask
        cand_df = optimals_df[cand_mask].reset_index(drop=True)
        cand_scores = optimals_scores[cand_mask.values, :]

        # Compute bundle EV for candidates
        bundle_ev = compute_bundle_ev(
            locked_scores=locked_scores,
            candidate_scores=cand_scores,
            field_scores=field_scores,
            payouts_df=payouts_df
        )

        cand_out = cand_df.copy()
        cand_out["BundleEV"] = bundle_ev
        # Show new Top-N (same control top_n as above)
        top_bundle = cand_out.sort_values("BundleEV", ascending=False).head(top_n)
        st.session_state.top_bundle = top_bundle

        st.success(f"Bundle EV computed for {len(cand_out):,} candidates. Showing top {top_n}.")
        st.dataframe(top_bundle, use_container_width=True)
        # Update top list for the Lock Teams section to use the new bundle results
        st.session_state.optimals_top = top_bundle

        # Optional download
        csv_bundle = top_bundle.to_csv(index=False).encode("utf-8")
        st.download_button("Download top bundle EV teams", csv_bundle, "bundle_ev_top.csv", "text/csv")
