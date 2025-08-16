# App_learn_ranker.py
# ============================================================
# 📚 Learner — HR Day Ranker from Leaderboards + Event Parquet
# - Upload merged leaderboards (your 7+ days) + event-level parquet (with hr_outcome)
# - Robust label join: tries (date+ID), then (date+name_norm), with dtype harmonization
# - Name normalization: accents, punctuation, common suffixes, known aliases
# - Trains the *small* LogisticRegression learner used by your prediction app:
#     features = ['base_prob','logit_p','log_overlay','ranker_z',
#                 'overlay_multiplier','final_multiplier','prob_2tb','prob_rbi']
# - Exports learning_ranker.pkl (model + features) and labeled_leaderboard.csv
# - No plots, no tuner, no graphs
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import io
import pickle
from datetime import datetime
from unidecode import unidecode

# ML
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# --------------------------- UI ---------------------------
st.set_page_config(page_title="📚 Learner — HR Day Ranker", layout="wide")
st.title("📚 Learner — HR Day Ranker from Leaderboards + Event Parquet")

st.markdown(
    "Upload your **merged leaderboard CSV** (combined days) and the **event-level Parquet** "
    "(must include `hr_outcome`). I’ll label each leaderboard row with the **true next-day HR outcome** "
    "and train the **learning ranker** your prediction app expects."
)

with st.expander("What this trains & exports", expanded=False):
    st.write("""
    - **Trains**: a small LogisticRegression model on your blended signals so your app can re-rank.
    - **Exports**: `learning_ranker.pkl` with:
        - `model`: fitted LogisticRegression
        - `features`: the exact feature list your prediction app uses
      And `labeled_leaderboard.csv` for audit/review.
    - **Preserves**: `prob_2tb`, `prob_rbi` if present.
    """)

lb_file = st.file_uploader("Merged leaderboard CSV (combined across days)", type=["csv"])
ev_file = st.file_uploader("Event-level Parquet (with hr_outcome)", type=["parquet"])

season_year = st.number_input(
    "Season year (used only if your leaderboard 'game_date' looks like '8_13')",
    min_value=2018, max_value=2100, value=2025, step=1
)

# --------------------------- Helpers ---------------------------
def to_date_ymd(x, season_year_hint=2025):
    """
    Accepts either already-YYYY-MM-DD or formats like '8_13'/'8-13'/'8/13'.
    Returns pd.Timestamp('YYYY-MM-DD') or NaT.
    """
    if pd.isna(x):
        return pd.NaT
    s = str(x).strip()
    # Already ISO?
    try:
        if len(s) >= 8 and "-" in s:
            return pd.to_datetime(s, errors="coerce", utc=False).date()
    except Exception:
        pass
    # M_D style (8_13, 8-13, 8/13)
    s2 = (
        s.replace("_", "-")
        .replace("/", "-")
        .replace("\\", "-")
        .replace(".", "-")
    )
    parts = s2.split("-")
    if len(parts) == 2 and all(p.isdigit() for p in parts):
        m, d = parts
        try:
            return datetime(int(season_year_hint), int(m), int(d)).date()
        except Exception:
            return pd.NaT
    # Try last fallback
    try:
        return pd.to_datetime(s, errors="coerce", utc=False).date()
    except Exception:
        return pd.NaT

SUFFIXES = [
    ", JR", " JR", " JR.", ", SR", " SR", " SR.", " II", " III", " IV",
    ".", ",", "'", '"'
]

ALIASES = {
    # common alias fixes if they ever appear
    "PETER ALONSO": "PETE ALONSO",
    "JULIO RODRIGUEZ": "JULIO RODRÍGUEZ",
    "J C KAYFUS": "C.J. KAYFUS",
    "CJ KAYFUS": "C.J. KAYFUS",
    "C. J. KAYFUS": "C.J. KAYFUS",
    "MICHAEL A TAYLOR": "MICHAEL A. TAYLOR",
    "J T REALMUTO": "J.T. REALMUTO",
}

def clean_name(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = unidecode(s).upper()
    # strip common suffixes and punctuation
    for suf in SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)]
    s = " ".join(s.split())  # collapse spaces
    s = ALIASES.get(s, s)
    return s

def std_team(x):
    if pd.isna(x):
        return ""
    return str(x).strip().upper()

def zscore(a):
    a = np.asarray(a, dtype=float)
    mu = np.nanmean(a); sd = np.nanstd(a) + 1e-9
    return (a - mu) / sd

# --------------------------- Pipeline ---------------------------
if lb_file is not None and ev_file is not None:
    # ---- Load ----
    try:
        lb = pd.read_csv(lb_file)
    except UnicodeDecodeError:
        lb = pd.read_csv(lb_file, encoding="latin1")
    ev = pd.read_parquet(ev_file)

    st.write(f"Leaderboard rows: {len(lb):,} | Event rows: {len(ev):,}")

    # ---- Minimal sanity: hr_outcome must exist in event parquet ----
    if "hr_outcome" not in ev.columns:
        st.error("Event-level Parquet is missing `hr_outcome`. Cannot proceed.")
        st.stop()

    # ---- Normalize leaderboard ----
    lb = lb.copy()

    # date
    if "game_date" not in lb.columns:
        st.error("Leaderboard CSV must have a `game_date` column.")
        st.stop()
    lb["game_date"] = lb["game_date"].apply(lambda s: to_date_ymd(s, season_year))

    # player & team normalization
    if "player_name" not in lb.columns:
        st.error("Leaderboard CSV must have `player_name`.")
        st.stop()
    lb["player_name_norm"] = lb["player_name"].astype(str).apply(clean_name)

    if "team_code" in lb.columns:
        lb["team_code_std"] = lb["team_code"].astype(str).apply(std_team)
    else:
        lb["team_code_std"] = ""

    # optional IDs if your leaderboard also includes them in the future
    if "batter_id" in lb.columns:
        # keep both raw and normalized string form
        lb["batter_id_join"] = lb["batter_id"].astype("Int64").astype(str)
    else:
        lb["batter_id_join"] = ""

    # ---- Normalize events ----
    ev = ev.copy()
    if "game_date" not in ev.columns:
        st.error("Event parquet must include `game_date`.")
        st.stop()

    ev["game_date"] = pd.to_datetime(ev["game_date"], errors="coerce", utc=False).dt.date

    # player_name exists in ev
    if "player_name" not in ev.columns:
        st.error("Event parquet must include `player_name`.")
        st.stop()
    ev["player_name_norm"] = ev["player_name"].astype(str).apply(clean_name)

    # batter_id optional in ev (usually present)
    if "batter_id" in ev.columns:
        ev["batter_id_join"] = ev["batter_id"].astype("Int64").astype(str)
    else:
        ev["batter_id_join"] = ""

    # reduce events to per-day labels for each batter
    # Each row should represent the batter's **game day** and whether they homered.
    ev_daily = (
        ev.groupby(["game_date", "player_name_norm", "batter_id_join"], dropna=False)["hr_outcome"]
        .max()
        .reset_index()
    )

    # ---------------------------------------------------------------
    # Robust JOIN: try (date+ID) when both sides have non-empty IDs w/ overlaps,
    # else fallback to (date+name_norm).
    # ---------------------------------------------------------------
    def safe_join(left, right):
        left = left.copy()
        right = right.copy()

        # Prepare keys
        # 1) Strong join: date + batter_id_join (non-empty on both)
        has_left_id = left["batter_id_join"].astype(str).str.len() > 0
        has_right_id = right["batter_id_join"].astype(str).str.len() > 0
        can_id_join = has_left_id.any() and has_right_id.any()

        merged = None
        joined_on = None

        if can_id_join:
            tmp_left = left[has_left_id].copy()
            tmp_right = right[has_right_id].copy()
            # Harmonize dtypes
            tmp_left["batter_id_join"] = tmp_left["batter_id_join"].astype(str)
            tmp_right["batter_id_join"] = tmp_right["batter_id_join"].astype(str)

            m1 = tmp_left.merge(
                tmp_right[["game_date", "player_name_norm", "batter_id_join", "hr_outcome"]],
                on=["game_date", "batter_id_join"],
                how="left",
                suffixes=("", "_y"),
            )
            # Put back the non-id rows untouched
            non_id = left[~has_left_id].copy()
            merged = pd.concat([m1, non_id], ignore_index=True)
            joined_on = "game_date + batter_id_join"

        # 2) Fallback join on (date + player_name_norm)
        if merged is None or merged["hr_outcome"].isna().all():
            m2 = left.merge(
                right[["game_date", "player_name_norm", "hr_outcome"]],
                on=["game_date", "player_name_norm"],
                how="left",
                suffixes=("", "_y"),
            )
            merged = m2
            if joined_on is None:
                joined_on = "game_date + player_name_norm"

        return merged, joined_on

    merged, joined_on = safe_join(lb, ev_daily)

    # --- Diagnostics ---
    total = len(merged)
    labeled = merged["hr_outcome"].notna().sum()
    st.subheader("2) Label join diagnostics")
    st.write(f"Join used: **{joined_on}**")
    st.write(f"Labeled rows: **{labeled}** / {total} ({(labeled/total*100 if total else 0):.1f}%)")

    if labeled == 0:
        st.error(
            "❌ No label matches found after join.\n\n"
            "Quick checks you can do in the CSV:\n"
            "• Ensure 'game_date' is YYYY-MM-DD (you did).\n"
            "• Keep 'player_name' as exported from your prediction app (accents/suffixes are normalized here).\n"
            "• Confirm your event parquet **covers the same dates** as your leaderboard.\n"
            "• (Optional) If you ever add 'batter_id' to the leaderboard, matching rate will jump."
        )
        st.stop()

    # ---------------------------------------------------------------
    # Build learner features (MUST match what the prediction app expects)
    # ---------------------------------------------------------------
    st.subheader("3) Train the learning ranker")

    # Safe numeric pulls with defaults:
    def get_num(colname, df=merged, default=0.0):
        if colname in df.columns:
            return pd.to_numeric(df[colname], errors="coerce").fillna(default).astype(float)
        return pd.Series(default, index=df.index, dtype=float)

    # Core signals from leaderboard
    base_prob = get_num("hr_probability_iso_T", merged, 0.0)
    ranked_probability = get_num("ranked_probability", merged, 0.0)

    overlay_multiplier = get_num("overlay_multiplier", merged, 1.0)
    final_multiplier   = get_num("final_multiplier", merged, 1.0)

    # Extra signals you asked to keep:
    prob_2tb = get_num("prob_2tb", merged, 0.0)
    prob_rbi = get_num("prob_rbi", merged, 0.0)

    # Derived as in your prediction app:
    def safe_logit(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1-1e-6)
        return np.log(p/(1.0-p))

    logit_p = pd.Series(safe_logit(base_prob), index=merged.index).astype(float)
    ranker_z = pd.Series(zscore(ranked_probability), index=merged.index).astype(float)
    log_overlay = pd.Series(np.log(final_multiplier + 1e-9), index=merged.index).astype(float)

    # Final training frame
    feat_cols = [
        "base_prob",
        "logit_p",
        "log_overlay",
        "ranker_z",
        "overlay_multiplier",
        "final_multiplier",
        "prob_2tb",
        "prob_rbi",
    ]
    train_df = pd.DataFrame({
        "base_prob": base_prob,
        "logit_p": logit_p,
        "log_overlay": log_overlay,
        "ranker_z": ranker_z,
        "overlay_multiplier": overlay_multiplier,
        "final_multiplier": final_multiplier,
        "prob_2tb": prob_2tb,
        "prob_rbi": prob_rbi,
        "hr_outcome": merged["hr_outcome"].fillna(0).astype(int),
        "game_date": merged["game_date"],
        "player_name": merged.get("player_name", ""),
        "team_code": merged.get("team_code", ""),
    })

    # Filter to labeled rows
    train_df = train_df[train_df["hr_outcome"].isin([0, 1])].dropna(subset=["hr_outcome"])
    n_labeled = len(train_df)
    st.write(f"Rows available for training after labeling: **{n_labeled}**")

    if n_labeled < 10:
        st.warning("Not enough labeled rows to fit a reliable learner (need ~50+). "
                   "You can still export the labeled CSV below.")
    else:
        # Fit the small LR meta (like prediction app expects)
        X = train_df[feat_cols].to_numpy(dtype=float)
        y = train_df["hr_outcome"].to_numpy(dtype=int)

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        lr = LogisticRegression(max_iter=1000, solver="lbfgs")
        lr.fit(Xs, y)

        st.success("✅ Learning ranker trained.")

        # ---- Export bundle (exact structure your prediction app expects) ----
        bundle = {
            "model": lr,
            "scaler": scaler,
            "features": feat_cols,
            # helpful for audit
            "trained_rows": int(n_labeled),
            "trained_at": datetime.now().isoformat(timespec="seconds"),
        }
        buf = io.BytesIO()
        pickle.dump(bundle, buf)
        buf.seek(0)
        st.download_button(
            label="⬇️ Download learning_ranker.pkl",
            data=buf,
            file_name="learning_ranker.pkl",
            mime="application/octet-stream",
        )

    # ---- Export merged + labels for audit ----
    out_csv = merged.copy()
    out_csv["hr_outcome"] = merged["hr_outcome"].fillna(0).astype(int)
    csv_buf = io.StringIO()
    out_csv.to_csv(csv_buf, index=False)
    st.download_button(
        label="⬇️ Download labeled_leaderboard.csv",
        data=csv_buf.getvalue(),
        file_name="labeled_leaderboard.csv",
        mime="text/csv",
    )

    st.caption(
        "Bundle contains the compact learner used by your prediction app. "
        "Keep using your XGB/LGB/CB stack in the predictor — this learner just re-orders the slate."
    )
