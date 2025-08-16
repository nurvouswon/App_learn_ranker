# App_learn_ranker.py
# ============================================================
# 📚 Learner — HR Day Ranker from Leaderboards + Event Parquet
# - Upload merged_leaderboards.csv (across days)
# - Upload event-level parquet (with hr_outcome)
# - Robust join by ['game_date','player_name_norm'] with normalization
# - Trains 3 rankers: LGBMRanker, XGBRanker, CatBoostRanker
# - Blends z-scored ranker predictions
# - Saves learning_ranker.pkl (models + features)
# - Lets you download the labeled merged leaderboard CSV
# - No plots
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
from sklearn.metrics import ndcg_score
import lightgbm as lgb
from xgboost import XGBRanker
from catboost import CatBoostRanker

st.set_page_config(page_title="📚 Learner — HR Day Ranker", layout="wide")
st.title("📚 Learner — HR Day Ranker from Leaderboards + Event Parquet")

# ------------- Helpers -------------
def normalize_name(s):
    if pd.isna(s):
        return ""
    x = str(s).strip()
    x = unidecode(x)                       # strip accents
    x = x.replace(".", "").replace("'", "")  # remove . and '
    x = x.replace("-", " ")                # hyphen -> space
    # common suffixes / noise
    bad = [" jr", " sr", " ii", " iii", " iv"]
    x_low = x.lower()
    for b in bad:
        if x_low.endswith(b):
            x = x[:-len(b)]
            x_low = x_low[:-len(b)]
    # collapse spaces
    x = " ".join(x.split())
    return x.lower()

def coerce_datetime_col(df, col):
    if col not in df.columns:
        return df
    # Accept YYYY-MM-DD or anything Pandas can parse
    df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    return df

def to_numeric_cols(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def zscore(a):
    a = np.asarray(a, dtype=np.float64)
    mu = np.nanmean(a); sd = np.nanstd(a) + 1e-9
    return (a - mu) / sd

st.markdown("### 1) Upload files")

lb_file = st.file_uploader("Merged leaderboard CSV (combined across days)", type=["csv"])
ev_file = st.file_uploader("Event-level Parquet (with hr_outcome)", type=["parquet"])

# Optional: if your historical leaderboards used short dates and you want to force a season year, you can expose this.
# For you, not needed since you already converted to YYYY-MM-DD.
# season_year = st.number_input("Season year (only used if your leaderboard has ambiguous short dates like '8_13')", 2025, 2030, 2025)

if lb_file and ev_file:
    with st.spinner("Loading files..."):
        lb = pd.read_csv(lb_file)
        ev = pd.read_parquet(ev_file)

    # Clean obvious junk index column if present
    if "Unnamed: 0" in lb.columns:
        lb = lb.drop(columns=["Unnamed: 0"], errors="ignore")

    # Ensure required columns exist
    need_lb = ["game_date", "player_name"]
    missing_lb = [c for c in need_lb if c not in lb.columns]
    if missing_lb:
        st.error(f"Leaderboard missing columns: {missing_lb}.")
        st.stop()

    if "player_name" not in ev.columns or "game_date" not in ev.columns:
        st.error("Event parquet must contain 'player_name' and 'game_date'.")
        st.stop()

    # Normalize names in BOTH
    lb["player_name_norm"] = lb["player_name"].apply(normalize_name)
    ev["player_name_norm"] = ev["player_name"].apply(normalize_name)

    # Parse dates (your leaderboard already in YYYY-MM-DD)
    lb = coerce_datetime_col(lb, "game_date")
    ev = coerce_datetime_col(ev, "game_date")

    # Build daily HR labels from the event file (max across each player's day)
    # hr_outcome must be present in event parquet.
    if "hr_outcome" not in ev.columns:
        st.error("Event parquet has no 'hr_outcome' column. Please export it in your event-level data.")
        st.stop()

    ev_daily = (
        ev[["game_date", "player_name_norm", "hr_outcome"]]
        .dropna(subset=["game_date", "player_name_norm"])
        .groupby(["game_date", "player_name_norm"], as_index=False)["hr_outcome"]
        .max()
    )

    # --- Join labels onto leaderboard
    merged = lb.merge(
        ev_daily,
        left_on=["game_date", "player_name_norm"],
        right_on=["game_date", "player_name_norm"],
        how="left",
        suffixes=("", "_y"),
    )

    # Report labeling coverage
    labeled = merged["hr_outcome"].notna().sum()
    st.info(f"Labels joined. Rows with labels: {labeled} (dropped {len(merged)-labeled} without event data).")

    # Keep labeled rows only for training
    train_df = merged.dropna(subset=["hr_outcome"]).copy()
    train_df["hr_outcome"] = train_df["hr_outcome"].astype(int)

    if train_df.empty:
        st.error("No label matches found after join.\n\n"
                 "Hints: ensure the leaderboard has 'game_date' in YYYY-MM-DD and names align; "
                 "this app normalizes accents and common suffixes. If needed, double-check that the "
                 "event parquet covers the same days and players.")
        st.stop()

    st.markdown("### 2) Select features for the ranker (auto-selected)")

    # Candidate features from leaderboard (use what's available)
    candidate_feats = [
        "ranked_probability",
        "hr_probability_iso_T",
        "final_multiplier",
        "overlay_multiplier",
        "weak_pitcher_factor",
        "hot_streak_factor",
        "rrf_aux",
        "model_disagreement",
        # keep 2TB/RBI if present
        "prob_2tb",
        "prob_rbi",
        # Optional extras if present (weather, raw mult)
        "final_multiplier_raw",
        "temp",
        "humidity",
        "wind_mph",
    ]
    # keep only columns that exist
    features_avail = [c for c in candidate_feats if c in train_df.columns]

    # ensure numeric
    to_numeric_cols(train_df, features_avail)

    # Drop any non-numeric among those
    features_avail = [c for c in features_avail if pd.api.types.is_numeric_dtype(train_df[c])]

    if not features_avail:
        st.error("No numeric features found to train the ranker.")
        st.stop()

    feats = st.multiselect("Features", options=features_avail, default=features_avail)
    if not feats:
        st.error("Please select at least one feature.")
        st.stop()

    # Build group vector by day
    # For LGBMRanker: group sizes list
    # For XGBRanker: same group (qid) representation
    # For CatBoostRanker: group_id column
    train_df = train_df.sort_values(["game_date", "player_name_norm"]).reset_index(drop=True)
    y = train_df["hr_outcome"].astype(int).to_numpy()

    # X matrix
    X = train_df[feats].copy().fillna(-1.0).to_numpy(dtype=np.float32)

    # LGB group sizes
    day_sizes = train_df.groupby("game_date").size().tolist()

    # XGB qid
    # create a numeric qid per date
    date_to_qid = {d:i for i, d in enumerate(train_df["game_date"].unique())}
    qid = train_df["game_date"].map(date_to_qid).to_numpy()

    # CatBoost group_id: we can reuse qid
    group_id = qid.copy()

    st.markdown("### 3) Train day-wise rankers (LGBM + XGB + CatBoost)")

    # ---- LightGBM Ranker
    rk_lgb = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg",
        n_estimators=600, learning_rate=0.05, num_leaves=63,
        feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=1,
        random_state=42
    )
    rk_lgb.fit(X, y, group=day_sizes)

    # ---- XGBoost Ranker
    rk_xgb = XGBRanker(
        objective="rank:pairwise",
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.5,
        random_state=42,
        tree_method="hist",
        n_jobs=2,
    )
    rk_xgb.fit(X, y, qid=qid)

    # ---- CatBoost Ranker
    rk_cb = CatBoostRanker(
        loss_function="YetiRank",
        iterations=1200,
        depth=7,
        learning_rate=0.05,
        l2_leaf_reg=5.0,
        random_seed=42,
        verbose=False
    )
    rk_cb.fit(X, y, group_id=group_id)

    # ---- Blend (z-scored average of the three)
    pred_lgb = rk_lgb.predict(X)
    pred_xgb = rk_xgb.predict(X)
    pred_cb  = rk_cb.predict(train_df[feats].fillna(-1.0))  # CatBoost accepts DataFrame

    blend = zscore(pred_lgb) + zscore(pred_xgb) + zscore(pred_cb)
    blend = blend / 3.0

    # Evaluate NDCG@30 (rough sanity check)
    # Build relevance per day chunks
    ndcgs = []
    start = 0
    for sz in day_sizes:
        end = start + sz
        rel = y[start:end]
        sc  = blend[start:end]
        # ndcg_score expects shape (1, n_samples)
        # relevance must be non-negative; here it's {0,1}
        if rel.sum() > 0:
            nd = ndcg_score(rel.reshape(1, -1), sc.reshape(1, -1), k=30)
            ndcgs.append(nd)
        start = end
    mean_ndcg30 = float(np.mean(ndcgs)) if ndcgs else 0.0
    st.success(f"Models trained. Mean NDCG@30 (in-sample, per-day): {mean_ndcg30:.4f}")

    # Attach learned rank back to the labeled leaderboard for download
    train_df["learned_rank_score"] = blend

    # ---------- Save bundle ----------
    bundle = {
        "models": {
            "lgb": rk_lgb,
            "xgb": rk_xgb,
            "cb":  rk_cb,
        },
        "features": feats,
        "note": "Day-wise ranker ensemble (LGBM+XGB+CatBoost), z-score blended.",
        "built_at": datetime.utcnow().isoformat() + "Z",
    }
    pkl_bytes = io.BytesIO()
    pickle.dump(bundle, pkl_bytes)
    pkl_bytes.seek(0)

    st.download_button(
        "⬇️ Download learning_ranker.pkl",
        data=pkl_bytes,
        file_name="learning_ranker.pkl",
        mime="application/octet-stream",
    )

    # ---------- Also let you download the labeled merged leaderboard ----------
    out_cols = ["game_date", "player_name", "team_code", "ranked_probability",
                "hr_probability_iso_T", "overlay_multiplier",
                "prob_2tb", "prob_rbi",
                "weak_pitcher_factor", "hot_streak_factor",
                "final_multiplier_raw", "final_multiplier",
                "temp", "humidity", "wind_mph",
                "rrf_aux", "model_disagreement",
                "hr_outcome", "learned_rank_score"]
    out_cols = [c for c in out_cols if c in train_df.columns]

    out_csv = train_df[out_cols].copy()
    csv_bytes = out_csv.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download labeled_merged_leaderboards.csv",
        data=csv_bytes,
        file_name="labeled_merged_leaderboards.csv",
        mime="text/csv",
    )

    st.success("✅ Done. Use learning_ranker.pkl in your prediction app’s optional ranker slot.")
