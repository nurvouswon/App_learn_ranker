# App_learn_ranker.py
# ============================================================
# 📚 Learner — HR Day Ranker from Leaderboards + Event Parquet
# - Inputs: merged leaderboards CSV + event-level parquet
# - Robust join: date parse, accent/period/suffix name normalization
# - Builds daily HR labels from events (max over day per player)
# - Uses 3 rankers: LGBMRanker, XGBRanker, CatBoostRanker
# - Stacks by z-score average → final learned day rank score
# - Features include ranked_probability, hr_probability_iso_T,
#   final_multiplier, overlay bits, rrf_aux, model_disagreement,
#   prob_2tb, prob_rbi (if present), plus simple weather cols.
# - Exports: labeled_merged_leaderboard.csv + learner bundle .pkl
# ============================================================

import io
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from unidecode import unidecode
from difflib import get_close_matches

# ML
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ndcg_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

st.set_page_config(page_title="📚 Learner — HR Day Ranker", layout="wide")
st.title("📚 Learner — HR Day Ranker from Leaderboards + Event Parquet")

# ------------------- Helpers -------------------
def normalize_name(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip()
    s = unidecode(s)  # remove accents
    s = s.replace(".", "")  # remove periods
    # common suffixes
    for suf in [" Jr", " Jr.", " SR", " Sr", " II", " III", " IV"]:
        s = s.replace(suf, "")
    # collapse double spaces
    s = " ".join(s.split())
    return s

def parse_date_any(x):
    # Expecting YYYY-MM-DD already; fallback to other patterns if needed
    try:
        return pd.to_datetime(x, errors="raise").date()
    except Exception:
        # Allow things like '8_13' if slipped in (needs year toggle above UI)
        try:
            if isinstance(x, str) and "_" in x:
                m, d = x.split("_")
                # Default to current season year if weird input appears
                yr = st.session_state.get("season_year", datetime.utcnow().year)
                return pd.to_datetime(f"{int(yr)}-{int(m)}-{int(d)}", errors="raise").date()
        except Exception:
            return pd.NaT
    return pd.NaT

def zscore(a):
    a = np.asarray(a, dtype=float)
    mu = np.nanmean(a)
    sd = np.nanstd(a) + 1e-9
    return (a - mu) / sd

def safe_float(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

# ------------------- UI: Uploads -------------------
lb_file = st.file_uploader("Merged leaderboard CSV (combined across days)", type=["csv"])
ev_file = st.file_uploader("Event-level Parquet (with hr_outcome)", type=["parquet"])

season_year = st.number_input("Season year (only used if your leaderboard dates are like '8_13')",
                              min_value=2018, max_value=2100, value=datetime.utcnow().year, step=1)
st.session_state.season_year = season_year

include_weather = st.checkbox("Include simple weather columns (temp, humidity, wind_mph)", value=True)

if not (lb_file and ev_file):
    st.stop()

# ------------------- Load data -------------------
lb = pd.read_csv(lb_file)
ev = pd.read_parquet(ev_file)

# Trim LB to relevant columns, keep extras if present
candidate_lb_cols = [
    "game_date", "player_name", "team_code", "ranked_probability", "hr_probability_iso_T",
    "final_multiplier", "overlay_multiplier", "weak_pitcher_factor", "hot_streak_factor",
    "rrf_aux", "model_disagreement", "prob_2tb", "prob_rbi",
    "final_multiplier_raw", "temp", "humidity", "wind_mph"
]
present_lb_cols = [c for c in candidate_lb_cols if c in lb.columns]
# Keep all original columns but ensure the above exist where possible
# (We’ll just work from originals; features are selected by presence later)
if "Unnamed: 0" in lb.columns:
    lb = lb.drop(columns=["Unnamed: 0"])

# Date + names normalize on LB
if "game_date" not in lb.columns:
    st.error("Leaderboard must have a 'game_date' column in YYYY-MM-DD.")
    st.stop()

lb["game_date"] = lb["game_date"].apply(parse_date_any)
lb = lb[~lb["game_date"].isna()].copy()
lb["player_name_norm"] = lb["player_name"].astype(str).apply(normalize_name)

# Build daily labels from events
need_cols = {"game_date", "player_name", "hr_outcome"}
missing_ev = [c for c in need_cols if c not in ev.columns]
if missing_ev:
    st.error(f"Event parquet is missing required columns: {missing_ev}")
    st.stop()

ev = ev.copy()
ev["game_date"] = pd.to_datetime(ev["game_date"], errors="coerce").dt.date
ev = ev[~pd.isna(ev["game_date"])].copy()
ev["player_name_norm"] = ev["player_name"].astype(str).apply(normalize_name)

# Make sure hr_outcome is numeric 0/1 and aggregate by day-player
ev["hr_outcome"] = pd.to_numeric(ev["hr_outcome"], errors="coerce").fillna(0).astype(int)
ev_daily = (
    ev.groupby(["game_date", "player_name_norm"], as_index=False)["hr_outcome"]
      .max()
)

# ------------------- Join labels -------------------
# Primary direct join on normalized name + date
merged = lb.merge(
    ev_daily,
    how="left",
    left_on=["game_date", "player_name_norm"],
    right_on=["game_date", "player_name_norm"],
    suffixes=("", "_y"),
)

# If we somehow still have all NaNs, try one extra fallback using team_code+date+approx name
if merged["hr_outcome"].isna().all():
    # attempt an approximate name match within each date (difflib)
    st.warning("Direct join produced no label matches. Trying approximate name matching per date…")
    approx_rows = []
    for dt, chunk in lb.groupby("game_date"):
        ev_names = ev_daily.loc[ev_daily["game_date"] == dt, "player_name_norm"].dropna().unique().tolist()
        if not ev_names:
            # No games this date in event parquet
            continue
        for _, row in chunk.iterrows():
            cand = row["player_name_norm"]
            best = get_close_matches(cand, ev_names, n=1, cutoff=0.92)  # high cutoff
            if best:
                approx_rows.append((dt, cand, best[0]))

    if approx_rows:
        approx_map = pd.DataFrame(approx_rows, columns=["game_date", "player_name_norm", "player_name_norm_ev"])
        ev_alt = ev_daily.merge(
            approx_map,
            how="right",
            left_on=["game_date", "player_name_norm"],
            right_on=["game_date", "player_name_norm_ev"],
        )[["game_date", "player_name_norm", "hr_outcome"]]

        merged = lb.drop(columns=["hr_outcome"], errors="ignore").merge(
            ev_alt, how="left", on=["game_date", "player_name_norm"]
        )

# Final sanity / report
matches = int(merged["hr_outcome"].notna().sum())
st.write(f"Labels joined. Rows with labels: {matches} (dropped {len(merged) - matches} without event data).")

if matches == 0:
    # Give immediate, actionable info
    u_lb_dates = pd.Series(lb["game_date"].unique())
    u_ev_dates = pd.Series(ev_daily["game_date"].unique())
    overlap_dates = pd.Series(list(set(u_lb_dates) & set(u_ev_dates)))
    st.error("❌ No label matches found after join.")
    st.write(f"Leaderboard dates (unique): {len(u_lb_dates)} | Event dates (unique): {len(u_ev_dates)} | Overlap: {len(overlap_dates)}")
    st.stop()

# Keep only labeled rows for training
train_df = merged.dropna(subset=["hr_outcome"]).copy()
train_df["hr_outcome"] = train_df["hr_outcome"].astype(int)

# ------------------- Feature selection -------------------
# We’ll auto-pick safe, present columns; 2tb/rbi included if present
feature_candidates = [
    "ranked_probability", "hr_probability_iso_T",
    "final_multiplier", "overlay_multiplier",
    "weak_pitcher_factor", "hot_streak_factor",
    "rrf_aux", "model_disagreement",
    "prob_2tb", "prob_rbi",
    "final_multiplier_raw"
]
if include_weather:
    feature_candidates += ["temp", "humidity", "wind_mph"]

features = [c for c in feature_candidates if c in train_df.columns]
if not features:
    st.error("No usable features found to train. Make sure your merged leaderboard has rank/prob columns.")
    st.stop()

st.subheader("Selected features for the ranker")
st.write(features)

# Prepare X/y and groups (day-wise)
X = train_df[features].copy()
safe_float(X, features)
X = X.fillna(0.0).astype(np.float32)

y = train_df["hr_outcome"].astype(int).values
# groups: count items per unique day in the same order
days = pd.to_datetime(train_df["game_date"]).dt.date
group_sizes = days.groupby(days).size().values.tolist()

# Safety: LightGBM expects 2D data with rows > 0
if X.shape[0] < 1 or X.shape[1] < 1:
    st.error("Input to the ranker is empty after filtering. Check your files.")
    st.stop()

# Standardize for XGB/Cat (LGBM doesn't require it, but safe to use)
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# ------------------- Train 3 rankers -------------------
st.subheader("Train day-wise rankers")

# 1) LightGBMRanker (LambdaRank)
rk_lgb = lgb.LGBMRanker(
    objective="lambdarank", metric="ndcg",
    n_estimators=600, learning_rate=0.05, num_leaves=63,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    random_state=42
)
rk_lgb.fit(X, y, group=group_sizes)

# Predict (same matrix) for score shaping
score_lgb = rk_lgb.predict(X).astype(float)

# 2) XGBRanker
rk_xgb = xgb.XGBRanker(
    n_estimators=600, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
    tree_method="hist", objective="rank:pairwise"
)
# XGBRanker wants group vector per row in order; we pass group as a list
rk_xgb.fit(Xs, y, group=group_sizes)
score_xgb = rk_xgb.predict(Xs).astype(float)

# 3) CatBoostRanker (YetiRank)
rk_cat = cb.CatBoostRanker(
    iterations=800, depth=6, learning_rate=0.05,
    loss_function="YetiRank", random_seed=42,
    verbose=False
)
# CatBoost grouping: list of arrays of indices per group; we’ll expand quickly
# Build group_id per row in order of days
day_to_id = {d: i for i, d in enumerate(pd.unique(days))}
group_id = days.map(day_to_id).astype(int).values
rk_cat.fit(Xs, y, group_id=group_id)
score_cat = rk_cat.predict(Xs).astype(float)

# Ensemble stack — average of z-scored ranker outputs
score_stack = (zscore(score_lgb) + zscore(score_xgb) + zscore(score_cat)) / 3.0
train_df["learned_rank_score"] = score_stack

# Quick NDCG@30 on each day then macro average (diagnostic)
def macro_ndcg_at_k(df, k=30, pred_col="learned_rank_score"):
    ndcgs = []
    for dt, chunk in df.groupby("game_date"):
        rel = chunk["hr_outcome"].values.astype(int)
        if rel.sum() == 0:
            continue
        y_true = rel.reshape(1, -1)
        y_score = chunk[pred_col].values.reshape(1, -1)
        ndcgs.append(ndcg_score(y_true, y_score, k=min(k, y_true.shape[1])))
    return float(np.mean(ndcgs)) if ndcgs else 0.0

ndcg30 = macro_ndcg_at_k(train_df, 30, "learned_rank_score")
st.success(f"Macro NDCG@30 (across labeled days): {ndcg30:.4f}")

# ------------------- Downloads -------------------
st.subheader("Downloads")

# 1) Labeled merged leaderboard
out_csv = train_df.sort_values(["game_date", "learned_rank_score"], ascending=[True, False])
csv_bytes = out_csv.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download labeled_merged_leaderboard.csv",
    data=csv_bytes,
    file_name="labeled_merged_leaderboard.csv",
    mime="text/csv"
)

# 2) Learner bundle (for use in prediction app)
bundle = dict(
    model_lgb=rk_lgb,
    model_xgb=rk_xgb,
    model_cat=rk_cat,
    scaler=scaler,
    features=features,
    meta=dict(
        ndcg30=ndcg30,
        trained_rows=int(X.shape[0]),
        trained_days=int(len(group_sizes)),
        season_year=season_year,
    ),
)
pkl_bytes = io.BytesIO()
pickle.dump(bundle, pkl_bytes)
pkl_bytes.seek(0)

st.download_button(
    "⬇️ Download learning_ranker.pkl",
    data=pkl_bytes,
    file_name="learning_ranker.pkl",
    mime="application/octet-stream"
)

st.caption("This learner does not change your prediction app logic—just produces a learning_ranker.pkl and a labeled leaderboard you can feed back in.")
