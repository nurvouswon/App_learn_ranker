# App_learn_ranker.py
# =============================================================================
# 📚 Learner — HR Day Ranker from Leaderboards + Event Parquet (ID-only join)
# - Deterministic join on game_date + batter_id (MLB id)
# - Trains LGB/XGB/Cat ranker ensemble; includes 2TB & RBI among features (if present)
# - Exports labeled CSV + learning_ranker.pkl
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle, io, re
from datetime import datetime

# ML
from sklearn.metrics import ndcg_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

st.set_page_config(page_title="📚 Learner — HR Day Ranker", layout="wide")
st.title("📚 Learner — HR Day Ranker from Leaderboards + Event Parquet")

# -------------------- Utilities --------------------
@st.cache_data(show_spinner=False)
def safe_read(fobj):
    name = str(getattr(fobj, "name", fobj)).lower()
    if name.endswith(".parquet"):
        return pd.read_parquet(fobj)
    try:
        return pd.read_csv(fobj, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(fobj, encoding="latin1", low_memory=False)

def to_date_ymd(s, season_year=None):
    if pd.isna(s): return pd.NaT
    ss = str(s).strip()
    # Try normal parse first
    try:
        return pd.to_datetime(ss, errors="raise").normalize()
    except Exception:
        pass
    # Fallback like "8_13" → use provided season_year
    m = re.match(r"^\s*(\d{1,2})[^\d]+(\d{1,2})\s*$", ss)
    if m and season_year:
        mm = int(m.group(1)); dd = int(m.group(2))
        try:
            return pd.to_datetime(f"{int(season_year):04d}-{mm:02d}-{dd:02d}").normalize()
        except Exception:
            return pd.NaT
    return pd.NaT

def col_any(df, candidates, required=False, err_label=""):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        st.error(f"Missing required column in {err_label}: one of {candidates}")
        st.stop()
    return None

# -------------------- UI --------------------
lb_file = st.file_uploader("Merged leaderboard CSV (combined across days)", type=["csv"])
ev_file = st.file_uploader("Event-level PARQUET/CSV (with hr_outcome)", type=["parquet", "csv"])
season_year = st.number_input(
    "Season year (only used if leaderboard 'game_date' is like '8_13')",
    min_value=2015, max_value=2100, value=2025, step=1
)

if not lb_file or not ev_file:
    st.info("Upload both files to continue.")
    st.stop()

# -------------------- Load --------------------
with st.spinner("Reading files..."):
    lb = safe_read(lb_file)
    ev = safe_read(ev_file)

st.write(f"Leaderboard rows: {len(lb):,} | Event rows: {len(ev):,}")

# -------------------- Validate & Normalize keys (ID-only path) --------------------
# Date columns
lb_date_col = col_any(lb, ["game_date", "date"], required=True, err_label="leaderboard")
ev_date_col = col_any(ev, ["game_date", "date"], required=True, err_label="event")

# Batter ID columns (MLB ID)
lb_id_col = col_any(lb, ["batter_id", "mlb_id", "mlb id", "batter"], required=True, err_label="leaderboard")
ev_id_col = col_any(ev, ["batter_id", "mlb_id", "mlb id", "batter"], required=True, err_label="event")

# Target column
if "hr_outcome" not in ev.columns:
    st.error("Event file missing required column: hr_outcome")
    st.stop()

# Normalize types
lb = lb.copy()
ev = ev.copy()

lb["game_date"] = lb[lb_date_col].apply(lambda s: to_date_ymd(s, season_year))
ev["game_date"] = pd.to_datetime(ev[ev_date_col], errors="coerce").dt.normalize()

# Force ID to string for safe merges
lb["batter_id"] = lb[lb_id_col].astype("Int64").astype(str).str.replace("<NA>", "", regex=False)
ev["batter_id"] = ev[ev_id_col].astype("Int64").astype(str).str.replace("<NA>", "", regex=False)

# -------------------- Quick diagnostics BEFORE join --------------------
lb_dates = pd.to_datetime(lb["game_date"]).dt.date.unique()
ev_dates = pd.to_datetime(ev["game_date"]).dt.date.unique()
date_overlap = sorted(set(lb_dates).intersection(set(ev_dates)))
st.write(f"🔍 Date overlap count: {len(date_overlap)}")
if len(date_overlap) == 0:
    st.error("❌ No date overlap between leaderboard and event parquet. Check the actual dates in each file.")
    st.stop()

st.write("Leaderboard date range:", str(pd.to_datetime(lb["game_date"]).min().date()), "→", str(pd.to_datetime(lb["game_date"]).max().date()))
st.write("Event date range:", str(pd.to_datetime(ev["game_date"]).min().date()), "→", str(pd.to_datetime(ev["game_date"]).max().date()))

# -------------------- Build per-day labels from event --------------------
# One label per (date, batter_id): any HR that day → 1
ev_daily = (
    ev.loc[ev["batter_id"].str.len() > 0, ["game_date", "batter_id", "hr_outcome"]]
      .groupby(["game_date", "batter_id"], dropna=False)["hr_outcome"]
      .max()
      .reset_index()
)

# -------------------- Deterministic join: date + batter_id --------------------
with st.spinner("Joining labels (ID-only)..."):
    merged = lb.merge(ev_daily, on=["game_date", "batter_id"], how="left", suffixes=("", "_y"))

labeled = merged[merged["hr_outcome"].notna()].copy()
st.write(f"🔎 Labeled rows: {len(labeled)} / {len(merged)}")

# Hard stop if still nothing usable
if len(labeled) == 0:
    st.error("❌ No label matches on (game_date, batter_id). Check that both files share the SAME date format and MLB IDs.")
    st.stop()

# -------------------- Feature set --------------------
# Keep your original feature roster (RBI & 2+TB included, if present)
candidate_feats = [
    "ranked_probability",
    "hr_probability_iso_T",
    "final_multiplier",
    "overlay_multiplier",
    "weak_pitcher_factor",
    "hot_streak_factor",
    "rrf_aux",
    "model_disagreement",
    "prob_2tb",
    "prob_rbi",
    "final_multiplier_raw",
    "temp", "humidity", "wind_mph",
]
avail = [c for c in candidate_feats if c in labeled.columns]
if not avail:
    st.error("No usable features found in leaderboard; need at least the core columns.")
    st.stop()

# build X/y/groups
X = labeled[avail].apply(pd.to_numeric, errors="coerce").fillna(-1).astype(np.float32)
y = labeled["hr_outcome"].astype(int).values

def groups_from_days(day_series: pd.Series):
    d = pd.to_datetime(day_series).dt.floor("D")
    return d.groupby(d.values).size().tolist()

groups = groups_from_days(labeled["game_date"])

# guard rails: groups must sum to n, and each group must have >=2 for ranking
n = len(y); gsum = int(np.sum(groups)) if len(groups) else 0
min_group = min(groups) if len(groups) else 0
if n < 10 or gsum != n or min_group < 2:
    st.error(
        "❌ Not enough labeled pairs for a ranker.\n"
        f"Rows labeled: {n} | groups sum: {gsum} | min group size: {min_group}.\n"
        "Fix: increase label matches (ensure (game_date, batter_id) overlap)."
    )
    # still let you download the partial labeled file
    labeled_out = labeled.sort_values(["game_date", "ranked_probability"], ascending=[True, False])
    csv_buf = io.StringIO(); labeled_out.to_csv(csv_buf, index=False)
    st.download_button("⬇️ Download Labeled Leaderboard CSV", csv_buf.getvalue(), "labeled_leaderboard.csv", "text/csv")
    st.stop()

# -------------------- Train ranker ensemble --------------------
st.subheader("Training day-wise ranker ensemble")
st.write(f"Features used ({len(avail)}): {', '.join(avail)}")

rk_lgb = lgb.LGBMRanker(
    objective="lambdarank", metric="ndcg",
    n_estimators=700, learning_rate=0.05, num_leaves=63,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    random_state=42
)
rk_lgb.fit(X, y, group=groups)
pred_lgb = rk_lgb.predict(X)

pred_xgb = None; rk_xgb = None
try:
    rk_xgb = xgb.XGBRanker(
        n_estimators=700, learning_rate=0.06, max_depth=6,
        subsample=0.85, colsample_bytree=0.85,
        objective="rank:pairwise", random_state=42, tree_method="hist"
    )
    rk_xgb.fit(X, y, group=groups, verbose=False)
    pred_xgb = rk_xgb.predict(X)
except Exception as e:
    st.warning(f"XGBRanker not used: {e}")

pred_cb = None; rk_cb = None
try:
    rk_cb = cb.CatBoost(
        iterations=1200, learning_rate=0.05, depth=7,
        loss_function="YetiRank", random_seed=42, verbose=False
    )
    rk_cb.fit(X, y, group_id=np.concatenate([[i]*g for i, g in enumerate(groups)]))
    pred_cb = rk_cb.predict(X).flatten()
except Exception as e:
    st.warning(f"CatBoost YetiRank not used: {e}")

preds = [p for p in [pred_lgb, pred_xgb, pred_cb] if p is not None]
ens_train = np.mean(np.column_stack(preds), axis=1) if len(preds) > 1 else pred_lgb

# Per-day ndcg (sanity)
try:
    ndcgs = []
    for day, df_day in labeled.groupby(pd.to_datetime(labeled["game_date"]).dt.floor("D")):
        idx = df_day.index
        y_true = df_day["hr_outcome"].values.reshape(1, -1)
        y_score = ens_train[idx].reshape(1, -1)
        nd = ndcg_score(y_true, y_score, k=min(10, y_true.shape[1]))
        ndcgs.append(float(nd))
    st.write(f"NDCG@10 (mean across days): {np.mean(ndcgs):.4f}")
except Exception:
    pass

st.success("✅ Ranker trained.")

# -------------------- Save artifacts --------------------
# Keep MLB id in labeled output for downstream use
labeled_out = labeled.copy()
labeled_out = labeled_out.sort_values(["game_date", "ranked_probability"], ascending=[True, False])

csv_buf = io.StringIO()
labeled_out.to_csv(csv_buf, index=False)
st.download_button(
    "⬇️ Download Labeled Leaderboard CSV",
    data=csv_buf.getvalue(),
    file_name="labeled_leaderboard.csv",
    mime="text/csv"
)

bundle = {
    "features": avail,
    "model_type": "ranker_ensemble",
    "models": {
        "lgb": rk_lgb,
        "xgb": rk_xgb,
        "cat": rk_cb,
    },
    "join_info": {
        "deterministic_strategy": "game_date + batter_id (MLB id)",
        "labeled_rows": int(len(labeled)),
        "total_rows": int(len(merged)),
    },
}
pkl_bytes = io.BytesIO()
pickle.dump(bundle, pkl_bytes)
pkl_bytes.seek(0)
st.download_button(
    "⬇️ Download learning_ranker.pkl",
    data=pkl_bytes,
    file_name="learning_ranker.pkl",
    mime="application/octet-stream"
)

st.caption("ID-only join. Ranker ensemble uses your leaderboard features (including 2+TB & RBI if present).")
