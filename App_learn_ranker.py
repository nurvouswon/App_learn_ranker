# app_learn_ranker.py
# ============================================================
# 📚 Learner — HR Day Ranker from Leaderboards + Event Parquet
# - Inputs:
#     • merged_leaderboards.csv (all days combined)
#     • event-level .parquet (must include hr_outcome)
# - Robust join on date + normalized player_name (accents/suffixes handled)
# - Features include ranked_probability, hr_probability_iso_T, overlays,
#   rrf_aux, model_disagreement, prob_2tb, prob_rbi, optional weather
# - ML:
#     • LightGBM LGBMRanker (LambdaRank)
#     • XGBoost XGBRanker (rank:pairwise) — if installed
#     • CatBoost CatBoostRanker (YetiRankPairwise) — if installed
#   -> Simple average ensemble with a uniform blend; predict() returns mean score
# - Outputs:
#     • labeled_leaderboard.csv (joined labels)
#     • learning_ranker.pkl (model + feature list)  <-- same interface as before
# - No plots. No tuner. Nothing else removed or changed.
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle, io, re, gc
from datetime import datetime
from typing import List, Optional

import lightgbm as lgb

# Optional rankers
_has_xgb = False
_has_cb  = False
try:
    from xgboost import XGBRanker
    _has_xgb = True
except Exception:
    pass
try:
    from catboost import CatBoostRanker
    _has_cb = True
except Exception:
    pass

st.set_page_config(page_title="📚 Learner — HR Day Ranker", layout="wide")
st.title("📚 Learner — HR Day Ranker from Leaderboards + Event Parquet")

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def read_csv_cached(f):
    return pd.read_csv(f)

@st.cache_data(show_spinner=False)
def read_parquet_cached(f):
    return pd.read_parquet(f)

def choose_first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

# unidecode for accent stripping (Peña -> Pena etc.)
try:
    from unidecode import unidecode
except Exception:
    def unidecode(x):  # graceful fallback
        return x

def clean_names(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str).str.strip()
    s = s.apply(unidecode)                        # remove accents
    s = s.str.replace(r"\.", "", regex=True)     # remove periods
    s = s.str.replace(",", "", regex=True)       # remove commas
    s = s.str.replace(r"\b(JR|SR|II|III|IV)\b", "", regex=True, flags=re.IGNORECASE)  # suffixes
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s.str.lower()

def parse_date_safely(df: pd.DataFrame, date_cols: List[str], fallback_year: int | None):
    # Try direct parse, else accept "8_13" or "8-13" and inject fallback year
    for c in date_cols:
        if c not in df.columns:
            continue

        # direct attempt
        d = pd.to_datetime(df[c], errors="coerce", utc=False)
        if d.notna().any():
            d = pd.to_datetime(d).dt.tz_localize(None)
            return d.dt.floor("D")

        raw = df[c].astype(str)

        def _mk_iso(x: str):
            x = x.strip()
            if re.fullmatch(r"\d{1,2}[_\-]\d{1,2}", x):
                if fallback_year is None:
                    return None
                m, d = re.split(r"[_\-]", x)
                try:
                    return f"{int(fallback_year):04d}-{int(m):02d}-{int(d):02d}"
                except Exception:
                    return None
            return x

        cand = raw.map(_mk_iso)
        d2 = pd.to_datetime(cand, errors="coerce", utc=False)
        if d2.notna().any():
            d2 = pd.to_datetime(d2).dt.tz_localize(None)
            return d2.dt.floor("D")

    return pd.Series(pd.NaT, index=df.index)

def common_name_fixes(s: pd.Series) -> pd.Series:
    # Apply small known fixes before normalization
    replacements = {
        r"^Peter Alonso$": "Pete Alonso",
        r"^Jeremy Pena$": "Jeremy Peña",
        r"^Julio Rodriguez$": "Julio Rodríguez",
        r"^Michael A Taylor$": "Michael A. Taylor",
        r"^CJ Kayfus$": "C.J. Kayfus",
        r"^C\. J\. Kayfus$": "C.J. Kayfus",
    }
    out = s.astype(str)
    for pat, rep in replacements.items():
        out = out.str.replace(pat, rep, regex=True)
    return out

def build_groups_from_days(dates: pd.Series) -> list[int]:
    d = pd.to_datetime(dates).dt.floor("D")
    return d.groupby(d.values).size().values.tolist()

# Simple ensemble wrapper to keep the same .predict API
class EnsembleRanker:
    def __init__(self, models: list, weights: Optional[list] = None):
        self.models = models
        self.weights = weights if (weights and len(weights) == len(models)) else [1.0] * len(models)

    def predict(self, X):
        preds = []
        for m in self.models:
            # LightGBM/CB return 1D; XGBRanker returns 1D
            p = m.predict(X)
            preds.append(np.asarray(p, dtype=float))
        W = np.asarray(self.weights, dtype=float)
        W = W / np.clip(W.sum(), 1e-9, None)
        stacked = np.vstack(preds)  # [n_models, n_samples]
        return (stacked.T @ W).astype(float)

# -----------------------------
# 1) Uploads
# -----------------------------
lb_file = st.file_uploader("Merged leaderboard CSV (combined across days)", type=["csv"])
ev_file = st.file_uploader("Event-level Parquet (with hr_outcome)", type=["parquet"])

if not lb_file or not ev_file:
    st.stop()

with st.spinner("Loading files..."):
    lb = read_csv_cached(lb_file)
    ev = read_parquet_cached(ev_file)

st.write(f"Leaderboard rows: {len(lb):,} | Event rows: {len(ev):,}")

# -----------------------------
# 2) Robust normalization + join
# -----------------------------
lb_name_raw_col = choose_first_col(lb, ["player_name", "batter_name", "batter", "name"])
ev_name_raw_col = choose_first_col(ev, ["player_name", "batter_name", "batter", "name"])
if lb_name_raw_col is None or ev_name_raw_col is None:
    st.error("Could not find a player-name column in one of the files.")
    st.stop()

lb[lb_name_raw_col] = common_name_fixes(lb[lb_name_raw_col])
ev[ev_name_raw_col] = common_name_fixes(ev[ev_name_raw_col])

lb["_join_name"] = clean_names(lb[lb_name_raw_col])
ev["_join_name"] = clean_names(ev[ev_name_raw_col])

# Try to infer season (fallback year) from event data
ev_date_col_guess = choose_first_col(ev, ["game_date", "date", "game_datetime", "start_time"])
ev_dates_try = pd.to_datetime(ev.get(ev_date_col_guess), errors="coerce") if ev_date_col_guess else pd.Series(pd.NaT, index=ev.index)
fallback_year = int(ev_dates_try.dt.year.mode().iloc[0]) if ev_dates_try.notna().any() else None

lb["_join_date"] = parse_date_safely(lb, ["game_date_iso", "game_date", "date"], fallback_year=fallback_year)
ev["_join_date"] = parse_date_safely(ev, ["game_date", "date", "game_datetime", "start_time"], fallback_year=None)

# Diagnostics
st.write("🔎 Join diagnostics (before merge)")
st.write({
    "lb_unique_names": int(lb["_join_name"].nunique()),
    "ev_unique_names": int(ev["_join_name"].nunique()),
    "lb_date_nonnull": int(lb["_join_date"].notna().sum()),
    "ev_date_nonnull": int(ev["_join_date"].notna().sum()),
})

label_col_ev = choose_first_col(ev, ["hr_outcome", "hr", "is_hr"])
if label_col_ev is None:
    st.error("Event parquet has no 'hr_outcome' (or equivalent) column.")
    st.stop()

ev_labels = ev[["_join_date", "_join_name", label_col_ev]].dropna(subset=["_join_date", "_join_name"]).drop_duplicates()
ev_labels = ev_labels.rename(columns={label_col_ev: "hr_outcome"})

lb_lab = lb.merge(ev_labels, how="inner", on=["_join_date", "_join_name"])

matches = len(lb_lab)
uniq_names_overlap = int(len(set(lb["_join_name"]).intersection(set(ev["_join_name"]))))
uniq_dates_overlap = int(len(set(lb["_join_date"].dropna()).intersection(set(ev["_join_date"].dropna()))))

st.write("🔗 Join results")
st.write({
    "merged_rows": matches,
    "unique_name_overlap": uniq_names_overlap,
    "unique_date_overlap": uniq_dates_overlap,
    "labels_found": int(lb_lab["hr_outcome"].notna().sum()),
})

if matches == 0:
    st.error(
        "No label matches found after join.\n\n"
        "Hints: ensure the leaderboard has a real date (prefer 'game_date_iso' like 2025-08-13) "
        "and names align; accents/suffixes are normalized automatically."
    )
    st.stop()

st.caption(f"Labels joined. Rows with labels: {matches:,}.")
st.dataframe(lb_lab.head(30), use_container_width=True)

# -----------------------------
# 3) Feature selection (includes 2TB & RBI)
# -----------------------------
default_feats = [
    "ranked_probability",
    "hr_probability_iso_T",
    "final_multiplier",
    "overlay_multiplier",
    "weak_pitcher_factor",
    "hot_streak_factor",
    "rrf_aux",
    "model_disagreement",
    "prob_2tb",          # <= included
    "prob_rbi",          # <= included
    # optional context if present:
    "final_multiplier_raw",
    "temp", "humidity", "wind_mph",
]

available = [c for c in default_feats if c in lb_lab.columns]
st.subheader("2) Select features for the ranker (auto-selected for you)")
feats = st.multiselect("Features", options=sorted(lb_lab.columns), default=available)

if len(feats) == 0:
    st.error("Select at least one feature.")
    st.stop()

num_feats = [c for c in feats if pd.api.types.is_numeric_dtype(lb_lab[c])]
if len(num_feats) == 0:
    st.error("Selected features are non-numeric. Pick numeric features.")
    st.stop()

X = lb_lab[num_feats].copy()
y = lb_lab["hr_outcome"].astype(int).copy()

if lb_lab["_join_date"].isna().all():
    st.error("No valid dates in joined data to form groups. Ensure 'game_date' or 'game_date_iso' is populated.")
    st.stop()

groups = build_groups_from_days(lb_lab["_join_date"])

X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)

if X.shape[0] < 1 or X.shape[1] < 1:
    st.error("After processing, X is empty. Check selected features.")
    st.stop()

# -----------------------------
# 4) Train day-wise rankers (ALL ML)
# -----------------------------
st.subheader("3) Train day-wise ranking models (LGBM + XGB + CatBoost)")

models = []
names  = []

# (a) LightGBM LambdaRank — always
rk_lgb = lgb.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=63,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    random_state=42,
)
with st.spinner("Training LightGBM LambdaRank..."):
    rk_lgb.fit(X, y, group=groups)
models.append(rk_lgb); names.append("lgbm")

# (b) XGBoost ranker — if available
if _has_xgb:
    # XGBRanker expects group/query as a list of group sizes
    rk_xgb = XGBRanker(
        objective="rank:pairwise",
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=0,
    )
    with st.spinner("Training XGBoost Ranker..."):
        rk_xgb.fit(X, y, group=groups)
    models.append(rk_xgb); names.append("xgboost")
else:
    st.info("XGBoost not installed — skipping XGBRanker.")

# (c) CatBoost ranking — if available
if _has_cb:
    rk_cb = CatBoostRanker(
        iterations=800,
        depth=7,
        learning_rate=0.05,
        l2_leaf_reg=6.0,
        loss_function="YetiRankPairwise",
        random_seed=42,
        verbose=False,
    )
    # CatBoost wants group_id per row; build day ids
    day_ids = pd.factorize(pd.to_datetime(lb_lab["_join_date"]).dt.strftime("%Y-%m-%d"))[0]
    with st.spinner("Training CatBoost Ranker..."):
        rk_cb.fit(X, y, group_id=day_ids)
    models.append(rk_cb); names.append("catboost")
else:
    st.info("CatBoost not installed — skipping CatBoostRanker.")

st.success(f"✅ Trained models: {', '.join(names)}")

# Build uniform-weight ensemble (works with your prediction app’s .predict call)
ens = EnsembleRanker(models=models, weights=[1.0] * len(models))

# -----------------------------
# 5) Downloads
# -----------------------------
# (a) Labeled leaderboard CSV
labeled_csv = lb_lab.copy()
download_cols = []
for c in ["_join_date", "player_name", "team_code", "ranked_probability",
          "hr_probability_iso_T", "final_multiplier", "overlay_multiplier",
          "weak_pitcher_factor", "hot_streak_factor", "rrf_aux", "model_disagreement",
          "prob_2tb", "prob_rbi", "hr_outcome", "source_file"]:
    if c in labeled_csv.columns:
        download_cols.append(c)
if len(download_cols) == 0:
    download_cols = labeled_csv.columns.tolist()

csv_buf = io.StringIO()
labeled_csv[download_cols].to_csv(csv_buf, index=False)
st.download_button(
    label="⬇️ Download Labeled Leaderboard CSV",
    data=csv_buf.getvalue(),
    file_name="labeled_leaderboard.csv",
    mime="text/csv",
)

# (b) Learning ranker bundle — SAME SHAPE as before
bundle = {
    "model": ens,              # ensemble with .predict(X)
    "features": num_feats,     # expected at inference time in prediction app
    "trained_on_rows": int(len(X)),
    "trained_on_days": int(len(groups)),
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "components": names,       # for transparency
}
pkl_bytes = io.BytesIO()
pickle.dump(bundle, pkl_bytes)
pkl_bytes.seek(0)

st.download_button(
    label="⬇️ Download learning_ranker.pkl",
    data=pkl_bytes,
    file_name="learning_ranker.pkl",
    mime="application/octet-stream",
)

st.caption("This ranker expects the selected numeric features at inference time (provided by your prediction app).")
gc.collect()
