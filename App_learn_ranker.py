# app_learn_ranker.py
# ============================================================
# 📚 Learner — HR Day Ranker from Leaderboards + Event Parquet
# - Upload merged leaderboards CSV (multi-day)
# - Upload event-level parquet (with hr_outcome)
# - Robust join: prefer ['game_date','batter_id'] else normalized ['game_date','player_name']
# - Uses all three rankers: LGBMRanker, XGBRanker, CatBoostRanker
# - Wraps them in AveragedRanker so prediction app can load 1 pkl
# - Includes prob_2tb and prob_rbi features
# - No plots; emits model pkl + labeled CSV
# ============================================================

import io
import gc
import pickle
import numpy as np
import pandas as pd
import streamlit as st

from datetime import datetime
from unidecode import unidecode
from sklearn.preprocessing import StandardScaler

# ML rankers
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRanker

st.set_page_config(page_title="📚 HR Learner — Day Ranker", layout="wide")
st.title("📚 Learner — HR Day Ranker from Leaderboards + Event Parquet")

st.markdown("**Upload files**")

lb_file = st.file_uploader("Merged leaderboard CSV (combined across days)", type=["csv"])
ev_file = st.file_uploader("Event-level Parquet (with hr_outcome)", type=["parquet"])

# ---------- helpers ----------
def _parse_date_col(s):
    # accept already-datetime, or strings like "2025-08-13"
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.date
    return pd.to_datetime(s, errors="coerce").dt.date

def _norm_name(x: str) -> str:
    if pd.isna(x):
        return ""
    x = str(x).strip()
    x = unidecode(x)  # remove accents
    # drop common suffixes
    x = x.replace(".", " ")
    for suf in (" JR", " SR", " III", " II"):
        if x.upper().endswith(suf):
            x = x[: -len(suf)]
    x = " ".join(x.split())
    return x

def _safe_float(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _ensure_numeric(df):
    # Only keep numeric columns; convert where possible
    for c in df.columns:
        if df[c].dtype == "O":
            df[c] = pd.to_numeric(df[c], errors="ignore")
    return df

def _z(x):
    x = np.asarray(x, dtype=np.float64)
    mu, sd = np.nanmean(x), np.nanstd(x) + 1e-9
    return (x - mu) / sd

class AveragedRanker:
    """Wrap 3 trained rankers and average their predictions."""
    def __init__(self, lgbm=None, xgbr=None, cat=None):
        self.lgbm = lgbm
        self.xgbr = xgbr
        self.cat = cat

    def predict(self, X):
        preds = []
        if self.lgbm is not None:
            preds.append(self.lgbm.predict(X))
        if self.xgbr is not None:
            preds.append(self.xgbr.predict(X))
        if self.cat is not None:
            # CatBoost expects np array
            preds.append(self.cat.predict(X))
        if not preds:
            raise RuntimeError("No models available inside AveragedRanker.")
        return np.mean(np.column_stack(preds), axis=1)

# ---------- main ----------
if lb_file is not None and ev_file is not None:
    with st.spinner("Loading data..."):
        lb = pd.read_csv(lb_file)
        ev = pd.read_parquet(ev_file)

    # Basic sanity for needed columns
    if "game_date" not in lb.columns:
        st.error("Leaderboard is missing 'game_date'.")
        st.stop()
    if "player_name" not in lb.columns:
        st.error("Leaderboard is missing 'player_name'.")
        st.stop()
    if "hr_outcome" not in ev.columns:
        st.error("Event parquet is missing 'hr_outcome'.")
        st.stop()
    if "game_date" not in ev.columns:
        st.error("Event parquet is missing 'game_date'.")
        st.stop()
    if "player_name" not in ev.columns and "batter_id" not in ev.columns:
        st.error("Event parquet needs 'player_name' or 'batter_id'.")
        st.stop()

    # Normalize dates
    lb["game_date"] = _parse_date_col(lb["game_date"])
    ev["game_date"] = _parse_date_col(ev["game_date"])

    # Try to carry batter_id from leaderboard if present, else leave absent
    has_batter_id_lb = "batter_id" in lb.columns and lb["batter_id"].notna().any()
    has_batter_id_ev = "batter_id" in ev.columns and ev["batter_id"].notna().any()

    # Build label table from event parquet (one row per game_date × identity)
    ev_lab = ev[["game_date", "player_name", "hr_outcome"]].copy()
    if has_batter_id_ev:
        ev_lab["batter_id"] = ev["batter_id"]

    # Normalize names for backup join
    ev_lab["player_name_norm"] = ev_lab["player_name"].map(_norm_name)

    # Aggregate to daily label (did the player hit an HR this date?)
    grp_keys = ["game_date"]
    if has_batter_id_ev:
        grp_keys += ["batter_id"]
    else:
        grp_keys += ["player_name_norm"]

    ev_daily = ev_lab.groupby(grp_keys, as_index=False)["hr_outcome"].max()

    # Prepare leaderboard keys
    lb["player_name_norm"] = lb["player_name"].map(_norm_name)
    if has_batter_id_lb:
        pass  # use it directly if present

    # Try primary join by ['game_date','batter_id']
    merged = None
    join_msg = ""
    if has_batter_id_lb and has_batter_id_ev:
        merged = lb.merge(
            ev_daily,
            on=["game_date", "batter_id"],
            how="left",
            validate="m:1",
            suffixes=("", "_y"),
        )
        join_msg = "Joined on ['game_date','batter_id']"

    # Fallback: join on ['game_date','player_name_norm']
    if merged is None or merged["hr_outcome"].isna().all():
        merged = lb.merge(
            ev_daily,
            left_on=["game_date", "player_name_norm"],
            right_on=["game_date", "player_name_norm"],
            how="left",
            validate="m:1",
            suffixes=("", "_y"),
        )
        if join_msg:
            join_msg += " | fallback to ['game_date','player_name_norm']"
        else:
            join_msg = "Joined on ['game_date','player_name_norm']"

    # Diagnostics if still no labels
    n_lab = merged["hr_outcome"].notna().sum()
    st.write(f"Labels joined. Rows with labels: {n_lab} (dropped without labels at train time).")
    st.caption(join_msg)

    if n_lab == 0:
        # Helpful overlap info
        lb_dates = pd.Series(lb["game_date"].unique())
        ev_dates = pd.Series(ev_daily["game_date"].unique())
        date_overlap = len(set(lb_dates) & set(ev_dates))

        lb_names = set(lb["player_name_norm"].unique())
        ev_names = set(ev_daily.get("player_name_norm", pd.Series([], dtype=str)).unique())
        name_overlap = len(lb_names & ev_names)

        st.error(
            f"No label matches found after join.\n\n"
            f"Overlap (unique dates): {date_overlap}\n"
            f"Overlap (unique normalized names): {name_overlap}\n\n"
            f"Tip: ensure leaderboard dates are YYYY-MM-DD and names are consistent. "
            f"If you can add 'batter_id' to the leaderboard, matching becomes bulletproof."
        )
        st.stop()

    # Drop unlabeled rows for training
    train_df = merged[merged["hr_outcome"].notna()].copy()
    train_df["hr_outcome"] = train_df["hr_outcome"].astype(int)

    # ===== Features =====
    # Keep your core signals + 2TB/RBI + simple weather numerics if present.
    # We'll select intersection so missing columns are fine.
    wanted = [
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
        "temp",
        "humidity",
        "wind_mph",
    ]
    # common junk to drop if present
    drop_cols = ["Unnamed: 0"]

    for c in drop_cols:
        if c in train_df.columns:
            train_df = train_df.drop(columns=c)

    feat_cols = [c for c in wanted if c in train_df.columns]
    if not feat_cols:
        st.error("No usable features found in the leaderboard after filtering.")
        st.stop()

    # Ensure numeric
    train_df = _ensure_numeric(train_df)
    train_df = _safe_float(train_df, feat_cols)

    # Build design matrix
    X = train_df[feat_cols].fillna(0.0).astype(np.float32)
    y = train_df["hr_outcome"].astype(int).values

    # Day-wise grouping for ranking
    if not pd.api.types.is_datetime64_any_dtype(train_df["game_date"]):
        # turn back into datetime to group consistently
        train_df["game_date"] = pd.to_datetime(train_df["game_date"], errors="coerce")

    # Sort by date to stabilize group splits
    order_idx = np.argsort(train_df["game_date"].values.astype("datetime64[ns]"))
    X = X.iloc[order_idx].reset_index(drop=True)
    y = y[order_idx]
    dates_sorted = train_df["game_date"].iloc[order_idx].dt.date

    # groups as counts (LGB/XGB) and group_id (CatBoost)
    group_sizes = pd.Series(1, index=dates_sorted).groupby(dates_sorted).sum().tolist()
    # Map each date to an integer id
    unique_dates = pd.Series(dates_sorted).astype(str)
    date_to_gid = {d: i for i, d in enumerate(sorted(unique_dates.unique()))}
    group_id = unique_dates.map(date_to_gid).values.astype(int)

    st.markdown("**Selected features for the ranker**")
    st.code(", ".join(feat_cols))

    # ===== Train all three rankers =====
    st.markdown("### Train day-wise rankers")
    with st.spinner("Training LightGBM, XGBoost, CatBoost rankers..."):
        models_ok = []

        # LightGBM Ranker
        try:
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
            rk_lgb.fit(X, y, group=group_sizes)
            models_ok.append("lgbm")
        except Exception as e:
            st.warning(f"LightGBM training skipped: {e}")
            rk_lgb = None

        # XGBoost Ranker
        try:
            rk_xgb = xgb.XGBRanker(
                n_estimators=700,
                max_depth=6,
                learning_rate=0.06,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="rank:pairwise",
                eval_metric="ndcg",
                random_state=42,
                tree_method="hist",
            )
            rk_xgb.fit(X, y, group=group_sizes)
            models_ok.append("xgboost")
        except Exception as e:
            st.warning(f"XGBoost training skipped: {e}")
            rk_xgb = None

        # CatBoost Ranker
        try:
            rk_cat = CatBoostRanker(
                iterations=1000,
                depth=7,
                learning_rate=0.06,
                l2_leaf_reg=6.0,
                loss_function="YetiRank",
                random_seed=42,
                verbose=False
            )
            rk_cat.fit(X, y, group_id=group_id)
            models_ok.append("catboost")
        except Exception as e:
            st.warning(f"CatBoost training skipped: {e}")
            rk_cat = None

    if not models_ok:
        st.error("All three rankers failed to train. Please check logs and inputs.")
        st.stop()

    st.success(f"Model(s) trained: {', '.join(models_ok)}")

    # ===== Package for prediction app =====
    # Your prediction app will pass features:
    #   base_prob (hr_probability_iso_T), logit_p, log_overlay, ranker_z, overlay_multiplier, final_multiplier
    # BUT we can keep using a flexible feature list. We'll store 'features' = feat_cols,
    # and the prediction app maps its internal signals to these names when applying the learner.
    bundle = {
        "model": AveragedRanker(lgbm=rk_lgb, xgbr=rk_xgb, cat=rk_cat),
        "features": feat_cols,
        "trained_on_dates": [str(d) for d in sorted(set(dates_sorted))],
        "note": "Day-wise ranking ensemble (LGB/XGB/Cat). Predict expects columns in 'features'."
    }

    # Download: model
    bio = io.BytesIO()
    pickle.dump(bundle, bio)
    bio.seek(0)
    st.download_button(
        label="⬇️ Download learning_ranker.pkl",
        data=bio,
        file_name="learning_ranker.pkl",
        mime="application/octet-stream"
    )

    # Download: merged labeled leaderboards
    merged_out = merged.copy()
    merged_out["hr_outcome"] = merged_out["hr_outcome"].fillna(-1).astype(int)
    csv_io = io.StringIO()
    merged_out.to_csv(csv_io, index=False)
    st.download_button(
        label="⬇️ Download merged_labeled_leaderboards.csv",
        data=csv_io.getvalue().encode("utf-8"),
        file_name="merged_labeled_leaderboards.csv",
        mime="text/csv"
    )

    st.success("✅ Done. Model and labeled CSV are ready.")
    gc.collect()
else:
    st.info("Upload both files to continue.")
