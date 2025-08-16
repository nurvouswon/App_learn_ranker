# app_learn_ranker.py
# ============================================================
# 📚 Learner — HR Day Ranker from Leaderboards + Event Parquet
# - Upload merged_leaderboards.csv (no labels) + event-level parquet (with hr_outcome at event level)
# - Create labeled leaderboard (per player/day) internally
# - Train LGBMRanker (LambdaRank), XGBRanker, CatBoostRanker
# - Save models -> multi_rankers.pkl
# - Offer labeled_leaderboard.csv for download
# - No plots, no tuner
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import unicodedata
import re
from datetime import datetime

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRanker

st.set_page_config(page_title="📚 HR Learner (Ranker)", layout="wide")
st.title("📚 Learner — HR Day Ranker from Leaderboards + Event Parquet")

# ------------------------------ Helpers ------------------------------

def _safe_read_csv(file):
    # Tolerant CSV loader
    try:
        return pd.read_csv(file)
    except UnicodeDecodeError:
        file.seek(0)
        return pd.read_csv(file, encoding="latin1")
    except Exception:
        file.seek(0)
        return pd.read_csv(file, engine="python")

def _norm_name(s: str) -> str:
    """Normalize player names consistently (strip, NFC, common fixes)."""
    if pd.isna(s):
        return ""
    s = str(s).strip()
    s = unicodedata.normalize("NFC", s)
    # Common canonicalizations
    fixes = {
        r"^Peter Alonso$": "Pete Alonso",
        r"^Jeremy Pena$": "Jeremy Peña",
        r"^Julio Rodriguez$": "Julio Rodríguez",
        r"^Michael A[\. ]? Taylor$": "Michael A. Taylor",
        r"^CJ Kayfus$": "C.J. Kayfus",
        r"^C[\. ]?J[\. ]? Kayfus$": "C.J. Kayfus",
    }
    for pat, rep in fixes.items():
        s = re.sub(pat, rep, s, flags=re.IGNORECASE)
    return s

def _norm_date_any(df, candidates, fallback=None):
    """
    Try to build a normalized date column (YYYY-MM-DD).
    Accepts list of candidate columns. Returns pd.Series of datetime64[ns].
    """
    out = None
    for col in candidates:
        if col in df.columns:
            v = df[col]
            # If looks like '2025-08-13'
            dt = pd.to_datetime(v, errors="coerce")
            if dt.notna().any():
                out = dt
                break
            # If looks like '8_13' or '08_13' -> infer year from another col if available
            # Try to parse M_D with current year if nothing else
            parsed = pd.to_datetime(v.astype(str).str.replace("_", "/", regex=False), errors="coerce")
            if parsed.notna().any():
                # If we have a year column somewhere, we could swap it in.
                # Otherwise, keep parsed (pandas assigns current year if missing).
                out = parsed
                break
    if out is None:
        if fallback is not None:
            out = pd.to_datetime(fallback, errors="coerce")
        else:
            out = pd.to_datetime(pd.Series([], dtype="object"))
    return out.dt.floor("D") if len(out) else pd.Series([], dtype="datetime64[ns]")

def _choose_features(df, preferred_order):
    """Return list of existing columns in df following preferred order."""
    return [c for c in preferred_order if c in df.columns]

def _groups_from_days(dates_series):
    """LightGBM/XGB group sizes by date."""
    d = pd.to_datetime(dates_series).dt.floor("D")
    return d.groupby(d.values).size().values.tolist()

def _to_float32_df(df):
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
    return df

def _ensure_2d(X):
    if isinstance(X, pd.Series):
        return X.to_frame()
    return X

# ------------------------------ File Uploads ------------------------------

st.markdown("### 1) Upload files")

lb_file = st.file_uploader("Merged leaderboard CSV (combined across days)", type=["csv"], key="merged_lb")
ev_file = st.file_uploader("Event-level Parquet (with hr_outcome at event level)", type=["parquet"], key="event_parquet")

if lb_file is not None and ev_file is not None:
    with st.spinner("Loading files..."):
        merged = _safe_read_csv(lb_file)
        events = pd.read_parquet(ev_file)

    st.write(f"Leaderboards shape: {merged.shape}")
    st.write(f"Events shape: {events.shape}")

    # ------------------------------ Normalize leaderboards ------------------------------
    merged = merged.copy()

    # Prefer ISO date if present, else try game_date, else parse what we can
    merged["game_date_norm"] = _norm_date_any(
        merged,
        candidates=["game_date_iso", "game_date"],
    )
    if merged["game_date_norm"].isna().all():
        st.error("Could not parse dates from merged leaderboards. Ensure 'game_date_iso' or 'game_date' is present.")
        st.stop()

    # Name normalization
    name_col_lb = "player_name" if "player_name" in merged.columns else None
    if name_col_lb is None:
        st.error("Merged leaderboard must include 'player_name'.")
        st.stop()
    merged["player_name_norm"] = merged[name_col_lb].astype(str).map(_norm_name)

    # ------------------------------ Build day labels from events ------------------------------
    ev = events.copy()

    # Derive/locate essential columns
    # - date
    ev_date_candidates = ["game_date", "date", "game_date_iso", "gameDate"]
    ev["game_date_norm"] = _norm_date_any(ev, ev_date_candidates)
    if ev["game_date_norm"].isna().all():
        st.error("Could not parse any usable date from events parquet. Expect a 'game_date'-like column.")
        st.stop()

    # - name
    ev_name_candidates = ["player_name", "batter_name", "batter", "name"]
    ev_name_col = None
    for c in ev_name_candidates:
        if c in ev.columns:
            ev_name_col = c
            break
    if ev_name_col is None:
        st.error("Could not find a batter/player name column in events parquet (e.g., 'player_name' or 'batter_name').")
        st.stop()
    ev["player_name_norm"] = ev[ev_name_col].astype(str).map(_norm_name)

    # - hr_outcome: if not present, try to synthesize from 'events' textual column
    label_col = None
    if "hr_outcome" in ev.columns:
        label_col = "hr_outcome"
    else:
        # try to infer from textual event-type columns
        text_cols = [c for c in ["events", "events_clean", "event", "play_desc"] if c in ev.columns]
        if text_cols:
            tmp = pd.Series(0, index=ev.index, dtype=np.int8)
            for c in text_cols:
                s = ev[c].astype(str).str.lower()
                tmp |= s.str.contains("home_run|homered|hr", regex=True).astype(np.int8)
            ev["hr_outcome"] = tmp
            label_col = "hr_outcome"
        else:
            st.error("No 'hr_outcome' in events parquet and no textual event column to infer from.")
            st.stop()

    # Aggregate to (player, day)
    day_labels = (
        ev.groupby(["game_date_norm", "player_name_norm"], dropna=False)[label_col]
          .max()
          .reset_index()
          .rename(columns={label_col: "hr_outcome"})
    )

    # ------------------------------ Join leaderboards with labels ------------------------------
    key_cols = ["game_date_norm", "player_name_norm"]
    to_join = merged.merge(day_labels, on=key_cols, how="left", validate="m:1")

    # Report overlap
    overlap_names = len(set(merged["player_name_norm"]) & set(day_labels["player_name_norm"]))
    overlap_dates = len(set(merged["game_date_norm"]) & set(day_labels["game_date_norm"]))

    labeled = to_join.dropna(subset=["hr_outcome"]).copy()
    labeled["hr_outcome"] = labeled["hr_outcome"].astype(int)

    st.info(
        f"Labels joined. Rows with labels: {len(labeled)} (dropped {len(to_join) - len(labeled)} without event data)."
        f"\nOverlap (unique names): {overlap_names} • Overlap (unique dates): {overlap_dates}"
    )

    if labeled.empty:
        st.error("No label matches found after join.\n\n"
                 "Hints:\n"
                 "• Make sure merged leaderboard has a real date (prefer 'game_date_iso' like 2025-08-13).\n"
                 "• Ensure names match (accents, suffixes). This app already applies common fixes.")
        st.stop()

    # ------------------------------ Feature selection ------------------------------
    st.markdown("### 2) Select features for the ranker (auto-selected for you)")

    preferred_features = [
        # Core from prediction app
        "ranked_probability",
        "hr_probability_iso_T",
        "final_multiplier",
        "overlay_multiplier",
        "weak_pitcher_factor",
        "hot_streak_factor",
        "rrf_aux",
        "model_disagreement",

        # Extras you asked to include
        "prob_2tb",
        "prob_rbi",
        "final_multiplier_raw",

        # Optional weather if present on leaderboard rows
        "temp",
        "humidity",
        "wind_mph",
    ]

    # Drop junk columns
    drop_noise = [c for c in labeled.columns if c.startswith("Unnamed")]
    if drop_noise:
        labeled = labeled.drop(columns=drop_noise, errors="ignore")

    # Candidate feature columns present
    auto_feats = _choose_features(labeled, preferred_features)

    # UI multi-select (pre-checked with auto)
    feats = st.multiselect(
        "Features",
        options=sorted([c for c in labeled.columns if pd.api.types.is_numeric_dtype(labeled[c])]),
        default=auto_feats,
    )

    if not feats:
        st.error("Please select at least one numeric feature.")
        st.stop()

    # ------------------------------ Train day-wise rankers ------------------------------
    st.markdown("### 3) Train day-wise LambdaRank model")

    # Prepare matrices
    train = labeled.copy()
    train = _to_float32_df(train)

    X = train[feats].copy()
    y = train["hr_outcome"].astype(np.float32).values
    days = pd.to_datetime(train["game_date_norm"]).dt.floor("D")
    groups = _groups_from_days(days)

    # Safety checks
    if len(X) < 1 or X.shape[1] < 1:
        st.error("Input data must be 2D and non-empty after feature selection.")
        st.stop()
    if np.sum(y) == 0:
        st.warning("All labels are 0. Rankers can still train, but evaluation will be limited.")

    # LightGBM Ranker (LambdaRank)
    rk_lgb = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=1,
        random_state=42,
    )
    rk_lgb.fit(X, y, group=groups)

    # XGBoost Ranker (pairwise)
    rk_xgb = xgb.XGBRanker(
        objective="rank:pairwise",
        n_estimators=600,
        learning_rate=0.06,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        tree_method="hist",
        random_state=42,
        n_jobs=0,
    )
    rk_xgb.fit(X, y, group=groups)

    # CatBoost Ranker
    rk_cb = CatBoostRanker(
        iterations=800,
        learning_rate=0.06,
        depth=7,
        loss_function="YetiRank",
        random_seed=42,
        verbose=False
    )
    # CatBoost needs query_id per row (here, per day). Map each day to an integer query id.
    day_to_qid = {d: i for i, d in enumerate(sorted(days.unique()))}
    qid = days.map(day_to_qid).astype(int).values
    rk_cb.fit(X, y, group_id=qid)

    st.success("Model trained.")

    # ------------------------------ Output: labeled CSV & model bundle ------------------------------
    st.markdown("### 4) Downloads")

    # Labeled leaderboard CSV
    labeled_out = labeled.copy()
    labeled_csv = labeled_out.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download labeled_leaderboard.csv",
        data=labeled_csv,
        file_name="labeled_leaderboard.csv",
        mime="text/csv",
    )

    # Model bundle for prediction app
    bundle = {
        "features": feats,
        "models": {
            "lgbm": rk_lgb,
            "xgbr": rk_xgb,
            "catb": rk_cb,
        },
        "meta": {
            "trained_rows": int(len(train)),
            "trained_days": int(len(np.unique(days))),
            "label_positive_count": int(np.sum(y)),
        },
    }
    buf = io.BytesIO()
    pickle.dump(bundle, buf)
    st.download_button(
        "⬇️ Download multi_rankers.pkl",
        data=buf.getvalue(),
        file_name="multi_rankers.pkl",
        mime="application/octet-stream",
    )

    # ------------------------------ Preview ------------------------------
    st.markdown("### 5) Preview (first 50 rows of the labeled data used for training)")
    st.dataframe(labeled_out.head(50), use_container_width=True)

else:
    st.caption("Upload both files to proceed.")
