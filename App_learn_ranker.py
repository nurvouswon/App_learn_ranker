# app_learn_ranker.py
# ============================================================
# 📚 Learner — HR Day Ranker from Leaderboards + Event Parquet
# - Upload 1) merged_leaderboards.csv  2) event-level parquet (with hr_outcome)
# - Robust name & date normalization (copies your prediction app style)
# - Joins labels by (date, batter_id) if available, else (date, player_name)
# - Uses ALL three ML rankers: LightGBM, XGBoost, CatBoost
# - Includes prob_2tb and prob_rbi as training features (if present)
# - Saves bundle: {"model": blended_ranker, "features": rk_feats}
# - Also lets you download the merged labeled dataset
# - No plots, no tuner, no heavy extras
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import io, gc, pickle, re, unicodedata
from datetime import datetime

# ML
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ndcg_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRanker

# ========= UI / Page =========
st.set_page_config(page_title="📚 Learner — HR Day Ranker", layout="wide")
st.title("📚 Learner — HR Day Ranker from Leaderboards + Event Parquet")

# ========= Helpers =========
def _strip_accents(s: str) -> str:
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def _normalize_name(raw: str) -> str:
    """Lowercase, remove accents, punctuation, suffixes, compress spaces."""
    s = "" if raw is None else str(raw)
    s = s.strip()
    s = _strip_accents(s)
    s = s.replace("-", " ")
    s = re.sub(r"[.,']", " ", s)
    # remove common suffixes
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s)
    s = s.lower().strip()
    # simple aliases (lightweight; add if you run into mismatches)
    aliases = {
        "peter alonso": "pete alonso",
        "cj kayfus": "c j kayfus",  # unify variants
        "c j kayfus": "cj kayfus",
        "michael a taylor": "michael a taylor",
    }
    return aliases.get(s, s)

def zscore(a):
    a = np.asarray(a, dtype=np.float64)
    m = np.nanmean(a)
    s = np.nanstd(a)
    if s == 0 or np.isnan(s):
        return np.zeros_like(a, dtype=np.float64)
    return (a - m) / s

def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1-1e-6)
    return np.log(p / (1 - p))

def _parse_game_date_series(s: pd.Series, season_year: int) -> pd.Series:
    """Return pandas datetime.date for a variety of formats."""
    if s is None:
        return pd.Series(pd.NaT, index=pd.RangeIndex(0))
    s = s.astype(str).str.strip()
    out = pd.to_datetime(s, errors="coerce", utc=False, infer_datetime_format=True)
    # Try M_D (like 8_13 or 8/13)
    mask = out.isna()
    if mask.any():
        s2 = s[mask].str.replace(r"[^0-9_/-]", "", regex=True)
        s2 = s2.str.replace("_", "/", regex=False)
        # Expect "M/D"
        def try_m_d(v):
            parts = v.split("/")
            if len(parts) == 2 and all(p.isdigit() for p in parts):
                m, d = int(parts[0]), int(parts[1])
                try:
                    return pd.Timestamp(datetime(season_year, m, d))
                except Exception:
                    return pd.NaT
            return pd.NaT
        converted = s2.apply(try_m_d)
        out.loc[mask] = converted
    # Return date (no time)
    return pd.to_datetime(out).dt.date

def _ranker_groups_from_dates(dates: pd.Series):
    """Return groups array (sizes) sorted by date, and sorted indices."""
    df = pd.DataFrame({"d": pd.to_datetime(dates).dt.date})
    df["_ord"] = df["d"].astype(str)
    df_sorted = df.sort_values(["_ord"]).reset_index()
    sizes = df_sorted.groupby("d").size().tolist()
    order_idx = df_sorted["index"].values
    return sizes, order_idx

def _per_day_z(x: pd.Series, dates: pd.Series):
    df = pd.DataFrame({"x": x.values, "d": pd.to_datetime(dates).dt.date})
    return df.groupby("d")["x"].transform(lambda v: pd.Series(zscore(v), index=v.index)).values

# ========= Uploads =========
st.markdown("### 1) Upload files")
lb_file = st.file_uploader("Merged leaderboard CSV (combined across days) — required", type=["csv"])
ev_file = st.file_uploader("Event-level Parquet (with hr_outcome) — required", type=["parquet"])

season_year = st.number_input("Season year (used if leaderboard dates are in M_D format like 8_13)", min_value=2000, max_value=2100, value=2025, step=1)

use_batter_id = st.checkbox("Prefer matching by batter_id when available", value=True)

if lb_file is not None and ev_file is not None:
    with st.spinner("Loading files..."):
        lb = pd.read_csv(lb_file)
        ev = pd.read_parquet(ev_file)

    # Basic sanity
    if "player_name" not in lb.columns:
        st.error("Leaderboard CSV must have 'player_name' column.")
        st.stop()

    # ---------- Normalize Leaderboard ----------
    # Date: accept either 'game_date' (best) or fallback to 'game_date_iso'
    if "game_date" in lb.columns:
        lb["game_date_norm"] = _parse_game_date_series(lb["game_date"], season_year=season_year)
    elif "game_date_iso" in lb.columns:
        lb["game_date_norm"] = _parse_game_date_series(lb["game_date_iso"], season_year=season_year)
    else:
        st.error("Leaderboard must have 'game_date' (YYYY-MM-DD or M_D) or 'game_date_iso'.")
        st.stop()

    lb["player_name_norm"] = lb["player_name"].astype(str).map(_normalize_name)

    # If leaderboards kept any accidental index column
    for bad in ["Unnamed: 0", "index"]:
        if bad in lb.columns:
            lb = lb.drop(columns=[bad])

    # ---------- Normalize Event Parquet ----------
    # Expect hr_outcome at event-level → roll up to player-day label (1 if any HR)
    # Date
    if "game_date" not in ev.columns:
        st.error("Event parquet must contain a 'game_date' column.")
        st.stop()
    ev["game_date_norm"] = _parse_game_date_series(ev["game_date"], season_year=season_year)

    # Prefer batter_id if requested and present
    have_batter_id = use_batter_id and ("batter_id" in ev.columns) and ("batter_id" in lb.columns)

    # Player name normalization for event data as fallback
    if "player_name" in ev.columns:
        ev["player_name_norm"] = ev["player_name"].astype(str).map(_normalize_name)
    else:
        # Sometimes event files call it 'batter_name'
        if "batter_name" in ev.columns:
            ev["player_name_norm"] = ev["batter_name"].astype(str).map(_normalize_name)
        else:
            ev["player_name_norm"] = ""

    # Build labels by player-day
    if "hr_outcome" not in ev.columns:
        st.error("Event parquet is missing 'hr_outcome'. It must be present to build labels.")
        st.stop()

    label_keys = ["game_date_norm", "batter_id"] if have_batter_id else ["game_date_norm", "player_name_norm"]
    labels = (
        ev.groupby(label_keys)["hr_outcome"]
          .max()  # any HR → 1
          .reset_index()
          .rename(columns={"hr_outcome": "hr_outcome_day"})
    )

    # ---------- Join labels onto leaderboard ----------
    # Prepare lb join keys
    if have_batter_id:
        if "batter_id" not in lb.columns:
            st.warning("Leaderboard missing 'batter_id'; falling back to (date, name) match.")
            have_batter_id = False

    if have_batter_id:
        lb["_k_date"] = lb["game_date_norm"]
        lb["_k_id"] = lb["batter_id"]
        labels["_k_date"] = labels["game_date_norm"]
        labels["_k_id"] = labels["batter_id"]
        merged = lb.merge(labels[["_k_date", "_k_id", "hr_outcome_day"]],
                          on=["_k_date", "_k_id"], how="left")
    else:
        lb["_k_date"] = lb["game_date_norm"]
        lb["_k_name"] = lb["player_name_norm"]
        labels["_k_date"] = labels["game_date_norm"]
        labels["_k_name"] = labels["player_name_norm"]
        merged = lb.merge(labels[["_k_date", "_k_name", "hr_outcome_day"]],
                          on=["_k_date", "_k_name"], how="left")

    # Diagnostics
    total = len(merged)
    with_labels = int(merged["hr_outcome_day"].notna().sum())
    st.write(f"Labels joined. Rows with labels: {with_labels} (dropped {total - with_labels} without event data).")

    if with_labels == 0:
        # Offer simple hints
        # Count overlaps quickly
        if have_batter_id:
            overlap_ids = len(set(lb["batter_id"].dropna().astype(str)) & set(ev["batter_id"].dropna().astype(str)))
            st.error(
                f"No label matches found after join.\n\n"
                f"Overlap (unique batter_id): {overlap_ids}\n"
                f"Hints: ensure both files share batter_id and that game_date formats align."
            )
        else:
            overlap_names = len(set(lb["player_name_norm"]) & set(ev["player_name_norm"]))
            overlap_dates = len(set(lb["game_date_norm"]) & set(ev["game_date_norm"]))
            st.error(
                "No label matches found after join.\n\n"
                f"Overlap (unique normalized names): {overlap_names}\n"
                f"Overlap (unique dates): {overlap_dates}\n"
                "Hints: make sure the leaderboard uses real dates (YYYY-MM-DD) in 'game_date' "
                "and that player names match the event file after normalization (accents/suffixes handled here)."
            )
        st.stop()

    # Keep only labeled rows
    merged = merged[merged["hr_outcome_day"].notna()].copy()
    merged["hr_outcome_day"] = merged["hr_outcome_day"].astype(int)

    # ========= Feature Build (mirrors prediction app semantics) =========
    # Core features expected by prediction app:
    # base_prob (from hr_probability_iso_T), logit_p, log_overlay (log(final_multiplier)),
    # ranker_z (per-day z of ranked_probability), overlay_multiplier, final_multiplier
    # Plus requested: prob_2tb, prob_rbi (if present)
    need_cols = ["ranked_probability", "hr_probability_iso_T", "final_multiplier", "overlay_multiplier"]
    missing = [c for c in need_cols if c not in merged.columns]
    if missing:
        st.error(f"Merged leaderboard missing required columns: {missing}")
        st.stop()

    merged["base_prob"] = merged["hr_probability_iso_T"].astype(float).clip(1e-6, 1-1e-6)
    merged["logit_p"] = logit(merged["base_prob"])
    merged["log_overlay"] = np.log(merged["final_multiplier"].astype(float) + 1e-9)

    # ranker_z = per-day z-score of ranked_probability
    merged["ranker_z"] = _per_day_z(merged["ranked_probability"].astype(float), merged["game_date_norm"])

    # Optional weather (if present)
    for w in ["temp", "humidity", "wind_mph"]:
        if w in merged.columns:
            merged[w] = pd.to_numeric(merged[w], errors="coerce")

    # Optional 2TB/RBI (strongly requested)
    for p in ["prob_2tb", "prob_rbi"]:
        if p not in merged.columns:
            merged[p] = 0.0  # still include feature; zeros if not provided

    # Build X/y in sorted day order (needed for rankers)
    rk_feats = [
        "base_prob",
        "logit_p",
        "log_overlay",
        "ranker_z",
        "overlay_multiplier",
        "final_multiplier",
        "prob_2tb",
        "prob_rbi",
    ]

    # Ensure float32
    for c in rk_feats:
        merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0.0).astype(np.float32)

    # Sort by date then by ranked_probability desc (stable)
    merged["_ord_date"] = pd.to_datetime(merged["game_date_norm"]).astype(np.int64)
    merged = merged.sort_values(["_ord_date", "ranked_probability"], ascending=[True, False]).reset_index(drop=True)

    X = merged[rk_feats].to_numpy(dtype=np.float32)
    y = merged["hr_outcome_day"].to_numpy(dtype=np.int32)

    # Ranker groups (sizes) in this order
    groups, order_idx = _ranker_groups_from_dates(merged["game_date_norm"])
    # (we already sorted merged by date, so no reindex needed)

    if len(groups) < 1 or X.shape[0] == 0:
        st.error("Input data must be 2D and non-empty after labeling & sorting.")
        st.stop()

    st.markdown("### 2) Train day-wise rankers (LGBM + XGB + CatBoost)")
    with st.spinner("Training LightGBM LambdaRank..."):
        lgbm = lgb.LGBMRanker(
            objective="lambdarank", metric="ndcg",
            n_estimators=600, learning_rate=0.05, num_leaves=63,
            feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=1,
            random_state=42
        )
        lgbm.fit(X, y, group=groups)

    with st.spinner("Training XGBoost Ranker..."):
        xgbr = xgb.XGBRanker(
            objective="rank:ndcg",
            eval_metric="ndcg",
            n_estimators=500,
            learning_rate=0.08,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            random_state=42,
            tree_method="hist"
        )
        xgbr.fit(X, y, group=groups)

    with st.spinner("Training CatBoost Ranker..."):
        # build group_id per row (same id for same day)
        days = pd.to_datetime(merged["game_date_norm"]).dt.date
        day_to_gid = {d:i for i, d in enumerate(sorted(days.unique()))}
        group_id = days.map(day_to_gid).astype(int).values
        cbr = CatBoostRanker(
            loss_function="YetiRank",
            iterations=1200,
            learning_rate=0.05,
            depth=7,
            l2_leaf_reg=6.0,
            random_seed=42,
            verbose=False
        )
        cbr.fit(X, y, group_id=group_id)

    st.success("Models trained.")

    # Simple blended ranker wrapper (mean of 3 rankers)
    class BlendedRanker:
        def __init__(self, lgbm_model, xgb_model, cb_model):
            self.lgbm = lgbm_model
            self.xgb = xgb_model
            self.cb  = cb_model

        def predict(self, X_input):
            p1 = self.lgbm.predict(X_input)
            p2 = self.xgb.predict(X_input)
            p3 = self.cb.predict(X_input)
            # Normalize each to z, then average, then rescale to 0..1 via sigmoid-ish transform
            P = np.column_stack([p1, p2, p3]).astype(float)
            Pz = (P - P.mean(axis=0, keepdims=True)) / (P.std(axis=0, keepdims=True) + 1e-9)
            s = Pz.mean(axis=1)
            # squash to 0..1 (ranking score on prob-like scale)
            return 1.0 / (1.0 + np.exp(-s))

    blended = BlendedRanker(lgbm, xgbr, cbr)

    # Quick NDCG@30 check (diagnostic only, not shown as plot)
    try:
        s_train = blended.predict(X)
        # compute per-day ndcg
        ndcgs = []
        start = 0
        for g in groups:
            y_g = y[start:start+g]
            s_g = s_train[start:start+g]
            # need 2D arrays for ndcg_score
            ndcgs.append(ndcg_score([y_g], [s_g], k=min(30, g)))
            start += g
        st.write(f"Mean NDCG@30 (train): {np.mean(ndcgs):.4f}")
    except Exception:
        pass

    # ========= Save bundle & labeled CSV =========
    bundle = {
        "model": blended,
        "features": rk_feats,  # exact order expected by prediction app
    }
    buf = io.BytesIO()
    pickle.dump(bundle, buf)
    buf.seek(0)

    st.download_button(
        label="⬇️ Download learning_ranker.pkl",
        data=buf,
        file_name="learning_ranker.pkl",
        mime="application/octet-stream"
    )

    # Also export the labeled training set used
    out_cols = ["game_date_norm", "player_name", "player_name_norm", "team_code",
                "ranked_probability", "hr_probability_iso_T", "overlay_multiplier",
                "final_multiplier", "prob_2tb", "prob_rbi", "hr_outcome_day"]
    out_cols = [c for c in out_cols if c in merged.columns]
    merged_out = merged[out_cols].copy()
    merged_out = merged_out.rename(columns={"game_date_norm": "game_date"})
    csv_buf = io.StringIO()
    merged_out.to_csv(csv_buf, index=False)
    st.download_button(
        label="⬇️ Download labeled_leaderboards_for_learning.csv",
        data=csv_buf.getvalue().encode("utf-8"),
        file_name="labeled_leaderboards_for_learning.csv",
        mime="text/csv"
    )

    st.success("✅ Done. Use learning_ranker.pkl in your prediction app (upload in the Learning Ranker slot).")

    # Light cleanup
    gc.collect()
