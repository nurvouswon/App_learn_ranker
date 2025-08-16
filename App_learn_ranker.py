# App_learn_ranker.py
# ============================================================
# 📚 Learner — HR Day Ranker from Leaderboards + Event Parquet
# - Upload merged leaderboards CSV (across days) + event-level parquet
# - Robust name/date normalization for label join
# - Build day-level labels (did player HR on that day?)
# - Select features (auto-selected; includes 2TB/RBI if present)
# - Train LightGBM LambdaRank day-wise
# - Download: trained ranker (pickle) + labeled merged leaderboard CSV
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import unicodedata

import lightgbm as lgb

# ===================== UI =====================
st.set_page_config(page_title="📚 Learner — HR Day Ranker", layout="wide")
st.title("📚 Learner — HR Day Ranker from Leaderboards + Event Parquet")

st.markdown("**1) Upload files**")

lb_file = st.file_uploader(
    "Merged leaderboard CSV (combined across days)", type=["csv"], key="lb_csv"
)
evpq_file = st.file_uploader(
    "Event-level Parquet (with hr_outcome)", type=["parquet"], key="ev_pq"
)

# ===================== Helpers =====================
@st.cache_data(show_spinner=False, max_entries=2)
def _safe_read_csv(file):
    try:
        return pd.read_csv(file, low_memory=False)
    except UnicodeDecodeError:
        file.seek(0)
        return pd.read_csv(file, low_memory=False, encoding="latin1")

@st.cache_data(show_spinner=False, max_entries=2)
def _safe_read_parquet(file):
    return pd.read_parquet(file)

def _safe_drop(df: pd.DataFrame, cols):
    return df.drop(columns=[c for c in cols if c in df.columns], errors="ignore")

def _normalize_name(s: pd.Series) -> pd.Series:
    s = s.astype(str).fillna("").str.strip().str.lower()
    # remove accents
    s = s.apply(lambda x: unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode("ascii"))
    # remove punctuation
    s = s.str.replace(r"[^\w\s-]", "", regex=True)
    # collapse spaces
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    # canonicalize a few known variants
    fixes = {
        "peter alonso": "pete alonso",
        "julio rodriguez": "julio rodriguez",
        "michael a taylor": "michael a taylor",
        "cj kayfus": "c j kayfus",
        "c. j. kayfus": "c j kayfus",
        "c j kayfus": "c j kayfus",
    }
    return s.map(lambda x: fixes.get(x, x))

def _normalize_date(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", utc=False)
    # strip timezone if any, then normalize to date
    try:
        dt = dt.dt.tz_localize(None)
    except Exception:
        pass
    return dt.dt.normalize()

def _prep_features(df: pd.DataFrame, feature_list):
    X = df[feature_list].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return X

# ===================== MAIN =====================
if lb_file is not None and evpq_file is not None:
    with st.spinner("Reading files..."):
        lb = _safe_read_csv(lb_file)
        ev = _safe_read_parquet(evpq_file)

    # Basic cleanup
    lb = _safe_drop(lb, ["Unnamed: 0", "index"])
    ev = _safe_drop(ev, ["Unnamed: 0", "index"])

    st.write(f"Leaderboard rows: {len(lb):,}  |  Event rows: {len(ev):,}")

    # Require columns
    need_lb = {"player_name", "game_date"}
    need_ev = {"player_name", "game_date", "hr_outcome"}
    miss_lb = [c for c in need_lb if c not in lb.columns]
    miss_ev = [c for c in need_ev if c not in ev.columns]
    if miss_lb:
        st.error(f"Leaderboard is missing required columns: {miss_lb}")
        st.stop()
    if miss_ev:
        st.error(f"Event parquet is missing required columns: {miss_ev}")
        st.stop()

    # Normalize join keys
    lb["key_name"] = _normalize_name(lb["player_name"])
    lb["key_date"] = _normalize_date(lb["game_date"])

    ev["key_name"] = _normalize_name(ev["player_name"])
    ev["key_date"] = _normalize_date(ev["game_date"])

    # Build day-level labels: did player HR on that day?
    labels = (
        ev.groupby(["key_date", "key_name"])["hr_outcome"]
          .max()  # 1 if any HR that day
          .reset_index()
    )

    # Join labels to leaderboard
    lb_lab = lb.merge(labels, on=["key_date", "key_name"], how="left")
    rows_with_labels = int(lb_lab["hr_outcome"].notna().sum())
    st.write(
        f"Labels joined. Rows with labels: **{rows_with_labels}** (dropped {len(lb_lab) - rows_with_labels} without event data)."
    )

    # If no labels matched, provide diagnostics and stop
    if rows_with_labels == 0:
        overlap_names = len(set(lb["key_name"]).intersection(set(ev["key_name"])))
        overlap_dates = len(set(lb["key_date"].dropna()).intersection(set(ev["key_date"].dropna())))
        st.error(
            "No label matches found after join.\n"
            f"- Overlap (unique names): {overlap_names}\n"
            f"- Overlap (unique dates): {overlap_dates}\n"
            "Check that both files use comparable player names and dates."
        )
        st.stop()

    # Keep only rows with labels
    lb_tr = lb_lab[lb_lab["hr_outcome"].notna()].copy()
    lb_tr["hr_outcome"] = lb_tr["hr_outcome"].astype(int)

    st.markdown("**2) Select features for the ranker (auto-selected for you)**")

    # Candidate feature list — includes 2TB/RBI, overlay bits, diagnostics, weather if present
    candidate_feats = [
        # core signals from prediction leaderboard
        "ranked_probability",
        "hr_probability_iso_T",
        "final_multiplier",
        "overlay_multiplier",
        "weak_pitcher_factor",
        "hot_streak_factor",
        "rrf_aux",
        "model_disagreement",
        # add-ons you requested
        "prob_2tb",
        "prob_rbi",
        "final_multiplier_raw",
        # optional weather if present
        "temp", "humidity", "wind_mph",
    ]
    # Only keep those that actually exist in the uploaded leaderboard
    candidate_feats = [c for c in candidate_feats if c in lb_tr.columns]

    # If nothing exists, abort
    if not candidate_feats:
        st.error("No usable features found in the merged leaderboard. Check column names.")
        st.stop()

    chosen_feats = st.multiselect(
        "Features", options=sorted(candidate_feats), default=candidate_feats, key="feat_ms"
    )

    if not chosen_feats:
        st.error("Select at least one feature.")
        st.stop()

    # Prepare X/y and groups (day-wise)
    X = _prep_features(lb_tr, chosen_feats)
    y = lb_tr["hr_outcome"].to_numpy(dtype=np.int32)

    # groups: each day is one query for LambdaRank
    grp_sizes = lb_tr.groupby("key_date").size().tolist()
    n_days = len(grp_sizes)

    if X.shape[0] == 0 or X.shape[1] == 0:
        st.error("Input data became empty after feature prep. Please check selected features.")
        st.stop()
    if n_days < 2:
        st.error("Need at least 2 days of data to train a day-wise ranker.")
        st.stop()
    if sum(grp_sizes) != len(lb_tr):
        st.error("Grouping mismatch; please check key_date construction.")
        st.stop()

    st.write(
        f"Prepared training data → X: {X.shape}, y: {y.shape}, days: {n_days}, total_rows: {len(lb_tr)}"
    )

    st.markdown("**3) Train day-wise LambdaRank model**")

    # LightGBM LambdaRank
    rk = lgb.LGBMRanker(
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
        rk.fit(X, y, group=grp_sizes)

    st.success("✅ Model trained.")

    # ---- Save outputs ----
    # 1) Trained model bundle (pickle)
    bundle = {
        "model": rk,
        "features": chosen_feats,
        "notes": "Day-wise LambdaRank trained on merged leaderboards + event labels (day-level HR).",
    }
    model_bytes = pickle.dumps(bundle)

    st.download_button(
        label="⬇️ Download learning_ranker.pkl",
        data=model_bytes,
        file_name="learning_ranker.pkl",
        mime="application/octet-stream",
    )

    # 2) Labeled merged leaderboard CSV (for your records / re-use)
    labeled_csv = lb_tr.copy()
    # keep a compact set of helpful columns if present
    keep_cols = [
        "game_date", "player_name",
        "ranked_probability", "hr_probability_iso_T",
        "final_multiplier", "overlay_multiplier", "weak_pitcher_factor", "hot_streak_factor",
        "rrf_aux", "model_disagreement",
        "prob_2tb", "prob_rbi",
        "final_multiplier_raw",
        "temp", "humidity", "wind_mph",
        "hr_outcome",
    ]
    keep_cols = [c for c in keep_cols if c in labeled_csv.columns]
    labeled_csv_out = labeled_csv[keep_cols].copy()
    st.download_button(
        label="⬇️ Download merged_leaderboards_labeled.csv",
        data=labeled_csv_out.to_csv(index=False),
        file_name="merged_leaderboards_labeled.csv",
        mime="text/csv",
    )

    st.markdown("**Preview (top 50 rows)**")
    st.dataframe(labeled_csv_out.head(50), use_container_width=True)
