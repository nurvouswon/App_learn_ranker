# ============================================================
# 📚 Learner App — Build a day-wise ranker from leaderboards + event parquet
# - Inputs: merged_leaderboard.csv, event_level.parquet
# - Creates player-day labels from parquet (hr_outcome)
# - Joins labels to leaderboard rows
# - Trains LGBMRanker (LambdaRank) grouped by game_date
# - Exports learning_ranker.pkl and merged_leaderboards_labeled.csv
# - No weather backfill needed; no charts; no tuner
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import io, pickle, gc
import lightgbm as lgb

st.set_page_config(page_title="Learner — HR Day Ranker", layout="wide")
st.title("📚 Learner — HR Day Ranker from Leaderboards + Event Parquet")

# ------------------------- Helpers -------------------------
@st.cache_data(show_spinner=False)
def _safe_read_any(path_or_file):
    name = getattr(path_or_file, "name", str(path_or_file)).lower()
    if name.endswith(".parquet"):
        return pd.read_parquet(path_or_file)
    if name.endswith(".csv"):
        return pd.read_csv(path_or_file, low_memory=False)
    # fallback: try parquet then csv
    try:
        return pd.read_parquet(path_or_file)
    except Exception:
        return pd.read_csv(path_or_file, low_memory=False)

def _to_day(s):
    s = pd.to_datetime(s, errors="coerce")
    return s.dt.tz_localize(None, nonexistent='NaT', ambiguous='NaT').dt.floor("D")

def _find_player_col(df):
    # Try common options; return first that exists
    for c in ["player_name", "batter_name", "batter_full_name", "batter"]:
        if c in df.columns:
            return c
    # Last resort: anything that looks like a name column
    name_like = [c for c in df.columns if "name" in c.lower()]
    return name_like[0] if name_like else None

def _build_labels_from_events(event_df, target_col="hr_outcome"):
    if target_col not in event_df.columns:
        raise ValueError(f"'{target_col}' not found in event parquet.")
    # date column
    dt_col = None
    for cand in ["game_date", "game_datetime", "game_date_time", "date"]:
        if cand in event_df.columns:
            dt_col = cand
            break
    if dt_col is None:
        raise ValueError("No date/time column found in event parquet (looked for game_date / game_datetime / date).")

    name_col = _find_player_col(event_df)
    if name_col is None:
        raise ValueError("No player name column found in event parquet (expected player_name / batter_name / batter_full_name).")

    df = event_df[[dt_col, name_col, target_col]].copy()
    df[dt_col] = _to_day(df[dt_col])
    df[name_col] = df[name_col].astype(str).str.strip()

    # Aggregate to player-day label: 1 if any HR that day
    labels = (
        df.groupby([dt_col, name_col], dropna=False)[target_col]
        .max()
        .reset_index()
        .rename(columns={dt_col: "game_date", name_col: "player_name", target_col: "hr_outcome"})
    )
    return labels

def _choose_feature_columns(leader_df):
    # Start with numeric columns and drop identifiers
    id_cols = {"game_date", "player_name", "team_code", "time"}
    num_cols = leader_df.select_dtypes(include=[np.number]).columns.tolist()

    # Prefer well-known leaderboard signals if they exist
    preferred = [
        "ranked_probability",
        "hr_probability_iso_T",
        "final_multiplier",
        "overlay_multiplier",
        "weak_pitcher_factor",
        "hot_streak_factor",
        "rrf_aux",
        "model_disagreement",
        # user occasionally wants RBI / 2B if present in leaderboard
        "RBI", "rbi", "RBIs",
    ]
    # Include any numeric columns that aren't obvious identifiers
    base = [c for c in num_cols if c not in id_cols]

    # Bring preferred to the front (if present), keep others after
    ordered = [c for c in preferred if c in base] + [c for c in base if c not in preferred]
    # If nothing numeric is left, that's an error
    if not ordered:
        raise ValueError("No numeric feature columns found in merged leaderboard.")
    return ordered

def _groups_from_days(day_series):
    # LightGBM expects group sizes (list of ints), in the same order as the data
    return day_series.groupby(day_series.values).size().values.tolist()

def _evaluate_hits_ndcg_per_day(df, score_col, label_col="hr_outcome", K=(10, 20, 30)):
    """Compute mean Hits@K and NDCG@K across days."""
    out = []
    for k in K:
        hits_list, ndcg_list = [], []
        for day, g in df.groupby("game_date"):
            g = g.sort_values(score_col, ascending=False)
            y = g[label_col].values.astype(int)
            # Hits@K
            hits_k = int(y[:k].sum()) if len(y) else 0
            hits_list.append(hits_k)
            # NDCG@K
            rel = y[:k]
            if len(rel) == 0:
                ndcg = 0.0
            else:
                discounts = 1.0 / np.log2(np.arange(2, 2 + len(rel)))
                dcg = float((rel * discounts).sum())
                ideal = np.sort(y)[::-1][:k]
                if ideal.sum() == 0:
                    ndcg = 0.0
                else:
                    idcg = float((ideal * discounts[:len(ideal)]).sum())
                    ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_list.append(ndcg)
        out.append({
            "K": k,
            "Mean Hits@K": float(np.mean(hits_list)) if hits_list else 0.0,
            "Mean NDCG@K": float(np.mean(ndcg_list)) if ndcg_list else 0.0,
        })
    return pd.DataFrame(out)

# ------------------------- Inputs -------------------------
st.subheader("1) Upload files")
merged_csv = st.file_uploader("Merged leaderboard CSV (combined across days)", type=["csv"])
event_parquet = st.file_uploader("Event-level Parquet (with hr_outcome)", type=["parquet"])

if merged_csv is not None and event_parquet is not None:
    with st.spinner("Reading files..."):
        leader = _safe_read_any(merged_csv)
        events = _safe_read_any(event_parquet)

    # Basic normalize
    if "game_date" not in leader.columns:
        st.error("Merged leaderboard CSV needs a 'game_date' column.")
        st.stop()
    if "player_name" not in leader.columns:
        st.error("Merged leaderboard CSV needs a 'player_name' column.")
        st.stop()

    leader = leader.copy()
    leader["game_date"] = _to_day(leader["game_date"])
    leader["player_name"] = leader["player_name"].astype(str).str.strip()

    # Build labels from event parquet
    with st.spinner("Building player-day labels (hr_outcome) from event parquet..."):
        labels = _build_labels_from_events(events, target_col="hr_outcome")

    # Join labels → keep only rows with label present (i.e., days you have events for)
    df = leader.merge(labels, on=["game_date", "player_name"], how="left")
    num_before = len(df)
    df = df.dropna(subset=["hr_outcome"]).copy()
    df["hr_outcome"] = df["hr_outcome"].astype(int)
    num_after = len(df)

    st.success(f"Labels joined. Rows with labels: {num_after} (dropped {num_before - num_after} without event data).")

    # Filter out days with only 1 row (ranker needs >1 per group)
    sizes = df.groupby("game_date").size()
    valid_days = sizes[sizes >= 2].index
    removed_days = set(df["game_date"].unique()) - set(valid_days)
    if removed_days:
        df = df[df["game_date"].isin(valid_days)].copy()
        st.info(f"Removed {len(removed_days)} day(s) with only one row (ranker requires groups of size ≥ 2).")

    # Pick features automatically (user can override)
    auto_feats = _choose_feature_columns(df)
    st.subheader("2) Select features for the ranker (auto-selected for you)")
    chosen_feats = st.multiselect("Features", options=auto_feats, default=auto_feats)

    if not chosen_feats:
        st.error("Please select at least one feature.")
        st.stop()

    # Prepare train matrix
    X = df[chosen_feats].astype(np.float32).copy()
    y = df["hr_outcome"].astype(int).values
    groups = _groups_from_days(df["game_date"])

    # Train-validation via simple chronological split by day (optional)
    # Here we fit one final model on all data (small datasets benefit from using everything)
    st.subheader("3) Train day-wise LambdaRank model")
    with st.spinner("Training LightGBM Ranker (LambdaRank)..."):
        rk = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=700,
            learning_rate=0.05,
            num_leaves=63,
            feature_fraction=0.85,
            bagging_fraction=0.85,
            bagging_freq=1,
            random_state=42,
        )
        rk.fit(X, y, group=groups)

    st.success("Model trained.")

    # Evaluate on the training data (diagnostic)
    df["_learn_score"] = rk.predict(X)
    metrics_df = _evaluate_hits_ndcg_per_day(df, score_col="_learn_score", label_col="hr_outcome", K=(10, 20, 30))
    st.write("Quick diagnostic (on training slates):")
    st.dataframe(metrics_df, use_container_width=True)

    # Save artifacts
    st.subheader("4) Download artifacts")
    # a) learning_ranker.pkl
    payload = {
        "model": rk,
        "features": chosen_feats,
        "notes": "Day-wise LambdaRank trained on merged leaderboards with labels from event parquet.",
    }
    pkl_bytes = io.BytesIO()
    pickle.dump(payload, pkl_bytes)
    pkl_bytes.seek(0)
    st.download_button(
        label="⬇️ Download learning_ranker.pkl",
        data=pkl_bytes,
        file_name="learning_ranker.pkl",
        mime="application/octet-stream",
    )

    # b) merged_leaderboards_labeled.csv
    out_df = df.drop(columns=["_learn_score"])
    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download merged_leaderboards_labeled.csv",
        data=csv_bytes,
        file_name="merged_leaderboards_labeled.csv",
        mime="text/csv",
    )

    st.caption("Tip: drop learning_ranker.pkl into your prediction app (it will auto-replace ranker_z when present).")

    gc.collect()
else:
    st.info("Upload your merged leaderboard CSV and event-level parquet to begin.")
