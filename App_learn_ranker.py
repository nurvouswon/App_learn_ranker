import streamlit as st
import pandas as pd
import numpy as np
import io, pickle
from datetime import datetime, date

# ML
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ndcg_score
import lightgbm as lgb
from scipy.special import logit

from unidecode import unidecode

st.set_page_config(page_title="📚 Learner — HR Day Ranker from Leaderboards + Event Parquet", layout="wide")
st.title("📚 Learner — HR Day Ranker from Leaderboards + Event Parquet")

st.caption("Upload your merged leaderboard CSV and the event-level Parquet (with `hr_outcome`). "
           "This will label each leaderboard row with the true next-day HR outcome and train the learning ranker "
           "(the small LR model your prediction app can use).")

# ---------------------------
# Helpers
# ---------------------------
def clean_name(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = unidecode(s)  # remove accents
    s = s.replace(".", " ")
    s = s.replace("-", " ")
    s = s.replace("  ", " ")
    s = s.lower()
    # common suffixes / noise
    for bad in [" jr", " sr", " ii", " iii", " iv"]:
        if s.endswith(bad):
            s = s[: -len(bad)]
    s = " ".join(s.split())
    return s

TEAM_FIX = {
    # add any known aliases here
    "laa": "LAA", "ari": "ARI", "phi": "PHI", "nyy": "NYY", "ny m": "NYM", "sd": "SDP", "sdg": "SDP",
}
def std_team(s: str) -> str:
    if s is None: return ""
    s2 = str(s).strip().upper()
    return TEAM_FIX.get(s2, s2)

def to_date_ymd(s, fallback_year=None):
    """
    Accepts robust formats, e.g. '2025-08-13', '8_13', '8/13', etc.
    If month/day only + fallback_year provided, attach that year.
    """
    if pd.isna(s):
        return pd.NaT
    s = str(s).strip()
    # already ISO date?
    try:
        dt = pd.to_datetime(s, errors="raise", utc=False, format="%Y-%m-%d")
        return dt
    except Exception:
        pass
    # try generic parse
    try:
        dt = pd.to_datetime(s, errors="raise", utc=False)
        return dt
    except Exception:
        pass
    # try  M_D with underscore
    if "_" in s:
        parts = s.split("_")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit() and fallback_year is not None:
            mm, dd = int(parts[0]), int(parts[1])
            try:
                return pd.to_datetime(f"{fallback_year:04d}-{mm:02d}-{dd:02d}")
            except Exception:
                return pd.NaT
    return pd.NaT

def zscore(a):
    a = np.asarray(a, dtype=np.float64)
    mu = np.nanmean(a); sd = np.nanstd(a) + 1e-9
    return (a - mu) / sd

st.markdown("### 1) Upload files")

lb_file = st.file_uploader("Merged leaderboard CSV (combined across days)", type=["csv"])
ev_file = st.file_uploader("Event-level Parquet (with hr_outcome)", type=["parquet"])

season_year = st.number_input("Season year (used only if your leaderboard 'game_date' is M_D like '8_13')",
                              min_value=2015, max_value=2100, value=date.today().year, step=1)

if lb_file and ev_file:
    # ---------------------------
    # Read files
    # ---------------------------
    try:
        lb = pd.read_csv(lb_file)
    except Exception:
        lb = pd.read_csv(lb_file, encoding="latin1")

    ev = pd.read_parquet(ev_file)

    st.write(f"Leaderboard rows: {len(lb):,} | Event rows: {len(ev):,}")

    # ---------------------------
    # Standardize Leaderboard
    # ---------------------------
    if "game_date" not in lb.columns or "player_name" not in lb.columns:
        st.error("Leaderboard must contain at least 'game_date' and 'player_name'.")
        st.stop()

    lb = lb.copy()
    # parse dates to real YYYY-MM-DD
    lb["game_date"] = lb["game_date"].apply(lambda s: to_date_ymd(s, season_year)).dt.date
    lb["player_name_norm"] = lb["player_name"].astype(str).apply(clean_name)

    # SAFE default for team_code (fixes your earlier crash)
    lb["team_code_std"] = lb.get("team_code", pd.Series([""] * len(lb))).astype(str).apply(std_team)

    # optional IDs passthrough (if present)
    if "batter_id" in lb.columns:
        lb["batter_id_join"] = lb["batter_id"]
    elif "batter" in lb.columns:
        lb["batter_id_join"] = lb["batter"]
    else:
        lb["batter_id_join"] = pd.Series([pd.NA] * len(lb))

    # ---------------------------
    # Build daily labels from event parquet
    # ---------------------------
    if "game_date" not in ev.columns:
        st.error("Event parquet must contain 'game_date'.")
        st.stop()
    if "player_name" not in ev.columns:
        st.error("Event parquet must contain 'player_name'.")
        st.stop()
    if "hr_outcome" not in ev.columns:
        st.error("Event parquet must contain 'hr_outcome' (0/1).")
        st.stop()

    ev = ev.copy()
    ev["game_date"] = pd.to_datetime(ev["game_date"], errors="coerce").dt.date
    ev["player_name_norm"] = ev["player_name"].astype(str).apply(clean_name)

    if "team_code" in ev.columns:
        ev["team_code_std"] = ev["team_code"].astype(str).apply(std_team)
    else:
        ev["team_code_std"] = pd.Series([""] * len(ev))

    # keep useful identifiers
    keep_cols = ["game_date", "player_name_norm", "team_code_std", "hr_outcome"]
    if "batter" in ev.columns:
        ev["batter_id_join"] = ev["batter"]
        keep_cols.append("batter_id_join")
    elif "batter_id" in ev.columns:
        ev["batter_id_join"] = ev["batter_id"]
        keep_cols.append("batter_id_join")

    ev_small = ev[keep_cols].copy()

    # aggregate per player x date — any HR that day = 1
    agg = {
        "hr_outcome": "max",
        "team_code_std": "first",
    }
    if "batter_id_join" in ev_small.columns:
        agg["batter_id_join"] = "first"

    ev_daily = ev_small.groupby(["game_date", "player_name_norm"], as_index=False).agg(agg)

    # ---------------------------
    # Join strategy (robust)
    # ---------------------------
    st.markdown("### 2) Label join diagnostics")

    merged = None

    # Prefer ID + date if available on both sides with at least some overlap
    can_id_join = ("batter_id_join" in lb.columns) and ("batter_id_join" in ev_daily.columns)
    if can_id_join:
        tmp = lb.merge(
            ev_daily[["game_date", "player_name_norm", "batter_id_join", "hr_outcome"]],
            on=["game_date", "player_name_norm", "batter_id_join"],
            how="left",
            suffixes=("", "_y"),
        )
        matches = tmp["hr_outcome"].notna().sum()
        st.write(f"Join using ['game_date','player_name_norm','batter_id_join'] → matches: {matches}")
        if matches > 0:
            merged = tmp

    if merged is None:
        # Next best: date + (normalized name + team)
        tmp = lb.merge(
            ev_daily[["game_date", "player_name_norm", "team_code_std", "hr_outcome"]],
            on=["game_date", "player_name_norm", "team_code_std"],
            how="left",
            suffixes=("", "_y"),
        )
        matches = tmp["hr_outcome"].notna().sum()
        st.write(f"Join using ['game_date','player_name_norm','team_code_std'] → matches: {matches}")
        if matches > 0:
            merged = tmp

    if merged is None:
        # Fallback: date + normalized name only
        tmp = lb.merge(
            ev_daily[["game_date", "player_name_norm", "hr_outcome"]],
            on=["game_date", "player_name_norm"],
            how="left",
            suffixes=("", "_y"),
        )
        matches = tmp["hr_outcome"].notna().sum()
        st.write(f"Join using ['game_date','player_name_norm'] → matches: {matches}")
        if matches > 0:
            merged = tmp

    if merged is None:
        st.error("❌ No label matches found after join.\n\n"
                 "Hints: ensure the leaderboard has 'game_date' in YYYY-MM-DD and names align; "
                 "accents and common suffixes are normalized automatically. Also confirm "
                 "your event parquet actually covers the same dates/players.")
        st.stop()

    # Drop rows without labels
    before = len(merged)
    merged = merged[merged["hr_outcome"].notna()].copy()
    merged["hr_outcome"] = merged["hr_outcome"].astype(int)
    after = len(merged)
    st.success(f"Labels joined. Rows with labels: {after} (dropped {before - after} without event data).")

    # keep a clean export (preserve your useful cols like prob_2tb / prob_rbi)
    export_cols = merged.columns.tolist()
    # move some to the front for readability
    front = ["game_date", "player_name", "team_code", "ranked_probability",
             "hr_probability_iso_T", "final_multiplier", "overlay_multiplier",
             "prob_2tb", "prob_rbi", "hr_outcome"]
    ordered = [c for c in front if c in export_cols] + [c for c in export_cols if c not in front]
    merged_export = merged[ordered].copy()

    st.download_button(
        "⬇️ Download merged_labeled_leaderboards.csv",
        data=merged_export.to_csv(index=False),
        file_name="merged_labeled_leaderboards.csv",
        mime="text/csv",
    )

    # ---------------------------
    # 3) Build features for the learning ranker (compat with prediction app)
    # ---------------------------
    st.markdown("### 3) Train the learning ranker (compatible with prediction app)")

    # required leaderboard fields
    req = ["ranked_probability", "hr_probability_iso_T", "final_multiplier", "overlay_multiplier"]
    missing = [c for c in req if c not in merged.columns]
    if missing:
        st.error("Your merged leaderboard is missing required columns: " + ", ".join(missing))
        st.stop()

    Xrk = pd.DataFrame({
        "base_prob": merged["hr_probability_iso_T"].astype(float).clip(1e-6, 1-1e-6),
        "logit_p": logit(merged["hr_probability_iso_T"].astype(float).clip(1e-6, 1-1e-6)),
        "log_overlay": np.log(merged["final_multiplier"].astype(float) + 1e-9),
        "ranker_z": zscore(merged["ranked_probability"].astype(float)),
        "overlay_multiplier": merged["overlay_multiplier"].astype(float),
        "final_multiplier": merged["final_multiplier"].astype(float),
    })

    y = merged["hr_outcome"].astype(int).values

    # train/test split by date (last day as small validation, purely for a quick check)
    all_days = pd.to_datetime(merged["game_date"]).dt.date
    unique_days = sorted(list(set(all_days)))
    if len(unique_days) >= 2:
        cutoff_day = unique_days[-1]
        tr_idx = np.where(all_days < cutoff_day)[0]
        va_idx = np.where(all_days == cutoff_day)[0]
    else:
        tr_idx = np.arange(len(merged))
        va_idx = np.array([], dtype=int)

    X_tr, y_tr = Xrk.iloc[tr_idx], y[tr_idx]
    X_va, y_va = Xrk.iloc[va_idx], y[va_idx] if len(va_idx) else (pd.DataFrame(columns=Xrk.columns), np.array([], dtype=int))

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr.values.astype(np.float32))
    if len(va_idx):
        X_va_s = scaler.transform(X_va.values.astype(np.float32))

    # Logistic regression (this is the exact model your prediction app expects to load and use)
    lr = LogisticRegression(max_iter=1000, solver="lbfgs")
    lr.fit(X_tr_s, y_tr)

    # quick sanity metrics (Hits@K via ndcg proxy on validation day)
    if len(va_idx):
        p_va = lr.predict_proba(X_va_s)[:, 1]
        # treat as ranking within the day
        try:
            # group by day for NDCG@K
            day_mask = (pd.to_datetime(merged["game_date"]).dt.date.values == cutoff_day)
            nd = ndcg_score(y_true=y_va.reshape(1, -1), y_score=p_va.reshape(1, -1), k=min(30, len(p_va)))
            st.info(f"Validation day {cutoff_day}: NDCG@30 ≈ {nd:.3f}")
        except Exception:
            pass

    # save bundle (exact keys your prediction app expects)
    bundle = {
        "model": lr,
        "features": ["base_prob", "logit_p", "log_overlay", "ranker_z", "overlay_multiplier", "final_multiplier"],
        "scaler": scaler,
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

    # ---------------------------
    # Optional: Day-wise LGBMRanker (diagnostic only)
    # ---------------------------
    with st.expander("🔬 (Optional) Train a day-wise LambdaRank model for diagnostics"):
        # Build ranking frame by day
        df_rank = merged.copy()
        df_rank["_qid"] = pd.to_datetime(df_rank["game_date"]).dt.date
        q_sizes = df_rank.groupby("_qid").size().values.tolist()

        # simple feature set from leaderboard for ranker diagnostics
        rank_feats = [
            "ranked_probability", "hr_probability_iso_T",
            "final_multiplier", "overlay_multiplier",
        ]
        rank_feats = [f for f in rank_feats if f in df_rank.columns]
        X_rank = df_rank[rank_feats].astype(float).fillna(0.0)
        y_rank = df_rank["hr_outcome"].astype(int).values

        rk = lgb.LGBMRanker(
            objective="lambdarank", metric="ndcg",
            n_estimators=600, learning_rate=0.05, num_leaves=63,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            random_state=42
        )
        rk.fit(X_rank, y_rank, group=q_sizes)
        st.write("Trained LGBMRanker on full labeled merged leaderboard (diagnostic).")

        # Attach diagnostic rank score to a preview
        df_out = df_rank.copy()
        df_out["rank_diag"] = rk.predict(X_rank)
        show_cols = ["game_date", "player_name", "team_code", "ranked_probability", "hr_probability_iso_T",
                     "final_multiplier", "overlay_multiplier", "prob_2tb", "prob_rbi",
                     "hr_outcome", "rank_diag"]
        show_cols = [c for c in show_cols if c in df_out.columns]
        st.dataframe(df_out[show_cols].sort_values(["game_date", "rank_diag"], ascending=[True, False]).head(50), use_container_width=True)

else:
    st.info("Upload both files to proceed.")
