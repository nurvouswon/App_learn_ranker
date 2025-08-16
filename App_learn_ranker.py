# ============================================================
# 📚 Learner — HR Day Ranker from Leaderboards + Event Parquet
# - Upload merged leaderboard CSV + event-level parquet (with hr_outcome)
# - Robust, prediction-app-style join (date + name + id/team fallback)
# - Build training set with the SAME feature names the pred app expects:
#     base_prob, logit_p, log_overlay, ranker_z, overlay_multiplier, final_multiplier
# - Train a compact meta model (scikit-learn Pipeline) and save:
#     learning_ranker.pkl  (core 6 features)
# - If 2TB/RBI probs exist, also save:
#     learning_ranker_plus.pkl (core 6 + prob_2tb + prob_rbi)
# - Download labeled CSV and model artifacts
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
from datetime import datetime
from unidecode import unidecode

# ML
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from scipy.special import logit

st.set_page_config(page_title="📚 Learner — HR Day Ranker", layout="wide")
st.title("📚 Learner — HR Day Ranker from Leaderboards + Event Parquet")
st.caption("Upload your merged leaderboard CSV and the event-level Parquet (with hr_outcome). This will label each leaderboard row with the true next-day HR outcome and train the learning ranker your prediction app can use.")

# ---------------- Utilities (match prediction app behavior) ----------------
TEAM_MAP = {
    "LAA":"LAA","ANA":"LAA",
    "ARI":"ARI","ARZ":"ARI",
    "ATL":"ATL","BRV":"ATL",
    "BAL":"BAL","BOS":"BOS",
    "CHC":"CHC","CHN":"CHC",
    "CWS":"CWS","CHW":"CWS",
    "CIN":"CIN","CLE":"CLE","CLV":"CLE",
    "COL":"COL","DET":"DET",
    "HOU":"HOU","KCR":"KC","KCE":"KC","KC":"KC",
    "LAD":"LAD","LAN":"LAD",
    "MIA":"MIA","FLA":"MIA",
    "MIL":"MIL","MIN":"MIN",
    "NYM":"NYM","NYN":"NYM",
    "NYY":"NYY","NYA":"NYY",
    "OAK":"OAK","PHI":"PHI","PHL":"PHI",
    "PIT":"PIT","SD":"SD","SDP":"SD",
    "SEA":"SEA","SF":"SF","SFG":"SF",
    "STL":"STL","SLN":"STL","TB":"TB","TBR":"TB",
    "TEX":"TEX","TOR":"TOR","WAS":"WSH","WSH":"WSH","WSN":"WSH"
}

NAME_FIXES = {
    r"^Peter Alonso$":"Pete Alonso",
    r"^CJ Kayfus$":"C.J. Kayfus",
    r"^C\. J\. Kayfus$":"C.J. Kayfus",
    r"^Julio Rodriguez$":"Julio Rodríguez",
    r"^Michael A Taylor$":"Michael A. Taylor",
    r"^Jeremy Pena$":"Jeremy Peña",
}

def clean_name(s: str) -> str:
    if s is None: return ""
    s = str(s).strip()
    for pat, rep in NAME_FIXES.items():
        if re.fullmatch(pat, s):
            s = rep
            break
    s = unidecode(s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\b(JR|SR|II|III|IV)\.?$", "", s, flags=re.IGNORECASE).strip()
    return s.upper()

def std_team(s: str) -> str:
    if s is None: return ""
    s = str(s).strip().upper()
    return TEAM_MAP.get(s, s)

def to_date_ymd(val, season_year: int) -> pd.Timestamp:
    if pd.isna(val): return pd.NaT
    s = str(val).strip()
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        return pd.to_datetime(s, errors="coerce")
    m2 = re.fullmatch(r"(\d{1,2})_(\d{1,2})", s)
    if m2:
        m_, d_ = int(m2.group(1)), int(m2.group(2))
        return pd.to_datetime(f"{season_year:04d}-{m_:02d}-{d_:02d}", errors="coerce")
    return pd.to_datetime(s, errors="coerce")

def coerce_id_str_series(s: pd.Series) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce").astype("Int64").astype(str)
    return out.replace("<NA>", "", regex=False)

def zscore(a):
    a = np.asarray(a, dtype=np.float64)
    mu = np.nanmean(a); sd = np.nanstd(a) + 1e-9
    return (a - mu) / sd

# ---------------- 1) Inputs ----------------
lb_file = st.file_uploader("Merged leaderboard CSV (combined across days)", type=["csv"], key="lb")
ev_file = st.file_uploader("Event-level Parquet (with hr_outcome)", type=["parquet"], key="ev")
season_year = st.number_input("Season year (used only if your leaderboard 'game_date' is M_D like '8_13')", min_value=2015, max_value=2100, value=2025, step=1)

if lb_file is not None and ev_file is not None:
    # Load
    lb = pd.read_csv(lb_file)
    ev = pd.read_parquet(ev_file)

    st.write(f"Leaderboard rows: {len(lb):,} | Event rows: {len(ev):,}")

    # ---------------- Clean leaderboard (match pred app) ----------------
    if "game_date" not in lb.columns or "player_name" not in lb.columns:
        st.error("Leaderboard must include at least 'game_date' and 'player_name' columns.")
        st.stop()

    lb = lb.copy()
    lb["game_date"] = lb["game_date"].apply(lambda s: to_date_ymd(s, int(season_year)))
    lb = lb[lb["game_date"].notna()].reset_index(drop=True)

    lb["player_name_norm"] = lb["player_name"].astype(str).apply(clean_name)
    if "team_code" in lb.columns:
        lb["team_code_std"] = lb["team_code"].astype(str).apply(std_team)
    else:
        lb["team_code_std"] = ""

    # optional id on LB
    if "batter_id" in lb.columns:
        lb["batter_id_join"] = coerce_id_str_series(lb["batter_id"])
    elif "batter_id_x" in lb.columns:
        lb["batter_id_join"] = coerce_id_str_series(lb["batter_id_x"])
    else:
        lb["batter_id_join"] = ""

    # ---------------- Prepare event daily HR outcomes ----------------
    if "hr_outcome" not in ev.columns:
        st.error("Event parquet must contain 'hr_outcome'.")
        st.stop()

    ev = ev.copy()
    ev["game_date"] = pd.to_datetime(ev["game_date"], errors="coerce")
    ev = ev[ev["game_date"].notna()]
    ev["player_name_norm"] = ev["player_name"].astype(str).apply(clean_name)
    if "team_code" in ev.columns:
        ev["team_code_std"] = ev["team_code"].astype(str).apply(std_team)
    else:
        ev["team_code_std"] = ""
    if "batter_id" in ev.columns:
        ev["batter_id_join"] = coerce_id_str_series(ev["batter_id"])
    else:
        ev["batter_id_join"] = ""

    ev_daily = (
        ev.groupby(["game_date","player_name_norm","team_code_std","batter_id_join"], dropna=False)["hr_outcome"]
          .max()
          .reset_index()
    )

    st.markdown("### 2) Label join diagnostics (prediction-app style)")
    merged = None

    def try_join(cols):
        cols_right = cols + ["hr_outcome"]
        tmp = lb.merge(ev_daily[cols_right], on=cols, how="left", suffixes=("","_y"))
        hits = int(tmp["hr_outcome"].notna().sum())
        st.write(f"Join on {cols} → matches: {hits}")
        return tmp, hits

    has_lb_ids = lb["batter_id_join"].astype(str).str.len().gt(0).any()
    has_ev_ids = ev_daily["batter_id_join"].astype(str).str.len().gt(0).any()

    # (1) date + name + id
    if has_lb_ids and has_ev_ids:
        tmp, hits = try_join(["game_date","player_name_norm","batter_id_join"])
        if hits > 0:
            merged = tmp

    # (2) date + name + team
    if merged is None:
        tmp, hits = try_join(["game_date","player_name_norm","team_code_std"])
        if hits > 0:
            merged = tmp

    # (3) date + name
    if merged is None:
        tmp, hits = try_join(["game_date","player_name_norm"])
        if hits > 0:
            merged = tmp

    if merged is None or merged["hr_outcome"].isna().all():
        st.error("❌ No label matches found after join.\n\n"
                 "Quick fixes you can do in the CSV:\n"
                 "• Ensure 'game_date' is YYYY-MM-DD (you did).\n"
                 "• Keep 'player_name' as exported from your prediction app (this code normalizes accents/suffixes).\n"
                 "• If possible, include 'batter_id' in the leaderboard export for near-perfect matching.\n"
                 "• Confirm your event parquet actually covers the same dates.")
        st.stop()

    lb_labeled = merged.copy()
    labeled_rows = int(lb_labeled["hr_outcome"].notna().sum())
    st.success(f"Labels joined. Rows with labels: {labeled_rows} (dropped {len(lb_labeled)-labeled_rows} without matches after join).")
    lb_labeled = lb_labeled[lb_labeled["hr_outcome"].notna()].copy()
    lb_labeled["hr_outcome"] = lb_labeled["hr_outcome"].astype(int)

    # ---------------- Build training features (names that pred app expects) ----------------
    must_cols = ["ranked_probability", "hr_probability_iso_T", "overlay_multiplier", "final_multiplier"]
    missing = [c for c in must_cols if c not in lb_labeled.columns]
    if missing:
        st.error(f"Your leaderboard is missing required columns: {missing}")
        st.stop()

    # Core features
    base_prob = lb_labeled["hr_probability_iso_T"].astype(float).clip(1e-6, 1-1e-6).values
    X_core = pd.DataFrame({
        "base_prob": base_prob,
        "logit_p": logit(base_prob),
        "log_overlay": np.log(lb_labeled["final_multiplier"].astype(float).clip(lower=1e-9)).values,
        "ranker_z": zscore(lb_labeled["ranked_probability"].astype(float).values),
        "overlay_multiplier": lb_labeled["overlay_multiplier"].astype(float).values,
        "final_multiplier": lb_labeled["final_multiplier"].astype(float).values,
    })

    y = lb_labeled["hr_outcome"].astype(int).values

    # Optional extras (2TB/RBI). These stay in the labeled CSV and
    # are included in a separate "plus" artifact so your pred app
    # can keep using the core 6-feature model unchanged.
    X_plus = X_core.copy()
    plus_feats = []
    if "prob_2tb" in lb_labeled.columns:
        X_plus["prob_2tb"] = pd.to_numeric(lb_labeled["prob_2tb"], errors="coerce").fillna(0.0).astype(float).values
        plus_feats.append("prob_2tb")
    if "prob_rbi" in lb_labeled.columns:
        X_plus["prob_rbi"] = pd.to_numeric(lb_labeled["prob_rbi"], errors="coerce").fillna(0.0).astype(float).values
        plus_feats.append("prob_rbi")

    # ---------------- Train the learner (compact, fast, stable) ----------------
    st.markdown("### 3) Train learning ranker (meta)")
    core_feats = list(X_core.columns)
    core_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, solver="lbfgs"))
    ])
    core_pipe.fit(X_core, y)
    core_auc = roc_auc_score(y, core_pipe.predict_proba(X_core)[:,1])
    st.success(f"Core learner trained (features={core_feats}). OOF AUC (on joined data): {core_auc:.4f}")

    # Save core artifact (this is the one your prediction app expects)
    core_bundle = {"model": core_pipe, "features": core_feats}
    core_bytes = pickle.dumps(core_bundle)
    st.download_button("⬇️ Download learning_ranker.pkl (core)",
                       data=core_bytes,
                       file_name="learning_ranker.pkl",
                       mime="application/octet-stream")

    # If extras exist, also train a plus model and save separately
    if plus_feats:
        plus_feat_list = list(X_plus.columns)
        plus_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, solver="lbfgs"))
        ])
        plus_pipe.fit(X_plus, y)
        plus_auc = roc_auc_score(y, plus_pipe.predict_proba(X_plus)[:,1])
        st.info(f"Plus learner trained (features={plus_feat_list}). OOF AUC: {plus_auc:.4f}")
        plus_bundle = {"model": plus_pipe, "features": plus_feat_list}
        plus_bytes = pickle.dumps(plus_bundle)
        st.download_button("⬇️ Download learning_ranker_plus.pkl (core + 2TB/RBI)",
                           data=plus_bytes,
                           file_name="learning_ranker_plus.pkl",
                           mime="application/octet-stream")

    # ---------------- Outputs ----------------
    st.markdown("### 4) Labeled rows preview")
    preview_cols = [
        "game_date","player_name","team_code","ranked_probability","hr_probability_iso_T",
        "overlay_multiplier","weak_pitcher_factor","hot_streak_factor",
        "final_multiplier_raw","final_multiplier","prob_2tb","prob_rbi","rrf_aux",
        "model_disagreement","hr_outcome"
    ]
    preview_cols = [c for c in preview_cols if c in lb_labeled.columns]
    st.dataframe(lb_labeled[preview_cols].head(50), use_container_width=True)

    # Labeled CSV download
    labeled_csv = lb_labeled.to_csv(index=False)
    st.download_button("⬇️ Download labeled_leaderboards.csv",
                       data=labeled_csv,
                       file_name="labeled_leaderboards.csv",
                       mime="text/csv")

    st.success("✅ Done. Load learning_ranker.pkl in your prediction app (it already knows how).")
