# App_learn_ranker.py
# ============================================================
# 📚 Learner — HR Day Ranker from Leaderboards + Event Parquet
# - Upload merged leaderboards CSV (multi-day)
# - Upload event-level parquet (with hr_outcome or infer from text cols)
# - Robust date parsing (handles '8_13' by stamping season year from events)
# - Aggressive name normalization (accents, suffixes, punctuation)
# - Two-pass join (normalized, then ASCII-folded) + diagnostics
# - Train LightGBM LambdaRank (day-wise groups)
# - Download labeled dataset and learning_ranker.pkl
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import re, gc, pickle, unicodedata
from io import BytesIO
from datetime import datetime

import lightgbm as lgb

# ===================== UI =====================
st.set_page_config(page_title="📚 Learner — HR Day Ranker", layout="wide")
st.title("📚 Learner — HR Day Ranker from Leaderboards + Event Parquet")

# ===================== Helpers =====================
@st.cache_data(show_spinner=False)
def safe_read_csv(file):
    fn = getattr(file, "name", "uploaded.csv").lower()
    try:
        return pd.read_csv(file, low_memory=False)
    except UnicodeDecodeError:
        file.seek(0)
        return pd.read_csv(file, encoding="latin1", low_memory=False)

@st.cache_data(show_spinner=False)
def safe_read_parquet(file):
    return pd.read_parquet(file)

def _strip_accents(s: str) -> str:
    if pd.isna(s): return ""
    return ''.join(ch for ch in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(ch))

def _norm_name(s: str, aggressive: bool = True) -> str:
    """Normalize player names consistently (strip, NFC→ASCII option, drop suffixes, unify spaces)."""
    if pd.isna(s):
        return ""
    s = str(s).strip()
    s = unicodedata.normalize("NFC", s)

    # common canonicalizations first
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

    if aggressive:
        # remove common suffixes
        s = re.sub(r"\b(JR\.?|SR\.?|II|III|IV)\b\.?", "", s, flags=re.IGNORECASE).strip()
        # collapse punctuation/extra spaces (keep letters, numbers, dot, hyphen)
        s = re.sub(r"[^\w\s\.-]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
    return s

def _norm_date_any(df, candidates, fallback=None, force_year: int | None = None):
    """
    Build a normalized date (YYYY-MM-DD).
    Accepts:
      - ISO (2025-08-13)
      - '8_13' (stamps force_year if provided)
    Returns pd.Series[datetime64[ns]] floored to day.
    """
    out = None
    for col in candidates:
        if col not in df.columns:
            continue
        v = df[col].astype(str).str.strip()

        # try direct parse
        dt = pd.to_datetime(v, errors="coerce")
        if dt.notna().any():
            out = dt
            break

        # try M_D → M/D then stamp year
        md = pd.to_datetime(v.str.replace("_", "/", regex=False), errors="coerce")
        if md.notna().any():
            if force_year is not None:
                md2 = pd.to_datetime(md.dt.strftime(f"{force_year}-%m-%d"), errors="coerce")
                out = md2
            else:
                out = md
            break

    if out is None:
        if fallback is not None:
            out = pd.to_datetime(fallback, errors="coerce")
        else:
            out = pd.to_datetime(pd.Series([], dtype="object"))

    return out.dt.floor("D") if len(out) else pd.Series([], dtype="datetime64[ns]")

def _ndcg_at_k(y_true_sorted, K: int) -> float:
    rels = np.asarray(y_true_sorted)[:K]
    if rels.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, 2 + len(rels)))
    dcg = float(np.sum(rels * discounts))
    ideal = np.sort(y_true_sorted)[::-1][:K]
    idcg = float(np.sum(ideal * discounts[: len(ideal)]))
    return (dcg / idcg) if idcg > 0 else 0.0

def _download_bytes(obj, filename: str) -> BytesIO:
    bio = BytesIO(obj)
    bio.seek(0)
    return bio

# ===================== Uploads =====================
lb_file = st.file_uploader("Merged leaderboard CSV (combined across days)", type=["csv"])
ev_file = st.file_uploader("Event-level Parquet (with hr_outcome, or text to infer)", type=["parquet"])

if lb_file is not None and ev_file is not None:
    with st.spinner("Loading files..."):
        merged = safe_read_csv(lb_file)
        events = safe_read_parquet(ev_file)

    # Basic cleansing
    merged = merged.dropna(axis=1, how="all")
    events = events.dropna(axis=1, how="all")

    # Parse events to determine season year
    ev = events.copy()
    ev["game_date_norm"] = _norm_date_any(ev, ["game_date", "date", "game_date_iso", "gameDate"])
    if ev["game_date_norm"].isna().all():
        st.error("Could not parse any usable date from events parquet. Expect a 'game_date'-like column.")
        st.stop()
    season_year = int(ev["game_date_norm"].dt.year.mode().iloc[0])

    # Parse leaderboard dates; stamp season_year for '8_13' style
    merged["game_date_norm"] = _norm_date_any(
        merged,
        candidates=["game_date_iso", "game_date"],
        force_year=season_year
    )
    if merged["game_date_norm"].isna().all():
        st.error("Could not parse dates from merged leaderboards. Ensure 'game_date_iso' or 'game_date' is present.")
        st.stop()

    # Normalize names (aggressive) in both
    name_col_lb = "player_name" if "player_name" in merged.columns else None
    if name_col_lb is None:
        st.error("Merged leaderboard must include 'player_name'.")
        st.stop()
    merged["player_name_norm"] = merged[name_col_lb].map(lambda x: _norm_name(x, aggressive=True))

    ev_name_candidates = ["player_name", "batter_name", "batter", "name"]
    ev_name_col = next((c for c in ev_name_candidates if c in ev.columns), None)
    if ev_name_col is None:
        st.error("Could not find a batter/player name column in events parquet (e.g., 'player_name' or 'batter_name').")
        st.stop()
    ev["player_name_norm"] = ev[ev_name_col].map(lambda x: _norm_name(x, aggressive=True))

    # Ensure we have labels in events
    label_col = "hr_outcome"
    if label_col not in ev.columns:
        # Try to infer from text columns if present
        text_cols = [c for c in ["events", "events_clean", "event", "play_desc"] if c in ev.columns]
        if text_cols:
            tmp = pd.Series(0, index=ev.index, dtype=np.int8)
            for c in text_cols:
                s = ev[c].astype(str).str.lower()
                tmp |= s.str.contains(r"home[_ ]?run|homered|\bhr\b", regex=True).astype(np.int8)
            ev[label_col] = tmp
        else:
            st.error("No 'hr_outcome' in events parquet and no textual event column to infer from.")
            st.stop()

    # Day-level labels (player-day)
    day_labels = (
        ev.groupby(["game_date_norm", "player_name_norm"], dropna=False)[label_col]
          .max()
          .reset_index()
          .rename(columns={label_col: "hr_outcome"})
    )

    # First join attempt
    key_cols = ["game_date_norm", "player_name_norm"]
    to_join = merged.merge(day_labels, on=key_cols, how="left", validate="m:1")
    labeled = to_join.dropna(subset=["hr_outcome"]).copy()

    # If no matches, try ASCII-folded names on both sides
    if labeled.empty:
        merged["player_name_norm_ascii"] = merged["player_name_norm"].map(_strip_accents).str.lower()
        day_labels["player_name_norm_ascii"] = day_labels["player_name_norm"].map(_strip_accents).str.lower()
        key2 = ["game_date_norm", "player_name_norm_ascii"]
        to_join2 = merged.merge(
            day_labels.drop(columns=["player_name_norm"]),
            on=key2, how="left", validate="m:1"
        )
        labeled = to_join2.dropna(subset=["hr_outcome"]).copy()

    overlap_names = len(set(merged.get("player_name_norm", [])) & set(day_labels.get("player_name_norm", [])))
    overlap_dates = len(set(merged["game_date_norm"]) & set(day_labels["game_date_norm"]))

    st.info(
        f"Labels joined. Rows with labels: {len(labeled)} (dropped {(len(merged) - len(labeled))} without event data). "
        f"Overlap (unique names): {overlap_names} • Overlap (unique dates): {overlap_dates}"
    )

    if labeled.empty:
        st.error("No label matches found after join.\n\n"
                 "Hints: • Make sure merged leaderboard has a real date (prefer 'game_date_iso' like 2025-08-13). "
                 "• Ensure names match (accents, suffixes). This app already applies common fixes.")
        # quick diagnostics
        st.write("Examples from leaderboard (first 10):")
        st.dataframe(merged[["game_date_norm","player_name"]].head(10))
        st.write("Examples from events (first 10):")
        st.dataframe(day_labels[["game_date_norm","player_name_norm"]].head(10))
        st.stop()

    # Remove junk cols commonly present
    for junk in ["Unnamed: 0", "source_file"]:
        if junk in labeled.columns:
            labeled = labeled.drop(columns=[junk])

    # ===================== Feature selection =====================
    st.subheader("2) Select features for the ranker (auto-selected for you)")

    # Build a friendly feature map (derived features + raw columns)
    feat_map = {}

    # Base prob (from leaderboard)
    if "hr_probability_iso_T" in labeled.columns:
        feat_map["base_prob"] = labeled["hr_probability_iso_T"].astype(float).to_numpy()
        # logit(base_prob)
        bp = np.clip(labeled["hr_probability_iso_T"].astype(float).to_numpy(), 1e-6, 1 - 1e-6)
        feat_map["logit_p"] = np.log(bp / (1 - bp))
    elif "calibrated_hr_probability" in labeled.columns:
        feat_map["base_prob"] = labeled["calibrated_hr_probability"].astype(float).to_numpy()
        bp = np.clip(feat_map["base_prob"], 1e-6, 1 - 1e-6)
        feat_map["logit_p"] = np.log(bp / (1 - bp))

    # Overlay-based features
    for c in ["final_multiplier", "overlay_multiplier", "weak_pitcher_factor", "hot_streak_factor",
              "final_multiplier_raw", "rrf_aux", "model_disagreement",
              "ranked_probability", "prob_2tb", "prob_rbi",
              "temp", "humidity", "wind_mph"]:
        if c in labeled.columns:
            # special: log_overlay
            if c == "final_multiplier":
                feat_map["log_overlay"] = np.log(np.maximum(1e-9, labeled[c].astype(float))).to_numpy()
            feat_map[c] = pd.to_numeric(labeled[c], errors="coerce").fillna(-1).astype(float).to_numpy()

    # Default feature list (keep it aligned with prediction app's feat_map keys)
    default_feats = [f for f in [
        "base_prob", "logit_p", "log_overlay",
        "overlay_multiplier", "final_multiplier",
        "ranked_probability", "rrf_aux", "model_disagreement",
        "prob_2tb", "prob_rbi",
        "temp", "humidity", "wind_mph"
    ] if f in feat_map]

    all_feat_options = list(feat_map.keys())
    feats = st.multiselect("Features", options=all_feat_options, default=default_feats)

    # Build X / y / groups
    if not feats:
        st.error("Please select at least one feature.")
        st.stop()

    X = np.column_stack([feat_map[f] for f in feats]).astype(np.float32)
    y = labeled["hr_outcome"].astype(int).to_numpy()

    # Day-wise grouping for LambdaRank
    if "game_date_norm" not in labeled.columns:
        st.error("Internal error: missing game_date_norm after merge.")
        st.stop()
    day_series = pd.to_datetime(labeled["game_date_norm"]).dt.floor("D")
    groups = day_series.groupby(day_series.values).size().values.tolist()

    # Sanity checks
    if X.ndim != 2 or X.shape[0] == 0:
        st.error("Input data must be 2D and non-empty (no rows after filtering).")
        st.stop()
    if len(y) != X.shape[0] or sum(groups) != X.shape[0]:
        st.error("Grouping mismatch. Ensure sum(groups) equals number of rows and y matches X.")
        st.stop()
    if y.sum() == 0:
        st.warning("All labels are 0. The ranker can still train, but quality will be limited.")

    # ===================== Train LambdaRank =====================
    st.subheader("3) Train day-wise LambdaRank model")
    with st.spinner("Training LGBMRanker..."):
        rk = lgb.LGBMRanker(
            objective="lambdarank", metric="ndcg",
            n_estimators=600, learning_rate=0.05, num_leaves=63,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            random_state=42
        )
        # Train on all data (no val to keep it simple and robust)
        rk.fit(X, y, group=groups)
    st.success("Model trained.")

    # Simple NDCG@30 on the training set (diagnostic only)
    scores = rk.predict(X)
    order = np.argsort(-scores)
    ndcg30 = _ndcg_at_k(y[order], K=30)
    st.write(f"Training NDCG@30 (diagnostic): **{ndcg30:.4f}**")

    # ===================== Downloads =====================
    st.subheader("4) Downloads")
    # Labeled dataset (what actually matched/joined)
    labeled_out = labeled.copy()
    # keep it small-ish: round some floats
    for col in ["ranked_probability", "hr_probability_iso_T", "final_multiplier",
                "overlay_multiplier", "weak_pitcher_factor", "hot_streak_factor",
                "final_multiplier_raw", "rrf_aux", "model_disagreement",
                "prob_2tb", "prob_rbi", "temp", "humidity", "wind_mph"]:
        if col in labeled_out.columns:
            labeled_out[col] = pd.to_numeric(labeled_out[col], errors="coerce").round(6)

    csv_bytes = labeled_out.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download labeled_merged_leaderboards.csv",
        data=_download_bytes(csv_bytes, "labeled_merged_leaderboards.csv"),
        file_name="labeled_merged_leaderboards.csv",
        mime="text/csv",
    )

    # Model bundle for prediction app
    bundle = {
        "model": rk,
        "features": feats,  # your prediction app will build this subset if available
        "trained_at": datetime.utcnow().isoformat() + "Z",
    }
    pkl_bytes = BytesIO()
    pickle.dump(bundle, pkl_bytes)
    pkl_bytes.seek(0)
    st.download_button(
        "⬇️ Download learning_ranker.pkl",
        data=pkl_bytes,
        file_name="learning_ranker.pkl",
        mime="application/octet-stream",
    )

    st.caption("This bundle includes the LightGBM ranker and the exact feature list you trained with.")

    gc.collect()
