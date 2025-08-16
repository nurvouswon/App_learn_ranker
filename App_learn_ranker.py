# app_learn_ranker.py
# ============================================================
# 📚 Learner — HR Day Ranker from Leaderboards + Event Parquet
# - Upload merged leaderboards CSV (multi-day)
# - Upload event-level parquet with hr_outcome
# - Robust join (ID+date preferred; fallback name+date)
# - Season-year helper for "8_13"-style dates
# - Keeps 2TB/RBI features (prob_2tb, prob_rbi) if present
# - Trains day-wise LambdaRank (LightGBM)
# - Exports learning_ranker.pkl (model + features)
# - Exports labeled merged leaderboard CSV
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import io, re, json, pickle
from datetime import datetime
from typing import List, Optional
import unicodedata

import lightgbm as lgb

# ---------- UI ----------
st.set_page_config(page_title="📚 Learner — HR Day Ranker", layout="wide")
st.title("📚 Learner — HR Day Ranker from Leaderboards + Event Parquet")

# ---------- Helpers ----------
def choose_first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def strip_bom_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    return df

def common_name_fixes(s: pd.Series) -> pd.Series:
    # Basic canonicalization (trim, fix common variants)
    rep = {
        r"^Peter Alonso$": "Pete Alonso",
        r"^Jeremy Pena$": "Jeremy Peña",
        r"^Julio Rodriguez$": "Julio Rodríguez",
        r"^Michael A Taylor$": "Michael A. Taylor",
        r"^CJ Kayfus$": "C.J. Kayfus",
        r"^C\. J\. Kayfus$": "C.J. Kayfus",
        r"\s+Jr\.?$": " Jr.",
        r"\s+II$": " II",
        r"\s+III$": " III",
    }
    out = s.astype(str).str.strip()
    for pat, repl in rep.items():
        out = out.str.replace(pat, repl, regex=True)
    return out

def clean_names(s: pd.Series) -> pd.Series:
    # Lowercase, strip, normalize accents, drop punctuation
    def _clean(x: str) -> str:
        x = (x or "").strip().lower()
        x = unicodedata.normalize("NFKD", x)
        x = "".join(ch for ch in x if not unicodedata.combining(ch))
        x = re.sub(r"[^a-z0-9\s\.]", " ", x)
        x = re.sub(r"\s+", " ", x).strip()
        return x
    return s.astype(str).map(_clean)

def parse_date_safely(df: pd.DataFrame, date_cols: List[str], fallback_year: Optional[int]) -> pd.Series:
    for c in date_cols:
        if c not in df.columns:
            continue
        # try direct parse
        d = pd.to_datetime(df[c], errors="coerce", utc=False)
        if d.notna().any():
            return pd.to_datetime(d).dt.tz_localize(None).dt.floor("D")
        raw = df[c].astype(str)

        def _mk_iso(x: str):
            x = x.strip()
            # match like 8_13, 08-13, 8-13
            if re.fullmatch(r"\d{1,2}[_\-]\d{1,2}", x) and fallback_year is not None:
                m, d = re.split(r"[_\-]", x)
                try:
                    return f"{int(fallback_year):04d}-{int(m):02d}-{int(d):02d}"
                except Exception:
                    return None
            return x

        cand = raw.map(_mk_iso)
        d2 = pd.to_datetime(cand, errors="coerce", utc=False)
        if d2.notna().any():
            return pd.to_datetime(d2).dt.tz_localize(None).dt.floor("D")
    return pd.Series(pd.NaT, index=df.index)

def groups_from_days(d: pd.Series) -> List[int]:
    # expects a date series
    g = d.groupby(d.values).size().values.tolist()
    if not isinstance(g, list):
        g = list(g)
    return g

def to_float32_df(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
    return df

# ---------- File inputs ----------
st.subheader("1) Upload files")
lb_file = st.file_uploader("Merged leaderboard CSV (combined across days)", type=["csv"], key="lb")
ev_file = st.file_uploader("Event-level Parquet (with hr_outcome)", type=["parquet"], key="ev")

if lb_file is None or ev_file is None:
    st.info("Upload both files to continue.")
    st.stop()

with st.spinner("Reading files..."):
    lb = pd.read_csv(lb_file)
    ev = pd.read_parquet(ev_file)

lb = strip_bom_cols(lb)
ev = strip_bom_cols(ev)

# ---------- Season year helper ----------
st.subheader("1½) Season year (for leaderboards like '8_13')")
_ev_date_col_guess = [c for c in ["game_date", "date", "game_datetime", "start_time"] if c in ev.columns]
_ev_dates_try = pd.to_datetime(ev[_ev_date_col_guess[0]], errors="coerce") if _ev_date_col_guess else pd.Series(pd.NaT, index=ev.index)
_inferred_year = int(_ev_dates_try.dt.year.mode().iloc[0]) if _ev_dates_try.notna().any() else None
season_year = st.number_input(
    "Season year (used when leaderboard has '8_13' style dates)",
    min_value=2000, max_value=2100,
    value=_inferred_year if _inferred_year else datetime.utcnow().year,
    step=1
)

# ---------- Name & ID columns ----------
lb_name_raw_col = choose_first_col(lb, ["player_name", "batter_name", "batter", "name"])
ev_name_raw_col = choose_first_col(ev, ["player_name", "batter_name", "batter", "name"])

if lb_name_raw_col:
    lb[lb_name_raw_col] = common_name_fixes(lb[lb_name_raw_col])
if ev_name_raw_col:
    ev[ev_name_raw_col] = common_name_fixes(ev[ev_name_raw_col])

id_candidates = ["batter_id", "player_id", "mlbam_id", "bat_id", "batterid", "mlb_id"]
lb_id_col = choose_first_col(lb, id_candidates)
ev_id_col = choose_first_col(ev, id_candidates)

# ---------- Normalized join keys ----------
if lb_name_raw_col:
    lb["_join_name"] = clean_names(lb[lb_name_raw_col])
else:
    lb["_join_name"] = ""

if ev_name_raw_col:
    ev["_join_name"] = clean_names(ev[ev_name_raw_col])
else:
    ev["_join_name"] = ""

lb["_join_date"] = parse_date_safely(lb, ["game_date_iso", "game_date", "date"], fallback_year=int(season_year))
ev["_join_date"] = parse_date_safely(ev, ["game_date", "date", "game_datetime", "start_time"], fallback_year=None)

st.write("🔎 Join diagnostics (before merge)")
st.write({
    "lb_unique_names": int(lb["_join_name"].nunique()),
    "ev_unique_names": int(ev["_join_name"].nunique()),
    "lb_date_nonnull": int(lb["_join_date"].notna().sum()),
    "ev_date_nonnull": int(ev["_join_date"].notna().sum()),
    "lb_has_id": bool(lb_id_col),
    "ev_has_id": bool(ev_id_col),
})

label_col_ev = choose_first_col(ev, ["hr_outcome", "hr", "is_hr"])
if label_col_ev is None:
    st.error("Event parquet has no 'hr_outcome' (or equivalent) column.")
    st.stop()

# Build labels view
ev_labels_cols = ["_join_date", label_col_ev]
if ev_id_col: ev_labels_cols.append(ev_id_col)
if ev_name_raw_col: ev_labels_cols.append("_join_name")

ev_labels = ev[ev_labels_cols].copy()
ev_labels = ev_labels.dropna(subset=["_join_date"]).copy()
ev_labels = ev_labels.rename(columns={label_col_ev: "hr_outcome"})

# ---------- Try ID+date join first ----------
lb_lab = pd.DataFrame()
join_used = "none"
if lb_id_col and ev_id_col:
    _ev_id_date = ev_labels.dropna(subset=[ev_id_col]).copy()
    _lb_id_date = lb.dropna(subset=[lb_id_col]).copy()
    if not _ev_id_date.empty and not _lb_id_date.empty:
        lb_lab = _lb_id_date.merge(
            _ev_id_date[["_join_date", ev_id_col, "hr_outcome"]],
            left_on=["_join_date", lb_id_col],
            right_on=["_join_date", ev_id_col],
            how="inner"
        )
        if not lb_lab.empty:
            join_used = f"ID+date ({lb_id_col} ~ {ev_id_col})"

# ---------- Fallback to name+date ----------
if lb_lab.empty:
    lb_lab = lb.merge(
        ev_labels[["_join_date", "_join_name", "hr_outcome"]].dropna(subset=["_join_name"]),
        how="inner",
        on=["_join_date", "_join_name"]
    )
    if not lb_lab.empty:
        join_used = "name+date"

matches = len(lb_lab)
uniq_names_overlap = int(len(set(lb["_join_name"]).intersection(set(ev["_join_name"]))))
uniq_dates_overlap = int(len(set(lb["_join_date"].dropna()).intersection(set(ev["_join_date"].dropna()))))

st.write("🔗 Join results")
st.write({
    "merged_rows": matches,
    "join_used": join_used,
    "unique_name_overlap": uniq_names_overlap,
    "unique_date_overlap": uniq_dates_overlap,
    "labels_found": int(lb_lab["hr_outcome"].notna().sum()) if matches else 0,
})

if matches == 0:
    st.error(
        "No label matches found after join.\n\n"
        "Hints: ensure the leaderboard has a real date (prefer 'game_date_iso' like 2025-08-13) "
        "or a 'game_date' like 8_13 with the correct Season year set above. "
        "If IDs exist, include the same ID field in both files."
    )
    st.stop()

st.success(f"Labels joined via **{join_used}**. Rows with labels: {matches:,}.")
st.caption("We’ll now select features (2TB/RBI kept if present).")

# ---------- Feature selection ----------
# Baseline feature whitelist (we'll pick what exists)
candidate_feats = [
    # From your prediction leaderboard
    "ranked_probability", "hr_probability_iso_T",
    "final_multiplier", "overlay_multiplier",
    "weak_pitcher_factor", "hot_streak_factor",
    "rrf_aux", "model_disagreement",
    # 2TB / RBI
    "prob_2tb", "prob_rbi",
    # sometimes present misc cols
    "final_multiplier_raw", "temp", "humidity", "wind_mph",
]

# remove obvious non-features if present
drop_non_feats = {"Unnamed: 0", "source_file", "_join_name", "_join_date"}

available_feats = [c for c in candidate_feats if c in lb_lab.columns]
if not available_feats:
    st.error("No expected leaderboard features found. Confirm your merged CSV includes columns like 'ranked_probability', etc.")
    st.stop()

# Let user optionally add/remove
st.subheader("2) Select features for the ranker (auto-selected for you)")
user_feats = st.multiselect("Features", options=sorted(set(available_feats + candidate_feats)),
                            default=available_feats)
if not user_feats:
    st.error("Please select at least one feature.")
    st.stop()

# ---------- Build training frame ----------
# Keep only selected features + hr_outcome + date
train_df = lb_lab.copy()

# Drop non-features if present
train_df = train_df.drop(columns=[c for c in drop_non_feats if c in train_df.columns], errors="ignore")

# Ensure we have the date grouping column
if "_join_date" not in train_df.columns:
    st.error("Internal error: missing _join_date.")
    st.stop()

# Keep only selected numeric features
X = train_df[user_feats].copy()
for c in X.columns:
    X[c] = pd.to_numeric(X[c], errors="coerce")
X = X.fillna(0.0)
X = to_float32_df(X)

# Remove zero-variance columns (LightGBM dislikes empty/constant cols)
nz = X.nunique()
const_cols = nz[nz <= 1].index.tolist()
if const_cols:
    X = X.drop(columns=const_cols)

y = pd.to_numeric(train_df["hr_outcome"], errors="coerce").fillna(0).astype(int)
d = pd.to_datetime(train_df["_join_date"], errors="coerce").dt.floor("D")
keep_mask = d.notna() & (y.isin([0, 1])) & (X.notna().all(axis=1))
X = X.loc[keep_mask].reset_index(drop=True)
y = y.loc[keep_mask].reset_index(drop=True)
d = d.loc[keep_mask].reset_index(drop=True)

# Group by day
groups = d.groupby(d).size().values.tolist()
if len(X) == 0 or len(groups) == 0:
    st.error("After cleaning, no rows remain for training. Check your selections and join results.")
    st.stop()

st.subheader("3) Train day-wise LambdaRank model")
st.write(f"Training rows: {len(X):,} | Days: {len(groups)}")
rk = lgb.LGBMRanker(
    objective="lambdarank", metric="ndcg",
    n_estimators=700, learning_rate=0.05, num_leaves=63,
    feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=1,
    random_state=42
)

# Fit
rk.fit(X, y, group=groups)
st.success("Model trained.")

# ---------- Exports ----------
# 1) Ranker pickle bundle
bundle = {
    "model": rk,
    "features": list(X.columns),
    "trained_on_rows": int(len(X)),
    "trained_on_days": int(len(groups)),
    "join_used": join_used,
    "season_year": int(season_year),
}
buf = io.BytesIO()
pickle.dump(bundle, buf)
buf.seek(0)
st.download_button(
    "⬇️ Download learning_ranker.pkl",
    data=buf,
    file_name="learning_ranker.pkl",
    mime="application/octet-stream",
)

# 2) Labeled merged leaderboard CSV (for your records)
lb_labeled_out = lb_lab.copy()
# Keep the hr_outcome and original leaderboard columns
# (avoid LightGBM-only columns like _join_name if you don't want them)
csv_buf = io.StringIO()
lb_labeled_out.to_csv(csv_buf, index=False)
st.download_button(
    "⬇️ Download merged_leaderboards_labeled.csv",
    data=csv_buf.getvalue(),
    file_name="merged_leaderboards_labeled.csv",
    mime="text/csv",
)

st.caption("Done. Use learning_ranker.pkl in your prediction app (it replaces ranker_z during blending).")
