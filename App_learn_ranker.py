import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from difflib import get_close_matches
from unidecode import unidecode
import io
import pickle
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="📚 Learner — HR Day Ranker from Leaderboards + Event Parquet", layout="wide")
st.title("📚 Learner — HR Day Ranker from Leaderboards + Event Parquet")

# ------------------------ Helpers ------------------------
def clean_name(s: str) -> str:
    if pd.isna(s): return ""
    s = unidecode(str(s)).strip()
    s = s.replace(".", "")
    for suf in (" JR", " SR", " II", " III", " IV"):
        if s.upper().endswith(suf):
            s = s[: -len(suf)]
    return " ".join(s.split()).upper()

TEAM_ALIASES = {
    "ANA":"LAA","LAA":"LAA",
    "TBD":"TB","TBR":"TB","TB":"TB",
    "FLA":"MIA","MIA":"MIA",
    "WSN":"WSH","WAS":"WSH","WSH":"WSH",
    "CHW":"CWS","CWS":"CWS",
    "KCR":"KC","KC":"KC",
    "SDP":"SD","SD":"SD",
    "SFG":"SF","SF":"SF",
    "LAN":"LAD","LAD":"LAD",
    "NYN":"NYM","NYM":"NYM",
    "SLN":"STL","STL":"STL",
}
def std_team(x: str) -> str:
    if pd.isna(x): return ""
    x = str(x).strip().upper()
    return TEAM_ALIASES.get(x, x)

def to_date_ymd(s, season_year):
    if pd.isna(s): return pd.NaT
    s = str(s).strip()
    if "_" in s:
        mm, dd = s.split("_")
        return pd.to_datetime(f"{season_year}-{int(mm):02d}-{int(dd):02d}", errors="coerce")
    return pd.to_datetime(s, errors="coerce")

def zscore(a):
    a = np.asarray(a, dtype=float)
    m = np.nanmean(a); s = np.nanstd(a) + 1e-9
    return (a - m) / s

# ------------------------ UI ------------------------
with st.sidebar:
    st.subheader("Settings")
    season_year = st.number_input("Season year (for dates like '8_13')", min_value=2015, max_value=2100, value=datetime.utcnow().year, step=1)
    st.session_state["season_year"] = season_year

st.markdown("### 1) Upload files")
lb_file = st.file_uploader("Merged leaderboard CSV (combined across days)", type=["csv"])
ev_file = st.file_uploader("Event-level Parquet (with hr_outcome)", type=["parquet"])

if not lb_file or not ev_file:
    st.stop()

# ------------------------ Load ------------------------
try:
    lb = pd.read_csv(lb_file)
except UnicodeDecodeError:
    lb = pd.read_csv(lb_file, encoding="latin1")
ev = pd.read_parquet(ev_file)

# Ensure expected columns exist
if "game_date" not in lb.columns or "player_name" not in lb.columns:
    st.error("Leaderboard must have at least 'game_date' and 'player_name'.")
    st.stop()
if "game_date" not in ev.columns or "player_name" not in ev.columns or "hr_outcome" not in ev.columns:
    st.error("Event parquet must have 'game_date', 'player_name', and 'hr_outcome'.")
    st.stop()

# Normalize/standardize keys
lb = lb.copy()
lb["game_date"] = lb["game_date"].apply(lambda s: to_date_ymd(s, season_year)).dt.date
lb["player_name_norm"] = lb["player_name"].astype(str).apply(clean_name)
lb["team_code_std"] = (lb.get("team_code") or pd.Series([""]*len(lb))).astype(str).apply(std_team)

# carry optional ids if present
lb["batter_id_join"] = (
    lb["batter_id"] if "batter_id" in lb.columns
    else lb["batter"] if "batter" in lb.columns
    else pd.Series([pd.NA]*len(lb))
)

# Events
ev = ev.copy()
ev["game_date"] = pd.to_datetime(ev["game_date"], errors="coerce").dt.date
ev["player_name_norm"] = ev["player_name"].astype(str).apply(clean_name)
if "team_code" not in ev.columns:
    # Some event dumps have home/away/team at bat; fall back to empty if missing
    ev["team_code"] = ""
ev["team_code_std"] = ev["team_code"].astype(str).apply(std_team)
ev["batter_id_join"] = (
    ev["batter_id"] if "batter_id" in ev.columns
    else ev["batter"] if "batter" in ev.columns
    else pd.Series([pd.NA]*len(ev))
)

# Daily label table
ev_daily = (
    ev.groupby(["game_date","player_name_norm","team_code_std","batter_id_join"], as_index=False)["hr_outcome"]
      .max()
)

# ------------------------ Robust Join ------------------------
st.markdown("### 2) Join labels")
merged = lb.merge(
    ev_daily,
    how="left",
    left_on=["game_date","batter_id_join"],
    right_on=["game_date","batter_id_join"],
    suffixes=("", "_ev")
)

if merged["hr_outcome"].isna().all():
    merged = lb.merge(
        ev_daily.drop(columns=["batter_id_join"]),
        how="left",
        left_on=["game_date","player_name_norm","team_code_std"],
        right_on=["game_date","player_name_norm","team_code_std"],
        suffixes=("", "_ev")
    )

if merged["hr_outcome"].isna().all():
    st.warning("Exact joins failed. Trying fuzzy name matching per (date, team)…")
    ev_idx = ev_daily.groupby(["game_date","team_code_std"])["player_name_norm"].apply(list)
    map_rows = []
    for (dt, tm), chunk in lb.groupby(["game_date","team_code_std"], dropna=False):
        candidates = ev_idx.get((dt, tm), [])
        if not candidates:
            continue
        for _, r in chunk.iterrows():
            nm = r["player_name_norm"]
            best = get_close_matches(nm, candidates, n=1, cutoff=0.88)
            if best:
                map_rows.append((dt, tm, nm, best[0]))
    if map_rows:
        approx = pd.DataFrame(map_rows, columns=["game_date","team_code_std","player_name_norm","player_name_norm_ev"])
        ev_alt = ev_daily.merge(
            approx,
            how="right",
            left_on=["game_date","team_code_std","player_name_norm"],
            right_on=["game_date","team_code_std","player_name_norm_ev"]
        )[["game_date","team_code_std","player_name_norm","hr_outcome"]]
        merged = lb.drop(columns=["hr_outcome"], errors="ignore").merge(
            ev_alt, how="left",
            on=["game_date","team_code_std","player_name_norm"]
        )

labels_found = int(merged["hr_outcome"].notna().sum())
st.write(f"Labels joined. Rows with labels: {labels_found} (dropped {len(merged)-labels_found} without event data).")

if labels_found == 0:
    st.error("❌ No label matches found after join.")
    st.write("Leaderboard sample:")
    st.dataframe(lb.head(12)[["game_date","player_name","player_name_norm","team_code","team_code_std"]])
    st.write("Events sample:")
    st.dataframe(ev.head(12)[["game_date","player_name","player_name_norm","team_code","team_code_std","hr_outcome"]])
    st.stop()

# Allow download of labeled merged CSV
labeled = merged.dropna(subset=["hr_outcome"]).copy()
labeled["hr_outcome"] = labeled["hr_outcome"].astype(int)
csv_buf = io.StringIO()
labeled.to_csv(csv_buf, index=False)
st.download_button("⬇️ Download labeled_merged.csv", data=csv_buf.getvalue(), file_name="labeled_merged.csv", mime="text/csv")

# ------------------------ Feature selection ------------------------
st.markdown("### 3) Select features for the ranker (auto-selected for you)")
default_feats = [
    "ranked_probability", "hr_probability_iso_T",
    "final_multiplier", "overlay_multiplier",
    "weak_pitcher_factor", "hot_streak_factor",
    "rrf_aux", "model_disagreement",
    "prob_2tb", "prob_rbi",
    # safe extras if present
    "final_multiplier_raw","temp","humidity","wind_mph"
]
present_feats = [c for c in default_feats if c in labeled.columns]
if not present_feats:
    # Fallback: keep numeric leaderboard columns except obvious non-features
    drop_like = {"hr_outcome","Unnamed: 0"}
    present_feats = [c for c in labeled.select_dtypes(include=[np.number]).columns if c not in drop_like]

feat_choices = st.multiselect("Features", options=sorted(labeled.columns), default=present_feats)
if not feat_choices:
    st.error("Pick at least one feature.")
    st.stop()

# ------------------------ Prepare training matrix ------------------------
train_df = labeled.dropna(subset=["hr_outcome"]).copy()
# sort by day so group arrays align with order
train_df = train_df.sort_values(["game_date","player_name_norm"]).reset_index(drop=True)

X = train_df[feat_choices].copy()
for c in X.columns:
    # best-effort coercion
    X[c] = pd.to_numeric(X[c], errors="coerce")
X = X.fillna(0.0).astype(np.float32)

y = train_df["hr_outcome"].astype(int).values

# groups: counts per day in current order
grp_sizes = train_df.groupby("game_date").size().tolist()

# safety
if len(X) == 0 or X.shape[1] == 0:
    st.error("Input data must be 2D and non-empty after feature prep.")
    st.stop()
if sum(grp_sizes) != len(X):
    st.error("Group sizes do not sum to dataset length — check game_date parsing.")
    st.stop()

# ------------------------ Train models (all 3 ML) ------------------------
st.markdown("### 4) Train day-wise ranker (LGB) + aux (XGB, Cat)")
# LGBMRanker (primary)
rk = lgb.LGBMRanker(
    objective="lambdarank", metric="ndcg",
    n_estimators=700, learning_rate=0.05, num_leaves=63,
    feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=1,
    random_state=42
)
rk.fit(X, y, group=grp_sizes)

# XGBoost aux (regression on y; used as auxiliary rank signal)
xgb_model = xgb.XGBRegressor(
    n_estimators=700, max_depth=6, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0,
    tree_method="hist", random_state=42
)
xgb_model.fit(X, y)

# CatBoost aux (regression on y; used as auxiliary rank signal)
cat_model = cb.CatBoostRegressor(
    iterations=1000, depth=7, learning_rate=0.05, l2_leaf_reg=6.0,
    loss_function="RMSE", random_seed=42, verbose=False
)
cat_model.fit(X, y)

# Blend z-scored predictions
lgb_scores = rk.predict(X)
xgb_scores = xgb_model.predict(X)
cat_scores = cat_model.predict(X)

blend = 0.5 * zscore(lgb_scores) + 0.25 * zscore(xgb_scores) + 0.25 * zscore(cat_scores)
train_df["learned_rank_score"] = blend

st.success("Model trained.")

# ------------------------ Download model bundle ------------------------
bundle = {
    "version": "ranker_v1",
    "season_year": season_year,
    "features": feat_choices,
    "model_lgb": rk,
    "model_xgb": xgb_model,
    "model_cat": cat_model,
    "blend_weights": {"lgb":0.5,"xgb":0.25,"cat":0.25}
}
pkl_bytes = io.BytesIO()
pickle.dump(bundle, pkl_bytes)
pkl_bytes.seek(0)
st.download_button("⬇️ Download learning_ranker.pkl", data=pkl_bytes, file_name="learning_ranker.pkl", mime="application/octet-stream")

# ------------------------ Preview learned ordering (no plots) ------------------------
st.markdown("### 5) Preview (first 30 rows by learned rank)")
preview_cols = ["game_date","player_name","team_code","ranked_probability","hr_probability_iso_T","prob_2tb","prob_rbi","learned_rank_score","hr_outcome"]
show_cols = [c for c in preview_cols if c in train_df.columns]
st.dataframe(train_df.sort_values(["game_date","learned_rank_score"], ascending=[True, False])[show_cols].head(30), use_container_width=True)
