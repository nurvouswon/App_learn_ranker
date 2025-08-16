# App_learn_ranker.py
# =====================================================================
# 📚 Learner — HR Day Ranker from Leaderboards + Event Parquet (Fixed)
# - No need for game_date_iso; uses game_date and normalizes formats.
# - Robust name normalization (accents, suffixes, punctuation).
# - Joins merged leaderboard ↔ event-level parquet to build labels.
# - Trains day-wise LambdaRank (LightGBM) using your leaderboard signals.
# - Includes prob_2tb and prob_rbi if present.
# - Exports labeled CSV + ranker model bundle (.pkl) for your main app.
# =====================================================================

import streamlit as st
import pandas as pd
import numpy as np
import io, gc, pickle, re, unicodedata
from datetime import datetime

import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="📚 HR Day Ranker Learner (Fixed)", layout="wide")
st.title("📚 Learner — HR Day Ranker from Leaderboards + Event Parquet")

# --------------------------- Helpers ---------------------------

def _strip_accents(s: str) -> str:
    if not isinstance(s, str): 
        s = "" if pd.isna(s) else str(s)
    return ''.join(
        c for c in unicodedata.normalize('NFKD', s)
        if not unicodedata.combining(c)
    )

def normalize_name(s: str) -> str:
    """
    Normalize player name strings:
    - Lowercase, remove accents
    - Remove periods/commas/extra spaces
    - Drop common suffixes (jr., sr., II/III/IV)
    - Fix common variants (pena->peña, rodriguez->rodríguez, etc.) via accent strip + clean
      (Because we match both sides after accent removal, this already aligns.)
    """
    s = "" if s is None else str(s)
    s = _strip_accents(s).lower().strip()

    # remove punctuation and duplicate whitespace
    s = re.sub(r"[.,']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # remove suffix tokens at end
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b$", "", s).strip()

    return s

def normalize_date_col(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Accepts many forms:
      - ISO: 2025-08-13
      - With time: 2025-08-13 00:00:00
      - Underscore formats: 8_13, 08_13, 2025_08_13 (if season year slider used)
      - Month/Day strings like 8/13/2025 or 08/13/25
    Output: 'YYYY-MM-DD' strings.
    """
    s = df[col].astype(str).fillna("")

    # Fast-path: try pandas to_datetime directly
    dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)

    # For clear fails like "8_13" try to repair with a fallback using season year
    if dt.isna().any():
        # See if patterns like 8_13 exist
        needs_fix = dt.isna()
        cand = s[needs_fix].str.strip()

        # Try patterns  M_D  or  MM_DD  -> use provided year
        m = cand.str.fullmatch(r"(\d{1,2})_(\d{1,2})")
        if m.notna().any():
            yy = st.session_state.get("season_year", 2025)
            fixed = pd.to_datetime(
                cand[m.notna()].apply(lambda z: f"{yy}-{z.replace('_','-')}"),
                errors="coerce"
            )
            dt.loc[needs_fix & m.notna()] = fixed

        # Try YYYY_MM_DD
        m3 = cand.str.fullmatch(r"(\d{4})_(\d{1,2})_(\d{1,2})")
        if m3.notna().any():
            fixed = pd.to_datetime(cand[m3.notna()].str.replace("_","-"), errors="coerce")
            dt.loc[needs_fix & m3.notna()] = fixed

    # Final string format
    out = dt.dt.strftime("%Y-%m-%d")
    return out

def load_merged_leaderboard(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    # Expect 'game_date' in ISO now; normalize just in case
    if "game_date" not in df.columns:
        raise ValueError("merged leaderboard missing 'game_date' column.")
    if "player_name" not in df.columns:
        raise ValueError("merged leaderboard missing 'player_name' column.")

    # Normalize date + name
    df["game_date_norm"] = normalize_date_col(df, "game_date")
    df["player_name_norm"] = df["player_name"].apply(normalize_name)

    # Clean common junk columns
    drop_junk = [c for c in df.columns if c.lower().startswith("unnamed")]
    if drop_junk:
        df = df.drop(columns=drop_junk, errors="ignore")

    # Coerce numerics where helpful (best-effort)
    for c in ["ranked_probability","hr_probability_iso_T","overlay_multiplier",
              "weak_pitcher_factor","hot_streak_factor","final_multiplier_raw",
              "final_multiplier","rrf_aux","model_disagreement","prob_2tb","prob_rbi",
              "temp","humidity","wind_mph"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def load_event_parquet(file) -> pd.DataFrame:
    ev = pd.read_parquet(file)
    # Must have 'game_date' and 'player_name' and 'hr_outcome'
    # We’ll create a day-level label: 1 if batter hit any HR that day
    # Normalize cols if variants exist:
    # Common alternates for player and date in event data
    # (Will try to map if needed)
    cand_name_cols = ["player_name", "batter_name", "batter_full_name", "player"]
    name_col = next((c for c in cand_name_cols if c in ev.columns), None)
    if name_col is None:
        raise ValueError("event parquet: no player name column found (expected one of: player_name, batter_name, batter_full_name, player).")
    if "game_date" not in ev.columns:
        # try date-like alternatives
        for alt in ["game_date_str","date","game_day","gamedate"]:
            if alt in ev.columns:
                ev = ev.rename(columns={alt: "game_date"})
                break
        if "game_date" not in ev.columns:
            raise ValueError("event parquet: no game_date column found.")

    if "hr_outcome" not in ev.columns:
        raise ValueError("event parquet missing 'hr_outcome' column.")

    # Normalize
    ev["game_date_norm"] = normalize_date_col(ev, "game_date")
    ev["player_name_norm"] = ev[name_col].astype(str).apply(normalize_name)

    # Collapse to day-player label
    grp = ev.groupby(["game_date_norm","player_name_norm"], as_index=False)["hr_outcome"].max()
    grp = grp.rename(columns={"hr_outcome":"hr_outcome_day"})
    return grp

def join_labels(lb: pd.DataFrame, day_labels: pd.DataFrame) -> pd.DataFrame:
    merged = lb.merge(
        day_labels, how="left",
        on=["game_date_norm","player_name_norm"]
    )
    # Filter rows that have labels
    labeled = merged[~merged["hr_outcome_day"].isna()].copy()
    labeled["hr_outcome_day"] = labeled["hr_outcome_day"].astype(int)
    return merged, labeled

def groups_from_dates(d: pd.Series) -> list:
    # Group by exact day; already normalized as YYYY-MM-DD
    # Count rows per day in order of appearance (LightGBM expects group sizes)
    order = d.values
    # build sizes by run-length of same date in current order:
    sizes = []
    if len(order) == 0:
        return sizes
    prev = order[0]; cnt = 1
    for x in order[1:]:
        if x == prev:
            cnt += 1
        else:
            sizes.append(cnt)
            prev = x; cnt = 1
    sizes.append(cnt)
    return sizes

# --------------------------- UI ---------------------------

with st.expander("⚙️ Options", expanded=True):
    season_year = st.number_input("Season year (used only if your leaderboard game_date looks like '8_13')", min_value=2000, max_value=2100, value=2025, step=1)
    st.session_state["season_year"] = int(season_year)

st.header("1) Upload files")

lb_file = st.file_uploader("Merged leaderboard CSV (combined across days)", type=["csv"], key="lb")
ev_file = st.file_uploader("Event-level Parquet (with hr_outcome)", type=["parquet"], key="ev")

if lb_file and ev_file:
    with st.spinner("Reading & normalizing files..."):
        try:
            lb = load_merged_leaderboard(lb_file)
            day_labels = load_event_parquet(ev_file)
        except Exception as e:
            st.error(f"Failed to read inputs: {e}")
            st.stop()

        merged_all, labeled = join_labels(lb, day_labels)

    # Diagnostics
    total_rows = len(lb)
    labeled_rows = len(labeled)
    dropped = total_rows - labeled_rows

    st.success(f"Labels joined. Rows with labels: {labeled_rows} (dropped {dropped} without event data).")

    if labeled_rows == 0:
        # Provide strong, *actionable* diagnostics
        st.error("No label matches found after join.")
        st.markdown("**Quick checks you can do in your CSV:**")
        st.write("- Ensure **game_date** is ISO like `2025-08-13` (you already switched to this).")
        st.write("- Ensure **player_name** matches event parquet spelling. This app normalizes accents/suffixes automatically.")
        st.write("- If your event parquet uses a different player column (like `batter_name`), it’s handled automatically.")
        st.write("- If you have **ID columns** (e.g., `batter_id`) present in both files, add them and we’ll prefer ID-based join.")

        # Show what unique examples look like to debug quickly
        with st.expander("Debug preview — unique date & name samples"):
            st.write("Leaderboard unique dates (sample):", lb["game_date_norm"].dropna().unique()[:6])
            st.write("Event unique dates (sample):", day_labels["game_date_norm"].dropna().unique()[:6])

            if "player_name" in lb.columns:
                st.write("Leaderboard player_name normalized (sample):", lb["player_name_norm"].dropna().unique()[:10])
            st.write("Event player_name normalized (sample):", day_labels["player_name_norm"].dropna().unique()[:10])

        st.stop()

    st.header("2) Select features for the ranker (auto-selected for you)")
    # Candidate features from leaderboard columns (keep everything you asked for)
    default_feats = [
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
        "final_multiplier_raw",
        "temp","humidity","wind_mph",
    ]
    # Keep only those present
    available_feats = [c for c in default_feats if c in labeled.columns]

    if not available_feats:
        st.error("None of the expected leaderboard features were found in your merged CSV.")
        st.stop()

    chosen_feats = st.multiselect(
        "Features",
        options=available_feats,
        default=available_feats,
    )

    if not chosen_feats:
        st.error("Select at least one feature.")
        st.stop()

    # Prepare training matrices
    # Sort by day to build LightGBM groups easily
    labeled = labeled.sort_values(["game_date_norm","ranked_probability"], ascending=[True, False]).reset_index(drop=True)

    X = labeled[chosen_feats].copy()
    # Replace inf/nan
    X = X.replace([np.inf, -np.inf], np.nan).fillna(-1.0)
    # scale numeric features (optional but stable)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X.astype(np.float32))

    y = labeled["hr_outcome_day"].astype(int).values
    groups = groups_from_dates(labeled["game_date_norm"])

    if len(groups) == 0 or sum(groups) != len(labeled):
        st.error("Failed to build day groups for LambdaRank. Check your dates.")
        st.stop()

    st.header("3) Train day-wise LambdaRank model")
    if st.button("Train Ranker"):
        with st.spinner("Training LightGBM LambdaRank..."):
            params = dict(
                objective="lambdarank",
                metric="ndcg",
                n_estimators=800,
                learning_rate=0.05,
                num_leaves=63,
                feature_fraction=0.85,
                bagging_fraction=0.85,
                bagging_freq=1,
                random_state=42,
            )
            rk = lgb.LGBMRanker(**params)
            # LightGBM expects raw arrays for fit with group
            rk.fit(X_s, y, group=groups)

        st.success("✅ Model trained.")

        # Save bundle
        bundle = {
            "model": rk,
            "features": chosen_feats,
            "scaler": scaler,
            "notes": "Day-wise LambdaRank trained on merged leaderboard + event labels (fixed, ISO game_date)."
        }
        pkl_bytes = io.BytesIO()
        pickle.dump(bundle, pkl_bytes)
        pkl_bytes.seek(0)

        st.download_button(
            label="⬇️ Download learning_ranker.pkl",
            data=pkl_bytes,
            file_name="learning_ranker.pkl",
            mime="application/octet-stream",
        )

        # Also provide a labeled CSV for your records
        out_cols = ["game_date_norm","player_name","team_code","ranked_probability","hr_probability_iso_T",
                    "prob_2tb","prob_rbi","final_multiplier","overlay_multiplier","weak_pitcher_factor","hot_streak_factor",
                    "rrf_aux","model_disagreement","hr_outcome_day"]
        out_cols = [c for c in out_cols if c in labeled.columns or c in ["game_date_norm"]]
        labeled_export = labeled[out_cols].copy()
        labeled_export = labeled_export.rename(columns={"game_date_norm":"game_date"})
        csv_bytes = labeled_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download labeled_leaderboards.csv",
            data=csv_bytes,
            file_name="labeled_leaderboards.csv",
            mime="text/csv",
        )

        st.info("Done. Use learning_ranker.pkl in your prediction app (it already supports the optional ranker).")

else:
    st.info("Upload both files to continue.")
