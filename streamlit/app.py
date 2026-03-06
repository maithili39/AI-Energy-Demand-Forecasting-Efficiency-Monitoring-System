"""
⚡ AI Energy Efficiency Monitoring System
Streamlit Dashboard — 6-page interactive analytics app
"""

import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Energy Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT        = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR    = os.path.join(ROOT, "data")
MODELS_DIR  = os.path.join(ROOT, "models")

METER_LABELS = {0: "Electricity", 1: "Chilled Water", 2: "Steam", 3: "Hot Water"}


# ─── Cached data loader ───────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading & preprocessing data…")
def load_data():
    t_dtypes = {"building_id": "int16", "meter": "int8", "meter_reading": "float32"}
    w_dtypes = {"site_id": "int8", "air_temperature": "float32", "cloud_coverage": "float32", "dew_temperature": "float32", "precip_depth_1_hr": "float32", "sea_level_pressure": "float32", "wind_direction": "float32", "wind_speed": "float32"}
    m_dtypes = {"site_id": "int8", "building_id": "int16", "square_feet": "int32"}
    
    train   = pd.read_csv(os.path.join(DATA_DIR, "train.csv"),           parse_dates=["timestamp"], dtype=t_dtypes, nrows=2000000)
    weather = pd.read_csv(os.path.join(DATA_DIR, "weather_train.csv"),   parse_dates=["timestamp"], dtype=w_dtypes)
    meta    = pd.read_csv(os.path.join(DATA_DIR, "building_metadata.csv"), dtype=m_dtypes)

    df = (train
          .merge(meta, on="building_id", how="left")
          .merge(weather, on=["site_id", "timestamp"], how="left")
          .drop_duplicates()
          .sort_values(["building_id", "meter", "timestamp"])
          .reset_index(drop=True))

    weather_cont = ["air_temperature","dew_temperature","wind_speed","wind_direction",
                    "cloud_coverage","sea_level_pressure","precip_depth_1_hr"]
    for col in weather_cont:
        if col in df.columns:
            df[col] = (df.groupby("site_id")[col]
                         .transform(lambda s: s.interpolate(method="linear",
                                                             limit_direction="both")))
    df["floor_count"] = df["floor_count"].fillna(df["floor_count"].median())
    df["year_built"]  = df["year_built"].fillna(df["year_built"].median())

    df["hour"]       = df["timestamp"].dt.hour
    df["day"]        = df["timestamp"].dt.day
    df["month"]      = df["timestamp"].dt.month
    df["weekday"]    = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["quarter"]    = df["timestamp"].dt.quarter
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"]  / 24)
    df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]    = np.sin(2 * np.pi * df["weekday"] / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * df["weekday"] / 7)

    grp = df.groupby(["building_id", "meter"])["meter_reading"]
    df["lag_1"]            = grp.shift(1)
    df["lag_24"]           = grp.shift(24)
    df["lag_168"]          = grp.shift(168)
    df["rolling_mean_24"]  = grp.transform(lambda x: x.shift(1).rolling(24,  min_periods=1).mean())
    df["rolling_mean_168"] = grp.transform(lambda x: x.shift(1).rolling(168, min_periods=1).mean())
    df["rolling_std_24"]   = grp.transform(lambda x: x.shift(1).rolling(24,  min_periods=2).std())

    df = pd.get_dummies(df, columns=["primary_use"], prefix="use", drop_first=False)

    cap99 = grp.transform(lambda x: x.quantile(0.99))
    df["log_target"] = np.log1p(df["meter_reading"].clip(upper=cap99))

    return df


@st.cache_resource(show_spinner="Loading model artifacts…")
def load_artifacts():
    out = {"model": None, "feat_cols": None, "model_name": "Not trained",
           "results_df": None, "iso": None, "building_profiles": None}

    feat_path = os.path.join(MODELS_DIR, "feature_cols.pkl")
    if not os.path.exists(feat_path):
        return out
    out["feat_cols"] = joblib.load(feat_path)

    for fname in ["lightgbm_tuned_model.pkl", "lightgbm_model.pkl",
                  "xgboost_model.pkl", "randomforest_model.pkl",
                  "linearregression_model.pkl"]:
        p = os.path.join(MODELS_DIR, fname)
        if os.path.exists(p):
            out["model"]      = joblib.load(p)
            out["model_name"] = fname.replace("_model.pkl","").replace("_"," ").title()
            break

    res = os.path.join(MODELS_DIR, "model_results.pkl")
    if os.path.exists(res):
        out["results_df"] = joblib.load(res)

    iso_p = os.path.join(MODELS_DIR, "anomaly_detector.pkl")
    if os.path.exists(iso_p):
        out["iso"] = joblib.load(iso_p)

    bp_p = os.path.join(MODELS_DIR, "building_profiles.pkl")
    if os.path.exists(bp_p):
        out["building_profiles"] = joblib.load(bp_p)

    return out


def safe_predict(model, feat_cols, subset_df):
    tmp = subset_df.reindex(columns=feat_cols, fill_value=0.0).astype(float)
    rows = tmp.dropna()
    preds = np.clip(np.expm1(model.predict(rows)), 0, None)
    return pd.Series(preds, index=rows.index)


# ─── Load ─────────────────────────────────────────────────────────────────────
df         = load_data()
arts       = load_artifacts()
model      = arts["model"]
feat_cols  = arts["feat_cols"]
model_name = arts["model_name"]
results_df = arts["results_df"]
bp         = arts["building_profiles"]
use_cols   = [c for c in df.columns if c.startswith("use_")]

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ AI Energy Monitor")
    st.markdown("---")
    page = st.radio("Navigate", [
        "📊 Overview",
        "🏢 Building Analysis",
        "🌤️ Weather vs Energy",
        "🤖 AI Forecast",
        "🚨 Anomaly Detection",
        "🔴 Live Simulation",
    ])
    st.markdown("---")
    st.markdown(f"**Rows:** {len(df):,}")
    st.markdown(f"**Buildings:** {df['building_id'].nunique():,}")
    st.markdown(f"**Model:** `{model_name}`")
    st.markdown("---")
    if model is None:
        st.warning("No model found.\nRun `notebooks/feature_engineering.ipynb` first.")
    else:
        st.success("Model loaded ✓")
    st.caption("AI Energy Efficiency Monitoring System · v2.0")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 Energy Consumption Overview")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Readings",  f"{len(df):,}")
    c2.metric("Buildings",       f"{df['building_id'].nunique():,}")
    c3.metric("Sites",           f"{df['site_id'].nunique():,}")
    c4.metric("Avg kWh / hour",  f"{df['meter_reading'].mean():.1f}")
    c5.metric("Date range",
              f"{df['timestamp'].min().strftime('%b %Y')} → "
              f"{df['timestamp'].max().strftime('%b %Y')}")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Average kWh by Meter Type")
        avg_m = df.groupby("meter")["meter_reading"].mean().rename(index=METER_LABELS)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        colors = sns.color_palette("husl", len(avg_m))
        bars = ax.bar(avg_m.index.astype(str), avg_m.values, color=colors, edgecolor="white", width=0.55)
        ax.set_ylabel("kWh")
        ax.tick_params(axis="x", rotation=10)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.subheader("24-Hour Energy Profile")
        hourly = df.groupby("hour")["meter_reading"].mean()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(hourly.index, hourly.values, color="#E91E63", marker="o", linewidth=2)
        ax.fill_between(hourly.index, hourly.values, alpha=0.15, color="#E91E63")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Avg kWh")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.subheader("Monthly Energy Trend")
    monthly = df.groupby("month")["meter_reading"].mean()
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.bar(range(1, 13), monthly.values, color="#3F51B5", edgecolor="white", width=0.7)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"])
    ax.set_ylabel("Avg kWh")
    st.pyplot(fig, use_container_width=True)
    plt.close()

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Weekend vs Weekday")
        wk = df.groupby("is_weekend")["meter_reading"].mean().rename({0: "Weekday", 1: "Weekend"})
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(wk.index, wk.values, color=["#1565C0","#E53935"], edgecolor="white", width=0.4)
        ax.set_ylabel("Avg kWh")
        st.pyplot(fig, use_container_width=True)
        plt.close()
    with col4:
        if results_df is not None:
            st.subheader("Model Leaderboard")
            st.dataframe(
                results_df.style
                    .highlight_min(subset=[c for c in ["MAE","RMSE","RMSLE"] if c in results_df.columns], color="#c8e6c9")
                    .highlight_max(subset=[c for c in ["R²"] if c in results_df.columns], color="#c8e6c9"),
                use_container_width=True)
        else:
            st.info("Train models in `feature_engineering.ipynb` to see leaderboard.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — BUILDING ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏢 Building Analysis":
    st.title("🏢 Building Analysis")

    st.subheader("Top 20 Buildings — Total Energy Consumption")
    top20 = df.groupby("building_id")["meter_reading"].sum().nlargest(20).sort_values()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.barh(top20.index.astype(str), top20.values, color="#FF6F00", edgecolor="white")
    ax.set_xlabel("Total kWh")
    ax.set_ylabel("Building ID")
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Energy by Building Type")
        if use_cols:
            type_energy = {uc.replace("use_",""):
                           df.loc[df[uc] == 1, "meter_reading"].mean()
                           for uc in use_cols}
            te = pd.Series(type_energy).sort_values(ascending=True).tail(12)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.barh(te.index, te.values, color="#5C6BC0", edgecolor="white")
            ax.set_xlabel("Avg kWh")
            ax.set_title("Avg kWh by Building Type")
            st.pyplot(fig, use_container_width=True)
            plt.close()

    with col2:
        st.subheader("Building Size vs Energy")
        samp = df.sample(min(40_000, len(df)), random_state=1)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(np.log1p(samp["square_feet"]), np.log1p(samp["meter_reading"]),
                   alpha=0.04, s=3, color="#26A69A")
        ax.set_xlabel("log1p(square_feet)")
        ax.set_ylabel("log1p(kWh)")
        ax.set_title("Building Size vs Energy")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("---")
    st.subheader("📈 Single Building Deep-Dive")
    c_a, c_b = st.columns(2)
    sel_bld = c_a.selectbox("Building", sorted(df["building_id"].unique()))
    sel_mtr = c_b.selectbox("Meter",
                              sorted(df[df["building_id"] == sel_bld]["meter"].unique()),
                              format_func=lambda m: METER_LABELS.get(m, str(m)))

    bld_ts = df[(df["building_id"] == sel_bld) & (df["meter"] == sel_mtr)].sort_values("timestamp")
    if len(bld_ts) > 0:
        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)
        axes[0].plot(bld_ts["timestamp"], bld_ts["meter_reading"],
                     color="#2196F3", linewidth=0.7)
        axes[0].set_title(f"Building {sel_bld} · {METER_LABELS.get(sel_mtr,'Meter')} — Full Series")
        axes[0].set_ylabel("kWh")
        daily = bld_ts.set_index("timestamp")["meter_reading"].resample("D").mean()
        axes[1].plot(daily.index, daily.values, color="#FF6F00", linewidth=1.5)
        axes[1].fill_between(daily.index, daily.values, alpha=0.2, color="#FF6F00")
        axes[1].set_title("Daily Average")
        axes[1].set_ylabel("kWh")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — WEATHER VS ENERGY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🌤️ Weather vs Energy":
    st.title("🌤️ Weather vs Energy Analysis")

    w_vars = [c for c in ["air_temperature","dew_temperature","cloud_coverage",
                           "wind_speed","sea_level_pressure","precip_depth_1_hr"]
              if c in df.columns]

    samp = df.sample(min(80_000, len(df)), random_state=42)
    samp = samp.copy()
    samp["log_reading"] = np.log1p(samp["meter_reading"])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Correlation with log(Energy)")
        corr = samp[w_vars + ["log_reading"]].corr()["log_reading"].drop("log_reading").sort_values()
        fig, ax = plt.subplots(figsize=(6, 4))
        bar_c = ["#F44336" if v < 0 else "#4CAF50" for v in corr.values]
        corr.plot(kind="barh", ax=ax, color=bar_c, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("Pearson Correlation")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.subheader("Full Correlation Matrix")
        corr_m = samp[w_vars + ["meter","log_reading"]].dropna().corr()
        fig, ax = plt.subplots(figsize=(7, 5))
        mask = np.triu(np.ones_like(corr_m, dtype=bool))
        sns.heatmap(corr_m, mask=mask, annot=True, fmt=".2f",
                    cmap="RdYlGn", center=0, ax=ax,
                    linewidths=0.4, annot_kws={"size": 7})
        ax.set_title("Feature Correlation Heatmap")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("---")
    c1, c2 = st.columns(2)
    sel_var  = c1.selectbox("Weather variable", w_vars)
    mtr_pair = c2.selectbox("Meter",
                              [("All", None)] + [(METER_LABELS[k], k) for k in sorted(METER_LABELS)],
                              format_func=lambda x: x[0])
    samp2 = samp if mtr_pair[1] is None else samp[samp["meter"] == mtr_pair[1]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(samp2[sel_var], samp2["log_reading"], alpha=0.04, s=3, color="#673AB7")
    axes[0].set_xlabel(sel_var)
    axes[0].set_ylabel("log1p(kWh)")
    axes[0].set_title(f"{sel_var} vs log(Energy)")

    try:
        bins = pd.qcut(samp2[sel_var].dropna(), q=6, duplicates="drop")
        box_df = samp2[["log_reading"]].copy()
        box_df["bin"] = bins.values
        box_df.boxplot(column="log_reading", by="bin", ax=axes[1], showfliers=False)
        axes[1].set_title(f"Energy across {sel_var} bins")
        axes[1].set_xlabel(sel_var)
        axes[1].set_ylabel("log1p(kWh)")
        plt.suptitle("")
        plt.xticks(rotation=30)
    except Exception:
        pass

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — AI FORECAST
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 AI Forecast":
    st.title("🤖 AI 24-Hour Energy Forecast")

    if model is None:
        st.error("No trained model found. Run `notebooks/feature_engineering.ipynb` first.")
    else:
        st.success(f"Model: **{model_name}**  |  Features: **{len(feat_cols)}**")

        col1, col2 = st.columns(2)
        sel_bld = col1.selectbox("Building", sorted(df["building_id"].unique()), key="fc_bld")
        sel_mtr = col2.selectbox("Meter",
                                  sorted(df[df["building_id"] == sel_bld]["meter"].unique()),
                                  key="fc_mtr",
                                  format_func=lambda m: METER_LABELS.get(m, str(m)))

        bld_df = (df[(df["building_id"] == sel_bld) & (df["meter"] == sel_mtr)]
                  .dropna(subset=["log_target"])
                  .sort_values("timestamp"))

        if len(bld_df) < 48:
            st.warning("Need ≥ 48 data rows. Try another building/meter.")
        else:
            inp   = bld_df.iloc[-48:-24]
            truth = bld_df.iloc[-24:]
            preds = safe_predict(model, feat_cols, inp)
            actual = np.expm1(truth["log_target"].values)

            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(truth["timestamp"].values, actual,
                    label="Actual", color="#1E88E5", linewidth=2.5)
            ax.plot(inp["timestamp"].values, preds.values,
                    label=f"{model_name} Forecast",
                    color="#E53935", linestyle="--", linewidth=2)
            ax.set_title(f"24-Hour Forecast — Building {sel_bld} · "
                         f"{METER_LABELS.get(sel_mtr,'Meter')}", fontsize=14)
            ax.set_ylabel("kWh")
            ax.legend()
            plt.xticks(rotation=20)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            if len(preds) > 0:
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                n    = min(len(preds), len(actual))
                mae  = mean_absolute_error(actual[:n], preds.values[:n])
                rmse = np.sqrt(mean_squared_error(actual[:n], preds.values[:n]))
                r2   = r2_score(actual[:n], preds.values[:n])
                m1, m2, m3 = st.columns(3)
                m1.metric("MAE",  f"{mae:.2f} kWh")
                m2.metric("RMSE", f"{rmse:.2f} kWh")
                m3.metric("R²",   f"{r2:.4f}")

        # Actual vs Predicted scatter
        st.markdown("---")
        st.subheader("Actual vs Predicted — Test Sample")
        split_ts   = df["timestamp"].quantile(0.8)
        test_samp  = df[df["timestamp"] > split_ts].dropna(subset=["log_target"])
        test_samp  = test_samp.sample(min(5000, len(test_samp)), random_state=42)
        test_preds = safe_predict(model, feat_cols, test_samp)
        test_actual = np.expm1(test_samp.loc[test_preds.index, "log_target"])

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(test_actual, test_preds.values, alpha=0.1, s=5, color="#1565C0")
        lim = max(test_actual.max(), test_preds.max()) * 1.05
        ax.plot([0, lim], [0, lim], "r--", linewidth=1.5, label="Perfect prediction")
        ax.set_xlabel("Actual kWh")
        ax.set_ylabel("Predicted kWh")
        ax.set_title("Actual vs Predicted (test set)")
        ax.legend()
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🚨 Anomaly Detection":
    st.title("🚨 Building Anomaly Detection")

    if bp is not None and "anomaly" in bp.columns:
        n_anom = bp["anomaly"].sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Buildings",     f"{len(bp):,}")
        c2.metric("Anomalous Buildings", f"{n_anom:,}")
        c3.metric("Anomaly Rate",        f"{n_anom/len(bp)*100:.1f}%")

        fig, ax = plt.subplots(figsize=(12, 6))
        norm = bp[~bp["anomaly"]]
        anom = bp[ bp["anomaly"]]
        ax.scatter(norm["mean_kwh"], norm["std_kwh"], alpha=0.4, s=15,
                   label="Normal", color="#4CAF50")
        ax.scatter(anom["mean_kwh"], anom["std_kwh"], alpha=0.9, s=60,
                   label="Anomaly", color="#E53935", marker="X")
        ax.set_xlabel("Mean kWh / hour")
        ax.set_ylabel("Std kWh / hour")
        ax.set_title("Building Anomaly Map (IsolationForest)", fontsize=13)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.subheader("Anomalous Buildings Table")
        disp = [c for c in ["building_id","mean_kwh","std_kwh","max_kwh","zero_pct","n_hours"]
                if c in bp.columns]
        st.dataframe(bp[bp["anomaly"]][disp].sort_values("mean_kwh", ascending=False)
                     .reset_index(drop=True), use_container_width=True)
    else:
        st.warning("No saved anomaly profiles. Run Step 10 in `notebooks/feature_engineering.ipynb`.")

    # Live IQR-based anomaly check
    st.markdown("---")
    st.subheader("Live Rule-Based Anomaly Check — Single Building")
    c_a, c_b = st.columns(2)
    sel_bld2 = c_a.selectbox("Building", sorted(df["building_id"].unique()), key="anom_bld")
    sel_mtr2 = c_b.selectbox("Meter",
                               sorted(df[df["building_id"] == sel_bld2]["meter"].unique()),
                               key="anom_mtr",
                               format_func=lambda m: METER_LABELS.get(m, str(m)))

    bld_check = df[(df["building_id"] == sel_bld2) & (df["meter"] == sel_mtr2)].copy()
    if len(bld_check) > 10:
        q1  = bld_check["meter_reading"].quantile(0.25)
        q3  = bld_check["meter_reading"].quantile(0.75)
        iqr = q3 - q1
        bld_check["outlier"] = (
            (bld_check["meter_reading"] < q1 - 1.5*iqr) |
            (bld_check["meter_reading"] > q3 + 1.5*iqr)
        )
        n_out = bld_check["outlier"].sum()
        st.info(f"Building {sel_bld2} — IQR outliers: **{n_out:,}** / {len(bld_check):,} "
                f"({n_out/len(bld_check)*100:.1f}%)")

        fig, ax = plt.subplots(figsize=(14, 4))
        normal_pts  = bld_check[~bld_check["outlier"]]
        outlier_pts = bld_check[ bld_check["outlier"]]
        ax.scatter(normal_pts["timestamp"],  normal_pts["meter_reading"],
                   s=2, alpha=0.4, color="#4CAF50", label="Normal")
        ax.scatter(outlier_pts["timestamp"], outlier_pts["meter_reading"],
                   s=15, alpha=0.8, color="#E53935", label="Outlier", marker="X")
        ax.set_title(f"Building {sel_bld2} · {METER_LABELS.get(sel_mtr2,'Meter')} — Anomaly Overlay")
        ax.set_ylabel("kWh")
        ax.legend()
        plt.xticks(rotation=15)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — LIVE SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔴 Live Simulation":
    st.title("🔴 Real-Time Energy Monitoring Simulation")

    if model is None:
        st.error("No model found. Run `notebooks/feature_engineering.ipynb` first.")
    else:
        st.markdown(
            "Replays held-out test rows **one at a time**, computing predictions and "
            "rolling metrics live — just like a real IoT sensor dashboard."
        )

        c1, c2 = st.columns(2)
        sel_bld = c1.selectbox("Building", sorted(df["building_id"].unique()), key="sim_bld")
        sel_mtr = c2.selectbox("Meter",
                                sorted(df[df["building_id"] == sel_bld]["meter"].unique()),
                                key="sim_mtr",
                                format_func=lambda m: METER_LABELS.get(m, str(m)))
        n_steps = st.slider("Steps to simulate", 10, 100, 30)
        speed   = st.slider("Delay between steps (seconds)", 0.05, 1.5, 0.2, step=0.05)

        if st.button("▶ Start Simulation", type="primary"):
            split_ts = df["timestamp"].quantile(0.8)
            sim_data = (df[(df["building_id"] == sel_bld) & (df["meter"] == sel_mtr) &
                           (df["timestamp"] > split_ts)]
                        .dropna(subset=["lag_168"])
                        .sort_values("timestamp")
                        .head(n_steps))

            if len(sim_data) < 5:
                st.error("Not enough test data after lag warm-up. Try a different building/meter.")
            else:
                kpi_ph   = st.empty()
                chart_ph = st.empty()
                table_ph = st.empty()

                actuals, preds_sim, errors = [], [], []
                log = []

                for i, (idx, row) in enumerate(sim_data.iterrows()):
                    act  = float(row["meter_reading"])
                    x    = (row[feat_cols].to_frame().T
                            .reindex(columns=feat_cols, fill_value=0.0)
                            .astype(float))
                    pred = float(np.clip(np.expm1(model.predict(x)[0]), 0, None))
                    err  = abs(act - pred)

                    actuals.append(act)
                    preds_sim.append(pred)
                    errors.append(err)
                    log.append({
                        "Step": i + 1,
                        "Timestamp": str(row["timestamp"])[:16],
                        "Actual kWh": round(act, 2),
                        "Predicted kWh": round(pred, 2),
                        "Error kWh": round(err, 2),
                    })

                    mae_live  = np.mean(errors)
                    rmse_live = np.sqrt(np.mean(np.square(errors)))

                    with kpi_ph.container():
                        k1, k2, k3 = st.columns(3)
                        k1.metric("Step",         f"{i+1} / {n_steps}")
                        k2.metric("Rolling MAE",  f"{mae_live:.2f} kWh")
                        k3.metric("Rolling RMSE", f"{rmse_live:.2f} kWh")

                    with chart_ph.container():
                        fig, ax = plt.subplots(figsize=(12, 4))
                        ax.plot(range(len(actuals)),   actuals,   label="Actual",
                                color="#1E88E5", linewidth=1.8)
                        ax.plot(range(len(preds_sim)), preds_sim, label="Predicted",
                                color="#E53935", linestyle="--", linewidth=1.8)
                        ax.set_title(f"Live — Step {i+1}/{n_steps}  |  "
                                     f"Building {sel_bld} · {METER_LABELS.get(sel_mtr,'Meter')}")
                        ax.set_ylabel("kWh")
                        ax.legend()
                        st.pyplot(fig, use_container_width=True)
                        plt.close()

                    table_ph.dataframe(
                        pd.DataFrame(log[-10:]).set_index("Step"),
                        use_container_width=True)

                    time.sleep(speed)

                st.success(
                    f"✅ Simulation complete!   "
                    f"Final MAE: {mae_live:.2f} kWh  |  RMSE: {rmse_live:.2f} kWh")
