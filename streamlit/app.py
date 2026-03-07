"""
⚡ AI Energy Efficiency Monitoring System
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")

st.set_page_config(
    page_title="AI Energy Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT       = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR   = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")

METER_LABELS = {0: "Electricity", 1: "Chilled Water", 2: "Steam", 3: "Hot Water"}


# ─── Shared feature engineering ──────────────────────────────────────────────
def _engineer_features(df, is_test=False):
    weather_cont = ["air_temperature", "dew_temperature", "wind_speed", "wind_direction",
                    "cloud_coverage", "sea_level_pressure", "precip_depth_1_hr"]
    for col in weather_cont:
        if col in df.columns:
            df[col] = (df.groupby("site_id")[col]
                         .transform(lambda s: s.interpolate(method="linear",
                                                             limit_direction="both")))
    if "floor_count" in df.columns:
        df["floor_count"] = df["floor_count"].fillna(df["floor_count"].median())
    if "year_built" in df.columns:
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

    if not is_test and "meter_reading" in df.columns:
        grp = df.groupby(["building_id", "meter"])["meter_reading"]
        df["lag_1"]            = grp.shift(1)
        df["lag_24"]           = grp.shift(24)
        df["lag_168"]          = grp.shift(168)
        df["rolling_mean_24"]  = grp.transform(lambda x: x.shift(1).rolling(24,  min_periods=1).mean())
        df["rolling_mean_168"] = grp.transform(lambda x: x.shift(1).rolling(168, min_periods=1).mean())
        df["rolling_std_24"]   = grp.transform(lambda x: x.shift(1).rolling(24,  min_periods=2).std())
        cap99 = grp.transform(lambda x: x.quantile(0.99))
        df["log_target"] = np.log1p(df["meter_reading"].clip(upper=cap99))

    if "primary_use" in df.columns:
        dummies = pd.get_dummies(df["primary_use"], prefix="use", drop_first=False)
        for c in dummies.columns:
            df[c] = dummies[c].values
        df.drop(columns=["primary_use"], inplace=True)


# ─── Data loaders ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading training data…")
def load_train():
    t_dtype = {"building_id": "int16", "meter": "int8", "meter_reading": "float32"}
    w_dtype = {"site_id": "int8", "air_temperature": "float32", "cloud_coverage": "float32",
               "dew_temperature": "float32", "precip_depth_1_hr": "float32",
               "sea_level_pressure": "float32", "wind_direction": "float32", "wind_speed": "float32"}
    m_dtype = {"site_id": "int8", "building_id": "int16", "square_feet": "int32"}

    train   = pd.read_csv(os.path.join(DATA_DIR, "train.csv"),
                          parse_dates=["timestamp"], dtype=t_dtype, nrows=2_000_000)
    weather = pd.read_csv(os.path.join(DATA_DIR, "weather_train.csv"),
                          parse_dates=["timestamp"], dtype=w_dtype)
    meta    = pd.read_csv(os.path.join(DATA_DIR, "building_metadata.csv"), dtype=m_dtype)

    df = (train
          .merge(meta,    on="building_id", how="left")
          .merge(weather, on=["site_id", "timestamp"], how="left")
          .drop_duplicates()
          .sort_values(["building_id", "meter", "timestamp"])
          .reset_index(drop=True))
    _engineer_features(df, is_test=False)
    return df


@st.cache_data(show_spinner="Loading test data…")
def load_test():
    w_dtype = {"site_id": "int8", "air_temperature": "float32", "cloud_coverage": "float32",
               "dew_temperature": "float32", "precip_depth_1_hr": "float32",
               "sea_level_pressure": "float32", "wind_direction": "float32", "wind_speed": "float32"}
    m_dtype = {"site_id": "int8", "building_id": "int16", "square_feet": "int32"}

    test    = pd.read_csv(os.path.join(DATA_DIR, "test.csv"),
                          parse_dates=["timestamp"], nrows=1_000_000)
    weather = pd.read_csv(os.path.join(DATA_DIR, "weather_test.csv"),
                          parse_dates=["timestamp"], dtype=w_dtype)
    meta    = pd.read_csv(os.path.join(DATA_DIR, "building_metadata.csv"), dtype=m_dtype)

    # Build lag proxy from train data — per building+meter averages
    train_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"),
                            usecols=["building_id", "meter", "meter_reading"],
                            dtype={"building_id": "int16", "meter": "int8",
                                   "meter_reading": "float32"})
    train_raw = train_raw[train_raw["meter_reading"] > 0]
    lag_proxy = (train_raw.groupby(["building_id", "meter"])["meter_reading"]
                 .agg(lag_1="mean", lag_24="mean", lag_168="mean",
                      rolling_mean_24="mean", rolling_mean_168="mean",
                      rolling_std_24="std")
                 .reset_index())

    df = (test
          .merge(meta,      on="building_id", how="left")
          .merge(weather,   on=["site_id", "timestamp"], how="left")
          .merge(lag_proxy, on=["building_id", "meter"], how="left")
          .drop_duplicates()
          .sort_values(["building_id", "meter", "timestamp"])
          .reset_index(drop=True))
    _engineer_features(df, is_test=True)
    return df


@st.cache_resource(show_spinner="Loading model artifacts…")
def load_artifacts():
    out = {"model": None, "feat_cols": None, "model_name": "Not trained", "results_df": None}

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
            out["model_name"] = fname.replace("_model.pkl", "").replace("_", " ").title()
            break

    res = os.path.join(MODELS_DIR, "model_results.pkl")
    if os.path.exists(res):
        out["results_df"] = joblib.load(res)

    return out


def safe_predict(model, feat_cols, subset_df):
    tmp = subset_df.reindex(columns=feat_cols, fill_value=0.0).fillna(0.0).astype(float)
    return pd.Series(np.clip(np.expm1(model.predict(tmp)), 0, None), index=tmp.index)


# ─── Load ─────────────────────────────────────────────────────────────────────
df         = load_train()
arts       = load_artifacts()
model      = arts["model"]
feat_cols  = arts["feat_cols"]
model_name = arts["model_name"]
results_df = arts["results_df"]

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ AI Energy Monitor")
    st.markdown("---")
    page = st.radio("Navigate", [
        "📊 Overview",
        "🏢 Building Forecast",
        "🔮 Test Set Predictions",
        "🚨 Anomaly Detection",
        "🎯 Live Prediction",
    ])
    st.markdown("---")
    st.markdown(f"**Train rows:** {len(df):,}")
    st.markdown(f"**Buildings:** {df['building_id'].nunique():,}")
    st.markdown(f"**Model:** `{model_name}`")
    st.markdown("---")
    if model is None:
        st.warning("No model found.\nRun `notebooks/feature_engineering.ipynb` first.")
    else:
        st.success("Model loaded ✓")
    st.caption("AI Energy Efficiency Monitoring · v2.0")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 Energy Consumption Overview")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Train Readings", f"{len(df):,}")
    c2.metric("Buildings",      f"{df['building_id'].nunique():,}")
    c3.metric("Sites",          f"{df['site_id'].nunique():,}")
    c4.metric("Avg kWh/hour",   f"{df['meter_reading'].mean():.1f}")
    c5.metric("Date Range",
              f"{df['timestamp'].min().strftime('%b %Y')} → "
              f"{df['timestamp'].max().strftime('%b %Y')}")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Avg kWh by Meter Type")
        avg_m = df.groupby("meter")["meter_reading"].mean().rename(index=METER_LABELS)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(avg_m.index.astype(str), avg_m.values,
               color=["#2196F3", "#26A69A", "#FF6F00", "#E53935"], edgecolor="white", width=0.55)
        ax.set_ylabel("kWh")
        st.pyplot(fig, width='stretch')
        plt.close()

    with col2:
        st.subheader("24-Hour Energy Profile")
        hourly = df.groupby("hour")["meter_reading"].mean()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(hourly.index, hourly.values, color="#E91E63", marker="o", linewidth=2)
        ax.fill_between(hourly.index, hourly.values, alpha=0.15, color="#E91E63")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Avg kWh")
        st.pyplot(fig, width='stretch')
        plt.close()

    st.markdown("---")
    st.subheader("Summary Statistics by Meter Type")
    summary = (df.groupby("meter")["meter_reading"]
               .agg(Readings="count", Mean="mean", Median="median",
                    Std="std", Max="max", Total="sum")
               .rename(index=METER_LABELS).round(2))
    summary.columns = ["# Readings", "Avg kWh", "Median kWh", "Std kWh", "Max kWh", "Total kWh"]
    st.dataframe(summary, width='stretch')

    if results_df is not None:
        st.markdown("---")
        st.subheader("Model Performance Leaderboard")
        lo_err = [c for c in ["MAE", "RMSE", "RMSLE"] if c in results_df.columns]
        hi_r2  = [c for c in ["R²"] if c in results_df.columns]
        styled = results_df.style
        if lo_err:
            styled = styled.highlight_min(subset=lo_err, color="#c8e6c9")
        if hi_r2:
            styled = styled.highlight_max(subset=hi_r2,  color="#c8e6c9")
        st.dataframe(styled, width='stretch')


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — BUILDING FORECAST
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏢 Building Forecast":
    st.title("🏢 Building Forecast")

    if model is None:
        st.error("No trained model found. Run `notebooks/feature_engineering.ipynb` first.")
    else:
        # Only offer buildings that have ≥ 24 non-zero readings for at least one meter
        valid_counts = (df[df["meter_reading"] > 0]
                        .groupby(["building_id", "meter"])
                        .size()
                        .reset_index(name="n"))
        valid_blds = sorted(valid_counts[valid_counts["n"] >= 24]["building_id"].unique())

        c1, c2 = st.columns(2)
        sel_bld = c1.selectbox("Building", valid_blds)

        # Only offer meters that have ≥ 24 non-zero readings for this building
        valid_mtrs = sorted(
            valid_counts[(valid_counts["building_id"] == sel_bld) &
                         (valid_counts["n"] >= 24)]["meter"].unique()
        )
        sel_mtr = c2.selectbox(
            "Meter", valid_mtrs,
            format_func=lambda m: METER_LABELS.get(m, str(m)),
        )

        bld_df = (df[(df["building_id"] == sel_bld) & (df["meter"] == sel_mtr)]
                  .dropna(subset=["log_target"])
                  .query("meter_reading > 0")
                  .sort_values("timestamp"))

        if len(bld_df) < 24:
            st.warning("Not enough data for this building/meter combination.")
        else:
            # last 7 days for chart
            last7  = bld_df.iloc[-168:]
            preds7 = safe_predict(model, feat_cols, last7)
            act7   = np.expm1(last7["log_target"].values)
            n      = min(len(preds7), len(act7))

            mae  = mean_absolute_error(act7[:n], preds7.values[:n])
            rmse = np.sqrt(mean_squared_error(act7[:n], preds7.values[:n]))
            r2   = r2_score(act7[:n], preds7.values[:n])

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("MAE",           f"{mae:.2f} kWh")
            k2.metric("RMSE",          f"{rmse:.2f} kWh")
            k3.metric("R²",            f"{r2:.4f}")
            k4.metric("Avg Predicted", f"{preds7.values[:n].mean():.2f} kWh")

            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(last7["timestamp"].values, act7,
                    label="Actual", color="#1E88E5", linewidth=1.8)
            ax.plot(last7.loc[preds7.index, "timestamp"].values, preds7.values,
                    label=f"{model_name} Predicted",
                    color="#E53935", linestyle="--", linewidth=1.8)
            ax.set_title(f"Building {sel_bld} · {METER_LABELS.get(sel_mtr, 'Meter')} — Last 7 Days")
            ax.set_ylabel("kWh")
            ax.legend()
            plt.xticks(rotation=20)
            plt.tight_layout()
            st.pyplot(fig, width='stretch')
            plt.close()

            # Last 24 h numeric table
            st.subheader("Hourly Predictions — Last 24 Hours")
            last24 = bld_df.iloc[-24:]
            p24    = safe_predict(model, feat_cols, last24)
            a24    = np.expm1(last24["log_target"].values)
            p24v   = p24.reindex(last24.index, fill_value=0.0).values
            mape   = np.where(a24 > 0.1,
                              np.round(np.abs(a24 - p24v) / a24 * 100, 1),
                              np.nan)
            tbl = pd.DataFrame({
                "Timestamp":     last24["timestamp"].dt.strftime("%Y-%m-%d %H:%M").values,
                "Actual kWh":    a24.round(2),
                "Predicted kWh": p24v.round(2),
                "Abs Error kWh": np.abs(a24 - p24v).round(2),
                "MAPE %":        mape,
            })
            st.dataframe(tbl.reset_index(drop=True), width='stretch')


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — TEST SET PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Test Set Predictions":
    st.title("🔮 Test Set Predictions")
    st.caption("Predictions generated from **test.csv** + **weather_test.csv** using the trained model.")

    if model is None or feat_cols is None:
        st.error("No trained model found. Run `notebooks/feature_engineering.ipynb` first.")
    else:
        with st.spinner("Running predictions on test.csv…"):
            test_df  = load_test()
            t_preds  = safe_predict(model, feat_cols, test_df)
            test_df["predicted_kwh"] = t_preds.reindex(test_df.index, fill_value=0.0).round(2)
            test_df["meter_label"]   = test_df["meter"].map(METER_LABELS)

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Test Rows",    f"{len(test_df):,}")
        k2.metric("Buildings",    f"{test_df['building_id'].nunique():,}")
        k3.metric("Avg kWh",      f"{test_df['predicted_kwh'].mean():.2f}")
        k4.metric("Max kWh",      f"{test_df['predicted_kwh'].max():.2f}")
        k5.metric("Total kWh",    f"{test_df['predicted_kwh'].sum():,.0f}")

        st.markdown("---")
        st.subheader("Predicted kWh Summary by Meter Type")
        msumm = (test_df.groupby("meter_label")["predicted_kwh"]
                 .agg(Rows="count", Mean="mean", Median="median", Max="max", Total="sum")
                 .round(2))
        msumm.columns = ["# Rows", "Avg Predicted kWh", "Median kWh", "Max kWh", "Total kWh"]
        st.dataframe(msumm, width='stretch')

        st.markdown("---")
        c1, c2 = st.columns(2)
        bld_opts      = ["All"] + [str(b) for b in sorted(test_df["building_id"].unique())]
        sel_bld       = c1.selectbox("Filter by Building", bld_opts, key="test_bld")
        mtr_opts      = ["All"] + sorted(test_df["meter_label"].dropna().unique().tolist())
        sel_mtr_label = c2.selectbox("Filter by Meter Type", mtr_opts, key="test_mtr")

        filt = test_df.copy()
        if sel_bld != "All":
            filt = filt[filt["building_id"] == int(sel_bld)]
        if sel_mtr_label != "All":
            filt = filt[filt["meter_label"] == sel_mtr_label]

        disp_cols = [c for c in ["building_id", "meter_label", "timestamp", "predicted_kwh",
                                  "air_temperature", "hour", "month"]
                     if c in filt.columns]
        filt_disp = filt[disp_cols].copy()
        filt_disp["timestamp"] = filt_disp["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        filt_disp = filt_disp.rename(columns={
            "building_id":    "Building",
            "meter_label":    "Meter Type",
            "timestamp":      "Timestamp",
            "predicted_kwh":  "Predicted kWh",
            "air_temperature": "Temp (°C)",
            "hour":           "Hour",
            "month":          "Month",
        })

        st.subheader(f"Row-Level Predictions ({len(filt_disp):,} rows — showing first 1,000)")
        st.dataframe(filt_disp.reset_index(drop=True).head(1_000), width='stretch')

        csv_bytes = filt_disp.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download filtered predictions as CSV",
                           csv_bytes, "test_predictions.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🚨 Anomaly Detection":
    st.title("🚨 Anomaly Detection")

    c_a, c_b = st.columns(2)
    sel_bld = c_a.selectbox("Building", sorted(df["building_id"].unique()), key="anom_bld")
    sel_mtr = c_b.selectbox(
        "Meter",
        sorted(df[df["building_id"] == sel_bld]["meter"].unique()),
        key="anom_mtr",
        format_func=lambda m: METER_LABELS.get(m, str(m)),
    )

    bld_check = df[(df["building_id"] == sel_bld) & (df["meter"] == sel_mtr)].copy()

    if len(bld_check) <= 10:
        st.warning("Not enough data points for this selection.")
    else:
        q1  = bld_check["meter_reading"].quantile(0.25)
        q3  = bld_check["meter_reading"].quantile(0.75)
        iqr = q3 - q1
        lo  = q1 - 1.5 * iqr
        hi  = q3 + 1.5 * iqr

        bld_check["status"] = "Normal"
        bld_check.loc[bld_check["meter_reading"] < lo, "status"] = "Low Outlier"
        bld_check.loc[bld_check["meter_reading"] > hi, "status"] = "High Outlier"

        n_out = (bld_check["status"] != "Normal").sum()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Readings", f"{len(bld_check):,}")
        k2.metric("Outliers",       f"{n_out:,}")
        k3.metric("Outlier Rate",   f"{n_out / len(bld_check) * 100:.1f}%")
        k4.metric("Normal Band",    f"{max(lo, 0):.1f} – {hi:.1f} kWh")

        fig, ax = plt.subplots(figsize=(14, 4))
        norm = bld_check[bld_check["status"] == "Normal"]
        out  = bld_check[bld_check["status"] != "Normal"]
        ax.scatter(norm["timestamp"], norm["meter_reading"],
                   s=2, alpha=0.4, color="#4CAF50", label="Normal")
        ax.scatter(out["timestamp"],  out["meter_reading"],
                   s=15, alpha=0.8, color="#E53935", label="Outlier", marker="X")
        ax.set_title(f"Building {sel_bld} · {METER_LABELS.get(sel_mtr, 'Meter')} — IQR Anomaly Detection")
        ax.set_ylabel("kWh")
        ax.legend()
        plt.xticks(rotation=15)
        plt.tight_layout()
        st.pyplot(fig, width='stretch')
        plt.close()

        st.subheader(f"Outlier Records ({n_out:,} rows)")
        out_disp = bld_check[bld_check["status"] != "Normal"][
            ["timestamp", "meter_reading", "status"]
        ].rename(columns={"meter_reading": "kWh"}).copy()
        out_disp["timestamp"] = out_disp["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(out_disp.reset_index(drop=True), width='stretch')


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — LIVE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Live Prediction":
    st.title("🎯 Live Prediction")
    st.caption("Select a building, set weather conditions and time, then hit **Predict** for an instant model inference.")

    if model is None or feat_cols is None:
        st.error("No trained model found. Run `notebooks/feature_engineering.ipynb` first.")
    else:
        @st.cache_data(show_spinner=False)
        def _live_meta():
            m_dtype = {"site_id": "int8", "building_id": "int16", "square_feet": "int32"}
            return pd.read_csv(os.path.join(DATA_DIR, "building_metadata.csv"), dtype=m_dtype)

        @st.cache_data(show_spinner=False)
        def _live_lag_stats():
            tr = pd.read_csv(
                os.path.join(DATA_DIR, "train.csv"),
                usecols=["building_id", "meter", "meter_reading"],
                dtype={"building_id": "int16", "meter": "int8", "meter_reading": "float32"},
            )
            tr = tr[tr["meter_reading"] > 0]
            return (
                tr.groupby(["building_id", "meter"])["meter_reading"]
                .agg(lag_1="mean", lag_24="mean", lag_168="mean",
                     rolling_mean_24="mean", rolling_mean_168="mean",
                     rolling_std_24="std")
                .reset_index()
            )

        meta_df  = _live_meta()
        lag_df   = _live_lag_stats()

        # ── Building & meter ──────────────────────────────────────────────────
        st.subheader("Building & Meter")
        bc1, bc2, bc3 = st.columns(3)
        all_blds  = sorted(meta_df["building_id"].unique().tolist())
        sel_bld   = bc1.selectbox("Building ID", all_blds, key="lv_bld")
        bld_meta  = meta_df[meta_df["building_id"] == sel_bld].iloc[0]

        avail_mtrs = sorted(
            df[df["building_id"] == sel_bld]["meter"].unique().tolist()
        ) if sel_bld in df["building_id"].values else [0, 1, 2, 3]
        sel_mtr = bc2.selectbox(
            "Meter Type", avail_mtrs,
            format_func=lambda m: METER_LABELS.get(m, str(m)),
            key="lv_mtr",
        )

        with bc3:
            yr  = bld_meta.get("year_built")
            fl  = bld_meta.get("floor_count")
            pu  = bld_meta.get("primary_use", "N/A")
            st.markdown(
                f"**Site:** {int(bld_meta['site_id'])}  \n"
                f"**Use:** {pu}  \n"
                f"**Sq ft:** {int(bld_meta['square_feet']):,}  \n"
                f"**Year:** {int(yr) if pd.notna(yr) else 'N/A'}  \n"
                f"**Floors:** {int(fl) if pd.notna(fl) else 'N/A'}"
            )

        st.markdown("---")

        # ── Date & time ───────────────────────────────────────────────────────
        st.subheader("Date & Time")
        dc1, dc2 = st.columns(2)
        pred_date = dc1.date_input("Date", value=pd.Timestamp("2016-06-15").date(), key="lv_date")
        pred_hour = dc2.slider("Hour of Day", 0, 23, 12, key="lv_hour",
                               format="%d:00")
        pred_ts   = pd.Timestamp(pred_date) + pd.Timedelta(hours=pred_hour)
        dc2.caption(f"Timestamp: **{pred_ts.strftime('%Y-%m-%d %H:%M')}**")

        st.markdown("---")

        # ── Weather inputs (default = site historical averages) ───────────────
        st.subheader("Weather Conditions")
        site_id = int(bld_meta["site_id"])
        site_wx = (
            df[df["site_id"] == site_id][
                ["air_temperature", "dew_temperature", "wind_speed",
                 "wind_direction", "cloud_coverage", "sea_level_pressure",
                 "precip_depth_1_hr"]
            ].mean()
            if "air_temperature" in df.columns
            else pd.Series({
                "air_temperature": 15.0, "dew_temperature": 10.0,
                "wind_speed": 3.0, "wind_direction": 180.0,
                "cloud_coverage": 4.0, "sea_level_pressure": 1013.0,
                "precip_depth_1_hr": 0.0,
            })
        )

        wc1, wc2, wc3, wc4 = st.columns(4)
        air_t  = wc1.number_input("Air Temp (°C)",           value=round(float(site_wx["air_temperature"]), 1),   step=0.5,  key="lv_airt")
        dew_t  = wc2.number_input("Dew Temp (°C)",           value=round(float(site_wx["dew_temperature"]), 1),   step=0.5,  key="lv_dewt")
        wnd_s  = wc3.number_input("Wind Speed (m/s)",        value=round(float(site_wx["wind_speed"]), 1),        step=0.5,  min_value=0.0, key="lv_wnds")
        wnd_d  = wc4.number_input("Wind Dir (°)",            value=round(float(site_wx["wind_direction"]), 0),    step=10.0, min_value=0.0, max_value=360.0, key="lv_wndd")

        wc5, wc6, wc7, _ = st.columns(4)
        cloud  = wc5.number_input("Cloud Cover (oktas)",     value=round(float(site_wx["cloud_coverage"]), 1),    step=1.0, min_value=0.0, max_value=9.0, key="lv_cld")
        slp    = wc6.number_input("Sea-Level Pressure (hPa)",value=round(float(site_wx["sea_level_pressure"]), 1), step=1.0, key="lv_slp")
        pcip   = wc7.number_input("Precipitation (mm/hr)",   value=0.0,                                          step=0.5, min_value=0.0, key="lv_pcp")

        st.markdown("---")

        # ── Lag features (pre-filled from training history) ───────────────────
        st.subheader("Historical Lag Features")
        st.caption("Pre-filled from training data averages for this building & meter. Adjust for scenario testing.")
        lag_row_df = lag_df[(lag_df["building_id"] == sel_bld) & (lag_df["meter"] == sel_mtr)]
        if len(lag_row_df):
            lr        = lag_row_df.iloc[0]
            d_l1      = round(float(lr["lag_1"]),   2)
            d_l24     = round(float(lr["lag_24"]),  2)
            d_l168    = round(float(lr["lag_168"]), 2)
            d_rm24    = round(float(lr["rolling_mean_24"]),  2)
            d_rm168   = round(float(lr["rolling_mean_168"]), 2)
            d_rs24    = round(float(lr["rolling_std_24"]) if pd.notna(lr["rolling_std_24"]) else 0.0, 2)
        else:
            d_l1 = d_l24 = d_l168 = d_rm24 = d_rm168 = d_rs24 = 0.0

        lc1, lc2, lc3 = st.columns(3)
        lag_1    = lc1.number_input("Lag 1h (kWh)",            value=d_l1,   step=1.0, min_value=0.0, key="lv_l1")
        lag_24   = lc2.number_input("Lag 24h (kWh)",           value=d_l24,  step=1.0, min_value=0.0, key="lv_l24")
        lag_168  = lc3.number_input("Lag 168h (kWh)",          value=d_l168, step=1.0, min_value=0.0, key="lv_l168")

        lc4, lc5, lc6 = st.columns(3)
        rm_24    = lc4.number_input("Rolling Mean 24h (kWh)",  value=d_rm24,  step=1.0, min_value=0.0, key="lv_rm24")
        rm_168   = lc5.number_input("Rolling Mean 168h (kWh)", value=d_rm168, step=1.0, min_value=0.0, key="lv_rm168")
        rs_24    = lc6.number_input("Rolling Std 24h (kWh)",   value=d_rs24,  step=0.1, min_value=0.0, key="lv_rs24")

        st.markdown("---")
        predict_clicked = st.button("⚡ Predict Energy Consumption", type="primary",
                                    use_container_width=True, key="lv_btn")

        if predict_clicked:
            med_yr = float(meta_df["year_built"].median())
            med_fl = float(meta_df["floor_count"].median())
            row = {
                "building_id":        sel_bld,
                "site_id":            int(bld_meta["site_id"]),
                "meter":              sel_mtr,
                "square_feet":        int(bld_meta["square_feet"]),
                "year_built":         float(bld_meta["year_built"]) if pd.notna(bld_meta.get("year_built")) else med_yr,
                "floor_count":        float(bld_meta["floor_count"]) if pd.notna(bld_meta.get("floor_count")) else med_fl,
                "air_temperature":    air_t,
                "dew_temperature":    dew_t,
                "wind_speed":         wnd_s,
                "wind_direction":     wnd_d,
                "cloud_coverage":     cloud,
                "sea_level_pressure": slp,
                "precip_depth_1_hr":  pcip,
                "lag_1":              lag_1,
                "lag_24":             lag_24,
                "lag_168":            lag_168,
                "rolling_mean_24":    rm_24,
                "rolling_mean_168":   rm_168,
                "rolling_std_24":     rs_24,
                "timestamp":          pred_ts,
                "primary_use":        str(bld_meta.get("primary_use", "Education")),
            }
            pred_row = pd.DataFrame([row])
            _engineer_features(pred_row, is_test=True)
            result_kwh = float(safe_predict(model, feat_cols, pred_row).iloc[0])

            # ── Result card ──────────────────────────────────────────────────
            st.markdown("---")
            _, mid, _ = st.columns([1, 2, 1])
            with mid:
                st.markdown(
                    f"""
                    <div style="background: linear-gradient(135deg, #1565C0, #AD1457);
                                border-radius: 16px; padding: 36px; text-align: center; color: white;">
                        <div style="font-size: 17px; opacity: 0.85; margin-bottom: 6px;">
                            Predicted Energy Consumption
                        </div>
                        <div style="font-size: 60px; font-weight: 800; letter-spacing: -1px; line-height: 1;">
                            {result_kwh:,.2f}
                        </div>
                        <div style="font-size: 24px; opacity: 0.80; margin-top: 4px;">kWh</div>
                        <div style="font-size: 13px; opacity: 0.65; margin-top: 14px;">
                            Building {sel_bld} &nbsp;·&nbsp;
                            {METER_LABELS.get(sel_mtr, 'Meter')} &nbsp;·&nbsp;
                            {pred_ts.strftime('%Y-%m-%d %H:%M')}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("")

            # ── Context metrics ──────────────────────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Predicted kWh", f"{result_kwh:,.2f}")
            if d_rm24 > 0:
                delta_pct = (result_kwh - d_rm24) / d_rm24 * 100
                m2.metric("vs 24h Rolling Avg", f"{d_rm24:,.2f} kWh",
                          delta=f"{delta_pct:+.1f}%")
            else:
                m2.metric("vs 24h Rolling Avg", "N/A")
            m3.metric("Meter Type", METER_LABELS.get(sel_mtr, str(sel_mtr)))
            m4.metric("Model", model_name)

            # ── Scenario comparison table ────────────────────────────────────
            st.subheader("Input Summary")
            summary_data = {
                "Parameter": [
                    "Building ID", "Meter", "Timestamp", "Air Temp (°C)",
                    "Dew Temp (°C)", "Wind Speed (m/s)", "Cloud Cover (oktas)",
                    "Sea-Level Pressure (hPa)", "Precipitation (mm/hr)",
                    "Lag 1h", "Lag 24h", "Rolling Mean 24h",
                ],
                "Value": [
                    sel_bld, METER_LABELS.get(sel_mtr, str(sel_mtr)),
                    pred_ts.strftime("%Y-%m-%d %H:%M"),
                    air_t, dew_t, wnd_s, cloud, slp, pcip,
                    lag_1, lag_24, rm_24,
                ],
            }
            st.dataframe(pd.DataFrame(summary_data), hide_index=True, width='stretch')
