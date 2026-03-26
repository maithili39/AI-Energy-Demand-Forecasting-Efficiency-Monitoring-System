"""
⚡ AI Energy Efficiency Monitoring System — Production v3.0
National Hackathon Edition
"""

import os, warnings, json, datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="⚡ AI Energy Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.metric-card {
    background: linear-gradient(135deg, #141926 0%, #1e2740 100%);
    border: 1px solid #2a3550;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover { transform: translateY(-3px); box-shadow: 0 8px 24px rgba(0,229,255,0.15); }
.metric-val { font-size: 2rem; font-weight: 800; color: #00E5FF; }
.metric-label { font-size: 0.78rem; color: #8892a4; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.08em; }
.hero-card {
    background: linear-gradient(135deg, #0a1628 0%, #1a2a4a 50%, #0d1f3c 100%);
    border: 1px solid #00E5FF33;
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
}
.section-header {
    font-size: 1.05rem; font-weight: 700; color: #00E5FF;
    text-transform: uppercase; letter-spacing: 0.1em;
    border-bottom: 2px solid #00E5FF33; padding-bottom: 6px; margin-bottom: 16px;
}
.badge-green { background:#0d3321; color:#00e676; border-radius:8px; padding:3px 10px; font-size:0.78rem; font-weight:600; }
.badge-red   { background:#3d1212; color:#ff5252; border-radius:8px; padding:3px 10px; font-size:0.78rem; font-weight:600; }
.badge-blue  { background:#0d1f3c; color:#40c4ff; border-radius:8px; padding:3px 10px; font-size:0.78rem; font-weight:600; }
stProgress > div > div > div { background: linear-gradient(90deg, #00E5FF, #7C4DFF); }
</style>
""", unsafe_allow_html=True)

# ─── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR   = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")

METER_LABELS  = {0: "Electricity", 1: "Chilled Water", 2: "Steam", 3: "Hot Water"}
METER_COLORS  = {0: "#00E5FF", 1: "#69F0AE", 2: "#FF6D00", 3: "#FF4081"}
CO2_FACTORS   = {0: 0.233, 1: 0.14,  2: 0.27,  3: 0.18}   # kg CO₂ / kWh
TARIFF_PER_KWH = 0.12   # USD — used for cost estimates

# ─── Feature Engineering ───────────────────────────────────────────────────────
def _engineer_features(df, is_test=False):
    weather_cont = ["air_temperature", "dew_temperature", "wind_speed", "wind_direction",
                    "cloud_coverage", "sea_level_pressure", "precip_depth_1_hr"]
    for col in weather_cont:
        if col in df.columns:
            df[col] = (df.groupby("site_id")[col]
                         .transform(lambda s: s.interpolate(method="linear", limit_direction="both")))
    if "floor_count" in df.columns:
        df["floor_count"] = df["floor_count"].fillna(df["floor_count"].median())
    if "year_built"  in df.columns:
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

    # Heating / Cooling Degree Days (industry-standard)
    BASE_TEMP = 18.0
    if "air_temperature" in df.columns:
        df["hdd"] = np.maximum(BASE_TEMP - df["air_temperature"], 0)
        df["cdd"] = np.maximum(df["air_temperature"] - BASE_TEMP, 0)
        df["feels_like"] = (df["air_temperature"]
                            - 0.4 * (df["air_temperature"] - 10)
                            * (1 - df.get("wind_speed", 0) / 200))

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

# ─── Data Loaders ──────────────────────────────────────────────────────────────
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
    df = (train.merge(meta, on="building_id", how="left")
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
    test    = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), parse_dates=["timestamp"], nrows=1_000_000)
    weather = pd.read_csv(os.path.join(DATA_DIR, "weather_test.csv"), parse_dates=["timestamp"], dtype=w_dtype)
    meta    = pd.read_csv(os.path.join(DATA_DIR, "building_metadata.csv"), dtype=m_dtype)
    train_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"),
                            usecols=["building_id", "meter", "meter_reading"],
                            dtype={"building_id": "int16", "meter": "int8", "meter_reading": "float32"})
    train_raw = train_raw[train_raw["meter_reading"] > 0]
    lag_proxy = (train_raw.groupby(["building_id", "meter"])["meter_reading"]
                 .agg(lag_1="mean", lag_24="mean", lag_168="mean",
                      rolling_mean_24="mean", rolling_mean_168="mean",
                      rolling_std_24="std").reset_index())
    df = (test.merge(meta, on="building_id", how="left")
              .merge(weather, on=["site_id", "timestamp"], how="left")
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
    for fname in ["lightgbm_tuned_model.pkl", "ensemble_model.pkl", "lightgbm_model.pkl",
                  "xgboost_model.pkl", "randomforest_model.pkl", "linearregression_model.pkl"]:
        p = os.path.join(MODELS_DIR, fname)
        if os.path.exists(p):
            out["model"]      = joblib.load(p)
            out["model_name"] = fname.replace("_model.pkl", "").replace("_", " ").title()
            break
    res = os.path.join(MODELS_DIR, "model_results.pkl")
    if os.path.exists(res):
        out["results_df"] = joblib.load(res)
    return out

@st.cache_data(show_spinner=False)
def _live_meta():
    return pd.read_csv(os.path.join(DATA_DIR, "building_metadata.csv"),
                       dtype={"site_id": "int8", "building_id": "int16", "square_feet": "int32"})

@st.cache_data(show_spinner=False)
def _live_lag_stats():
    tr = pd.read_csv(os.path.join(DATA_DIR, "train.csv"),
                     usecols=["building_id", "meter", "meter_reading"],
                     dtype={"building_id": "int16", "meter": "int8", "meter_reading": "float32"})
    tr = tr[tr["meter_reading"] > 0]
    return (tr.groupby(["building_id", "meter"])["meter_reading"]
              .agg(lag_1="mean", lag_24="mean", lag_168="mean",
                   rolling_mean_24="mean", rolling_mean_168="mean",
                   rolling_std_24="std").reset_index())

def safe_predict(model, feat_cols, subset_df):
    tmp = subset_df.reindex(columns=feat_cols, fill_value=0.0).fillna(0.0).astype(float)
    return pd.Series(np.clip(np.expm1(model.predict(tmp)), 0, None), index=tmp.index)

def load_registry():
    reg_path = os.path.join(MODELS_DIR, "model_registry.json")
    if os.path.exists(reg_path):
        with open(reg_path) as f:
            return json.load(f)
    return {}

def plotly_theme():
    return dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,25,38,0.8)",
        font=dict(family="Inter", color="#E8EAED"),
        margin=dict(l=12, r=12, t=40, b=12),
    )

# ─── Load ──────────────────────────────────────────────────────────────────────
df         = load_train()
arts       = load_artifacts()
model      = arts["model"]
feat_cols  = arts["feat_cols"]
model_name = arts["model_name"]
results_df = arts["results_df"]
registry   = load_registry()

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ AI Energy Monitor")
    st.markdown('<p style="color:#8892a4;font-size:0.78rem;margin-top:-12px;">Production v3.0 · Hackathon Edition</p>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigate", [
        "📊 Overview",
        "🏢 Building Forecast",
        "🌿 Carbon & Sustainability",
        "🏆 Building Benchmarking",
        "🔮 Multi-Horizon Forecast",
        "🔴 Test Set Predictions",
        "🚨 Anomaly Detection",
        "🎯 Live Prediction",
    ])
    st.markdown("---")
    st.markdown(f"**Train rows:** {len(df):,}")
    st.markdown(f"**Buildings:** {df['building_id'].nunique():,}")
    st.markdown(f"**Model:** `{model_name}`")
    if registry:
        versions = list(registry.keys())
        st.markdown(f"**Versions:** {len(versions)} saved")
    st.markdown("---")
    if model is None:
        st.warning("No model found.\nRun `notebooks/feature_engineering.ipynb`.")
    else:
        st.success("✅ Model loaded")
    st.caption("AI Energy Efficiency Monitor · v3.0")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown('<h1 style="background:linear-gradient(90deg,#00E5FF,#7C4DFF);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">📊 Energy Consumption Overview</h1>', unsafe_allow_html=True)

    total_kwh  = df["meter_reading"].sum()
    total_co2  = sum(df[df["meter"] == m]["meter_reading"].sum() * CO2_FACTORS.get(m, 0.2)
                     for m in df["meter"].unique()) / 1e6  # tonnes
    total_cost = total_kwh * TARIFF_PER_KWH / 1e6  # $M

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, label in [
        (c1, f"{len(df):,}", "Train Readings"),
        (c2, f"{df['building_id'].nunique():,}", "Buildings"),
        (c3, f"{df['meter_reading'].mean():.1f}", "Avg kWh/Hour"),
        (c4, f"{total_co2:,.1f}k t", "CO₂ Emitted (tonnes)"),
        (c5, f"${total_cost:.2f}M", "Est. Energy Cost"),
    ]:
        col.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">⚡ Avg kWh by Meter Type</div>', unsafe_allow_html=True)
        avg_m = df.groupby("meter")["meter_reading"].mean().rename(index=METER_LABELS).reset_index()
        avg_m.columns = ["Meter", "Avg kWh"]
        fig = px.bar(avg_m, x="Meter", y="Avg kWh",
                     color="Meter", color_discrete_sequence=list(METER_COLORS.values()),
                     text_auto=".1f")
        fig.update_layout(**plotly_theme(), showlegend=False, height=300)
        fig.update_traces(marker_line_width=0, textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">🕐 24-Hour Energy Profile</div>', unsafe_allow_html=True)
        hourly = df.groupby("hour")["meter_reading"].mean().reset_index()
        fig = px.area(hourly, x="hour", y="meter_reading",
                      color_discrete_sequence=["#00E5FF"])
        fig.update_layout(**plotly_theme(), height=300, xaxis_title="Hour", yaxis_title="Avg kWh")
        fig.update_traces(line_width=2.5, fillcolor="rgba(0,229,255,0.12)")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-header">📅 Monthly Trend</div>', unsafe_allow_html=True)
        monthly = df.groupby("month")["meter_reading"].mean().reset_index()
        monthly["month_name"] = monthly["month"].map({1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                                                       7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"})
        fig = px.line(monthly, x="month_name", y="meter_reading", markers=True,
                      color_discrete_sequence=["#FF4081"])
        fig.update_layout(**plotly_theme(), height=280, xaxis_title="Month", yaxis_title="Avg kWh")
        fig.update_traces(line_width=2)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header">🏗️ Energy by Building Type</div>', unsafe_allow_html=True)
        use_cols = [c for c in df.columns if c.startswith("use_")]
        if use_cols:
            use_avg = {c.replace("use_","").replace("_"," ").title():
                       df[df[c]==1]["meter_reading"].mean() for c in use_cols if df[c].sum() > 100}
            use_df  = pd.DataFrame(use_avg.items(), columns=["Type","Avg kWh"]).sort_values("Avg kWh")
            fig = px.bar(use_df, x="Avg kWh", y="Type", orientation="h",
                         color="Avg kWh", color_continuous_scale="Teal")
            fig.update_layout(**plotly_theme(), height=280, showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    if results_df is not None:
        st.markdown('<div class="section-header">🏅 Model Leaderboard</div>', unsafe_allow_html=True)
        lo_err = [c for c in ["MAE","RMSE","RMSLE"] if c in results_df.columns]
        hi_r2  = [c for c in ["R²"] if c in results_df.columns]
        styled = results_df.style
        if lo_err: styled = styled.highlight_min(subset=lo_err, color="#0d3321")
        if hi_r2:  styled = styled.highlight_max(subset=hi_r2,  color="#0d3321")
        st.dataframe(styled, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — BUILDING FORECAST
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏢 Building Forecast":
    st.title("🏢 Building Forecast")
    if model is None:
        st.error("No trained model found. Run `notebooks/feature_engineering.ipynb` first.")
    else:
        valid_counts = (df[df["meter_reading"] > 0]
                        .groupby(["building_id","meter"]).size().reset_index(name="n"))
        valid_blds = sorted(valid_counts[valid_counts["n"] >= 24]["building_id"].unique())
        c1, c2 = st.columns(2)
        sel_bld = c1.selectbox("Building", valid_blds)
        valid_mtrs = sorted(valid_counts[(valid_counts["building_id"]==sel_bld)
                                         & (valid_counts["n"]>=24)]["meter"].unique())
        sel_mtr = c2.selectbox("Meter", valid_mtrs, format_func=lambda m: METER_LABELS.get(m, str(m)))

        bld_df = (df[(df["building_id"]==sel_bld) & (df["meter"]==sel_mtr)]
                  .dropna(subset=["log_target"]).query("meter_reading > 0").sort_values("timestamp"))
        if len(bld_df) < 24:
            st.warning("Not enough data for this building/meter.")
        else:
            last7  = bld_df.iloc[-168:]
            preds7 = safe_predict(model, feat_cols, last7)
            act7   = np.expm1(last7["log_target"].values)
            n      = min(len(preds7), len(act7))
            mae  = mean_absolute_error(act7[:n], preds7.values[:n])
            rmse = np.sqrt(mean_squared_error(act7[:n], preds7.values[:n]))
            r2   = r2_score(act7[:n], preds7.values[:n])

            k1, k2, k3, k4 = st.columns(4)
            for col, lbl, val in [(k1,"MAE",f"{mae:.2f} kWh"),(k2,"RMSE",f"{rmse:.2f} kWh"),
                                   (k3,"R²",f"{r2:.4f}"),(k4,"Avg Predicted",f"{preds7.values[:n].mean():.2f} kWh")]:
                col.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=last7["timestamp"], y=act7, name="Actual",
                                     line=dict(color="#00E5FF", width=2)))
            fig.add_trace(go.Scatter(x=last7.loc[preds7.index, "timestamp"], y=preds7.values,
                                     name=f"{model_name} Predicted",
                                     line=dict(color="#FF4081", width=2, dash="dash")))
            # Confidence band ±10%
            fig.add_trace(go.Scatter(x=pd.concat([last7["timestamp"], last7["timestamp"][::-1]]),
                                     y=np.concatenate([preds7.values*1.10, preds7.values[::-1]*0.90]),
                                     fill="toself", fillcolor="rgba(255,64,129,0.08)",
                                     line=dict(color="rgba(0,0,0,0)"), name="±10% Band", showlegend=True))
            fig.update_layout(**plotly_theme(), height=360,
                              title=f"Building {sel_bld} · {METER_LABELS.get(sel_mtr,'Meter')} — Last 7 Days",
                              xaxis_title="Time", yaxis_title="kWh")
            st.plotly_chart(fig, use_container_width=True)

            # Last-24h table
            st.subheader("Hourly Predictions — Last 24 Hours")
            last24 = bld_df.iloc[-24:]
            p24    = safe_predict(model, feat_cols, last24)
            a24    = np.expm1(last24["log_target"].values)
            p24v   = p24.reindex(last24.index, fill_value=0.0).values
            mape   = np.where(a24>0.1, np.round(np.abs(a24-p24v)/a24*100, 1), np.nan)
            tbl = pd.DataFrame({
                "Timestamp":     last24["timestamp"].dt.strftime("%Y-%m-%d %H:%M").values,
                "Actual kWh":    a24.round(2),
                "Predicted kWh": p24v.round(2),
                "Abs Error kWh": np.abs(a24-p24v).round(2),
                "MAPE %":        mape,
            })
            st.dataframe(tbl.reset_index(drop=True), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CARBON & SUSTAINABILITY  ★ NEW
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌿 Carbon & Sustainability":
    st.markdown('<h1 style="background:linear-gradient(90deg,#00E676,#69F0AE);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">🌿 Carbon & Sustainability Dashboard</h1>', unsafe_allow_html=True)
    st.caption("Quantified environmental impact of buildings across the dataset.")

    # CO₂ per meter type
    co2_df = df.copy()
    co2_df["co2_kg"] = co2_df.apply(lambda r: r["meter_reading"] * CO2_FACTORS.get(r["meter"], 0.2), axis=1)
    co2_df["cost_usd"] = co2_df["meter_reading"] * TARIFF_PER_KWH

    total_co2_t   = co2_df["co2_kg"].sum() / 1e6
    total_co2_per = co2_df.groupby("building_id")["co2_kg"].sum().mean() / 1000
    savings_pct   = 15.0  # assumed potential with AI-driven optimization
    saved_co2     = total_co2_t * savings_pct / 100

    k1, k2, k3, k4 = st.columns(4)
    for col, val, lbl in [
        (k1, f"{total_co2_t:,.0f}k t", "Total CO₂ Emitted"),
        (k2, f"{total_co2_per:.1f} t", "Avg CO₂ / Building"),
        (k3, f"{saved_co2:,.1f}k t", "Potential CO₂ Saved"),
        (k4, f"${co2_df['cost_usd'].sum()/1e6:.2f}M", "Total Energy Cost"),
    ]:
        col.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">🌍 CO₂ Emissions by Meter Type</div>', unsafe_allow_html=True)
        co2_by_meter = co2_df.groupby("meter")["co2_kg"].sum().rename(index=METER_LABELS).reset_index()
        co2_by_meter.columns = ["Meter", "CO2_kg"]
        co2_by_meter["CO2_tonnes"] = co2_by_meter["CO2_kg"] / 1000
        fig = px.pie(co2_by_meter, values="CO2_tonnes", names="Meter",
                     color_discrete_sequence=["#00E676","#69F0AE","#1DE9B6","#00BFA5"],
                     hole=0.55)
        fig.update_layout(**plotly_theme(), height=320, showlegend=True)
        fig.update_traces(textinfo="percent+label", textfont_size=12)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">📈 Monthly CO₂ Trend</div>', unsafe_allow_html=True)
        monthly_co2 = co2_df.groupby("month")["co2_kg"].sum().reset_index()
        monthly_co2["co2_tonnes"] = monthly_co2["co2_kg"] / 1000
        monthly_co2["month_name"] = monthly_co2["month"].map(
            {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
             7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"})
        fig = px.bar(monthly_co2, x="month_name", y="co2_tonnes",
                     color="co2_tonnes", color_continuous_scale="Greens",
                     labels={"co2_tonnes": "CO₂ (tonnes)"})
        fig.update_layout(**plotly_theme(), height=320, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # Top 20 carbon-emitting buildings
    st.markdown('<div class="section-header">🏭 Top 20 Carbon-Emitting Buildings</div>', unsafe_allow_html=True)
    top_co2 = (co2_df.groupby("building_id")["co2_kg"].sum() / 1000).reset_index()
    top_co2.columns = ["building_id", "CO2_tonnes"]
    top_co2 = top_co2.nlargest(20, "CO2_tonnes")

    # Merge sq_ft if available
    if "square_feet" in co2_df.columns:
        sqft = co2_df.groupby("building_id")["square_feet"].first().reset_index()
        top_co2 = top_co2.merge(sqft, on="building_id", how="left")
        top_co2["CO2_per_sqft"] = (top_co2["CO2_tonnes"] * 1000 / top_co2["square_feet"]).round(3)

    fig = px.bar(top_co2, x="CO2_tonnes", y=top_co2["building_id"].astype(str),
                 orientation="h", color="CO2_tonnes",
                 color_continuous_scale="RdYlGn_r",
                 labels={"CO2_tonnes":"CO₂ (tonnes)", "y":"Building ID"})
    fig.update_layout(**plotly_theme(), height=420, coloraxis_showscale=False, yaxis_title="Building ID")
    st.plotly_chart(fig, use_container_width=True)

    # Decarbonization Trajectory
    st.markdown('<div class="section-header">🎯 Decarbonization Trajectory</div>', unsafe_allow_html=True)
    baseline = total_co2_t
    years    = list(range(2020, 2036))
    reduction_rate = 0.06  # 6% per year with AI optimization
    traj = [baseline * ((1 - reduction_rate) ** (y - 2020)) for y in years]
    net_zero_yr = next((y for y, v in zip(years, traj) if v < baseline * 0.1), 2035)
    traj_df = pd.DataFrame({"Year": years, "CO2_ktonnes": [t/1000 for t in traj]})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=traj_df["Year"], y=traj_df["CO2_ktonnes"],
                             mode="lines+markers", name="AI-Optimized Path",
                             line=dict(color="#00E676", width=3)))
    fig.add_trace(go.Scatter(x=traj_df["Year"], y=[baseline/1000]*len(years),
                             mode="lines", name="Business As Usual",
                             line=dict(color="#FF5252", width=2, dash="dash")))
    fig.add_hline(y=baseline/10000, line_dash="dot", line_color="#FFD740",
                  annotation_text="Net-Zero Target", annotation_position="bottom right")
    fig.update_layout(**plotly_theme(), height=340, xaxis_title="Year", yaxis_title="CO₂ (kilo-tonnes)")
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"🎯 With AI-driven 6%/year reduction, the portfolio approaches net-zero by **{net_zero_yr}**, saving an estimated **{saved_co2:,.0f}k tonnes CO₂**.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — BUILDING BENCHMARKING  ★ NEW
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Building Benchmarking":
    st.title("🏆 Building Benchmarking & Ranking")
    st.caption("Efficiency scores, peer comparisons, and top/bottom performers across 1,400+ buildings.")

    @st.cache_data(show_spinner="Computing efficiency scores…")
    def compute_scores(_df):
        grp = _df.groupby("building_id")
        scores = pd.DataFrame({
            "total_kwh":       grp["meter_reading"].sum(),
            "avg_kwh":         grp["meter_reading"].mean(),
            "std_kwh":         grp["meter_reading"].std(),
            "anomaly_pct":     grp.apply(lambda g: (
                (g["meter_reading"] > g["meter_reading"].quantile(0.95)).sum() / max(len(g), 1) * 100
            )),
        }).reset_index()
        if "square_feet" in _df.columns:
            sqft = _df.groupby("building_id")["square_feet"].first().reset_index()
            scores = scores.merge(sqft, on="building_id", how="left")
            scores["energy_intensity"] = scores["avg_kwh"] / scores["square_feet"] * 1000
        else:
            scores["energy_intensity"] = scores["avg_kwh"]
        # Normalize 0-100 (lower intensity = higher score)
        ei = scores["energy_intensity"]
        scores["efficiency_score"] = (100 * (1 - (ei - ei.min()) / (ei.max() - ei.min() + 1e-6))).round(1)
        return scores.sort_values("efficiency_score", ascending=False).reset_index(drop=True)

    scores_df = compute_scores(df)

    k1, k2, k3, k4 = st.columns(4)
    for col, val, lbl in [
        (k1, f"{scores_df['efficiency_score'].mean():.1f}/100", "Avg Efficiency Score"),
        (k2, f"{scores_df['efficiency_score'].max():.0f}", "Best Score"),
        (k3, f"{scores_df['efficiency_score'].min():.0f}", "Worst Score"),
        (k4, f"{(scores_df['efficiency_score']>=70).sum()}", "Buildings ≥70 Score"),
    ]:
        col.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">🏆 Top 15 Most Efficient Buildings</div>', unsafe_allow_html=True)
        top15 = scores_df.head(15)[["building_id","efficiency_score","avg_kwh","energy_intensity"]]
        top15.columns = ["Building", "Score", "Avg kWh", "Energy Intensity"]
        fig = px.bar(top15, x="Score", y=top15["Building"].astype(str), orientation="h",
                     color="Score", color_continuous_scale="Greens",
                     labels={"x":"Score","y":"Building"}, text="Score")
        fig.update_layout(**plotly_theme(), height=380, coloraxis_showscale=False, yaxis_title="Building ID")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">⚠️ Bottom 15 Least Efficient</div>', unsafe_allow_html=True)
        bot15 = scores_df.tail(15)[["building_id","efficiency_score","avg_kwh","energy_intensity"]]
        bot15.columns = ["Building", "Score", "Avg kWh", "Energy Intensity"]
        fig = px.bar(bot15, x="Score", y=bot15["Building"].astype(str), orientation="h",
                     color="Score", color_continuous_scale="Reds_r",
                     labels={"x":"Score","y":"Building"}, text="Score")
        fig.update_layout(**plotly_theme(), height=380, coloraxis_showscale=False, yaxis_title="Building ID")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">📊 Efficiency Score Distribution</div>', unsafe_allow_html=True)
    fig = px.histogram(scores_df, x="efficiency_score", nbins=40,
                       color_discrete_sequence=["#7C4DFF"],
                       labels={"efficiency_score":"Efficiency Score (0-100)"})
    fig.add_vline(x=scores_df["efficiency_score"].mean(), line_dash="dash", line_color="#FFD740",
                  annotation_text="Portfolio Average")
    fig.update_layout(**plotly_theme(), height=280)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">🔍 Full Building Scorecard</div>', unsafe_allow_html=True)
    min_s, max_s = st.slider("Filter by Efficiency Score", 0, 100, (0, 100), step=5)
    filtered = scores_df[(scores_df["efficiency_score"] >= min_s) & (scores_df["efficiency_score"] <= max_s)]
    cols_show = [c for c in ["building_id","efficiency_score","avg_kwh","energy_intensity","anomaly_pct","square_feet"] if c in filtered.columns]
    st.dataframe(filtered[cols_show].rename(columns={
        "building_id":"Building", "efficiency_score":"Score (0-100)",
        "avg_kwh":"Avg kWh", "energy_intensity":"Energy Intensity",
        "anomaly_pct":"Anomaly %", "square_feet":"Sq Ft"
    }).reset_index(drop=True), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MULTI-HORIZON FORECAST  ★ NEW
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Multi-Horizon Forecast":
    st.title("🔮 Multi-Horizon Forecast & What-If Analysis")
    st.caption("7-day and 30-day recursive forecasts with confidence bands and temperature sensitivity analysis.")
    if model is None or feat_cols is None:
        st.error("Run `notebooks/feature_engineering.ipynb` first.")
    else:
        valid_blds = sorted(df[df["meter_reading"]>0].groupby("building_id").size()[
            df[df["meter_reading"]>0].groupby("building_id").size()>=168].index)
        c1, c2, c3 = st.columns(3)
        sel_bld  = c1.selectbox("Building", valid_blds, key="mh_bld")
        valid_mt = sorted(df[df["building_id"]==sel_bld]["meter"].unique())
        sel_mtr  = c2.selectbox("Meter", valid_mt, format_func=lambda m: METER_LABELS.get(m,str(m)), key="mh_mtr")
        horizon  = c3.selectbox("Forecast Horizon", ["7 Days (168 hours)","30 Days (720 hours)"])
        n_hours  = 168 if "7" in horizon else 720

        st.markdown("---")
        st.markdown("#### 🌡️ What-If: Adjust Future Temperature")
        temp_delta = st.slider("Temperature Offset (°C vs historical avg)", -10.0, 10.0, 0.0, 0.5)

        bld_df = (df[(df["building_id"]==sel_bld)&(df["meter"]==sel_mtr)]
                  .dropna(subset=["log_target"]).query("meter_reading>0")
                  .sort_values("timestamp"))
        if len(bld_df) < 168:
            st.warning("Not enough data for multi-horizon forecast.")
        else:
            seed   = bld_df.iloc[-168:].copy()
            future_preds, future_ts, lower, upper = [], [], [], []
            roll_buf = list(np.expm1(seed["log_target"].values[-24:]))

            for h in range(n_hours):
                ts_h  = seed["timestamp"].iloc[-1] + pd.Timedelta(hours=h+1)
                mod   = seed.iloc[-1:].copy()
                mod["timestamp"] = ts_h
                mod["hour"]      = ts_h.hour
                mod["day"]       = ts_h.day
                mod["month"]     = ts_h.month
                mod["weekday"]   = ts_h.dayofweek
                mod["is_weekend"]= int(ts_h.dayofweek >= 5)
                mod["quarter"]   = ts_h.quarter
                mod["hour_sin"]  = np.sin(2*np.pi*ts_h.hour/24)
                mod["hour_cos"]  = np.cos(2*np.pi*ts_h.hour/24)
                if "air_temperature" in mod.columns and temp_delta != 0:
                    mod["air_temperature"] = mod["air_temperature"] + temp_delta
                    mod["hdd"] = max(18 - float(mod["air_temperature"].iloc[0]), 0)
                    mod["cdd"] = max(float(mod["air_temperature"].iloc[0]) - 18, 0)
                mod["lag_1"]           = roll_buf[-1]
                mod["lag_24"]          = roll_buf[-24] if len(roll_buf) >= 24 else roll_buf[0]
                mod["lag_168"]         = bld_df["meter_reading"].iloc[max(0,len(bld_df)-168+h-168)]\
                                         if h < 168 else roll_buf[-min(168,len(roll_buf))]
                mod["rolling_mean_24"] = np.mean(roll_buf[-24:])
                mod["rolling_std_24"]  = np.std(roll_buf[-24:]) if len(roll_buf)>1 else 0
                p = float(safe_predict(model, feat_cols, mod).iloc[0])
                sigma = np.std(roll_buf[-24:]) * 0.5 if len(roll_buf)>1 else p*0.1
                future_preds.append(p)
                future_ts.append(ts_h)
                lower.append(max(p - 1.96*sigma, 0))
                upper.append(p + 1.96*sigma)
                roll_buf.append(p)
                if len(roll_buf) > 200: roll_buf = roll_buf[-200:]

            actual_last = np.expm1(bld_df["log_target"].values[-72:])
            actual_ts   = bld_df["timestamp"].values[-72:]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=actual_ts, y=actual_last, name="Historical",
                                     line=dict(color="#00E5FF", width=2)))
            fig.add_trace(go.Scatter(x=future_ts, y=future_preds, name="Forecast",
                                     line=dict(color="#FF4081", width=2.5)))
            # Confidence band
            fig.add_trace(go.Scatter(
                x=future_ts + future_ts[::-1],
                y=upper + lower[::-1],
                fill="toself", fillcolor="rgba(255,64,129,0.12)",
                line=dict(color="rgba(0,0,0,0)"), name="95% CI"
            ))
            title_suffix = f" (+{temp_delta}°C)" if temp_delta != 0 else ""
            fig.update_layout(**plotly_theme(), height=420,
                              title=f"Building {sel_bld} · {METER_LABELS.get(sel_mtr,'Meter')} — {horizon}{title_suffix}",
                              xaxis_title="Time", yaxis_title="kWh")
            st.plotly_chart(fig, use_container_width=True)

            tot = sum(future_preds)
            k1, k2, k3, k4 = st.columns(4)
            for col, val, lbl in [
                (k1, f"{np.mean(future_preds):.2f}", "Avg Predicted kWh"),
                (k2, f"{max(future_preds):.2f}", "Peak kWh"),
                (k3, f"{tot:,.0f}", "Total Forecasted kWh"),
                (k4, f"${tot*TARIFF_PER_KWH:,.2f}", "Estimated Cost (USD)"),
            ]:
                col.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if temp_delta != 0:
                direction = "increase" if temp_delta > 0 else "decrease"
                pct_diff = (np.mean(future_preds) / np.expm1(bld_df["log_target"].values[-72:]).mean() - 1)*100
                st.info(f"🌡️ A **{abs(temp_delta):.1f}°C {direction}** in temperature results in a **{pct_diff:+.1f}%** change in average energy demand.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — TEST SET PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔴 Test Set Predictions":
    st.title("🔴 Test Set Predictions")
    st.caption("Predictions on **test.csv** + **weather_test.csv** using the trained model.")
    if model is None or feat_cols is None:
        st.error("No trained model found. Run `notebooks/feature_engineering.ipynb` first.")
    else:
        with st.spinner("Running predictions on test.csv…"):
            test_df = load_test()
            t_preds = safe_predict(model, feat_cols, test_df)
            test_df["predicted_kwh"] = t_preds.reindex(test_df.index, fill_value=0.0).round(2)
            test_df["meter_label"]   = test_df["meter"].map(METER_LABELS)
            test_df["co2_kg"]        = test_df.apply(lambda r: r["predicted_kwh"]*CO2_FACTORS.get(r["meter"],0.2), axis=1)

        k1,k2,k3,k4,k5 = st.columns(5)
        for col,val,lbl in [
            (k1,f"{len(test_df):,}","Test Rows"),
            (k2,f"{test_df['building_id'].nunique():,}","Buildings"),
            (k3,f"{test_df['predicted_kwh'].mean():.2f}","Avg kWh"),
            (k4,f"{test_df['predicted_kwh'].max():.2f}","Max kWh"),
            (k5,f"{test_df['co2_kg'].sum()/1000:,.0f} t","Est. CO₂ (t)"),
        ]:
            col.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-header">📊 Predicted kWh Distribution</div>', unsafe_allow_html=True)
            fig = px.histogram(test_df[test_df["predicted_kwh"]<test_df["predicted_kwh"].quantile(0.99)],
                               x="predicted_kwh", nbins=60, color="meter_label",
                               color_discrete_map={v:METER_COLORS[k] for k,v in METER_LABELS.items()})
            fig.update_layout(**plotly_theme(), height=300, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown('<div class="section-header">📋 Summary by Meter Type</div>', unsafe_allow_html=True)
            msumm = (test_df.groupby("meter_label")["predicted_kwh"]
                     .agg(Rows="count", Mean="mean", Median="median", Max="max", Total="sum").round(2))
            msumm.columns = ["# Rows","Avg kWh","Median kWh","Max kWh","Total kWh"]
            st.dataframe(msumm, use_container_width=True)

        c1, c2 = st.columns(2)
        bld_opts = ["All"]+[str(b) for b in sorted(test_df["building_id"].unique())]
        sel_bld  = c1.selectbox("Filter by Building", bld_opts, key="test_bld")
        mtr_opts = ["All"]+sorted(test_df["meter_label"].dropna().unique().tolist())
        sel_ml   = c2.selectbox("Filter by Meter", mtr_opts, key="test_mtr")
        filt = test_df.copy()
        if sel_bld != "All": filt = filt[filt["building_id"]==int(sel_bld)]
        if sel_ml  != "All": filt = filt[filt["meter_label"]==sel_ml]
        disp_cols = [c for c in ["building_id","meter_label","timestamp","predicted_kwh","air_temperature","hour","month"] if c in filt.columns]
        filt_disp = filt[disp_cols].copy()
        filt_disp["timestamp"] = filt_disp["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        st.subheader(f"Row-Level Predictions ({len(filt_disp):,} rows — first 1,000)")
        st.dataframe(filt_disp.reset_index(drop=True).head(1000), use_container_width=True)
        st.download_button("⬇ Download CSV", filt_disp.to_csv(index=False).encode(),
                           "test_predictions.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚨 Anomaly Detection":
    st.title("🚨 Anomaly Detection")
    st.caption("IQR-based statistical anomaly detection with interactive time-series visualization.")

    c_a, c_b = st.columns(2)
    sel_bld = c_a.selectbox("Building", sorted(df["building_id"].unique()), key="anom_bld")
    sel_mtr = c_b.selectbox("Meter", sorted(df[df["building_id"]==sel_bld]["meter"].unique()),
                             key="anom_mtr", format_func=lambda m: METER_LABELS.get(m,str(m)))
    bld_check = df[(df["building_id"]==sel_bld)&(df["meter"]==sel_mtr)].copy()

    if len(bld_check) <= 10:
        st.warning("Not enough data points.")
    else:
        q1 = bld_check["meter_reading"].quantile(0.25)
        q3 = bld_check["meter_reading"].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
        bld_check["status"] = "Normal"
        bld_check.loc[bld_check["meter_reading"]<lo, "status"] = "Low Outlier"
        bld_check.loc[bld_check["meter_reading"]>hi, "status"] = "High Outlier"
        n_out = (bld_check["status"]!="Normal").sum()

        k1,k2,k3,k4 = st.columns(4)
        for col,val,lbl in [
            (k1,f"{len(bld_check):,}","Total Readings"),
            (k2,f"{n_out:,}","Anomalies Detected"),
            (k3,f"{n_out/len(bld_check)*100:.1f}%","Anomaly Rate"),
            (k4,f"{max(lo,0):.1f}–{hi:.1f}","Normal Band (kWh)"),
        ]:
            col.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        normal  = bld_check[bld_check["status"]=="Normal"]
        outlier = bld_check[bld_check["status"]!="Normal"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=normal["timestamp"], y=normal["meter_reading"],
                                 mode="markers", name="Normal",
                                 marker=dict(color="#00E676", size=3, opacity=0.5)))
        fig.add_trace(go.Scatter(x=outlier["timestamp"], y=outlier["meter_reading"],
                                 mode="markers", name="Anomaly",
                                 marker=dict(color="#FF5252", size=8, symbol="x")))
        fig.add_hline(y=hi, line_dash="dash", line_color="#FFD740",
                      annotation_text="Upper Bound")
        fig.add_hline(y=max(lo,0), line_dash="dash", line_color="#FFD740",
                      annotation_text="Lower Bound")
        fig.update_layout(**plotly_theme(), height=400,
                          title=f"Building {sel_bld} · {METER_LABELS.get(sel_mtr,'Meter')} — IQR Anomaly Detection",
                          xaxis_title="Time", yaxis_title="kWh")
        st.plotly_chart(fig, use_container_width=True)

        if n_out > 0:
            st.subheader(f"Anomaly Records ({n_out:,} rows)")
            out_disp = bld_check[bld_check["status"]!="Normal"][["timestamp","meter_reading","status"]].copy()
            out_disp["timestamp"] = out_disp["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
            out_disp.columns = ["Timestamp","kWh","Status"]
            st.dataframe(out_disp.reset_index(drop=True), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — LIVE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Live Prediction":
    st.title("🎯 Live Prediction")
    st.caption("Select a building, configure weather & time, and get instant AI-powered energy prediction.")
    if model is None or feat_cols is None:
        st.error("No trained model found. Run `notebooks/feature_engineering.ipynb` first.")
    else:
        meta_df = _live_meta()
        lag_df  = _live_lag_stats()

        st.markdown("#### 🏢 Building & Meter")
        bc1, bc2, bc3 = st.columns(3)
        all_blds = sorted(meta_df["building_id"].unique().tolist())
        sel_bld  = bc1.selectbox("Building ID", all_blds, key="lv_bld")
        bld_meta = meta_df[meta_df["building_id"]==sel_bld].iloc[0]
        avail_mt = sorted(df[df["building_id"]==sel_bld]["meter"].unique().tolist()) \
                   if sel_bld in df["building_id"].values else [0,1,2,3]
        sel_mtr  = bc2.selectbox("Meter Type", avail_mt, format_func=lambda m:METER_LABELS.get(m,str(m)), key="lv_mtr")
        with bc3:
            yr = bld_meta.get("year_built"); fl = bld_meta.get("floor_count"); pu = bld_meta.get("primary_use","N/A")
            st.markdown(f"**Site:** {int(bld_meta['site_id'])}  \n**Use:** {pu}  \n"
                        f"**Sq ft:** {int(bld_meta['square_feet']):,}  \n"
                        f"**Year:** {int(yr) if pd.notna(yr) else 'N/A'}  \n"
                        f"**Floors:** {int(fl) if pd.notna(fl) else 'N/A'}")

        st.markdown("---")
        st.markdown("#### 📅 Date & Time")
        dc1, dc2 = st.columns(2)
        pred_date = dc1.date_input("Date", value=pd.Timestamp("2016-06-15").date(), key="lv_date")
        pred_hour = dc2.slider("Hour of Day", 0, 23, 12, key="lv_hour", format="%d:00")
        pred_ts   = pd.Timestamp(pred_date) + pd.Timedelta(hours=pred_hour)
        dc2.caption(f"Timestamp: **{pred_ts.strftime('%Y-%m-%d %H:%M')}**")

        st.markdown("---")
        st.markdown("#### 🌤️ Weather Conditions")
        site_id = int(bld_meta["site_id"])
        site_wx = (df[df["site_id"]==site_id][["air_temperature","dew_temperature","wind_speed",
                                                "wind_direction","cloud_coverage","sea_level_pressure","precip_depth_1_hr"]].mean()
                   if "air_temperature" in df.columns else
                   pd.Series({"air_temperature":15.0,"dew_temperature":10.0,"wind_speed":3.0,
                               "wind_direction":180.0,"cloud_coverage":4.0,"sea_level_pressure":1013.0,"precip_depth_1_hr":0.0}))

        wc1,wc2,wc3,wc4 = st.columns(4)
        air_t = wc1.number_input("Air Temp (°C)",       value=round(float(site_wx["air_temperature"]),1),    step=0.5,  key="lv_airt")
        dew_t = wc2.number_input("Dew Temp (°C)",       value=round(float(site_wx["dew_temperature"]),1),    step=0.5,  key="lv_dewt")
        wnd_s = wc3.number_input("Wind Speed (m/s)",    value=round(float(site_wx["wind_speed"]),1),         step=0.5,  min_value=0.0, key="lv_wnds")
        wnd_d = wc4.number_input("Wind Dir (°)",        value=round(float(site_wx["wind_direction"]),0),     step=10.0, min_value=0.0, max_value=360.0, key="lv_wndd")
        wc5,wc6,wc7,_ = st.columns(4)
        cloud = wc5.number_input("Cloud Cover (oktas)", value=round(float(site_wx["cloud_coverage"]),1),     step=1.0,  min_value=0.0, max_value=9.0, key="lv_cld")
        slp   = wc6.number_input("Sea-Level P (hPa)",   value=round(float(site_wx["sea_level_pressure"]),1), step=1.0,  key="lv_slp")
        pcip  = wc7.number_input("Precipitation (mm)",  value=0.0,                                           step=0.5,  min_value=0.0, key="lv_pcp")

        st.markdown("---")
        st.markdown("#### 🧠 Historical Lag Features")
        lag_row = lag_df[(lag_df["building_id"]==sel_bld)&(lag_df["meter"]==sel_mtr)]
        if len(lag_row):
            lr   = lag_row.iloc[0]
            d_l1 = round(float(lr["lag_1"]),2); d_l24=round(float(lr["lag_24"]),2)
            d_l168=round(float(lr["lag_168"]),2); d_rm24=round(float(lr["rolling_mean_24"]),2)
            d_rm168=round(float(lr["rolling_mean_168"]),2)
            d_rs24=round(float(lr["rolling_std_24"]) if pd.notna(lr["rolling_std_24"]) else 0.0,2)
        else:
            d_l1=d_l24=d_l168=d_rm24=d_rm168=d_rs24=0.0
        lc1,lc2,lc3 = st.columns(3)
        lag_1   = lc1.number_input("Lag 1h (kWh)",           value=d_l1,   step=1.0, min_value=0.0, key="lv_l1")
        lag_24  = lc2.number_input("Lag 24h (kWh)",          value=d_l24,  step=1.0, min_value=0.0, key="lv_l24")
        lag_168 = lc3.number_input("Lag 168h (kWh)",         value=d_l168, step=1.0, min_value=0.0, key="lv_l168")
        lc4,lc5,lc6 = st.columns(3)
        rm_24   = lc4.number_input("Rolling Mean 24h (kWh)", value=d_rm24,  step=1.0, min_value=0.0, key="lv_rm24")
        rm_168  = lc5.number_input("Rolling Mean 168h (kWh)",value=d_rm168, step=1.0, min_value=0.0, key="lv_rm168")
        rs_24   = lc6.number_input("Rolling Std 24h (kWh)",  value=d_rs24,  step=0.1, min_value=0.0, key="lv_rs24")

        st.markdown("---")
        predict_clicked = st.button("⚡ Predict Energy Consumption", type="primary",
                                    use_container_width=True, key="lv_btn")
        if predict_clicked:
            med_yr = float(meta_df["year_built"].median()); med_fl = float(meta_df["floor_count"].median())
            row = {
                "building_id": sel_bld, "site_id": int(bld_meta["site_id"]), "meter": sel_mtr,
                "square_feet": int(bld_meta["square_feet"]),
                "year_built":  float(bld_meta["year_built"]) if pd.notna(bld_meta.get("year_built")) else med_yr,
                "floor_count": float(bld_meta["floor_count"]) if pd.notna(bld_meta.get("floor_count")) else med_fl,
                "air_temperature": air_t, "dew_temperature": dew_t, "wind_speed": wnd_s,
                "wind_direction": wnd_d, "cloud_coverage": cloud, "sea_level_pressure": slp,
                "precip_depth_1_hr": pcip, "lag_1": lag_1, "lag_24": lag_24, "lag_168": lag_168,
                "rolling_mean_24": rm_24, "rolling_mean_168": rm_168, "rolling_std_24": rs_24,
                "timestamp": pred_ts, "primary_use": str(bld_meta.get("primary_use","Education")),
            }
            pred_row = pd.DataFrame([row])
            _engineer_features(pred_row, is_test=True)
            result_kwh = float(safe_predict(model, feat_cols, pred_row).iloc[0])
            co2_kg     = result_kwh * CO2_FACTORS.get(sel_mtr, 0.2)
            cost_usd   = result_kwh * TARIFF_PER_KWH
            sigma_est  = d_rs24 * 0.5 if d_rs24 > 0 else result_kwh * 0.08
            lo_ci, hi_ci = max(result_kwh - 1.96*sigma_est, 0), result_kwh + 1.96*sigma_est

            st.markdown("---")
            _, mid, _ = st.columns([1,2,1])
            with mid:
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#0d1f3c,#1a2a4a,#0d3321);
                            border:1px solid #00E5FF44;border-radius:20px;padding:36px;text-align:center;color:white;">
                    <div style="font-size:0.9rem;opacity:0.7;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.1em">
                        Predicted Energy Consumption
                    </div>
                    <div style="font-size:4rem;font-weight:800;color:#00E5FF;line-height:1;">
                        {result_kwh:,.2f}
                    </div>
                    <div style="font-size:1.3rem;opacity:0.7;margin-top:4px;">kWh</div>
                    <div style="margin-top:16px;font-size:0.82rem;opacity:0.55;">
                        Building {sel_bld} &nbsp;·&nbsp; {METER_LABELS.get(sel_mtr,'Meter')} &nbsp;·&nbsp; {pred_ts.strftime('%Y-%m-%d %H:%M')}
                    </div>
                    <div style="margin-top:12px;font-size:0.82rem;color:#8892a4;">
                        95% CI: [{lo_ci:,.2f} — {hi_ci:,.2f}] kWh
                    </div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            m1,m2,m3,m4 = st.columns(4)
            for col,val,lbl in [
                (m1,f"{result_kwh:,.2f} kWh","Predicted"),
                (m2,f"{co2_kg:.2f} kg","CO₂ Emitted"),
                (m3,f"${cost_usd:.4f}","Est. Cost (USD)"),
                (m4,model_name,"Model"),
            ]:
                col.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)
            if d_rm24 > 0:
                delta_pct = (result_kwh - d_rm24) / d_rm24 * 100
                st.metric("vs 24h Rolling Average", f"{d_rm24:,.2f} kWh", delta=f"{delta_pct:+.1f}%")

            st.subheader("Input Summary")
            params = {"Building":sel_bld,"Meter":METER_LABELS.get(sel_mtr,str(sel_mtr)),
                      "Timestamp":pred_ts.strftime("%Y-%m-%d %H:%M"),
                      "Air Temp (°C)":air_t,"Dew Temp (°C)":dew_t,"Wind Speed":wnd_s,
                      "Cloud Cover":cloud,"Pressure":slp,"Precipitation":pcip,
                      "Lag 1h":lag_1,"Lag 24h":lag_24,"Rolling Mean 24h":rm_24}
            st.dataframe(pd.DataFrame(params.items(), columns=["Parameter","Value"]),
                         hide_index=True, use_container_width=True)

