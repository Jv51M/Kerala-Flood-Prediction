"""
app.py  —  Streamlit Frontend
Flood Prediction System – Thrissur, Kerala

Multi-page app:
  Landing → Predict Flood / Report Flood / View Reports / Emergency Help
"""

import sqlite3
import os
import math
import streamlit as st
import requests
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from datetime import datetime
from streamlit_js_eval import get_geolocation

# ─── config ──────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"
DB_PATH  = os.path.join(os.path.dirname(__file__), "flood_reports.db")

st.set_page_config(
    page_title="Flood Prediction — Thrissur",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── SQLite ──────────────────────────────────────────────────────────────────
def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS flood_reports (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            latitude      REAL    NOT NULL,
            longitude     REAL    NOT NULL,
            location_name TEXT,
            severity      TEXT    NOT NULL,
            description   TEXT,
            reported_at   TEXT    NOT NULL
        )
    """)
    con.commit()
    con.close()

init_db()

def insert_report(lat, lon, name, severity, desc):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO flood_reports (latitude, longitude, location_name, severity, description, reported_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (lat, lon, name, severity, desc, datetime.now().isoformat(timespec="seconds"))
    )
    con.commit()
    con.close()

def get_all_reports():
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM flood_reports ORDER BY reported_at DESC", con
    )
    con.close()
    return df

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# ─── API helpers ─────────────────────────────────────────────────────────────
def api_post(path, payload):
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=120)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach the backend. Is FastAPI running on port 8000?"
    except requests.exceptions.HTTPError as e:
        try:    detail = e.response.json().get("detail", str(e))
        except: detail = str(e)
        return None, detail
    except Exception as e:
        return None, str(e)

def api_get(path, params=None):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=30)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach the backend. Is FastAPI running on port 8000?"
    except Exception as e:
        return None, str(e)

# ─── colour helpers ───────────────────────────────────────────────────────────
def prob_color(p):
    if p < 0.25: return "#4caf50"
    if p < 0.50: return "#ff9800"
    if p < 0.75: return "#f44336"
    return "#9c27b0"

def risk_badge(level):
    return f'<span class="badge-{level.lower()}">{level}</span>'

def point_color(p):
    if p < 0.25: return "#00e676"
    if p < 0.50: return "#ffab00"
    if p < 0.75: return "#ff1744"
    return "#d500f9"

SEVERITY_COLOR = {
    "Low":      "#4caf50",
    "Moderate": "#ff9800",
    "High":     "#f44336",
    "Extreme":  "#9c27b0",
}

# ─── session state nav ───────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"
if "result" not in st.session_state:
    st.session_state.result = None
if "pred_error" not in st.session_state:
    st.session_state.pred_error = None

def go(page):
    st.session_state.page = page

# ─── Global CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap');

* { font-family: 'Inter', sans-serif !important; }
.stApp { background: linear-gradient(160deg, #050d18, #0d1b2a, #091525) !important; }

/* ── NAV CARD ── */
.nav-card {
    background: linear-gradient(135deg, rgba(0,188,212,0.12), rgba(124,77,255,0.08));
    border: 1px solid rgba(0,188,212,0.25);
    border-radius: 20px;
    padding: 2rem 1.6rem;
    text-align: center;
    cursor: pointer;
    transition: transform .2s, box-shadow .2s, border-color .2s;
    backdrop-filter: blur(14px);
    min-height: 200px;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
}
.nav-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0,188,212,0.25);
    border-color: rgba(0,188,212,0.55);
}
.nav-card .nav-icon { font-size: 3rem; margin-bottom: .7rem; }
.nav-card .nav-title { color: #e0f7fa; font-size: 1.15rem; font-weight: 700; margin-bottom: .35rem; }
.nav-card .nav-desc  { color: #78909c; font-size: .85rem; line-height: 1.4; }

/* ── HERO ── */
.hero-wrap {
    text-align: center;
    padding: 3.5rem 1rem 2rem;
}
.hero-wrap h1 {
    font-size: 3rem; font-weight: 900;
    background: linear-gradient(90deg, #00bcd4, #7c4dff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 .5rem;
}
.hero-subtitle { color: #78909c; font-size: 1.05rem; margin-bottom: 2.5rem; }

/* ── PREDICT page ── */
.hero-card {
    background: linear-gradient(135deg, rgba(0,150,200,0.15), rgba(100,0,200,0.1));
    border: 1px solid rgba(0,188,212,0.3);
    border-radius: 16px; padding: 1.6rem 2rem; margin-bottom: 1.2rem;
    backdrop-filter: blur(10px);
}
.hero-card h2 { color: #e0f7fa; margin: 0 0 .3rem 0; font-size: 1.05rem;
                text-transform: uppercase; letter-spacing: .08em; }
.hero-prob   { font-size: 3.2rem; font-weight: 900; line-height: 1.1; }

.metric-row  { display: flex; gap: 1rem; margin-bottom: 1rem; flex-wrap: wrap; }
.metric-box  { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1);
               border-radius: 12px; padding: .9rem 1.2rem; flex: 1; min-width: 130px; }
.metric-box .label { color: #78909c; font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; }
.metric-box .val   { color: #e0f7fa; font-size: 1.5rem; font-weight: 700; margin-top: .2rem; }

.model-row { display: flex; gap: .8rem; margin-bottom: 1.2rem; flex-wrap: wrap; }
.model-box { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.09);
             border-radius: 12px; padding: .9rem 1.1rem; flex: 1; min-width: 150px; text-align: center; }
.model-box .mtitle { color: #90caf9; font-size: .8rem; text-transform: uppercase; letter-spacing: .06em; }
.model-box .mprob  { font-size: 2rem; font-weight: 800; margin: .3rem 0; }
.model-box .mpred  { font-size: .85rem; }

.badge-low      { background:#1b5e20; color:#a5d6a7; padding:3px 11px; border-radius:20px; font-weight:600; font-size:.85rem; }
.badge-moderate { background:#e65100; color:#ffe0b2; padding:3px 11px; border-radius:20px; font-weight:600; font-size:.85rem; }
.badge-high     { background:#b71c1c; color:#ffcdd2; padding:3px 11px; border-radius:20px; font-weight:600; font-size:.85rem; }
.badge-extreme  { background:#4a148c; color:#e1bee7; padding:3px 11px; border-radius:20px; font-weight:600; font-size:.85rem; }

/* ── REPORT page ── */
.report-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px; padding: 1.8rem;
}

/* ── VIEW REPORTS page ── */
.report-row {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 12px; padding: 1rem 1.3rem; margin-bottom: .7rem;
    display: flex; align-items: center; gap: 1rem;
}
.rr-sev { font-size: 1.4rem; }
.rr-main { flex: 1; }
.rr-loc  { color: #e0f7fa; font-size: .95rem; font-weight: 600; }
.rr-desc { color: #78909c; font-size: .82rem; margin-top: .15rem; }
.rr-date { color: #546e7a; font-size: .78rem; text-align: right; min-width: 140px; }

/* ── EMERGENCY page ── */
.em-card {
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1rem;
    display: flex; align-items: flex-start; gap: 1.2rem;
}
.em-icon  { font-size: 2.5rem; flex-shrink: 0; }
.em-body  {}
.em-title { color: #e0f7fa; font-size: 1.05rem; font-weight: 700; }
.em-num   { font-size: 1.6rem; font-weight: 900; letter-spacing: .04em; margin: .2rem 0; }
.em-desc  { color: #78909c; font-size: .82rem; }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] { background: rgba(0,0,0,0.4) !important; }
section[data-testid="stSidebar"] * { color: #eceff1 !important; }

/* ── Typography ── */
h1, h2, h3 { color: #e0f7fa !important; }
p, li, label { color: #b0bec5 !important; }
.stProgress > div > div { background: linear-gradient(90deg,#00bcd4,#7c4dff) !important; }
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ── BACK BTN ── */
.back-btn { margin-bottom: 1.2rem; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  PAGE: LANDING
# ════════════════════════════════════════════════════════════════════════════
def show_landing():
    st.markdown("""
    <div class="hero-wrap">
        <div style="font-size:4rem">🌊</div>
        <h1>Kerala Flood Watch</h1>
        <p class="hero-subtitle">Real-time flood prediction & community reporting for Thrissur, Kerala</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4, gap="large")

    with c1:
        st.markdown("""
        <div class="nav-card">
            <div class="nav-icon">🔍</div>
            <div class="nav-title">Predict Flood</div>
            <div class="nav-desc">AI-powered flood risk prediction using RF, LSTM & GNN models for any location</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open →", key="btn_predict", width='stretch', type="primary"):
            go("predict")
            st.rerun()

    with c2:
        st.markdown("""
        <div class="nav-card">
            <div class="nav-icon">📝</div>
            <div class="nav-title">Report Flood</div>
            <div class="nav-desc">Submit a community flood report so others can see current conditions</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open →", key="btn_report", width='stretch', type="primary"):
            go("report")
            st.rerun()

    with c3:
        st.markdown("""
        <div class="nav-card">
            <div class="nav-icon">🗺️</div>
            <div class="nav-title">View Reports</div>
            <div class="nav-desc">Browse all community flood reports on an interactive map with filters</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open →", key="btn_view", width='stretch', type="primary"):
            go("view_reports")
            st.rerun()

    with c4:
        st.markdown("""
        <div class="nav-card">
            <div class="nav-icon">🚨</div>
            <div class="nav-title">Emergency Help</div>
            <div class="nav-desc">Critical contact numbers for Fireforce, NDRF, police & flood rescue teams</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open →", key="btn_emergency", width='stretch', type="primary"):
            go("emergency")
            st.rerun()

    st.markdown("---")
    st.markdown(
        '<p style="text-align:center; color:#37474f; font-size:.8rem">'
        'Built with FastAPI · Streamlit · Folium · Open-Meteo · sklearn · TensorFlow · PyTorch'
        '</p>',
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════
#  PAGE: PREDICT FLOOD
# ════════════════════════════════════════════════════════════════════════════
def show_predict():
    # ── Sidebar controls ─────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🌊 Flood Predictor")
        st.markdown("*Thrissur, Kerala — Real-time*")
        st.divider()

        presets = {
            "Custom":                None,
            "Thrissur City Centre":  (10.5276, 76.2144),
            "Chalakudy":             (10.2999, 76.3311),
            "Irinjalakuda":          (10.3435, 76.2101),
            "Guruvayur":             (10.5944, 76.0460),
            "Kodungallur":           (10.2324, 76.1991),
            "Kunnamkulam":           (10.6521, 76.0705),
        }
        preset = st.selectbox("📍 Quick-select", list(presets.keys()))
        dlat, dlon = presets[preset] if presets[preset] else (10.5276, 76.2144)

        st.divider()
        st.markdown("#### 📍 GPS Location")
        user_loc = get_geolocation()
        if user_loc:
            if st.button("🛰️ Use Current location", width='stretch'):
                st.session_state["lat_input"] = float(user_loc['coords']['latitude'])
                st.session_state["lon_input"] = float(user_loc['coords']['longitude'])
                st.rerun()
        else:
            st.caption("Waiting for GPS permission...")

        lat = st.number_input("Latitude",  value=dlat, format="%.6f", step=0.001, key="lat_input")
        lon = st.number_input("Longitude", value=dlon, format="%.6f", step=0.001, key="lon_input")
        radius_km  = st.slider("Search radius (km)", 0.5, 10.0, 3.0, 0.5)

        st.divider()
        st.markdown("#### Map display")
        show_heatmap  = st.checkbox("Heatmap layer",      value=True)
        show_markers  = st.checkbox("Point markers",      value=True)
        show_user     = st.checkbox("Show your location", value=True)
        show_reports  = st.checkbox("Community reports",  value=True, help="Overlay flood reports within 3 km")

        st.divider()
        st.markdown("#### ⛈️ Heavy Rain Simulation")
        sim_enabled = st.toggle(
            "Enable simulation",
            value=False,
            help="Override live rainfall and recompute risk scores on the map."
        )
        sim_rain_mm = 0.0
        if sim_enabled:
            sim_rain_mm = st.slider(
                "Simulated rainfall (mm/day)",
                min_value=0.0, max_value=400.0,
                value=150.0, step=5.0,
                help="150 mm = heavy rain; 300+ mm = extreme flood scenario"
            )
            st.caption(f"🌧️ Simulating **{sim_rain_mm:.0f} mm/day** — live rainfall ignored on map")

        st.divider()
        hdata, herr = api_get("/health")
        if herr:
            st.error(f"⚠️ API offline\n{herr}")
        else:
            st.success("✅ API online")
            for name, ok in hdata.get("models_loaded", {}).items():
                st.markdown(f"{'✅' if ok else '❌'} {name.replace('_',' ').title()}")
            st.caption(f"RF rows: {hdata.get('rf_dataset_rows',0):,}")

        st.divider()
        run_btn = st.button("🔍 Run Prediction", type="primary", width='stretch')
        st.divider()
        if st.button("🏠 Back to Home", width='stretch'):
            go("home")
            st.rerun()

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown("# 🌊 Flood Prediction System")
    st.caption(f"{datetime.now().strftime('%A, %d %B %Y  ·  %H:%M IST')}")

    tab_map, tab_table, tab_rain, tab_about = st.tabs(
        ["🗺️ Heatmap", "📊 Point Data", "🌧️ Rainfall History", "ℹ️ About"]
    )

    if run_btn:
        with st.spinner("Fetching rainfall & running all models on every nearby point …"):
            payload = {
                "latitude":  lat,
                "longitude": lon,
                "radius_km": radius_km,
            }
            if sim_enabled:
                payload["simulated_rainfall_mm"] = sim_rain_mm
            result, err = api_post("/predict", payload)
        st.session_state.result     = result
        st.session_state.pred_error = err

    # ── TAB 1: HEATMAP ────────────────────────────────────────────────────────
    with tab_map:
        if st.session_state.pred_error:
            st.error(f"**Error:** {st.session_state.pred_error}")

        elif st.session_state.result is None:
            m = folium.Map(location=[lat, lon], zoom_start=13, tiles="CartoDB dark_matter")
            folium.Marker(
                [lat, lon], tooltip="Your location",
                icon=folium.Icon(color="blue", icon="crosshairs", prefix="fa"),
            ).add_to(m)
            st.markdown(
                '<div style="color:#90a4ae; margin-bottom:.6rem">'
                'Set your location in the sidebar and click <strong>Run Prediction</strong>.'
                '</div>', unsafe_allow_html=True
            )
            st_folium(m, width="100%", height=520, returned_objects=[])

        else:
            result = st.session_state.result
            points = result["points"]
            n      = result["point_count"]

            def _sim_risk(p):
                if p < 0.25: return "LOW"
                if p < 0.50: return "MODERATE"
                if p < 0.75: return "HIGH"
                return "EXTREME"

            def _sim_point(pt):
                if not sim_enabled: return pt
                rain_factor  = min(sim_rain_mm / 200.0, 2.0)
                def _boost(p):
                    if p is None: return None
                    b = p + (1.0 - p) * rain_factor * 0.65
                    return round(min(b, 0.99), 4)
                boosted_ens = _boost(pt["ensemble_probability"])
                return {**pt,
                        "ensemble_probability": boosted_ens,
                        "rf_probability":       _boost(pt.get("rf_probability")),
                        "lstm_probability":     _boost(pt.get("lstm_probability")),
                        "gnn_probability":      _boost(pt.get("gnn_probability")),
                        "flood_predicted":      boosted_ens >= 0.5,
                        "risk_level":           _sim_risk(boosted_ens),
                        "rainfall_today_mm":    sim_rain_mm}

            disp_points = [_sim_point(pt) for pt in points]

            ens   = sum(p["ensemble_probability"] for p in disp_points) / n if n else 0.0
            risk  = _sim_risk(ens)
            flood = ens >= 0.5
            color = prob_color(ens)

            if result["random_forest"]["available"]:
                v = [p["rf_probability"] for p in disp_points if p.get("rf_probability") is not None]
                result["random_forest"]["mean_probability"] = sum(v)/len(v) if v else 0.0
                result["random_forest"]["flood_predicted"] = result["random_forest"]["mean_probability"] >= 0.5

            if result["lstm"]["available"]:
                v = [p["lstm_probability"] for p in disp_points if p.get("lstm_probability") is not None]
                result["lstm"]["mean_probability"] = sum(v)/len(v) if v else 0.0
                result["lstm"]["flood_predicted"] = result["lstm"]["mean_probability"] >= 0.5

            if result["gnn"]["available"]:
                v = [p["gnn_probability"] for p in disp_points if p.get("gnn_probability") is not None]
                result["gnn"]["mean_probability"] = sum(v)/len(v) if v else 0.0
                result["gnn"]["flood_predicted"] = result["gnn"]["mean_probability"] >= 0.5

            rain_info   = result.get("rainfall", {})
            rain_today  = rain_info.get("today_mm", 0.0)
            rain_total  = rain_info.get("total_15day_mm", 0.0)
            rain_source = rain_info.get("source", "unknown")
            source_icon = {"open_meteo": "🌧️", "dataset_fallback": "📊", "regional_mean": "📌"}.get(rain_source, "❓")
            if sim_enabled:
                rain_today = sim_rain_mm
                rain_source = "simulation"
                source_icon = "⛈️"

            st.markdown(f"""
            <div class="hero-card">
                <h2>Ensemble Flood Assessment — {n} points within {result['radius_km']} km</h2>
                <div class="hero-prob" style="color:{color}">{ens*100:.1f}%</div>
                <div style="margin:.5rem 0">
                    {'🔴 FLOOD LIKELY' if flood else '🟢 LOW RISK'}
                    &nbsp;&nbsp;{risk_badge(risk)}
                    &nbsp;&nbsp;<span style="color:#78909c; font-size:.9rem">
                        {source_icon} Rain today: {rain_today} mm &nbsp;·&nbsp;
                        15-day total: {rain_total} mm &nbsp;·&nbsp; source: {rain_source}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            def mbox(title, icon, summ):
                if not summ["available"]:
                    return (f'<div class="model-box"><div class="mtitle">{icon} {title}</div>'
                            f'<div style="color:#546e7a; margin-top:.4rem">Not loaded</div></div>')
                p    = summ["mean_probability"]
                c    = prob_color(p)
                pred = "🔴 Flood" if summ["flood_predicted"] else "🟢 No Flood"
                err  = f'<div style="color:#ef9a9a;font-size:.7rem">{summ["error"][:60]}</div>' \
                       if summ.get("error") else ""
                return (f'<div class="model-box"><div class="mtitle">{icon} {title}</div>'
                        f'<div class="mprob" style="color:{c}">{p*100:.1f}%</div>'
                        f'<div class="mpred">{pred}</div>{err}</div>')

            st.markdown(
                '<div class="model-row">'
                + mbox("Random Forest", "🌲", result["random_forest"])
                + mbox("LSTM",          "🔁", result["lstm"])
                + mbox("GNN",           "🕸️",  result["gnn"])
                + '</div>',
                unsafe_allow_html=True,
            )
            st.progress(ens)
            st.divider()

            center_lat = np.mean([p["latitude"]  for p in points])
            center_lon = np.mean([p["longitude"] for p in points])

            m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="CartoDB dark_matter")

            if sim_enabled:
                st.warning(
                    f"⛈️ **Simulation active** — displaying results for **{sim_rain_mm:.0f} mm/day** "
                    f"artificial rainfall. Live weather data is not used for this view."
                )

            if show_heatmap:
                heat_data = [
                    [p["latitude"], p["longitude"], p["ensemble_probability"]]
                    for p in disp_points
                ]
                HeatMap(
                    heat_data, min_opacity=0.35, max_zoom=18, radius=28, blur=20,
                    gradient={
                        "0.0":  "#00e676",
                        "0.25": "#ffab00",
                        "0.5":  "#ff5722",
                        "0.75": "#d500f9",
                        "1.0":  "#b71c1c",
                    },
                    name="Flood Risk Heatmap",
                ).add_to(m)

            if show_markers:
                for pt in disp_points:
                    ens_pt    = pt["ensemble_probability"]
                    is_flood  = pt["flood_predicted"]
                    clr       = point_color(ens_pt)
                    rain_lbl  = (f"{sim_rain_mm:.0f} mm ⚡ simulated"
                                 if sim_enabled else f"{pt['rainfall_today_mm']} mm")
                    popup_html = f"""
                    <div style="font-family:monospace; font-size:12px; min-width:220px">
                        <b style="font-size:13px">{'🔴 FLOOD RISK' if is_flood else '🟢 NO FLOOD'}</b>
                        <hr style="margin:4px 0">
                        <b>Ensemble prob:</b> {ens_pt*100:.1f}% — <b>{pt['risk_level']}</b><br>
                        <b>RF prob:</b>   {f"{pt['rf_probability']*100:.1f}%"   if pt['rf_probability']   is not None else 'N/A'}<br>
                        <b>LSTM prob:</b> {f"{pt['lstm_probability']*100:.1f}%" if pt['lstm_probability'] is not None else 'N/A'}<br>
                        <b>GNN prob:</b>  {f"{pt['gnn_probability']*100:.1f}%"  if pt['gnn_probability']  is not None else 'N/A'}<br>
                        <hr style="margin:4px 0">
                        <b>Elevation:</b>    {pt['elevation']} m<br>
                        <b>Slope:</b>        {pt['slope']}°<br>
                        <b>Dist to river:</b>{pt['dist_to_river']} m<br>
                        <b>Rain today:</b>   {rain_lbl}<br>
                        <b>Distance:</b>     {pt['distance_km']} km<br>
                        <b>Coords:</b>       {pt['latitude']:.5f}, {pt['longitude']:.5f}
                    </div>
                    """
                    if is_flood:
                        # Flood → solid filled circle
                        folium.CircleMarker(
                            location=[pt["latitude"], pt["longitude"]],
                            radius=7, color=clr, fill=True, fill_color=clr,
                            fill_opacity=0.90, weight=2,
                            popup=folium.Popup(popup_html, max_width=260),
                            tooltip=f"🔴 FLOOD {pt['risk_level']} ({ens_pt*100:.0f}%)",
                        ).add_to(m)
                    else:
                        # No-flood → hollow circle with cross marker using DivIcon
                        folium.Marker(
                            location=[pt["latitude"], pt["longitude"]],
                            popup=folium.Popup(popup_html, max_width=260),
                            tooltip=f"🟢 NO FLOOD ({ens_pt*100:.0f}%)",
                            icon=folium.DivIcon(
                                html=f"""
                                <div style="
                                    width:14px; height:14px;
                                    border: 2.5px solid {clr};
                                    border-radius: 3px;
                                    background: rgba(0,0,0,0.35);
                                    margin-top:-7px; margin-left:-7px;
                                "></div>""",
                                icon_size=(14, 14),
                                icon_anchor=(7, 7),
                            ),
                        ).add_to(m)

            # ── Community reports overlay ────────────────────────────────────
            if show_reports:
                reports_df = get_all_reports()
                nearby = reports_df[
                    reports_df.apply(
                        lambda row: haversine_km(lat, lon, row["latitude"], row["longitude"]) <= 3.0,
                        axis=1
                    )
                ] if not reports_df.empty else pd.DataFrame()

                for _, row in nearby.iterrows():
                    sev_clr = SEVERITY_COLOR.get(row["severity"], "#ff9800")
                    popup_html = f"""
                    <div style="font-family:sans-serif; font-size:12px; min-width:200px">
                        <b style="color:#ff7043">🚨 Community Report</b><hr style="margin:4px 0">
                        <b>Location:</b> {row['location_name'] or 'Unknown'}<br>
                        <b>Severity:</b> {row['severity']}<br>
                        <b>Description:</b> {row['description'] or '—'}<br>
                        <b>Reported:</b> {row['reported_at']}
                    </div>
                    """
                    folium.CircleMarker(
                        location=[row["latitude"], row["longitude"]],
                        radius=10,
                        color=sev_clr,
                        fill=True,
                        fill_color=sev_clr,
                        fill_opacity=0.7,
                        weight=2.5,
                        popup=folium.Popup(popup_html, max_width=250),
                        tooltip=f"⚠️ Report: {row['severity']} — {row['location_name'] or 'Unknown'}",
                    ).add_to(m)

                if not nearby.empty:
                    st.info(f"🚨 **{len(nearby)} community report(s)** within 3 km of your location shown on map.")

            if show_user:
                folium.Marker(
                    [lat, lon], tooltip="Your location",
                    icon=folium.Icon(color="blue", icon="crosshairs", prefix="fa"),
                ).add_to(m)
                folium.Circle(
                    location=[lat, lon], radius=radius_km * 1000,
                    color="#00bcd4", fill=False, weight=2, dash_array="6 4",
                    tooltip=f"{radius_km} km radius",
                ).add_to(m)

            sim_tag = " ⛈️ SIM" if sim_enabled else ""
            legend_html = f"""
            <div style="position:fixed; bottom:30px; left:30px; z-index:9999;
                        background:rgba(13,27,42,0.92); border:1px solid #37474f;
                        border-radius:10px; padding:12px 16px; font-family:sans-serif;
                        font-size:12px; color:#eceff1; min-width:180px">
                <b style="font-size:13px">Risk Level{sim_tag}</b><br>
                <span style="color:#00e676">●</span> LOW      (&lt;25%)<br>
                <span style="color:#ffab00">●</span> MODERATE (25–50%)<br>
                <span style="color:#ff1744">●</span> HIGH     (50–75%)<br>
                <span style="color:#d500f9">●</span> EXTREME  (&gt;75%)<br>
                <hr style="border-color:#37474f; margin:6px 0">
                <b style="font-size:12px">Point type</b><br>
                <span style="color:#eceff1">● filled</span> = Flood predicted<br>
                <span style="color:#eceff1">▢ hollow</span> = No flood<br>
                <hr style="border-color:#37474f; margin:6px 0">
                <b style="font-size:12px">Community Reports</b><br>
                <span style="color:#4caf50">●</span> Low &nbsp;
                <span style="color:#ff9800">●</span> Moderate<br>
                <span style="color:#f44336">●</span> High &nbsp;
                <span style="color:#9c27b0">●</span> Extreme
            </div>
            """
            m.get_root().html.add_child(folium.Element(legend_html))
            folium.LayerControl().add_to(m)

            flood_pts = sum(1 for p in disp_points if p["flood_predicted"])
            st.markdown(
                f"**{n} data points** predicted · "
                f"🔴 **{flood_pts} flood** · 🟢 **{n-flood_pts} safe** · "
                f"click any marker for details"
            )
            st_folium(m, width="100%", height=570, returned_objects=[])

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total points",    n)
            c2.metric("Flood predicted", flood_pts,
                      delta=f"{flood_pts/n*100:.0f}% of area", delta_color="inverse")
            c3.metric("Safe points",     n - flood_pts)
            c4.metric("Max probability",
                      f"{max(p['ensemble_probability'] for p in disp_points)*100:.1f}%")

    # ── TAB 2: POINT DATA ────────────────────────────────────────────────────
    with tab_table:
        if st.session_state.result is None:
            st.info("Run a prediction first.")
        else:
            result = st.session_state.result
            points = result["points"]
            df = pd.DataFrame(points).rename(columns={
                "latitude":             "Lat",
                "longitude":            "Lon",
                "distance_km":          "Dist (km)",
                "elevation":            "Elev (m)",
                "slope":                "Slope (°)",
                "dist_to_river":        "Dist River (m)",
                "rainfall_today_mm":    "Rain (mm)",
                "rf_probability":       "RF Prob",
                "lstm_probability":     "LSTM Prob",
                "gnn_probability":      "GNN Prob",
                "ensemble_probability": "Ensemble",
                "flood_predicted":      "Flood?",
                "risk_level":           "Risk",
            })

            st.markdown(f"### All {len(df)} predicted points")
            filt_col, sort_col = st.columns(2)
            with filt_col:
                risk_filter = st.multiselect(
                    "Filter by risk level",
                    ["LOW", "MODERATE", "HIGH", "EXTREME"],
                    default=["LOW", "MODERATE", "HIGH", "EXTREME"],
                )
            with sort_col:
                sort_by = st.selectbox("Sort by", ["Ensemble", "Dist (km)", "Rain (mm)", "Elev (m)"])

            df_display = df[df["Risk"].isin(risk_filter)].sort_values(sort_by, ascending=False)

            def highlight_risk(val):
                if val < 0.25: return "background-color:#1b5e20; color:#a5d6a7"
                if val < 0.50: return "background-color:#bf360c; color:#ffe0b2"
                if val < 0.75: return "background-color:#b71c1c; color:#ffcdd2"
                return                "background-color:#4a148c; color:#e1bee7"

            styled = (
                df_display.style
                .format({
                    "Lat": "{:.5f}", "Lon": "{:.5f}",
                    "Dist (km)": "{:.3f}", "Elev (m)": "{:.0f}",
                    "Slope (°)": "{:.2f}", "Dist River (m)": "{:.0f}",
                    "Rain (mm)": "{:.1f}",
                    "RF Prob":   lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A",
                    "LSTM Prob": lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A",
                    "GNN Prob":  lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A",
                    "Ensemble":  "{:.1%}",
                })
                .applymap(highlight_risk, subset=["Ensemble"])
            )
            st.dataframe(styled, width='stretch', height=480)

            csv = df_display.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download CSV", data=csv,
                file_name=f"flood_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )

    # ── TAB 3: RAINFALL ──────────────────────────────────────────────────────
    with tab_rain:
        st.markdown("### 15-Day Rainfall History")
        st.markdown("Live data from Open-Meteo (free API, no key required).")
        rc1, rc2 = st.columns(2)
        with rc1: r_lat = st.number_input("Latitude",  value=lat, key="rl",  format="%.6f")
        with rc2: r_lon = st.number_input("Longitude", value=lon, key="rlo", format="%.6f")

        if st.button("🌧️ Fetch Rainfall", key="rain_btn"):
            with st.spinner("Calling Open-Meteo …"):
                data, err = api_get("/rainfall-history", params={"lat": r_lat, "lon": r_lon})
            if err:
                st.error(err)
            else:
                seq   = data["rainfall_sequence_mm"]
                today = data["today_mm"]
                total = data["total_15day_mm"]
                m1, m2, m3 = st.columns(3)
                m1.metric("Today",        f"{today:.1f} mm")
                m2.metric("15-Day Total", f"{total:.1f} mm")
                m3.metric("Daily Avg",    f"{total/15:.1f} mm")
                days   = list(range(14, -1, -1))
                labels = [f"D-{d}" if d > 0 else "Today" for d in days]
                rdf    = pd.DataFrame({"Day": labels, "Rainfall (mm)": seq})
                st.bar_chart(rdf.set_index("Day"), color="#00bcd4")

    # ── TAB 4: ABOUT ────────────────────────────────────────────────────────
    with tab_about:
        st.markdown("""
        ### How It Works

        Every geographic data point within the search radius gets its own flood probability
        from all three models. The results are plotted as a **Folium heatmap** so you can
        see *where exactly* flood risk is concentrated, not just a single number.

        #### Data flow
        ```
        User location  →  Haversine filter  →  N nearby points (RF dataset, 126k rows)
                                                   │
                                         Open-Meteo API (free)
                                         15-day daily rainfall per point
                                                   │
                                   ┌────────────────┼────────────────┐
                                RF model         LSTM model       GNN model
                              (terrain +       (15-day seq +    (spatial graph +
                               today rain)      terrain)         terrain + rain)
                                   │                │                │
                                   └────────────────┴────────────────┘
                                               Ensemble average
                                               per-point probability
                                                   │
                                              Folium heatmap
        ```

        #### Models
        | Model | Features | Strength |
        |-------|----------|----------|
        | **Random Forest** | elevation, slope, dist_water, landcover, lat, lon, rainfall | Fast, interpretable |
        | **LSTM** | 15-day rainfall sequence + elevation, slope, dist_river | Temporal patterns |
        | **GNN** | elevation, slope, dist_river, precipitation + spatial graph | Spatial propagation |

        #### Risk Levels
        | Level    | Probability |
        |----------|-------------|
        | 🟢 LOW      | 0 – 25%  |
        | 🟠 MODERATE | 25 – 50% |
        | 🔴 HIGH     | 50 – 75% |
        | 🟣 EXTREME  | 75 – 100% |
        """)
        st.caption("Built with FastAPI · Streamlit · Folium · Open-Meteo · sklearn · TensorFlow · PyTorch")


# ════════════════════════════════════════════════════════════════════════════
#  PAGE: REPORT FLOOD
# ════════════════════════════════════════════════════════════════════════════
def show_report():
    if st.button("← Back to Home", key="back_report"):
        go("home")
        st.rerun()

    st.markdown("## 📝 Report a Flood")
    st.markdown("Help your community by submitting what you are witnessing on the ground.")

    col_form, col_info = st.columns([2, 1], gap="large")

    PRESET_LOCATIONS = {
        "Custom (enter manually)": None,
        "Thrissur City Centre":    (10.5276, 76.2144),
        "Chalakudy":               (10.2999, 76.3311),
        "Irinjalakuda":            (10.3435, 76.2101),
        "Guruvayur":               (10.5944, 76.0460),
        "Kodungallur":             (10.2324, 76.1991),
        "Kunnamkulam":             (10.6521, 76.0705),
        "Wadakkanchery":           (10.6519, 76.2175),
        "Chavakkad":               (10.5578, 76.0241),
        "Mala":                    (10.5006, 76.4662),
    }

    with col_form:
        st.markdown("#### 📍 Location")
        user_loc_report = get_geolocation()
        if user_loc_report:
            if st.button("🛰️ Detect My Location", key="gps_report_btn"):
                st.session_state["rep_lat_input"] = float(user_loc_report['coords']['latitude'])
                st.session_state["rep_lon_input"] = float(user_loc_report['coords']['longitude'])
                st.session_state["rep_preset_sel"] = "Custom (enter manually)"
                st.rerun()

        # ── Preset selector lives OUTSIDE the form so it can update session_state
        # before the number_input widgets render (Streamlit ignores `value=` when
        # the key already exists in session_state).
        preset_sel = st.selectbox(
            "Select a location preset",
            list(PRESET_LOCATIONS.keys()),
            key="rep_preset_sel",
        )
        preset_coords = PRESET_LOCATIONS[preset_sel]
        if preset_coords is not None:
            # Only overwrite when the user actively chose a named preset
            # (guard prevents overwriting GPS-injected values on every rerun)
            if st.session_state.get("_last_preset") != preset_sel:
                st.session_state["rep_lat_input"] = preset_coords[0]
                st.session_state["rep_lon_input"] = preset_coords[1]
                st.session_state["_last_preset"] = preset_sel
        elif preset_sel == "Custom (enter manually)":
            # Reset the tracker so re-selecting a named preset works again
            if st.session_state.get("_last_preset") != "Custom (enter manually)":
                st.session_state["_last_preset"] = "Custom (enter manually)"

        # Initialise defaults on very first load
        if "rep_lat_input" not in st.session_state:
            st.session_state["rep_lat_input"] = 10.5276
        if "rep_lon_input" not in st.session_state:
            st.session_state["rep_lon_input"] = 76.2144

        with st.form("flood_report_form", clear_on_submit=True):
            lc1, lc2 = st.columns(2)
            with lc1:
                rep_lat = st.number_input(
                    "Latitude",
                    value=st.session_state["rep_lat_input"],
                    format="%.6f", step=0.001, key="rep_lat_input"
                )
            with lc2:
                rep_lon = st.number_input(
                    "Longitude",
                    value=st.session_state["rep_lon_input"],
                    format="%.6f", step=0.001, key="rep_lon_input"
                )

            loc_name = st.text_input(
                "Location name / landmark",
                value=preset_sel if preset_sel != "Custom (enter manually)" else "",
                placeholder="e.g. Near Thrissur Railway Station"
            )

            st.markdown("#### 🌊 Flood Details")
            severity = st.select_slider(
                "Flood severity",
                options=["Low", "Moderate", "High", "Extreme"],
                value="Moderate",
            )

            description = st.text_area(
                "Description",
                placeholder="Describe what you see — water level, roads blocked, people affected…",
                height=120,
            )

            submitted = st.form_submit_button("🚨 Submit Report", type="primary", width='stretch')

            # Capture values INSIDE the form before clear_on_submit wipes them
            if submitted:
                _lat  = rep_lat
                _lon  = rep_lon
                _name = loc_name
                _sev  = severity
                _desc = description

        if submitted:
            insert_report(_lat, _lon, _name, _sev, _desc)
            st.success("✅ Flood report submitted! It will appear on the map for others in your area.")
            st.balloons()

    with col_info:
        st.markdown("""
        <div class="report-card">
            <div style="font-size:2rem; margin-bottom:.5rem">📌</div>
            <div style="color:#e0f7fa; font-weight:700; font-size:1rem; margin-bottom:.5rem">Severity Guide</div>
            <div style="color:#a5d6a7; font-weight:600; margin-bottom:.2rem">🟢 Low</div>
            <div style="color:#78909c; font-size:.83rem; margin-bottom:.6rem">Minor waterlogging, no major disruption</div>
            <div style="color:#ffcc80; font-weight:600; margin-bottom:.2rem">🟡 Moderate</div>
            <div style="color:#78909c; font-size:.83rem; margin-bottom:.6rem">Roads partially flooded, some areas inaccessible</div>
            <div style="color:#ef9a9a; font-weight:600; margin-bottom:.2rem">🔴 High</div>
            <div style="color:#78909c; font-size:.83rem; margin-bottom:.6rem">Major roads blocked, properties at risk</div>
            <div style="color:#ce93d8; font-weight:600; margin-bottom:.2rem">🟣 Extreme</div>
            <div style="color:#78909c; font-size:.83rem">Evacuation needed, life-threatening conditions</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="report-card" style="margin-top:1rem">
            <div style="font-size:2rem; margin-bottom:.5rem">🔒</div>
            <div style="color:#e0f7fa; font-weight:700; font-size:1rem; margin-bottom:.5rem">Privacy</div>
            <div style="color:#78909c; font-size:.83rem">
                Reports are anonymous. Only location, severity, and description are stored.
                No personal data is collected.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  PAGE: VIEW FLOOD REPORTS
# ════════════════════════════════════════════════════════════════════════════
def show_view_reports():
    if st.button("← Back to Home", key="back_view"):
        go("home")
        st.rerun()

    st.markdown("## 🗺️ Community Flood Reports")
    st.caption(f"Last updated: {datetime.now().strftime('%d %B %Y, %H:%M IST')}")

    df = get_all_reports()

    if df.empty:
        st.info("No flood reports yet. Be the first to submit a report!")
        return

    # ── Filters ──────────────────────────────────────────────────────────────
    st.markdown("#### 🔍 Filter Reports")
    fc1, fc2 = st.columns(2)
    with fc1:
        severity_filter = st.multiselect(
            "Severity", ["Low", "Moderate", "High", "Extreme"],
            default=["Low", "Moderate", "High", "Extreme"],
            key="sev_filter"
        )
    with fc2:
        sort_reports = st.selectbox("Sort by", ["Newest first", "Oldest first", "Severity (High→Low)"])

    dff = df[df["severity"].isin(severity_filter)].copy()
    if sort_reports == "Newest first":
        dff = dff.sort_values("reported_at", ascending=False)
    elif sort_reports == "Oldest first":
        dff = dff.sort_values("reported_at", ascending=True)
    else:
        sev_order = {"Extreme": 0, "High": 1, "Moderate": 2, "Low": 3}
        dff["_sev_order"] = dff["severity"].map(sev_order)
        dff = dff.sort_values("_sev_order").drop(columns=["_sev_order"])

    # ── Stats row ─────────────────────────────────────────────────────────────
    total_reports = len(df)
    extreme_count = len(df[df["severity"] == "Extreme"])
    high_count    = len(df[df["severity"] == "High"])
    recent_report = df["reported_at"].max() if not df.empty else "—"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Reports",   total_reports)
    m2.metric("Extreme",         extreme_count)
    m3.metric("High Severity",   high_count)
    m4.metric("Latest Report",   recent_report[:16] if recent_report != "—" else "—")

    # ── Map ──────────────────────────────────────────────────────────────────
    st.markdown("### Interactive Map")
    center_lat = dff["latitude"].mean() if not dff.empty else 10.5276
    center_lon = dff["longitude"].mean() if not dff.empty else 76.2144
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB dark_matter")

    for _, row in dff.iterrows():
        sev_clr = SEVERITY_COLOR.get(row["severity"], "#ff9800")
        popup_html = f"""
        <div style="font-family:sans-serif; font-size:12px; min-width:210px">
            <b style="font-size:14px">🚨 {row['severity']} Flood</b><hr style="margin:4px 0">
            <b>Location:</b> {row['location_name'] or 'Unknown'}<br>
            <b>Description:</b> {row['description'] or '—'}<br>
            <b>Reported:</b> {row['reported_at']}<br>
            <b>Coords:</b> {row['latitude']:.5f}, {row['longitude']:.5f}
        </div>
        """
        icon_color = {
            "Low": "green", "Moderate": "orange", "High": "red", "Extreme": "purple"
        }.get(row["severity"], "orange")
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=f"⚠️ {row['severity']}: {row['location_name'] or 'Unknown'}",
            icon=folium.Icon(color=icon_color, icon="exclamation-triangle", prefix="fa"),
        ).add_to(m)

    # Map legend
    legend = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:9999;
                background:rgba(13,27,42,0.92); border:1px solid #37474f;
                border-radius:10px; padding:12px 16px; font-family:sans-serif;
                font-size:12px; color:#eceff1; min-width:150px">
        <b>Flood Severity</b><br>
        <span style="color:#4caf50">●</span> Low<br>
        <span style="color:#ff9800">●</span> Moderate<br>
        <span style="color:#f44336">●</span> High<br>
        <span style="color:#9c27b0">●</span> Extreme
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))
    st_folium(m, width="100%", height=480, returned_objects=[])

    # ── Report list ──────────────────────────────────────────────────────────
    st.markdown(f"### Report List  ({len(dff)} showing)")
    SEV_ICONS = {"Low": "🟢", "Moderate": "🟡", "High": "🔴", "Extreme": "🟣"}

    for _, row in dff.iterrows():
        icon = SEV_ICONS.get(row["severity"], "⚠️")
        loc  = row["location_name"] or f"{row['latitude']:.4f}, {row['longitude']:.4f}"
        desc = row["description"] or "No description provided."
        dt   = row["reported_at"].replace("T", " ")[:16]
        sev_clr = SEVERITY_COLOR.get(row["severity"], "#ff9800")

        st.markdown(f"""
        <div class="report-row">
            <div class="rr-sev" style="color:{sev_clr}">{icon}</div>
            <div class="rr-main">
                <div class="rr-loc">{row['severity']} — {loc}</div>
                <div class="rr-desc">{desc[:140]}{"…" if len(desc) > 140 else ""}</div>
            </div>
            <div class="rr-date">🕐 {dt}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Export ───────────────────────────────────────────────────────────────
    csv = dff.drop(columns=["id"], errors="ignore").to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Export Reports CSV", data=csv,
        file_name=f"flood_reports_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )


# ════════════════════════════════════════════════════════════════════════════
#  PAGE: EMERGENCY HELP
# ════════════════════════════════════════════════════════════════════════════
def show_emergency():
    if st.button("← Back to Home", key="back_emergency"):
        go("home")
        st.rerun()

    st.markdown("## 🚨 Emergency Contacts")
    st.markdown(
        '<p style="color:#ef9a9a; font-weight:600">⚠️ If you are in immediate danger, call 112 (National Emergency Number) right now.</p>',
        unsafe_allow_html=True
    )
    st.divider()

    contacts = [
        # (icon, title, number, description, bg_color, num_color)
        ("🚒", "Kerala Fire & Rescue Services",        "101",
         "Flood rescue, boat deployment, evacuation support across Kerala",
         "rgba(183,28,28,0.18)", "#ef5350"),
        ("🆘", "National Emergency Number",            "112",
         "Single pan-India emergency number — connects to Police, Fire & Ambulance",
         "rgba(74,20,140,0.18)", "#ce93d8"),
        ("🚔", "Thrissur District Police",             "100 / 0487-2420100",
         "Law enforcement, flood control room, emergency coordination",
         "rgba(13,71,161,0.18)", "#90caf9"),
        ("🩺", "Ambulance / DMER Kerala",              "108",
         "Free ambulance service — medical emergencies and flood victims",
         "rgba(0,96,100,0.18)", "#80deea"),
        ("🌊", "NDRF (National Disaster Response Force)", "011-24363260",
         "Specialised flood & disaster rescue teams deployed across Kerala",
         "rgba(230,81,0,0.18)", "#ffb74d"),
        ("🏥", "Thrissur District Hospital",           "0487-2361100",
         "Primary government hospital, 24×7 emergency trauma care",
         "rgba(38,50,56,0.35)", "#b0bec5"),
        ("🌐", "Kerala State Disaster Management Authority", "1070 / 0471-2331639",
         "SDMA control room — flood warnings, relief camp info, helpline",
         "rgba(0,60,50,0.25)", "#80cbc4"),
        ("🚁", "Coast Guard (Marine Flood Rescue)",    "1554",
         "Sea / backwater rescue operations in coastal Thrissur areas",
         "rgba(1,87,155,0.2)", "#81d4fa"),
    ]

    col1, col2 = st.columns(2, gap="large")
    for i, (icon, title, number, desc, bg, num_clr) in enumerate(contacts):
        target = col1 if i % 2 == 0 else col2
        with target:
            st.markdown(f"""
            <div class="em-card" style="background:{bg}; border:1px solid rgba(255,255,255,0.1);">
                <div class="em-icon">{icon}</div>
                <div class="em-body">
                    <div class="em-title">{title}</div>
                    <div class="em-num" style="color:{num_clr}">{number}</div>
                    <div class="em-desc">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style="background:rgba(255,152,0,0.12); border:1px solid rgba(255,152,0,0.35);
                border-radius:12px; padding:1.2rem 1.5rem; margin-top:.5rem">
        <div style="color:#ffcc80; font-weight:700; font-size:1rem; margin-bottom:.5rem">
            🧭 During a Flood — Stay Safe
        </div>
        <ul style="color:#b0bec5; font-size:.88rem; line-height:1.8; margin:0; padding-left:1.2rem">
            <li>Move to higher ground immediately if water levels are rising</li>
            <li>Do <strong>not</strong> walk or drive through floodwater — just 15 cm can knock you down</li>
            <li>Turn off electricity, gas and water at the mains before evacuating</li>
            <li>Keep a waterproof bag with ID, medicines, water, food & phone charger</li>
            <li>Follow instructions from officials and do not return until declared safe</li>
            <li>After a flood, watch for contaminated water, broken gas lines, and unstable structures</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  ROUTER
# ════════════════════════════════════════════════════════════════════════════
page = st.session_state.page

if page == "home":
    show_landing()
elif page == "predict":
    show_predict()
elif page == "report":
    show_report()
elif page == "view_reports":
    show_view_reports()
elif page == "emergency":
    show_emergency()
else:
    go("home")
    st.rerun()
