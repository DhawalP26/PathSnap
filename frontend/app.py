"""PathSnap Route Studio ΓÇö a map-first Streamlit UI for the evaluator API.

Run from the repository root:
    python -m streamlit run frontend/app.py

Environment:
    PATHSNAP_API_URL (default http://127.0.0.1:8000)

"""
from __future__ import annotations

import json
import os
from html import escape

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

from frontend.route_utils import route_lines

API_URL = os.getenv("PATHSNAP_API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT_SECONDS = 15

# Deliberately map-first: off-white canvas, editorial type, and restrained teal.
st.set_page_config(
    page_title="PathSnap ΓÇö Route Studio",
    page_icon="Γùê",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
/* Hide Streamlit's unused chrome, while preserving interactive app controls. */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; height: 1.5rem; }
[data-testid="stSidebar"] { display: none; }
[data-testid="stAppViewContainer"], .stApp { background: #f6f5f1; }
[data-testid="stAppViewContainer"] .block-container {
    max-width: 1510px; padding: 2.2rem 3.2rem 3.4rem;
}
html, body, [class*="css"], .stApp {
    font-family: "Inter", "Segoe UI", "Arial", sans-serif;
    color: #172925;
}
/* Header ΓÇö product identity, not a generic hero. */
.ps-header {
    display:flex; align-items:center; justify-content:space-between;
    border-bottom:1px solid #dfded8; padding:0 0 20px; margin:0 0 32px;
    gap:16px; flex-wrap:wrap;
}
.ps-brand {display:flex;align-items:center;gap:12px;}
.ps-logomark {
    display:flex;align-items:center;justify-content:center;
    width:39px;height:39px;border-radius:10px;
    background:#173f39;color:#fcf8ed;font-size:25px;font-weight:700;
    line-height:1;
}
.ps-brandname {font-size:22px;letter-spacing:-1.2px;font-weight:750;color:#152e29;}
.ps-brandname span {color:#d56b4a;}
.ps-brand-sub {font-size:11px;font-weight:650;letter-spacing:1.5px;
    text-transform:uppercase;color:#8b918a;margin-left:12px;
    padding-left:18px;border-left:1px solid #d8d8d1;}
.ps-header-meta {display:flex;align-items:center;gap:12px;font-size:12px;color:#68756e;}
.ps-online {
    background:#e5eee9;border:1px solid #c9ddd0;color:#255c45;
    border-radius:40px;padding:7px 12px;font-weight:650;
}
.ps-offline {
    background:#f8e8e2;border:1px solid #e9c9bb;color:#98503d;
    border-radius:40px;padding:7px 12px;font-weight:650;
}
.ps-dot {display:inline-block;width:7px;height:7px;background:#41866b;
    border-radius:50%;margin-right:8px;vertical-align:1px;}
.ps-topline {font-size:11px;font-weight:750;letter-spacing:1.6px;
    color:#a66b4d;text-transform:uppercase;margin-bottom:8px;}
.ps-title {font-size:clamp(30px,3vw,46px);font-weight:720;letter-spacing:-2px;
    line-height:1.06;color:#1a302a;margin:0 0 10px;}
.ps-description {font-size:14px;line-height:1.6;color:#748079;
    margin:0 0 26px;max-width:580px;}
.ps-section-kicker {font-size:11px;letter-spacing:1.8px;font-weight:750;
    text-transform:uppercase;color:#a16849;margin:0 0 10px;}
.ps-panel-title {font-size:16px;font-weight:750;letter-spacing:-.4px;
    margin:0;color:#1b312b;}
.ps-panel-intro {font-size:12px;color:#7d8981;margin:8px 0 22px;
    line-height:1.6;}
.ps-caption {font-size:11px;color:#8a938d;line-height:1.6;margin-top:13px;}
.ps-rule {border-top:1px solid #e5e4dd;margin:18px 0;}
.ps-field-label {font-size:11px;font-weight:700;text-transform:uppercase;
    letter-spacing:1.2px;color:#62746a;margin:0 0 7px;}
.ps-kpi-row {display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:16px;}
.ps-kpi {background:#fafaf8;border:1px solid #e8e8e1;border-radius:10px;
    padding:16px 16px 14px;min-width:0;}
.ps-kpi-value {font-size:25px;letter-spacing:-1px;font-weight:750;color:#18372f;}
.ps-kpi-name {font-size:11px;color:#78867b;margin-top:7px;letter-spacing:.2px;}
.ps-info-note {background:#ecf1ed;border-radius:9px;padding:14px;
    color:#466259;font-size:12px;line-height:1.7; margin-top:15px;}
.ps-map-header {display:flex;align-items:center;justify-content:space-between;
    gap:12px;flex-wrap:wrap;margin-bottom:16px;}
.ps-legend {display:flex;align-items:center;gap:15px;color:#63736a;font-size:11px;}
.ps-legend-line {display:inline-block;width:24px;height:3px;vertical-align:4px;
    background:#3178f6;margin-right:6px;border-radius:4px;}
.ps-legend-line--match {background:#159c68;}
.ps-small-meta {font-size:11px;color:#78867c;letter-spacing:.4px;}
/* Inputs and buttons */
div[data-testid="stSelectbox"] label {display:none;}
div[data-testid="stSelectbox"] > div > div {
    background:#fff; border-color:#d9ddd6; border-radius:9px;
}
div[data-testid="stButton"] button[kind="primary"] {
    width:100%;min-height:46px;background:#173f39;color:#fff;
    border-radius:9px;border:1px solid #173f39;font-weight:700;
    letter-spacing:.1px;transition:background 150ms ease;
}
div[data-testid="stButton"] button[kind="primary"]:hover {
    background:#245c52;border-color:#245c52;color:#fff;
}
div[data-testid="stButton"] button[kind="primary"]:focus {
    box-shadow:0 0 0 2px #d0e4d7;
}
/* Map itself comes from the normal Folium backend; no mock data. */
[data-testid="stIFrame"] {border:1px solid #dedfd8;border-radius:11px;overflow:hidden;}
[data-testid="stAlert"] {border-radius:9px;}
@media(max-width:850px) {
 [data-testid="stAppViewContainer"] .block-container {padding:1.5rem 1rem 2rem;}
 .ps-brand-sub {display:none;}
 .ps-header {margin-bottom:20px;}
 .ps-title {letter-spacing:-1.3px;}
}
</style>
""",
    unsafe_allow_html=True,
)


def call_api(method: str, path: str, **kwargs) -> dict:
    """Call our own backend. All trip IDs come from the server, not user text."""
    response = requests.request(
        method,
        API_URL + path,
        timeout=kwargs.pop("timeout", TIMEOUT_SECONDS),
        **kwargs,
    )
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"API {response.status_code}: {detail}")
    return response.json()


def safe_card(value: str, label: str) -> str:
    return (
        '<div class="ps-kpi">'
        f'<div class="ps-kpi-value">{escape(str(value))}</div>'
        f'<div class="ps-kpi-name">{escape(label)}</div>'
        '</div>'
    )


try:
    health = call_api("GET", "/health")
    trips_response = call_api("GET", "/trips")
except (requests.RequestException, RuntimeError, ValueError) as exc:
    st.markdown(
        '<div class="ps-header"><div class="ps-brand"><div class="ps-logomark">P</div>'
        '<div class="ps-brandname">PathSnap<span>.</span></div></div>'
        '<span class="ps-offline">API unavailable</span></div>',
        unsafe_allow_html=True,
    )
    st.error(f"Cannot connect to the PathSnap API at {API_URL}. Details: {exc}")
    st.caption("Check that FastAPI is running and PATHSNAP_API_URL points to the backend computer.")
    st.stop()

trip_ids = trips_response.get("trips", [])
trip_count = int(trips_response.get("count", len(trip_ids)))
partition = str(trips_response.get("partition", "unknown"))
data_ready = bool(health.get("data_root_exists"))
model_ready = bool(health.get("model_ready"))

st.markdown(
    '<div class="ps-header">'
    '<div class="ps-brand"><div class="ps-logomark">P</div>'
    '<div class="ps-brandname">PathSnap<span>.</span></div>'
    '<div class="ps-brand-sub">Route Intelligence Studio</div></div>'
    '<div class="ps-header-meta">'
    '<span>WORKSPACE / 01</span>'
    '<span class="ps-online"><span class="ps-dot"></span>API CONNECTED</span>'
    '</div></div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="ps-topline">Map matching / Evaluation workspace</div>', unsafe_allow_html=True)
st.markdown('<div class="ps-title">GPS route workspace.</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="ps-description">Inspect recorded GPS trips, run map matching, and '
    'review predicted routes on the map.</div>',
    unsafe_allow_html=True,
)

if not data_ready:
    st.warning("Dataset root is missing on the backend computer. Set PATHSNAP_DATA_DIR and restart FastAPI.")
if not model_ready:
    st.warning("Install the validated 40-feature production model on the backend to enable predictions.")
if not trip_ids:
    st.info("No trips available in the configured partition. Check PATHSNAP_PARTITION and your dataset directory.")
    st.stop()

controls, canvas = st.columns([1, 2.25], gap="large")

with controls:
    with st.container(border=True):
        st.markdown('<div class="ps-section-kicker">01 / Source data</div>', unsafe_allow_html=True)
        st.markdown('<div class="ps-panel-title">Choose a trajectory</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="ps-panel-intro">Select any available trip from the configured dataset.</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="ps-field-label">Trip identifier</div>', unsafe_allow_html=True)
        selected = st.selectbox("Trip identifier", trip_ids, label_visibility="collapsed")
        st.markdown(
            f'<div class="ps-small-meta">PARTITION: {escape(partition.upper())} '
            f'&nbsp;&nbsp;┬╖&nbsp;&nbsp; {trip_count} AVAILABLE TRIPS</div>',
            unsafe_allow_html=True,
        )

    try:
        preview = call_api("GET", f"/preview/{selected}")
    except (requests.RequestException, RuntimeError, ValueError) as exc:
        st.error(f"Could not load GPS observations for {selected}: {exc}")
        st.stop()

    with st.container(border=True):
        st.markdown('<div class="ps-section-kicker">02 / Processing</div>', unsafe_allow_html=True)
        st.markdown('<div class="ps-panel-title">Route reconstruction</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="ps-panel-intro">Run the connected ML model and Viterbi decoder when available.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="ps-kpi-row">'
            + safe_card(f"{int(preview['total_observations']):,}", "GPS observations")
            + safe_card(f"{trip_count}", "Trips in partition")
            + '</div>',
            unsafe_allow_html=True,
        )
        st.write("")
        run_clicked = st.button(
            "Predict Route", type="primary", use_container_width=True, disabled=not model_ready,
        )
        if run_clicked:
            with st.spinner("Requesting real route inferenceΓÇª"):
                try:
                    st.session_state["pathsnap_prediction"] = call_api(
                        "POST", "/predict", json={"trip_id": selected}, timeout=300,
                    )
                    st.session_state["pathsnap_prediction_error"] = None
                except (requests.RequestException, RuntimeError, ValueError) as exc:
                    st.session_state["pathsnap_prediction"] = None
                    st.session_state["pathsnap_prediction_error"] = str(exc)
                    st.session_state["pathsnap_error_trip"] = selected

        predicted = st.session_state.get("pathsnap_prediction")
        if predicted and predicted.get("trip_id") != selected:
            predicted = None

        error = st.session_state.get("pathsnap_prediction_error")
        if error and st.session_state.get("pathsnap_error_trip") == selected:
            st.error(error)

        st.markdown('<div class="ps-rule"></div>', unsafe_allow_html=True)
        lines = []
        if predicted:
            try:
                lines = route_lines(predicted)
            except ValueError as exc:
                st.error(f"Invalid reconstructed route geometry: {exc}")
            else:
                if not lines:
                    st.warning("The pipeline returned no road segments for this trip.")
            st.markdown(
                '<div class="ps-field-label">Inference result</div>',
                unsafe_allow_html=True,
            )
            if lines:
                st.success("Route reconstructed successfully")
            st.markdown(
                '<div class="ps-kpi-row">'
                + safe_card(str(len(predicted.get("predicted_arc_ids", []))), "Reconstructed arcs")
                + safe_card(str(len(lines)), "Route polylines")
                + safe_card(str(predicted.get("route_breaks", 0)), "Decoder restarts")
                + safe_card(str(predicted.get("duration_ms", "ΓÇö")), "Processing time (ms)")
                + '</div>',
                unsafe_allow_html=True,
            )
            st.download_button(
                "Export prediction JSON",
                data=json.dumps(predicted, indent=2),
                file_name=f"pathsnap_prediction_{selected}.json",
                mime="application/json",
                use_container_width=True,
            )
            metrics = predicted.get("metrics", {})
            if metrics:
                with st.expander("Prediction diagnostics"):
                    st.json(metrics)
        else:
            st.markdown(
                '<div class="ps-info-note">'
                '<strong>Observation mode</strong><br>'
                'Displaying real recorded GPS data. Reconstructed road geometry '
                'will appear in green after a successful prediction.'
                '</div>',
                unsafe_allow_html=True,
            )

with canvas:
    with st.container(border=True):
        st.markdown(
            '<div class="ps-map-header">'
            '<div><div class="ps-section-kicker">03 / Geospatial view</div>'
            '<div class="ps-panel-title">Trajectory explorer</div></div>'
            '<div class="ps-legend">'
            '<span><span class="ps-legend-line"></span>Observed GPS</span>'
            '<span><span class="ps-legend-line ps-legend-line--match"></span>Matched route</span>'
            '</div></div>',
            unsafe_allow_html=True,
        )
        points = preview.get("gps_points", [])
        if not points:
            st.error("There are no valid GPS points for this trip.")
            st.stop()

        # Restore the exact behavior of the first frontend map, without changing
        # the Route Studio layout, controls, or prediction workflow around it.
        m = folium.Map(
            location=[points[0][0], points[0][1]],
            zoom_start=13,
            tiles="OpenStreetMap",
        )
        gps_layer = folium.FeatureGroup(name="Raw GPS (blue)", show=True)
        folium.PolyLine(
            [(p[0], p[1]) for p in points],
            color="#3178f6", weight=2, opacity=0.6,
            tooltip="Raw GPS points",
        ).add_to(gps_layer)
        for lat, lon, timestamp in points[::max(1, len(points) // 120)]:
            folium.CircleMarker(
                [lat, lon], radius=2, color="#3178f6", fill=True,
                tooltip=f"Time {timestamp:g}",
            ).add_to(gps_layer)
        gps_layer.add_to(m)

        bounds = [[point[0], point[1]] for point in points]
        if lines:
            route_layer = folium.FeatureGroup(name="Predicted route (green)", show=True)
            for index, segment in enumerate(lines, start=1):
                if len(segment) > 1:
                    folium.PolyLine(
                        segment, color="#159c68", weight=5,
                        tooltip=f"Predicted road segment {index}",
                    ).add_to(route_layer)
                else:
                    folium.CircleMarker(
                        segment[0], radius=4, color="#159c68",
                        tooltip=f"Predicted point {index}",
                    ).add_to(route_layer)
                bounds.extend(segment)
            route_layer.add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)
        m.fit_bounds([
            [min(p[0] for p in bounds), min(p[1] for p in bounds)],
            [max(p[0] for p in bounds), max(p[1] for p in bounds)],
        ])
        st_folium(
            m,
            height=625,
            use_container_width=True,
            returned_objects=[],
            key=f"route_map_{selected}_{'predicted' if predicted else 'raw'}",
        )
        st.markdown(
            '<div class="ps-caption">Blue: observed GPS points &nbsp;&nbsp; '
            'Green: predicted road route (after inference). '
            'Ground-truth routes remain server-side.</div>',
            unsafe_allow_html=True,
        )

st.markdown(
    '<div style="border-top:1px solid #dfded8;margin-top:32px;padding-top:17px;'
    'display:flex;justify-content:space-between;flex-wrap:wrap;gap:10px;'
    'font-size:11px;color:#8a948c">'
    '<span>PATHSNAP / DISTRIBUTED MAP MATCHING</span>'
    f'<span>BACKEND ┬╖ {escape(API_URL)}</span></div>',
    unsafe_allow_html=True,
)
