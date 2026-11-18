"""PathSnap real Streamlit route reconstruction UI.

Run: python -m streamlit run frontend/app.py
"""
from __future__ import annotations

import os
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium

from frontend.route_utils import route_lines

API_URL = os.getenv("PATHSNAP_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="PathSnap | Route Reconstruction", page_icon="🗺️", layout="wide")
st.title("🗺️ PathSnap")
st.caption("Distributed map matching · 40-feature model · directed Viterbi reconstruction")
st.sidebar.header("Connection")
st.sidebar.write(f"API: `{API_URL}`")


def call_api(method: str, path: str, **kwargs):
    response = requests.request(method, API_URL + path, timeout=kwargs.pop("timeout", 20), **kwargs)
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"API {response.status_code}: {detail}")
    return response.json()


try:
    health = call_api("GET", "/health")
    trips_response = call_api("GET", "/trips")
except (requests.RequestException, RuntimeError, ValueError) as exc:
    st.error(f"Cannot connect to backend. Check PATHSNAP_API_URL and FastAPI. Details: {exc}")
    st.stop()

st.sidebar.success("Backend connected")
st.sidebar.write(f"Partition: `{trips_response['partition']}`")
st.sidebar.write(f"Trips: **{trips_response['count']}**")
if not health.get("data_root_exists"):
    st.warning("PATHSNAP_DATA_DIR does not exist on the backend computer.")
if not health.get("model_ready"):
    st.warning("The validated 40-feature production model is not installed. Predictions will return HTTP 503.")

trip_ids = trips_response.get("trips", [])
if not trip_ids:
    st.warning("No trip inputs found in the configured partition.")
    st.stop()

selected = st.selectbox("Select a trip", trip_ids)
try:
    preview = call_api("GET", f"/preview/{selected}")
except (requests.RequestException, RuntimeError, ValueError) as exc:
    st.error(f"Could not load GPS preview: {exc}")
    st.stop()

m1, m2 = st.columns(2)
m1.metric("GPS observations", preview["total_observations"])
m2.metric("Trip", selected)

if st.button("Predict route", type="primary", disabled=not health.get("model_ready", False)):
    with st.spinner("Running feature generation, ML scoring and Viterbi reconstruction..."):
        try:
            st.session_state.prediction = call_api(
                "POST", "/predict", json={"trip_id": selected}, timeout=300
            )
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            st.session_state.prediction = None
            st.error(str(exc))

predicted = st.session_state.get("prediction")
if predicted and predicted.get("trip_id") != selected:
    predicted = None

st.subheader("GPS observations and reconstructed road route")
points = preview["gps_points"]
location = [points[0][0], points[0][1]]
m = folium.Map(location=location, zoom_start=13, tiles="OpenStreetMap")
gps_layer = folium.FeatureGroup(name="Raw GPS (blue)", show=True)
folium.PolyLine([(p[0], p[1]) for p in points], color="#3178f6", weight=2,
                opacity=0.65, tooltip="Raw GPS observations").add_to(gps_layer)
for lat, lon, timestamp in points[::max(1, len(points) // 120)]:
    folium.CircleMarker([lat, lon], radius=2, color="#3178f6", fill=True,
                        tooltip=f"GPS time {timestamp:g}").add_to(gps_layer)
gps_layer.add_to(m)

bounds = [[p[0], p[1]] for p in points]
if predicted:
    try:
        lines = route_lines(predicted)
    except ValueError as exc:
        st.error(f"Invalid reconstructed route geometry: {exc}")
        lines = []
    if lines:
        route_layer = folium.FeatureGroup(name="Reconstructed road (green)", show=True)
        for i, segment in enumerate(lines, start=1):
            if len(segment) > 1:
                folium.PolyLine(segment, color="#159c68", weight=5,
                                tooltip=f"Predicted road segment {i}").add_to(route_layer)
            else:
                folium.CircleMarker(segment[0], radius=4, color="#159c68",
                                    tooltip=f"Predicted point {i}").add_to(route_layer)
            bounds.extend(segment)
        route_layer.add_to(m)
    else:
        st.warning("The pipeline returned no road segments for this trip.")
    st.success(f"Prediction completed in {predicted.get('duration_ms', '?')} ms")
    a, b, c = st.columns(3)
    a.metric("Reconstructed arcs", len(predicted.get("predicted_arc_ids", [])))
    b.metric("Route polylines", len(lines))
    c.metric("Decoder restarts", predicted.get("route_breaks", 0))
    metrics = predicted.get("metrics", {})
    if metrics:
        st.subheader("Prediction diagnostics (not accuracy)")
        st.json(metrics)

folium.LayerControl(collapsed=False).add_to(m)
if bounds:
    m.fit_bounds([[min(p[0] for p in bounds), min(p[1] for p in bounds)],
                  [max(p[0] for p in bounds), max(p[1] for p in bounds)]])
st_folium(m, height=530, use_container_width=True, key=f"map_{selected}")
st.caption("The model reads only .track, .nodes and .arcs at inference. Reference .route labels are not used or sent to the UI. Disconnected road segments are drawn separately.")
