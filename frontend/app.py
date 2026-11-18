"""PathSnap evaluator UI. Run: python -m streamlit run frontend/app.py"""
from __future__ import annotations
import os
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium

API_URL = os.getenv("PATHSNAP_API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT_SECONDS = 15

st.set_page_config(page_title="PathSnap | Route Reconstruction", page_icon="🗺️", layout="wide")
st.title("🗺️ PathSnap")
st.caption("Distributed map matching and route reconstruction")
st.sidebar.header("Connection")
st.sidebar.write(f"API: `{API_URL}`")


def call_api(method: str, path: str, **kwargs):
    response = requests.request(method, API_URL + path, timeout=kwargs.pop("timeout", TIMEOUT_SECONDS), **kwargs)
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
    st.error(f"Cannot connect to backend. Check IP, Wi-Fi, and FastAPI status. Details: {exc}")
    st.stop()

st.sidebar.success("Backend connected")
st.sidebar.write(f"Partition: `{trips_response['partition']}`")
st.sidebar.write(f"Trips: **{trips_response['count']}**")
if not health.get("data_root_exists"):
    st.warning("API is running, but PATHSNAP_DATA_DIR does not exist on Laptop 1.")
trip_ids = trips_response["trips"]
if not trip_ids:
    st.warning("No trip data found. Set PATHSNAP_DATA_DIR on the backend computer.")
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

if st.button("Predict route", type="primary"):
    with st.spinner("Requesting live prediction from Laptop 1..."):
        try:
            st.session_state.prediction = call_api("POST", "/predict", json={"trip_id": selected}, timeout=180)
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            st.session_state.prediction = None
            st.error(str(exc))

predicted = st.session_state.get("prediction")
if predicted and predicted.get("trip_id") != selected:
    predicted = None

st.subheader("GPS observations and predicted route")
points = preview["gps_points"]
m = folium.Map(location=[points[0][0], points[0][1]], zoom_start=13, tiles="OpenStreetMap")
gps_layer = folium.FeatureGroup(name="Raw GPS (blue)", show=True)
folium.PolyLine([(p[0], p[1]) for p in points], color="#3178f6", weight=2,
                opacity=0.6, tooltip="Raw GPS points").add_to(gps_layer)
for lat, lon, timestamp in points[::max(1, len(points)//120)]:
    folium.CircleMarker([lat, lon], radius=2, color="#3178f6", fill=True,
                        tooltip=f"Time {timestamp:g}").add_to(gps_layer)
gps_layer.add_to(m)

if predicted:
    route_coords = predicted.get("route_coordinates", [])
    if route_coords:
        route_layer = folium.FeatureGroup(name="Predicted route (green)", show=True)
        folium.PolyLine(route_coords, color="#159c68", weight=5, tooltip="Model prediction").add_to(route_layer)
        route_layer.add_to(m)
    st.success(f"Prediction complete in {predicted.get('duration_ms', '?')} ms")
    metrics = predicted.get("metrics", {})
    if metrics:
        st.subheader("Server-side evaluation")
        st.json(metrics)
    st.caption(f"Predicted {len(predicted.get('predicted_arc_ids', []))} road segments")

folium.LayerControl(collapsed=False).add_to(m)
m.fit_bounds([[min(p[0] for p in points), min(p[1] for p in points)],
              [max(p[0] for p in points), max(p[1] for p in points)]])
st_folium(m, height=500, use_container_width=True, key=f"map_{selected}")
st.caption("Only raw GPS observations and model predictions are visualized. No reference-route labels are sent to the UI.")
