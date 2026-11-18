"""Create a Folium map of a trip's GPS track and reference route."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from reconstruction.road_graph import RoadGraph
from reconstruction.route_analysis import read_route, read_track, route_geometry


def create_trip_map(
    nodes_path: str | Path,
    arcs_path: str | Path,
    track_path: str | Path,
    route_path: str | Path,
    output_path: str | Path,
    predicted_arc_ids: Iterable[int] | None = None,
    predicted_route_segments: Iterable[Iterable[int]] | None = None,
    predicted_route_name: str = "Predicted route",
) -> Path:
    try:
        import folium
    except ImportError as exc:
        raise RuntimeError(
            "Folium is required to create maps. Install project dependencies with "
            "'python -m pip install -r requirements.txt'."
        ) from exc

    graph = RoadGraph.from_files(nodes_path, arcs_path)
    observations = read_track(track_path)
    route = read_route(route_path)
    if not observations:
        raise ValueError(f"{track_path}: no GPS observations")

    center = (
        sum(latitude for _, latitude, _ in observations) / len(observations),
        sum(longitude for longitude, _, _ in observations) / len(observations),
    )
    map_view = folium.Map(location=center, control_scale=True, zoom_start=14)
    gps_layer = folium.FeatureGroup(name="GPS observations", show=True)
    folium.PolyLine(
        [(latitude, longitude) for longitude, latitude, _ in observations],
        color="#3388ff",
        weight=3,
        opacity=0.65,
        name="GPS track",
    ).add_to(gps_layer)
    for index, (longitude, latitude, timestamp) in enumerate(observations):
        if index in (0, len(observations) - 1) or index % 20 == 0:
            folium.CircleMarker(
                location=(latitude, longitude),
                radius=4,
                tooltip=f"GPS {index}, t={timestamp:g}",
                color="#145da0",
                fill=True,
                fill_opacity=0.8,
            ).add_to(gps_layer)
    gps_layer.add_to(map_view)

    route_layer = folium.FeatureGroup(name="Reference route", show=True)
    for line in route_geometry(graph, route):
        folium.PolyLine(
            [(latitude, longitude) for longitude, latitude in line],
            color="#d62728",
            weight=5,
            opacity=0.9,
        ).add_to(route_layer)
    route_layer.add_to(map_view)

    if predicted_arc_ids is not None and predicted_route_segments is not None:
        raise ValueError("Provide predicted_arc_ids or predicted_route_segments, not both")
    if not predicted_route_name.strip():
        raise ValueError("predicted_route_name must not be empty")
    if predicted_arc_ids is not None or predicted_route_segments is not None:
        route_segments = (
            [predicted_arc_ids]
            if predicted_arc_ids is not None
            else predicted_route_segments
        )
        predicted_layer = folium.FeatureGroup(name=predicted_route_name, show=True)
        for route_segment in route_segments:
            for line in route_geometry(graph, route_segment):
                folium.PolyLine(
                    [(latitude, longitude) for longitude, latitude in line],
                    color="#159c68",
                    weight=6,
                    opacity=0.9,
                ).add_to(predicted_layer)
        predicted_layer.add_to(map_view)

    folium.Marker(
        (observations[0][1], observations[0][0]),
        tooltip="Start",
        icon=folium.Icon(color="green"),
    ).add_to(map_view)
    folium.Marker(
        (observations[-1][1], observations[-1][0]),
        tooltip="End",
        icon=folium.Icon(color="red"),
    ).add_to(map_view)
    folium.LayerControl().add_to(map_view)
    map_view.fit_bounds(
        [(latitude, longitude) for longitude, latitude, _ in observations]
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    map_view.save(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trip-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--predicted-route",
        type=Path,
        help="Optional text file containing predicted arc IDs, one per line",
    )
    args = parser.parse_args()
    trip_id = args.trip_dir.name
    prefix = args.trip_dir / trip_id
    result = create_trip_map(
        prefix.with_suffix(".nodes"),
        prefix.with_suffix(".arcs"),
        prefix.with_suffix(".track"),
        prefix.with_suffix(".route"),
        args.output,
        read_route(args.predicted_route) if args.predicted_route else None,
    )
    print(f"Map written to {result}")


if __name__ == "__main__":
    main()
