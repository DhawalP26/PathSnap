"""Pure helper for validating and preserving disconnected route polylines."""
from __future__ import annotations

import math


def route_lines(payload: dict) -> list[list[tuple[float, float]]]:
    """Convert the Member 3 API's nested [lat, lon] lines without joining them."""
    geometry = payload.get("route_coordinates", [])
    if not isinstance(geometry, list):
        raise ValueError("route_coordinates must be a list of polylines")
    lines = []
    for line in geometry:
        if not isinstance(line, list):
            raise ValueError("Each polyline must be a list")
        coordinates = []
        for point in line:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError("Route point must be [latitude, longitude]")
            lat, lon = point
            if isinstance(lat, bool) or isinstance(lon, bool):
                raise ValueError("Route coordinates must be numeric")
            try:
                lat, lon = float(lat), float(lon)
            except (ValueError, TypeError) as exc:
                raise ValueError("Route coordinates must be numeric") from exc
            if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError("Route coordinates must be finite WGS84 values")
            coordinates.append((lat, lon))
        if coordinates:
            lines.append(coordinates)
    return lines
