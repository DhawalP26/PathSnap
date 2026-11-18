"""Local metric road geometry and exact radius queries."""
import numpy as np
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree


class RoadNetwork:
    def __init__(self, trip):
        self.nodes = trip.nodes
        self.arcs = trip.arcs
        # Circular mean keeps a trip spanning the antimeridian centered locally.
        longitude = np.radians(trip.track[:, 0])
        lon = np.degrees(np.arctan2(np.sin(longitude).mean(), np.cos(longitude).mean()))
        lat = trip.track[:, 1].mean()
        crs = CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m")
        self.transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        xy = self.project(trip.nodes)
        self.starts = xy[self.arcs[:, 0]]
        self.vectors = xy[self.arcs[:, 1]] - self.starts
        self.lengths = np.linalg.norm(self.vectors, axis=1)
        self.headings = np.degrees(np.arctan2(self.vectors[:, 0], self.vectors[:, 1])) % 360
        self.headings[self.lengths == 0] = np.nan

        # Group midpoints by segment length so long arcs don't widen every search.
        self.trees = []
        buckets = np.floor(np.log2(np.maximum(self.lengths, 1))).astype(int)
        for bucket in np.unique(buckets):
            ids = np.flatnonzero(buckets == bucket)
            self.trees.append((ids, self.lengths[ids].max() / 2,
                               cKDTree(self.starts[ids] + self.vectors[ids] / 2)))

    def project(self, coordinates):
        if len(coordinates) == 1:
            lon, lat = coordinates[0, :2]
            xy = np.asarray([self.transformer.transform(float(lon), float(lat))])
        else:
            xy = np.column_stack(self.transformer.transform(coordinates[:, 0], coordinates[:, 1]))
        if not np.isfinite(xy).all():
            raise ValueError("coordinates are outside the local metric projection")
        return xy

    def distances(self, point, ids):
        """Return point-to-segment distance and fraction along each directed arc."""
        vector = self.vectors[ids]
        offset = point - self.starts[ids]
        squared = self.lengths[ids] ** 2
        fraction = np.divide(np.einsum("ij,ij->i", offset, vector), squared,
                             out=np.zeros_like(squared), where=squared > 0)
        fraction = np.clip(fraction, 0, 1)
        return np.linalg.norm(offset - vector * fraction[:, None], axis=1), fraction

    def candidates(self, points, radius_m, top_k):
        """Yield (arc IDs, distances, fractions) for each observation, nearest first."""
        if not np.isfinite(radius_m) or radius_m <= 0 or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("radius_m must be finite and positive, and top_k a positive integer")
        hits = [ids for ids, _, _ in self.trees]
        neighborhoods = [tree.query_ball_point(points, np.nextafter(radius_m + half, np.inf))
                         for _, half, tree in self.trees]
        for point, groups in zip(points, zip(*neighborhoods)):
            ids = np.concatenate([ids[group] for ids, group in zip(hits, groups)])
            distances, fractions = self.distances(point, ids)
            order = np.lexsort((ids, distances))
            order = order[distances[order] <= radius_m][:top_k]
            yield ids[order], distances[order], fractions[order]
