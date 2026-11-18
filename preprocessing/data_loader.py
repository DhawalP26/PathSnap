
from pathlib import Path
import pandas as pd

# Dataset location
DATASET_DIR = Path(
    r"C:\Users\ganes\Downloads\Student folder\Student folder"
)


def load_trip(trip_id, partition="train"):
    """Load GPS, road nodes, road arcs and reference route."""

    trip_id = str(trip_id).zfill(8)
    trip_dir = DATASET_DIR / partition / trip_id

    if not trip_dir.exists():
        raise FileNotFoundError(f"Trip folder not found: {trip_dir}")

    gps = pd.read_csv(
        trip_dir / f"{trip_id}.track",
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude", "timestamp"]
    )

    nodes = pd.read_csv(
        trip_dir / f"{trip_id}.nodes",
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude"]
    )

    arcs = pd.read_csv(
        trip_dir / f"{trip_id}.arcs",
        sep=r"\s+",
        header=None,
        names=["source_node", "target_node"]
    )

    # Provisional arc IDs based on row position.
    arcs.insert(0, "arc_id", range(len(arcs)))

    route = pd.read_csv(
        trip_dir / f"{trip_id}.route",
        header=None,
        names=["arc_id"]
    )

    return gps, nodes, arcs, route


if __name__ == "__main__":

    gps, nodes, arcs, route = load_trip("00000001")

    print("\n--- DATASET VALIDATION ---")

    print("GPS points:", len(gps))
    print("Road nodes:", len(nodes))
    print("Road arcs:", len(arcs))
    print("Route segments:", len(route))

    # Validate road node references.
    valid_node_ids = set(range(len(nodes)))

    invalid_sources = ~arcs["source_node"].isin(valid_node_ids)
    invalid_targets = ~arcs["target_node"].isin(valid_node_ids)

    print("\nInvalid source nodes:", invalid_sources.sum())
    print("Invalid target nodes:", invalid_targets.sum())

    # Check provisional route ID mapping.
    valid_arc_ids = set(arcs["arc_id"])
    invalid_route_ids = ~route["arc_id"].isin(valid_arc_ids)

    print("\nMinimum route ID:", route["arc_id"].min())
    print("Maximum route ID:", route["arc_id"].max())
    print("Invalid route IDs:", invalid_route_ids.sum())

    if invalid_route_ids.sum() == 0:
        print("All route IDs match provisional arc indices.")
    else:
        print("WARNING: Some route IDs do not match arc indices.")

    print("\n--- FIRST FIVE GPS OBSERVATIONS ---")
    print(gps.head())

    print("\n--- FIRST FIVE ROAD NODES ---")
    print(nodes.head())

    print("\n--- FIRST FIVE ROAD ARCS ---")
    print(arcs.head())

    print("\n--- FIRST FIVE ROUTE SEGMENTS ---")
    print(route.head())
