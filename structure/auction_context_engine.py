import json
from typing import Any


INPUT_PATH = "session_profiles_202604.jsonl"
OUTPUT_PATH = "auction_context_202604.jsonl"


def _round_float(value: float, ndigits: int) -> float:
    """
    Deterministically round numeric output and force native Python float type.
    """
    return float(round(float(value), ndigits))


def load_session_profiles(input_path: str) -> list[dict[str, Any]]:
    """
    Load session profiles from JSONL with one JSON object per line.
    """
    rows: list[dict[str, Any]] = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    return rows


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0
    return float(numerator / denominator)


def compute_topology_prominence(volume_profile: list[dict[str, Any]], idx: int) -> float:
    """
    Compute structural prominence of one bin versus nearby structure.

    topology_prominence = current_bin_volume / local_neighbor_mean

    Local neighborhood uses ±1 and ±2 bins (where available) so the score
    reflects visible protrusion from immediate profile topology.
    """
    current = float(volume_profile[idx].get("total_volume", 0.0))

    neighbors: list[float] = []
    for j in (idx - 2, idx - 1, idx + 1, idx + 2):
        if 0 <= j < len(volume_profile):
            neighbors.append(float(volume_profile[j].get("total_volume", 0.0)))

    if not neighbors:
        return 0.0

    local_neighbor_mean = float(sum(neighbors) / len(neighbors))
    return _safe_div(current, max(local_neighbor_mean, 1e-12))


def classify_edge_proximity(bin_row: dict[str, Any], val: float, vah: float, bin_width: float) -> str:
    """
    Classify whether a bin sits near value-area edges or inside value.
    """
    price_low = float(bin_row["bin_low"])
    price_high = float(bin_row["bin_high"])

    # If bin overlaps near VAH/VAL within one bin width, tag as edge-proximate.
    if abs(price_high - vah) <= bin_width or abs(price_low - vah) <= bin_width:
        return "VAH"

    if abs(price_low - val) <= bin_width or abs(price_high - val) <= bin_width:
        return "VAL"

    return "INSIDE_VALUE"


def compute_neighbor_divergence(volume_profile: list[dict[str, Any]], idx: int) -> float:
    """
    Measure how different current bin participation is versus adjacent bins.

    Uses total_volume relative difference against local neighbor mean.
    """
    current = float(volume_profile[idx].get("total_volume", 0.0))

    neighbors = []
    if idx - 1 >= 0:
        neighbors.append(float(volume_profile[idx - 1].get("total_volume", 0.0)))
    if idx + 1 < len(volume_profile):
        neighbors.append(float(volume_profile[idx + 1].get("total_volume", 0.0)))

    if not neighbors:
        return 0.0

    neighbor_mean = float(sum(neighbors) / len(neighbors))
    return _safe_div(current - neighbor_mean, max(neighbor_mean, 1e-12))


def detect_watch_zones(session_row: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Detect structurally interesting bins from previous session as context watch zones.

    Watch zones are NOT predictions.
    They are contextual reference locations only.
    """
    profile = session_row.get("volume_profile", [])
    if not profile:
        return []

    val = float(session_row["val"])
    vah = float(session_row["vah"])
    bin_width = float(session_row.get("bin_width", 0.0))

    all_volumes = [float(b.get("total_volume", 0.0)) for b in profile]
    global_mean_volume = _safe_div(sum(all_volumes), max(len(all_volumes), 1))
    sorted_volumes = sorted(all_volumes)
    q75_idx = int(0.75 * (len(sorted_volumes) - 1)) if sorted_volumes else 0
    q75_volume = float(sorted_volumes[q75_idx]) if sorted_volumes else 0.0

    candidates: list[dict[str, Any]] = []

    for i, bin_row in enumerate(profile):
        current_volume = float(bin_row.get("total_volume", 0.0))
        current_delta = abs(float(bin_row.get("delta", 0.0)))

        neighbor_vols = []
        neighbor_deltas = []

        if i - 1 >= 0:
            neighbor_vols.append(float(profile[i - 1].get("total_volume", 0.0)))
            neighbor_deltas.append(abs(float(profile[i - 1].get("delta", 0.0))))
        if i + 1 < len(profile):
            neighbor_vols.append(float(profile[i + 1].get("total_volume", 0.0)))
            neighbor_deltas.append(abs(float(profile[i + 1].get("delta", 0.0))))

        if not neighbor_vols:
            continue

        mean_neighbor_vol = float(sum(neighbor_vols) / len(neighbor_vols))
        mean_neighbor_abs_delta = float(sum(neighbor_deltas) / len(neighbor_deltas)) if neighbor_deltas else 0.0

        topology_prominence = compute_topology_prominence(profile, i)
        neighbor_divergence = compute_neighbor_divergence(profile, i)

        local_volume_anomaly = (
            current_volume >= (1.35 * mean_neighbor_vol)
            and current_volume >= (1.05 * global_mean_volume)
        )
        delta_asymmetry = current_delta >= (1.50 * max(mean_neighbor_abs_delta, 1e-12))

        edge_proximity = classify_edge_proximity(
            bin_row=bin_row,
            val=val,
            vah=vah,
            bin_width=bin_width,
        )

        # Topology-first trigger logic:
        # - Strong structural protrusion, OR
        # - Local structural anomaly + divergence, OR
        # - High absolute volume bulge with moderate protrusion.
        # Delta cannot trigger watch zones alone (secondary metadata only).
        strong_protrusion = topology_prominence >= 1.60
        anomaly_plus_divergence = local_volume_anomaly and neighbor_divergence >= 0.30
        high_volume_bulge = (
            current_volume >= q75_volume
            and topology_prominence >= 1.20
            and neighbor_divergence >= 0.12
        )

        is_structurally_interesting = (
            strong_protrusion
            or anomaly_plus_divergence
            or high_volume_bulge
        )

        if is_structurally_interesting:
            candidates.append(
                {
                    "bin_index": int(bin_row["bin_index"]),
                    "price_low": _round_float(float(bin_row["bin_low"]), 2),
                    "price_high": _round_float(float(bin_row["bin_high"]), 2),
                    "features": {
                        "local_volume_anomaly": bool(local_volume_anomaly),
                        "delta_asymmetry": bool(delta_asymmetry),
                        "edge_proximity": str(edge_proximity),
                        "neighbor_divergence": _round_float(neighbor_divergence, 3),
                        "topology_prominence": _round_float(topology_prominence, 3),
                    },
                }
            )

    # Aggressive noise reduction:
    # Keep only the strongest few protrusions per session.
    candidates.sort(
        key=lambda z: (
            float(z["features"]["topology_prominence"]),
            float(z["features"]["neighbor_divergence"]),
        ),
        reverse=True,
    )

    max_watch_zones = 6
    return candidates[:max_watch_zones]


def export_context_row(session_row: dict[str, Any]) -> dict[str, Any]:
    """
    Build export object for one session.
    """
    return {
        "session_id": str(session_row["session_id"]),
        "watch_zones": detect_watch_zones(session_row),
    }


def build_auction_context(input_path: str = INPUT_PATH, output_path: str = OUTPUT_PATH) -> None:
    """
    Consume structure session profiles and export contextual auction reference rows.
    """
    session_profiles = load_session_profiles(input_path)

    with open(output_path, "w", encoding="utf-8") as f:
        for row in session_profiles:
            context_row = export_context_row(row)
            f.write(json.dumps(context_row) + "\n")


if __name__ == "__main__":
    build_auction_context()
