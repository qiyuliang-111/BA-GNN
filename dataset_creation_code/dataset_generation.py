"""

import glob
import h5py
import json
import numpy as np
import os
import pandas as pd
import platform
import time
import torch
from torch_geometric.data import Data
from scipy.spatial import cKDTree

# ---------------------- Configuration ----------------------
ROOT = r"dataset"
AIRFOIL_PREFIX = None  # Set to "NACA" to process only NACA directories; None processes all airfoil directories.
AIRFOIL_NAMES = sorted([
    d for d in os.listdir(ROOT)
    if os.path.isdir(os.path.join(ROOT, d))
    and (AIRFOIL_PREFIX is None or d.upper().startswith(AIRFOIL_PREFIX.upper()))
])
AOA_LIST = [float(i) for i in range(1, 21)]
OUTPUT_NAME_FMT = "{airfoil}_aoa_{aoa:.1f}.pt"
TOL = 1e-4
USE_AOA_AS_FEATURE = True
FARFIELD_THRESHOLD = 0.78

# For reviewer-oriented timing, keep this False so that automatic recognition
# is actually executed even if edge_type.npy already exists.
USE_SAVED_EDGE_TYPE = False

# Output files
TIMING_CSV = "boundary_recognition_timing.csv"
TIMING_SUMMARY_JSON = "boundary_recognition_timing_summary.json"
DATASET_INDEX_CSV = "dataset_index.csv"

# Edge types
EDGE_TYPE_INTERNAL = 0
EDGE_TYPE_AIRFOIL = 1
EDGE_TYPE_FARFIELD = 2

# Node types
NODE_TYPE_INTERIOR = 0
NODE_TYPE_AIRFOIL = 1
NODE_TYPE_FARFIELD = 2


# ---------------------- Utility functions ----------------------
def smart_read(f, path):
    if path not in f:
        raise KeyError(f"{path} not found in CGNS")
    obj = f[path]
    if isinstance(obj, h5py.Dataset):
        return obj[()]
    if isinstance(obj, h5py.Group):
        datasets = [k for k, v in obj.items() if isinstance(v, h5py.Dataset)]
        if not datasets:
            raise KeyError(f"{path} is a Group but contains no Dataset")
        return obj[datasets[0]][()]
    raise TypeError(f"Unsupported object type: {type(obj)}")


def read_cgns_nodes_and_cells(cgns_path):
    with h5py.File(cgns_path, "r") as f:
        base = "Base/dom-1"
        x = smart_read(f, f"{base}/GridCoordinates/CoordinateX")
        y = smart_read(f, f"{base}/GridCoordinates/CoordinateY")
        coords = np.vstack([x, y]).T

        tri, quad = None, None
        if f"{base}/TriElements" in f:
            arr = smart_read(f, f"{base}/TriElements/ElementConnectivity")
            tri = arr.reshape(-1, 3) - 1
        if f"{base}/QuadElements" in f:
            arr = smart_read(f, f"{base}/QuadElements/ElementConnectivity")
            quad = arr.reshape(-1, 4) - 1

    return coords, tri, quad


def cells_to_edges(tri, quad):
    """Construct unique undirected edges and count cell occurrences."""
    edge_count = {}

    def add_edge(a, b):
        key = (min(int(a), int(b)), max(int(a), int(b)))
        edge_count[key] = edge_count.get(key, 0) + 1

    if tri is not None:
        for elem in tri:
            for i in range(3):
                add_edge(elem[i], elem[(i + 1) % 3])

    if quad is not None:
        for elem in quad:
            for i in range(4):
                add_edge(elem[i], elem[(i + 1) % 4])

    if not edge_count:
        return np.zeros((2, 0), dtype=int), np.zeros((0,), dtype=int)

    edges_list = np.array(list(edge_count.keys()), dtype=int)
    counts = np.array([edge_count[tuple(e)] for e in edges_list], dtype=int)
    return edges_list.T, counts


def detect_edge_types(coords, edges, counts, farfield_threshold=0.78):
    """Classify unique edges as internal, airfoil boundary, or far-field."""
    edge_type = np.zeros(edges.shape[1], dtype=np.int64)

    x_min, y_min = np.min(coords, axis=0)
    x_max, y_max = np.max(coords, axis=0)
    x_range, y_range = x_max - x_min, y_max - y_min

    x_thresh_min = x_min + (1.0 - farfield_threshold) * x_range / 2.0
    x_thresh_max = x_max - (1.0 - farfield_threshold) * x_range / 2.0
    y_thresh_min = y_min + (1.0 - farfield_threshold) * y_range / 2.0
    y_thresh_max = y_max - (1.0 - farfield_threshold) * y_range / 2.0

    for i in range(edges.shape[1]):
        if counts[i] != 1:
            edge_type[i] = EDGE_TYPE_INTERNAL
            continue

        u, v = edges[0, i], edges[1, i]
        u_coord, v_coord = coords[u], coords[v]

        is_farfield = (
            u_coord[0] <= x_thresh_min or u_coord[0] >= x_thresh_max or
            u_coord[1] <= y_thresh_min or u_coord[1] >= y_thresh_max or
            v_coord[0] <= x_thresh_min or v_coord[0] >= x_thresh_max or
            v_coord[1] <= y_thresh_min or v_coord[1] >= y_thresh_max
        )
        edge_type[i] = EDGE_TYPE_FARFIELD if is_farfield else EDGE_TYPE_AIRFOIL

    return edge_type


def assign_node_types(num_nodes, edges, edge_type):
    """Assign node types from the recognized edge types."""
    node_type = np.zeros(num_nodes, dtype=np.int64)

    for i in range(edges.shape[1]):
        u, v = edges[0, i], edges[1, i]

        if edge_type[i] == EDGE_TYPE_AIRFOIL:
            node_type[u] = NODE_TYPE_AIRFOIL
            node_type[v] = NODE_TYPE_AIRFOIL

        elif edge_type[i] == EDGE_TYPE_FARFIELD:
            if node_type[u] != NODE_TYPE_AIRFOIL:
                node_type[u] = NODE_TYPE_FARFIELD
            if node_type[v] != NODE_TYPE_AIRFOIL:
                node_type[v] = NODE_TYPE_FARFIELD

    return node_type


def recognize_boundaries_with_timing(
    coords,
    tri,
    quad,
    saved_edge_type=None,
    farfield_threshold=0.78,
):
    """
    Execute and time the automatic boundary-recognition procedure.

    File I/O and subsequent graph construction are excluded from the timing.
    """
    # 1) Cell connectivity -> unique edges and edge occurrence counts
    t0 = time.perf_counter()
    edges, counts = cells_to_edges(tri, quad)
    t1 = time.perf_counter()

    # 2) Edge-type recognition
    used_saved_edge_type = False
    if (
        USE_SAVED_EDGE_TYPE
        and saved_edge_type is not None
        and saved_edge_type.shape[0] == edges.shape[1]
    ):
        edge_type = saved_edge_type.astype(np.int64)
        used_saved_edge_type = True
        t2 = time.perf_counter()
    else:
        edge_type = detect_edge_types(
            coords,
            edges,
            counts,
            farfield_threshold=farfield_threshold,
        )
        t2 = time.perf_counter()

    # 3) Edge types -> node types
    node_type = assign_node_types(coords.shape[0], edges, edge_type)
    t3 = time.perf_counter()

    timing = {
        "time_cells_to_edges_s": t1 - t0,
        "time_detect_edge_types_s": t2 - t1,
        "time_assign_node_types_s": t3 - t2,
        "time_total_boundary_recognition_s": t3 - t0,
        "used_saved_edge_type": int(used_saved_edge_type),
    }
    return edges, counts, edge_type, node_type, timing


def build_edge_attr(coords, edges, counts=None):
    """
    Return five-dimensional edge features:
    [delta_x_norm, delta_y_norm, length_norm, unit_x, unit_y].
    """
    src, dst = edges
    vec = coords[dst] - coords[src]
    length = np.linalg.norm(vec, axis=1, keepdims=True)
    unit_vec = vec / (length + 1e-12)

    max_len = np.max(length) + 1e-12
    vec_norm = vec / max_len
    length_norm = length / max_len

    return np.hstack([vec_norm, length_norm, unit_vec]).astype(np.float32)


def read_dat(dat_path):
    df = pd.read_csv(dat_path, header=None, sep=",", comment="#", engine="python")
    if df.shape[1] < 6:
        # Compatibility with the user's original data format.
        df = pd.read_csv(
            dat_path,
            header=None,
            sep=r"\s+",
            comment="#",
            engine="python",
        )

    df = df.iloc[:, :6]
    df.columns = ["id", "x", "y", "cp", "u", "v"]
    df["id"] = df["id"].astype(float).astype(int) - 1
    return df


def simple_data_matching(coords, dat_df, tol=1e-4):
    dat_coords = dat_df[["x", "y"]].values
    tree = cKDTree(coords)
    dist, idx = tree.query(dat_coords, k=1)

    mismatch_count = int(np.sum(dist > tol))
    if mismatch_count > 0:
        print(f"Warning: {mismatch_count} matching distances exceed {tol}")

    return idx, dist


def find_dat_path(ascii_dir, aoa):
    patterns = [
        f"sf_aoa_{aoa:.1f}",
        f"sf_aoa_{int(aoa)}.0",
        f"sf_aoa_{int(aoa)}",
        f"aoa_{aoa:.1f}",
    ]
    for pattern in patterns:
        matches = glob.glob(os.path.join(ascii_dir, pattern + "*"))
        if matches:
            return matches[0]
    return None


def build_pyg_data_from_precomputed(
    coords,
    edges,
    edge_attr_np,
    edge_type_np,
    node_type_np,
    dat_path,
    aoa,
    tol=1e-4,
):
    """Build one AoA graph using geometry information computed once per mesh."""
    dat_df = read_dat(dat_path)
    matched_indices, dist = simple_data_matching(coords, dat_df, tol)

    y = np.zeros((coords.shape[0], 3), dtype=np.float32)
    y[matched_indices] = dat_df[["cp", "u", "v"]].values.astype(np.float32)

    num_nodes = coords.shape[0]
    if USE_AOA_AS_FEATURE:
        aoa_feat = np.full((num_nodes, 1), aoa, dtype=np.float32)
        x_feat = np.hstack([coords, node_type_np.reshape(-1, 1), aoa_feat])
    else:
        x_feat = np.hstack([coords, node_type_np.reshape(-1, 1)])

    data = Data(
        x=torch.tensor(x_feat, dtype=torch.float32),
        edge_index=torch.tensor(edges, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr_np, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.float32),
        node_type=torch.tensor(node_type_np, dtype=torch.long),
        edge_type=torch.tensor(edge_type_np, dtype=torch.long),
    )
    data.matched_indices = matched_indices
    data.matching_distances = dist
    return data


def save_timing_outputs(root_dir, timing_rows, num_graph_samples):
    timing_path = os.path.join(root_dir, TIMING_CSV)
    timing_df = pd.DataFrame(timing_rows)
    timing_df.to_csv(timing_path, index=False, encoding="utf-8-sig")

    if timing_df.empty:
        print("No boundary-recognition timing records were generated.")
        return

    total_time = float(timing_df["time_total_boundary_recognition_s"].sum())
    mean_time = float(timing_df["time_total_boundary_recognition_s"].mean())
    std_time = float(timing_df["time_total_boundary_recognition_s"].std(ddof=1)) if len(timing_df) > 1 else 0.0
    median_time = float(timing_df["time_total_boundary_recognition_s"].median())

    summary = {
        "timing_scope": (
            "cells_to_edges + detect_edge_types + assign_node_types; "
            "file I/O, KD-tree matching, edge attributes, tensor conversion, and saving excluded"
        ),
        "num_airfoil_meshes": int(len(timing_df)),
        "num_generated_graph_samples": int(num_graph_samples),
        "total_boundary_recognition_time_s": total_time,
        "mean_time_per_airfoil_mesh_s": mean_time,
        "std_time_per_airfoil_mesh_s": std_time,
        "median_time_per_airfoil_mesh_s": median_time,
        "min_time_per_airfoil_mesh_s": float(timing_df["time_total_boundary_recognition_s"].min()),
        "max_time_per_airfoil_mesh_s": float(timing_df["time_total_boundary_recognition_s"].max()),
        "amortized_time_per_generated_graph_sample_s": (
            total_time / num_graph_samples if num_graph_samples > 0 else None
        ),
        "total_nodes_across_meshes": int(timing_df["N_nodes"].sum()),
        "total_unique_edges_across_meshes": int(timing_df["E_unique"].sum()),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }

    summary_path = os.path.join(root_dir, TIMING_SUMMARY_JSON)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("Automatic boundary-recognition timing summary")
    print("=" * 80)
    print(f"Airfoil meshes processed : {summary['num_airfoil_meshes']}")
    print(f"Generated graph samples  : {summary['num_generated_graph_samples']}")
    print(f"Total time               : {total_time:.6f} s")
    print(f"Mean time per mesh       : {mean_time:.6f} s")
    print(f"Std. time per mesh       : {std_time:.6f} s")
    print(f"Median time per mesh     : {median_time:.6f} s")
    if num_graph_samples > 0:
        print(
            "Amortized time per graph : "
            f"{summary['amortized_time_per_generated_graph_sample_s']:.8f} s"
        )
    print(f"Timing CSV               : {timing_path}")
    print(f"Summary JSON             : {summary_path}")
    print("=" * 80)

    # Ready-to-copy sentence for the reviewer response.
    print("\nReviewer-response sentence template:")
    print(
        f"Automatic boundary recognition was performed for {len(timing_df)} airfoil "
        f"meshes in {total_time:.4f} s, corresponding to "
        f"{mean_time:.6f} ± {std_time:.6f} s per mesh."
    )


def build_dataset_for_root(root_dir, out_format=OUTPUT_NAME_FMT, tol=TOL):
    index_rows = []
    timing_rows = []

    for airfoil in AIRFOIL_NAMES:
        air_dir = os.path.join(root_dir, airfoil)
        cgns_dir = os.path.join(air_dir, "CGNS")
        ascii_dir = os.path.join(air_dir, "ASCII")

        cgns_files = glob.glob(os.path.join(cgns_dir, "*.cgns"))
        if not cgns_files:
            continue
        cgns_path = cgns_files[0]

        # File reading is intentionally outside the boundary-recognition timer.
        try:
            coords, tri, quad = read_cgns_nodes_and_cells(cgns_path)
        except Exception as exc:
            print(f"Failed to read mesh for {airfoil}: {exc}")
            continue

        saved_edge_type = None
        edge_type_path = os.path.join(air_dir, "edge_type.npy")
        if os.path.exists(edge_type_path):
            saved_edge_type = np.load(edge_type_path)

        try:
            edges, counts, edge_type, node_type, timing = recognize_boundaries_with_timing(
                coords,
                tri,
                quad,
                saved_edge_type=saved_edge_type,
                farfield_threshold=FARFIELD_THRESHOLD,
            )
        except Exception as exc:
            print(f"Boundary recognition failed for {airfoil}: {exc}")
            continue

        # Edge attributes are not part of the boundary-recognition timing.
        edge_attr_np = build_edge_attr(coords, edges, counts)

        num_tri = 0 if tri is None else int(tri.shape[0])
        num_quad = 0 if quad is None else int(quad.shape[0])
        num_boundary_edges = int(np.sum(counts == 1))

        timing_rows.append({
            "airfoil": airfoil,
            "cgns_file": os.path.basename(cgns_path),
            "N_nodes": int(coords.shape[0]),
            "F_tri": num_tri,
            "F_quad": num_quad,
            "F_total": num_tri + num_quad,
            "E_unique": int(edges.shape[1]),
            "B_boundary_edges": num_boundary_edges,
            "edge_type_internal": int(np.sum(edge_type == EDGE_TYPE_INTERNAL)),
            "edge_type_airfoil": int(np.sum(edge_type == EDGE_TYPE_AIRFOIL)),
            "edge_type_farfield": int(np.sum(edge_type == EDGE_TYPE_FARFIELD)),
            **timing,
            "time_per_node_us": timing["time_total_boundary_recognition_s"] / max(coords.shape[0], 1) * 1e6,
            "time_per_unique_edge_us": timing["time_total_boundary_recognition_s"] / max(edges.shape[1], 1) * 1e6,
        })

        print(
            f"[{airfoil}] boundary recognition: "
            f"N={coords.shape[0]}, E={edges.shape[1]}, "
            f"time={timing['time_total_boundary_recognition_s']:.6f} s"
        )

        # Reuse the same recognized geometry for every AoA case.
        for aoa in AOA_LIST:
            dat_path = find_dat_path(ascii_dir, aoa)
            if dat_path is None:
                continue

            try:
                data = build_pyg_data_from_precomputed(
                    coords=coords,
                    edges=edges,
                    edge_attr_np=edge_attr_np,
                    edge_type_np=edge_type,
                    node_type_np=node_type,
                    dat_path=dat_path,
                    aoa=aoa,
                    tol=tol,
                )
            except Exception as exc:
                print(f"Failed: {airfoil}, AoA={aoa}: {exc}")
                continue

            save_dir = os.path.join(root_dir, airfoil, "dataset_pyg")
            os.makedirs(save_dir, exist_ok=True)

            out_name = out_format.format(airfoil=airfoil, aoa=aoa)
            out_path = os.path.join(save_dir, out_name)
            torch.save(data, out_path)

            index_rows.append({
                "airfoil": airfoil,
                "aoa": aoa,
                "N_nodes": int(data.x.shape[0]),
                "E_edges": int(data.edge_index.shape[1]),
                "edge_type_internal": int(torch.sum(data.edge_type == EDGE_TYPE_INTERNAL)),
                "edge_type_airfoil": int(torch.sum(data.edge_type == EDGE_TYPE_AIRFOIL)),
                "edge_type_farfield": int(torch.sum(data.edge_type == EDGE_TYPE_FARFIELD)),
            })

    index_df = pd.DataFrame(index_rows)
    index_path = os.path.join(root_dir, DATASET_INDEX_CSV)
    index_df.to_csv(index_path, index=False, encoding="utf-8-sig")

    save_timing_outputs(root_dir, timing_rows, num_graph_samples=len(index_rows))
    print(f"\nDataset creation completed. Index saved to: {index_path}")


if __name__ == "__main__":
    build_dataset_for_root(ROOT, out_format=OUTPUT_NAME_FMT, tol=TOL)
