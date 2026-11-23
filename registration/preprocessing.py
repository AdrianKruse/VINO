import pickle
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

from third_party.sc2_pcr.benchmark_utils_predator import read_trajectory


SCENES = (
    "7-scenes-redkitchen",
    "sun3d-home_at-home_at_scan1_2013_jan_1",
    "sun3d-home_md-home_md_scan9_2012_sep_30",
    "sun3d-hotel_uc-scan3",
    "sun3d-hotel_umd-maryland_hotel1",
    "sun3d-hotel_umd-maryland_hotel3",
    "sun3d-mit_76_studyroom-76-1studyroom2",
    "sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika",
)


def benchmark_dir(data_root, benchmark, scene):
    root = Path(data_root)
    if benchmark == "3dmatch":
        return root / "3DMatch" / "gt_result" / f"{scene}-evaluation"
    return root / "benchmarks" / "3DLoMatch" / scene


def read_pairs(data_root, benchmark):
    if benchmark == "3dmatch":
        pairs = []
        for scene in SCENES:
            keys, _ = read_trajectory(benchmark_dir(data_root, benchmark, scene) / "gt.log")
            pairs.extend((scene, int(key[0]), int(key[1])) for key in keys)
        return pairs
    with open(Path(data_root) / "3DLoMatch.pkl", "rb") as file:
        metadata = pickle.load(file)
    return [
        (source.split("/")[1], int(Path(source).stem.split("_")[-1]),
         int(Path(target).stem.split("_")[-1]))
        for source, target in zip(metadata["src"], metadata["tgt"])
        if source.split("/")[1] in SCENES
    ]


def grid_sample(coord, grid_size):
    grid = np.floor(coord / grid_size).astype(np.int64)
    grid -= grid.min(0)
    key = np.full(len(grid), 14695981039346656037, dtype=np.uint64)
    for axis in range(3):
        key *= np.uint64(1099511628211)
        key ^= grid[:, axis].astype(np.uint64)
    order = np.argsort(key)
    _, count = np.unique(key[order], return_counts=True)
    selection = np.cumsum(np.insert(count, 0, 0)[:-1])
    selection += np.random.randint(0, count.max(), count.size) % count
    indices = order[selection]
    return indices, grid[indices]


def estimate_normals(coord):
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(coord))
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    cloud.orient_normals_towards_camera_location(np.zeros(3))
    return np.asarray(cloud.normals).astype(np.float32)


def load_fragment(data_root, scene, fragment):
    path = Path(data_root) / "3DMatch" / "fragments" / scene / f"cloud_bin_{fragment}.ply"
    cloud = o3d.io.read_point_cloud(str(path))
    coord = np.asarray(cloud.points).copy()
    indices, grid = grid_sample(coord, 0.02)
    coord = coord[indices]
    normals = estimate_normals(coord)
    inputs = dict(
        feat=torch.from_numpy(np.concatenate([np.zeros_like(normals), normals], axis=1)),
        grid_coord=torch.from_numpy(grid),
        batch=torch.zeros(len(coord), dtype=torch.long),
    )
    return torch.from_numpy(coord.astype(np.float32)), inputs
