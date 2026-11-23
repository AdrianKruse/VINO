"""
NYUv2 Dataset
"""

import os
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.io import loadmat
from scipy.ndimage import distance_transform_edt, maximum_filter, minimum_filter

from .builder import DATASETS
from .defaults import DefaultDataset

# RGB camera of NYU Depth v2 (camera_params.m of the NYU toolbox), shared by all frames
INTRINSICS = [
    [518.857901, 0.0, 325.582449],
    [0.0, 519.469611, 253.736166],
    [0.0, 0.0, 1.0],
]


def backproject_depth(depth, intrinsic, rtilt):
    height, width = depth.shape
    pixel_x, pixel_y = np.meshgrid(
        np.arange(1, width + 1, dtype=np.float32),
        np.arange(1, height + 1, dtype=np.float32),
    )
    valid = np.isfinite(depth) & (depth > 0)
    depth = np.where(valid, depth, 0.0).astype(np.float32)
    x = (pixel_x - intrinsic[0, 2]) * depth / intrinsic[0, 0]
    y = (pixel_y - intrinsic[1, 2]) * depth / intrinsic[1, 1]
    points = np.einsum("ij,hwj->hwi", rtilt, np.stack((x, depth, -y), axis=-1))
    points[~valid] = 0
    return np.ascontiguousarray(points, dtype=np.float32), valid


def organized_normals(points, valid):
    down = np.zeros_like(points)
    right = np.zeros_like(points)
    down[:-1] = points[:-1] - points[1:]
    right[:, :-1] = points[:, :-1] - points[:, 1:]
    normals = np.cross(down, right)
    supported = np.zeros_like(valid)
    supported[:-1, :-1] = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:]
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = np.divide(
        normals, norm, out=np.zeros_like(normals), where=(norm > 0) & supported[..., None]
    )
    return np.ascontiguousarray(normals, dtype=np.float32)


def depth_edges(depth, valid, ltol=0.01):
    # inverse-depth Laplacian criterion of MoGe's utils3d depth_map_edge
    with np.errstate(divide="ignore", invalid="ignore"):
        disparity = np.where(valid, 1.0 / depth, np.nan)
        values = np.pad(np.nan_to_num(disparity, nan=0.0), 1, mode="edge")
        counts = np.pad(valid, 1, mode="edge")
        for axis in (0, 1):
            values = np.lib.stride_tricks.sliding_window_view(values, 3, axis=axis).sum(-1)
            counts = np.lib.stride_tricks.sliding_window_view(counts, 3, axis=axis).sum(-1)
        laplacian = (disparity - values / counts) / disparity
    high = maximum_filter(
        np.where(np.isnan(laplacian), -np.inf, laplacian), size=3, mode="constant", cval=-np.inf
    )
    low = minimum_filter(
        np.where(np.isnan(laplacian), np.inf, laplacian), size=3, mode="constant", cval=np.inf
    )
    return (high > ltol) & (low < -ltol)


def nearest_kept_pixel(keep):
    index = np.full(keep.shape, -1, dtype=np.int64)
    index[keep] = np.arange(keep.sum())
    if keep.all():
        return index.reshape(-1)
    nearest = distance_transform_edt(~keep, return_distances=False, return_indices=True)
    return index[nearest[0], nearest[1]].reshape(-1)


@DATASETS.register_module()
class NYUv2Dataset(DefaultDataset):
    def __init__(
        self,
        depth_root="data/nyuv2/depth",
        intrinsic=INTRINSICS,
        edge_filter=True,
        edge_ltol=0.01,
        **kwargs,
    ):
        root = Path(kwargs.get("data_root", "data/nyuv2"))
        self.labeled_file = root / "nyu_depth_v2_labeled.mat"
        self.depth_root = Path(depth_root)
        # gravity alignment (Rtilt) of all 1449 frames, taken from the SUN RGB-D metadata
        self.rtilt = np.load(
            Path(__file__).parent / "preprocessing" / "nyuv2" / "meta_data" / "rtilt.npy"
        )
        self.edge_filter = edge_filter
        self.edge_ltol = edge_ltol
        mapping = loadmat(root / "classMapping40.mat", squeeze_me=True)["mapClass"]
        self.label_map = np.concatenate(([-1], np.asarray(mapping) - 1)).astype(np.int64)
        split = loadmat(root / "splits.mat")
        self.splits = dict(
            train=split["trainNdxs"].reshape(-1).astype(int),
            val=split["testNdxs"].reshape(-1).astype(int),
            test=split["testNdxs"].reshape(-1).astype(int),
        )
        self.intrinsic = np.asarray(intrinsic, dtype=np.float32)
        self._h5 = None
        self._pid = None
        super().__init__(**kwargs)

    def get_data_list(self):
        return self.splits[self.split].tolist()

    def get_data_name(self, idx):
        return f"NYU{self.data_list[idx % len(self.data_list)]:04d}"

    def get_split_name(self, idx):
        return self.split

    def _labeled(self):
        if self._h5 is None or self._pid != os.getpid():
            self._h5 = h5py.File(self.labeled_file, "r")
            self._pid = os.getpid()
        return self._h5

    def get_data(self, idx):
        frame = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        h5 = self._labeled()
        color = np.asarray(h5["images"][frame - 1]).transpose(2, 1, 0).astype(np.float32)
        segment = self.label_map[np.asarray(h5["labels"][frame - 1]).T]
        depth = np.load(self.depth_root / name / "depth.npy")
        points, valid = backproject_depth(depth, self.intrinsic, self.rtilt[frame - 1])
        normal = organized_normals(points, valid)
        keep = valid
        if self.edge_filter:
            keep = valid & ~depth_edges(depth, valid, self.edge_ltol)
        data_dict = dict(
            coord=points[keep],
            color=color[keep],
            normal=normal[keep],
            segment=segment[keep],
            name=name,
            split=self.split,
        )
        if self.split != "train" or self.test_mode:
            data_dict["pixel_segment"] = segment.reshape(-1)
            data_dict["pixel_inverse"] = nearest_kept_pixel(keep)
        return data_dict

    def prepare_train_data(self, idx):
        data_dict = self.get_data(idx)
        pixel_segment = data_dict.pop("pixel_segment", None)
        pixel_inverse = data_dict.pop("pixel_inverse", None)
        data_dict = self.transform(data_dict)
        if pixel_segment is not None:
            data_dict["origin_segment"] = torch.from_numpy(pixel_segment)
            data_dict["inverse"] = data_dict["inverse"][torch.from_numpy(pixel_inverse)]
        return data_dict

    def prepare_test_data(self, idx):
        data_dict = self.get_data(idx)
        pixel_segment = data_dict.pop("pixel_segment")
        pixel_inverse = data_dict.pop("pixel_inverse")
        data_dict = self.transform(data_dict)
        result_dict = dict(
            segment=data_dict.pop("segment"),
            name=data_dict.pop("name"),
            origin_segment=pixel_segment,
            inverse=data_dict.pop("inverse")[pixel_inverse],
        )
        data_dict.pop("origin_segment", None)

        fragment_list = []
        for aug in self.aug_transform:
            data_part_list = self.test_voxelize(aug(deepcopy(data_dict)))
            for data_part in data_part_list:
                if self.test_crop is not None:
                    fragment_list += self.test_crop(data_part)
                else:
                    fragment_list.append(data_part)
        result_dict["fragment_list"] = [self.post_transform(f) for f in fragment_list]
        return result_dict

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        state["_pid"] = None
        return state
