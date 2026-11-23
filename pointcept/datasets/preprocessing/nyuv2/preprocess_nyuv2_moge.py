"""
Predict metric depth for the labelled NYUv2 frames with MoGe-3
"""

import argparse
import math
from pathlib import Path

import h5py
import numpy as np
import torch
from moge.model.v3 import MoGeModel

# focal length fx of the NYU Depth v2 RGB camera (camera_params.m of the NYU toolbox)
FOCAL_X = 518.857901


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="data/nyuv2")
    parser.add_argument("--output_root", default="data/nyuv2/depth")
    parser.add_argument("--checkpoint", required=True, help="MoGe-3 ViT-G checkpoint")
    parser.add_argument("--resolution_level", type=int, default=30)
    parser.add_argument("--refine_steps", type=int, default=5)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    model = MoGeModel.from_pretrained(args.checkpoint).cuda().eval()
    fov_x = math.degrees(2 * math.atan(640 / (2 * FOCAL_X)))
    with h5py.File(Path(args.dataset_root) / "nyu_depth_v2_labeled.mat", "r") as f:
        images = f["images"]
        for index in range(len(images)):
            output_path = output_root / f"NYU{index + 1:04d}" / "depth.npy"
            if output_path.exists():
                continue
            image = np.asarray(images[index]).transpose(2, 1, 0)
            image = torch.from_numpy(image.copy()).cuda().permute(2, 0, 1).float() / 255
            output = model.infer(
                image,
                resolution_level=args.resolution_level,
                refine_steps=args.refine_steps,
                use_fp16=False,
                fov_x=fov_x,
                apply_mask=True,
            )
            output_path.parent.mkdir(exist_ok=True)
            np.save(output_path, output["depth"].cpu().numpy().astype(np.float32))
            print(f"[{index + 1}/{len(images)}] {output_path.parent.name}", flush=True)


if __name__ == "__main__":
    main()
