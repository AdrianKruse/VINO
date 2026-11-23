import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pointcept.models.volt import Volt
from preprocessing import benchmark_dir, load_fragment, read_pairs
from third_party.sc2_pcr.SC2_PCR import Matcher
from third_party.sc2_pcr.benchmark_utils_predator import (
    evaluate_registration,
    read_trajectory,
    read_trajectory_info,
)


def load_model(checkpoint):
    model = Volt(
        in_channels=6, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        init_values=1e-5, qk_norm=True, up_mlp_dim=None, num_register_tokens=16,
        tokenizer_mean_pool=True,
    )
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(state_dict)
    return model


def extract(model, args, scene, fragment):
    coord, inputs = load_fragment(args.data_root, scene, fragment)
    feature = model({key: value.cuda() for key, value in inputs.items()})
    feature = F.normalize(feature.float(), dim=1)
    index = torch.randperm(len(coord))[:5000] if len(coord) > 5000 else torch.arange(len(coord))
    return coord[index][None].cuda(), feature[index.cuda()][None]


def score(data_root, benchmark, pairs, transforms):
    recalls = {}
    for scene in sorted({pair[0] for pair in pairs}):
        root = benchmark_dir(data_root, benchmark, scene)
        gt_pairs, gt = read_trajectory(root / "gt.log")
        n_fragments, info = read_trajectory_info(root / "gt.info")
        gt_keys = {tuple(map(int, pair[:2])) for pair in gt_pairs}
        result_pairs, poses = [], []
        for (pair_scene, source, target), pose in zip(pairs, transforms):
            if pair_scene != scene:
                continue
            pose = np.asarray(pose, dtype=float)
            if (source, target) in gt_keys:
                pose = np.linalg.inv(pose)
            else:
                source, target = target, source
                assert (source, target) in gt_keys
            result_pairs.append((source, target, n_fragments))
            poses.append(pose)
        _, recall, _ = evaluate_registration(
            n_fragments, np.asarray(poses), np.asarray(result_pairs),
            gt_pairs, gt, info, err2=0.2,
        )
        recalls[scene] = float(recall)
        print(f"{scene}: RR {100 * recall:.2f}%", flush=True)
    return recalls


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("3dmatch", "3dlomatch"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, help="seed NumPy and torch to repeat a run exactly")
    args = parser.parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    model = load_model(args.checkpoint).cuda().eval()
    matcher = Matcher(
        inlier_threshold=0.1, num_node="all", use_mutual=False, d_thre=0.1,
        num_iterations=10, ratio=0.2, nms_radius=0.1, max_points=8000,
        k1=30, k2=20,
    )
    pairs = read_pairs(args.data_root, args.benchmark)
    transforms = []
    for index, (scene, source, target) in enumerate(pairs):
        source_coord, source_feature = extract(model, args, scene, source)
        target_coord, target_feature = extract(model, args, scene, target)
        transform, _, _, _ = matcher.estimator(
            source_coord, target_coord, source_feature, target_feature,
        )
        transforms.append(transform[0].cpu().numpy())
        print(f"[{index + 1}/{len(pairs)}] {scene}: {source} -> {target}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output / "predictions.npz", transform=np.asarray(transforms),
        scene=[p[0] for p in pairs], source=[p[1] for p in pairs],
        target=[p[2] for p in pairs],
    )
    recalls = score(args.data_root, args.benchmark, pairs, transforms)
    results = dict(
        checkpoint=args.checkpoint, benchmark=args.benchmark, seed=args.seed,
        pairs=len(pairs), scene_rr=recalls, rr=float(np.mean(list(recalls.values()))),
    )
    print(f"Mean RR: {100 * results['rr']:.2f}%", flush=True)
    with (args.output / "results.json").open("w") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
