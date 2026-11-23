# Registration

Extract VINO descriptors from raw fragments, estimate poses with
[SC2-PCR](https://github.com/ZhiChen902/SC2-PCR), and report scene-averaged
registration recall (RR) on 3DMatch or 3DLoMatch.

## Data

Both benchmarks use the same raw fragment clouds from the SC2-PCR test set
([3DMatch.zip](https://drive.google.com/file/d/1kRwuTHlNPr9siENcEMddCO23Oaq0cz-X/view?usp=sharing),
unzipped into `DATA_ROOT`; its precomputed descriptors are not used). For 3DLoMatch, also copy
`3DLoMatch.pkl` and the `benchmarks` folder from the [SC2-PCR repository](https://github.com/ZhiChen902/SC2-PCR):

```text
DATA_ROOT/
├── 3DMatch/
│   ├── fragments/SCENE/cloud_bin_ID.ply
│   └── gt_result/SCENE-evaluation/{gt.log,gt.info}
├── 3DLoMatch.pkl
└── benchmarks/3DLoMatch/SCENE/{gt.log,gt.info}
```

## Run

```bash
export PYTHONPATH=.
python registration/evaluate.py --checkpoint weights/vino/vino_indoor_stage2.pth \
  --data-root DATA_ROOT --benchmark 3dmatch --output exp/registration/3dmatch

python registration/evaluate.py --checkpoint weights/vino/vino_indoor_stage2.pth \
  --data-root DATA_ROOT --benchmark 3dlomatch --output exp/registration/3dlomatch
```

`predictions.npz` stores source-to-target poses and pair IDs. `results.json` stores
per-scene and mean RR as fractions; console output uses percentages. Scoring uses
SC2-PCR's `gt.info`-based error at 0.2 m, not the rotation/translation-threshold
recall.
