#!/usr/bin/env python3
"""
extract_predictions.py  (FACT on our_dataset)
=====================================================================
Loads the best checkpoint (best_ckpt.gz, selected during training by
FACT's own eval_every logic on held-out F1@0.50) and dumps one
frame-level prediction .txt per test video into
results/our_dataset/split_1/<id>.txt, using FACT's own
expand_frame_label (the identical op its compute_metrics uses) to
upsample the model's downsampled pred back to groundTruth length, then
mapping int class ids back to class-name strings via mapping.txt.

Run (from this folder):
    python extract_predictions.py

mapping.txt is read from the shared repo-root dataset/ (see
dataset_paths.py); the checkpoint is fully self-contained (it caches its
own predictions + groundtruth from training-time validation), so this
script has no other dependency on dataset/.
=====================================================================
"""
import os
import sys
import numpy as np

import utils
import utils.evaluate
import utils.utils
from utils.evaluate import Checkpoint
from utils.utils import expand_frame_label
import dataset_paths

# best_ckpt.gz was pickled back when this folder was named FACT_ourdataset,
# so its pickled class references resolve module paths under that name.
# Alias FACT_ourdataset -> this package (now just "utils") so unpickling
# finds them without needing to keep the old folder name.
sys.modules.setdefault("FACT_ourdataset", sys.modules[__name__])
sys.modules["FACT_ourdataset.utils"] = utils
sys.modules["FACT_ourdataset.utils.evaluate"] = utils.evaluate
sys.modules["FACT_ourdataset.utils.utils"] = utils.utils

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "log", "our_dataset", "split1", "our_dataset", "1", "best_ckpt.gz")
RESULTS_DIR = os.path.join(HERE, "results", "our_dataset", "split_1")
MAP_FILE = dataset_paths.mapping_path()

os.makedirs(RESULTS_DIR, exist_ok=True)

index2label = {}
with open(MAP_FILE) as f:
    for line in f:
        parts = line.split()
        if len(parts) == 2:
            index2label[int(parts[0])] = parts[1]

ckpt = Checkpoint.load(CKPT)
print(f"Loaded checkpoint: iteration={ckpt.iteration}, {len(ckpt.videos)} videos")

for vname, video in ckpt.videos.items():
    pred = expand_frame_label(video.pred, len(video.gt_label))
    pred = pred.numpy() if hasattr(pred, "numpy") else np.asarray(pred)
    labels = [index2label[int(p)] for p in pred]
    with open(os.path.join(RESULTS_DIR, f"{vname}.txt"), "w") as f:
        f.write("\n".join(labels) + "\n")

print(f"Wrote {len(ckpt.videos)} prediction files to {RESULTS_DIR}")
