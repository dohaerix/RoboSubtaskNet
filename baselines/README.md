# Baselines on RoboSubtask

Five publicly-known temporal action segmentation architectures (MS-TCN,
MS-TCN++, ASFormer, DiffAct, FACT), each trained on the same RoboSubtask
data as the proposed model, so results in the paper's Table III can be
reproduced directly from this repository.

## Shared dataset — no per-model data copies

All five folders below read features, groundtruth, splits, and the class
mapping **directly from the shared `../dataset/`** at the repo root (see
each folder's `dataset_paths.py`) — none of them keep a local copy. Their
architectures expect a single concatenated 2048-d RGB+flow feature per
frame (the standard MS-TCN-family input format), so `dataset_paths.py`
loads `features_rgb/<vid>.npy` + `features_flow/<vid>.npy` and
concatenates them in memory at load time (padding the shorter stream by
repeating its last frame if RGB/flow differ by a frame or two, a common
I3D artifact). There is exactly one dataset in this repository.

```
baselines/<Model>/dataset_paths.py   # shared-dataset loader (identical across all 5)
```

## Reproducing results

Each folder is runnable standalone from within itself:

| Model | Predict command | Eval command |
|---|---|---|
| MSTCN | `python main.py --action predict --dataset our_dataset --split 1` | `python eval_metrics.py` |
| MSTCN2 | `python main.py --action predict --dataset our_dataset --split 1 --num_epochs 50 --num_layers_PG 11 --num_layers_R 10 --num_R 3` | `python eval_metrics.py` |
| ASFormer | `python main.py --action predict --dataset our_dataset --split 1` | `python eval_metrics.py` |
| DiffAct | `python predict_best.py --config configs/our_dataset.json --device 0 --epoch 40` | `python eval_metrics.py` |
| FACT | `python extract_predictions.py` | `python eval_metrics.py` |

`eval_metrics.py` in each folder writes `metrics_per_activity.csv`,
`flops_params.csv`, and `results_table.png/.pdf` into that same folder —
these are already included (from the run used to verify this
reorganization), so you only need to rerun anything if you want to
reproduce it yourself.

Install dependencies first: `pip install -r ../requirements.txt -r requirements.txt`.

## Checkpoint provenance

Each folder ships only the single checkpoint that actually produced its
results (not every training epoch — mirrors the released
`RoboSubtaskNet` checkpoint's own convention):

| Model | Checkpoint kept | Why this one |
|---|---|---|
| MSTCN | `models/our_dataset/split_1/epoch-50.model` | `main.py` trains 50 epochs; predict loads the final epoch |
| MSTCN2 | `models/our_dataset/split_1/epoch-50.model` | highest epoch actually present after training |
| ASFormer | `models/our_dataset/split_1/epoch-50.model` | `main.py` trains 50 epochs; predict loads the final epoch |
| DiffAct | `result/OurDataset-Trained-S1/epoch-40.model` | best checkpoint by decoder-agg test accuracy (94.70 @ ep40 vs. 94.29 @ ep90) |
| FACT | `log/.../best_ckpt.gz` + `log/.../ckpts/network.iter-96000.net` | `best_ckpt.gz` is FACT's own best-checkpoint selection (also caches its training-time predictions, used directly by `extract_predictions.py`); the final iteration net is included for anyone who wants to run fresh inference rather than replay cached predictions |

## Results (test set, 160 videos / 40 per task)

Regenerated from the checkpoints above against the shared `dataset/`
(numbers here are the freshly-reproduced ones — see caveats below for
where and why they differ very slightly from the paper's Table III):

| Model | Params (M) | GFLOPs | Acc (%) | Edit (%) | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|---|---|
| MS-TCN | 0.795 | 0.087 | 93.05 | 98.52 | 98.82 | 98.69 | 95.94 |
| MS-TCN++ | 0.993 | 0.109 | 93.20 | 98.48 | 98.56 | 98.56 | 94.64 |
| ASFormer | 1.130 | 0.153 | 94.40 | 99.27 | 99.21 | 99.21 | 97.76 |
| DiffAct | 1.204 | 0.418 | 90.45 | 98.67 | 98.68 | 97.88 | 91.93 |
| FACT | 8.896 | 0.585 | 94.47 | 99.48 | 99.34 | 99.34 | 97.63 |
| **RoboSubtaskNet** (../) | 0.819 | 0.090 | 93.92 | 99.30 | 99.21 | 99.21 | 97.76 |

## Known reproduction caveats

- **`pick_pour` task, MS-TCN/MS-TCN++/ASFormer/DiffAct only**: predictions
  and metrics for `mopping`, `pick_place`, and `pick_give` are exact,
  byte-for-byte reproductions of the originally-reported numbers.
  `pick_pour` differs by roughly 0.2–1 point in Acc/F1 (predictions differ
  for a subset of the 40 test videos). This task was re-extracted at some
  point during dataset development (its internal name in these
  checkpoints' training pipeline was `pick_pour_new`); the canonical
  `dataset/` shipped in this repo and what these specific checkpoints were
  originally trained on appear to be very close but not byte-identical
  versions of that re-extraction. It does not change which model ranks
  where. FACT is unaffected (see below).
- **FACT**: its checkpoint internally names this task's videos
  `pick_pour_new_*` rather than `pick_pour_*`; `results/our_dataset/split_1/`
  keeps that native naming (`eval_metrics.py` bridges it to the canonical
  `pick_pour_*` groundtruth automatically). Because FACT's
  `extract_predictions.py` replays cached predictions from the checkpoint
  itself rather than re-running inference on features, its results are an
  **exact** match to the original Table III numbers, including `pick_pour`.
- **DiffAct**: uses DDIM diffusion sampling (25 steps) at inference time.
  Even with `set_sampling_seed: true`, GPU kernel non-determinism across
  runs/environments produces small (~1–4 point) metric differences from
  the originally-reported numbers — expected behavior for diffusion-based
  segmentation, not a data-loading issue (all 4 tasks show similar-sized
  differences, not just `pick_pour`).

## Notes

- `eval.py` in each folder (distinct from `eval_metrics.py`) is the
  original upstream reference evaluation script kept for provenance; some
  are Python 2-only artifacts of the upstream codebases they were adapted
  from. `eval_metrics.py` is the actively-used, Python 3 script that
  produced everything in this README.
- Architectures (`model.py` in each folder) are unmodified from their
  upstream implementations — only data-loading source paths were changed
  to point at the shared `dataset/`, plus the naming/import fixes needed
  to run standalone in this repo layout (see each folder's `dataset_paths.py`
  and, for FACT, the module-alias note at the top of `extract_predictions.py`).
