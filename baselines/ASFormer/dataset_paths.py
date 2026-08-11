"""Shared-dataset loader.

Reads I3D RGB + optical-flow features and groundtruth directly from the
repo-root ``dataset/`` (features_rgb, features_flow, groundtruth, splits,
mapping.txt) and concatenates RGB+flow into the (2048, T) feature this
architecture expects, in memory. This baseline keeps no local copy of the
dataset -- ``dataset/`` at the repo root is the single shared source for
all six models in this repository.
"""
import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", "dataset"))
_SPLITS = ("train", "test")


def _find(split_subdir, vid, ext):
    for split in _SPLITS:
        p = os.path.join(DATASET_ROOT, split, split_subdir, vid + ext)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"{vid}{ext} not found under dataset/{{train,test}}/{split_subdir}/ "
        f"(looked in {DATASET_ROOT})"
    )


def load_combined_features(vid):
    """vid: base video id, e.g. 'mopping_1' (no extension).

    Returns the (2048, T) RGB+flow concatenated feature array -- the input
    format this architecture (and the other MS-TCN-family baselines) expects.
    """
    rgb = np.load(_find("features_rgb", vid, ".npy"))    # (T_rgb, 1024)
    flow = np.load(_find("features_flow", vid, ".npy"))  # (T_flow, 1024)
    # RGB and flow I3D streams can differ by a frame or two; pad the shorter
    # one by repeating its last frame (matching the original feature_combiner.py
    # pipeline) rather than truncating, so lengths always match the longer one.
    T = max(rgb.shape[0], flow.shape[0])
    if rgb.shape[0] < T:
        rgb = np.concatenate([rgb, np.repeat(rgb[-1:], T - rgb.shape[0], axis=0)], axis=0)
    if flow.shape[0] < T:
        flow = np.concatenate([flow, np.repeat(flow[-1:], T - flow.shape[0], axis=0)], axis=0)
    return np.concatenate([rgb.T, flow.T], axis=0).astype(np.float32)  # (2048, T)


def groundtruth_path(vid):
    """vid: base video id, e.g. 'mopping_1' (no extension)."""
    return _find("groundtruth", vid, ".txt")


def mapping_path():
    return os.path.join(DATASET_ROOT, "mapping.txt")


def split_bundle_path(split):
    """split: 'train' or 'test'."""
    return os.path.join(DATASET_ROOT, "splits", f"{split}.bundle")
