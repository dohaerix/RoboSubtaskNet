#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_fig5_all.py
=====================================================================
Reproduce Fig. 5 (RoboSubtaskNet paper) style qualitative segmentation
strips comparing GROUND TRUTH against multiple models on ONE video:

    GT  |  MS-TCN  |  MS-TCN++  |  ASFormer  |  FACT  |  DiffAct  |  RoboSubtaskNet

All 6 models are looked up by the same "<task>_<local_id>" filename
against the shared repo-root dataset/ groundtruth (FACT's checkpoint
internally names pick_pour videos "pick_pour_new_*" -- handled via an
alias, see baselines/README.md).

USAGE
-----
    python visualize_fig5_all.py                        # both tasks, video 483
    python visualize_fig5_all.py --task pick_place --robo_idx 483
    python visualize_fig5_all.py --task pick_pour  --robo_idx 483
    python visualize_fig5_all.py --task pick_place --find_best        # auto-pick best video
    python visualize_fig5_all.py --task pick_place --find_best --topk 5   # show top-5 candidates

Output PNGs are written under figures/fig5_visualizations/.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# =====================================================================
# 1. PATH CONFIGURATION  -- edit here if any folder moves
# =====================================================================

# All 6 models trained on our_dataset share IDENTICAL filenames
# ("<task>_<local_id>", e.g. pick_place_483) against the shared repo-root
# dataset/ ground truth -- every model's row is looked up by the same
# name. (Exception: FACT's checkpoint internally names pick_pour videos
# "pick_pour_new_*" -- see baselines/README.md -- handled below.)
HERE_ROOT = Path(__file__).parent
_DATASET = HERE_ROOT / "dataset"


def _gt_dir_for(vid_name):
    """dataset/ splits groundtruth into train/ and test/ -- resolve which."""
    for split in ("test", "train"):
        d = _DATASET / split / "groundtruth"
        if (d / f"{vid_name}.txt").is_file():
            return d
    return _DATASET / "test" / "groundtruth"  # fallback for error messages


class _GTDirProxy:
    """Path-like object that resolves train/ vs test/ groundtruth per file."""
    def __truediv__(self, fname):
        vid_name = fname[:-4] if fname.endswith(".txt") else fname
        return _gt_dir_for(vid_name) / fname

    def __str__(self):
        return str(_DATASET / "{train,test}" / "groundtruth")


GLOBAL_GT = _GTDirProxy()

# Per-model PREDICTION directories.
PRED_DIRS = {
    "MS-TCN":         HERE_ROOT / "baselines/MSTCN/results/our_dataset/split_1",
    "MS-TCN++":       HERE_ROOT / "baselines/MSTCN2/results/our_dataset/split_1",
    "ASFormer":       HERE_ROOT / "baselines/ASFormer/results/our_dataset/split_1",
    "FACT":           HERE_ROOT / "baselines/FACT/results/our_dataset/split_1",
    "DiffAct":        HERE_ROOT / "baselines/DiffAct/result/OurDataset-Trained-S1/prediction",
    "RoboSubtaskNet": HERE_ROOT / "results/bottleneck_h128fm48_dropout03",
}

# RoboSubtaskNet uses the SAME ground truth dir as everything else now.
ROBO_GT = GLOBAL_GT

# Row order in the figure (top -> bottom). Rows whose pred dir is None
# or whose file is missing are skipped automatically.
ROW_ORDER = ["MS-TCN", "MS-TCN++", "ASFormer", "DiffAct", "FACT", "RoboSubtaskNet"]

# Where to drop the PNGs.
OUT_DIR = HERE_ROOT / "figures" / "fig5_visualizations"

# Map a friendly task name -> the RoboSubtaskNet file-name prefix.
ROBO_TASK_PREFIX = {
    "pick_place": "pick_place",
    "pick_pour":  "pick_pour",
}


# =====================================================================
# 2. COLOR / LABEL SCHEME (matches the paper figure)
# =====================================================================

COLORS = {
    "Reach":    "#e9969d",
    "Pick":     "#8ca65a",
    "Move":     "#30326f",
    "Place":    "#f2c45b",
    "Withdraw": "#d98bd4",
    "Tilt":     "#d98b45",
    "Give":     "#d98b45",
    "Wipe":     "#8a8a8a",
}

DISPLAY = {
    "Reach": "Reach", "Pick": "Pick", "Move": "Move", "Place": "Place",
    "Withdraw": "Retract", "Tilt": "Pour", "Give": "Give", "Wipe": "Wipe",
}

LEGEND_ORDER = {
    "pick_place": ["Reach", "Pick", "Move", "Place", "Withdraw"],
    "pick_pour":  ["Reach", "Pick", "Move", "Tilt", "Place", "Withdraw"],
}

# Synonyms / alternate spellings some repos emit -> canonical name.
SYNONYMS = {
    "retract": "Withdraw", "withdraw": "Withdraw",
    "pour": "Tilt", "tilt": "Tilt",
    "reach": "Reach", "pick": "Pick", "move": "Move",
    "place": "Place", "give": "Give", "wipe": "Wipe",
}


def load_id_to_name(mapping_file):
    """Optional: load a 'mapping.txt' of '<id> <name>' lines (MS-TCN/ASFormer
    sometimes emit numeric class ids instead of words)."""
    m = {}
    if mapping_file and Path(mapping_file).is_file():
        for line in Path(mapping_file).read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                m[parts[0]] = parts[1]
    return m


ID_TO_NAME = load_id_to_name(str(Path(__file__).parent / "dataset" / "mapping.txt"))


def normalize(tok):
    """Map a raw token to a canonical label name."""
    tok = tok.strip()
    if not tok:
        return None
    if tok.isdigit() and tok in ID_TO_NAME:    # numeric class id -> name
        tok = ID_TO_NAME[tok]
    low = tok.lower()
    if low in SYNONYMS:
        return SYNONYMS[low]
    # already canonical (capitalised) ?
    if tok in COLORS:
        return tok
    cap = tok.capitalize()
    return cap if cap in COLORS else tok


# =====================================================================
# 3. FILE READERS
# =====================================================================

def read_labels(path):
    """Read a label file in either layout:
       (a) one label per line  (ground truth)
       (b) a header line starting with '#', then space-separated labels
           on one or more lines  (MS-TCN / ASFormer / RoboSubtaskNet preds)
    Returns a list of canonical labels."""
    toks = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for t in line.split():
            n = normalize(t)
            if n is not None:
                toks.append(n)
    return toks


def find_pred_file(pred_dir, *candidates):
    """Return the first existing file among candidate names inside pred_dir."""
    for name in candidates:
        p = pred_dir / name
        if p.is_file():
            return p
    return None


def segments(labels):
    """Compress a per-frame label list into (start, end, label) runs."""
    out, start, cur = [], 0, labels[0]
    for i, lab in enumerate(labels[1:], 1):
        if lab != cur:
            out.append((start, i, cur))
            start, cur = i, lab
    out.append((start, len(labels), cur))
    return out


# =====================================================================
# 4. RESOLVE  RoboSubtaskNet name  ->  global chunk index
# =====================================================================

def _resample(seq, n):
    """Resample a label list to exactly n points (nearest-neighbour).
    Makes the comparison invariant to temporal resolution / fps / stride."""
    L = len(seq)
    if L == 0:
        return []
    return [seq[min(L - 1, int(i * L / n))] for i in range(n)]


def resolve_global_id(robo_gt_seq, forced=None, topk=8, N=1000, robo_name=None):
    """CURRENT RUN: all 6 models share identical filenames and byte-identical
    ground truth, so no content-matching resolver is needed -- the 'global
    id' IS just robo_name. Kept as a function (same call sites) for minimal
    diff; `forced` still works if ever needed."""
    if forced is not None:
        return forced
    return robo_name


# =====================================================================
# 5. BEST-VIDEO FINDER
# =====================================================================

def frame_acc(gt, pred):
    n = min(len(gt), len(pred))
    if n == 0:
        return 0.0, 0
    return 100.0 * sum(a == b for a, b in zip(gt[:n], pred[:n])) / n, n


def find_best_video(task, topk=5, beat_models=("MS-TCN", "MS-TCN++")):
    """Scan RoboSubtaskNet prediction files for this task and find a
    representative video where RoboSubtaskNet beats the specified baseline
    models (default: MS-TCN and MS-TCN++).

    Rules:
      1. ALL models in `beat_models` must have a prediction file for the video.
      2. robo_acc > acc for EVERY model in beat_models.
      3. Rank survivors by the MINIMUM margin over beat_models (how clearly
         we beat the weakest of the two), then pick the video at the
         (topk//2)-th position — a solid representative, not an outlier.

    Returns the robo_idx string of the chosen candidate and prints a table.
    """
    import io, contextlib

    robo_prefix = ROBO_TASK_PREFIX[task]
    robo_pred_dir = PRED_DIRS.get("RoboSubtaskNet")
    if robo_pred_dir is None or not robo_pred_dir.is_dir():
        print("[find_best] RoboSubtaskNet prediction dir not found.")
        return None

    # Collect all prediction files for this task prefix.
    # RoboSubtaskNet writes prediction files WITHOUT .txt extension.
    candidates = sorted([c for c in robo_pred_dir.glob(f"{robo_prefix}_*")
                         if c.is_file()])
    if not candidates:
        print(f"[find_best] No files matching '{robo_prefix}_*' in {robo_pred_dir}")
        all_files = list(robo_pred_dir.iterdir())[:10]
        print(f"[find_best] Files in dir: {[f.name for f in all_files]}")
        return None

    print(f"[find_best] scanning {len(candidates)} '{robo_prefix}_*' videos "
          f"(require beating: {list(beat_models)}) ...")

    qualified = []   # videos where robo beats ALL beat_models
    skipped_no_pred = 0
    skipped_not_beat = 0

    for pf in candidates:
        stem = pf.stem
        robo_idx = stem.replace(f"{robo_prefix}_", "")
        robo_name = f"{robo_prefix}_{robo_idx}"

        # RoboSubtaskNet GT
        robo_gt_file = find_pred_file(ROBO_GT, f"{robo_name}.txt", robo_name)
        if robo_gt_file is None:
            skipped_no_pred += 1
            continue
        robo_gt = read_labels(robo_gt_file)
        if not robo_gt:
            skipped_no_pred += 1
            continue

        robo_pred = read_labels(pf)
        robo_acc, _ = frame_acc(robo_gt, robo_pred)

        # Resolve global chunk id (quiet, same N=1000 as build_figure uses)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            global_id = resolve_global_id(robo_gt, topk=1, N=1000, robo_name=robo_name)
        if global_id is None:
            skipped_no_pred += 1
            continue

        gt_file = find_pred_file(GLOBAL_GT, f"{global_id}.txt")
        if gt_file is None:
            skipped_no_pred += 1
            continue
        GT = read_labels(gt_file)

        # Compute accuracy for ALL models in ROW_ORDER
        all_accs = {}
        for model in ROW_ORDER:
            if model == "RoboSubtaskNet":
                continue
            pdir = PRED_DIRS.get(model)
            if pdir is None:
                continue
            mpf = find_pred_file(pdir, f"{global_id}.txt", str(global_id))
            if mpf is None:
                continue
            acc, _ = frame_acc(GT, read_labels(mpf))
            all_accs[model] = acc

        # --- FILTER: all beat_models must have predictions AND robo must beat them ---
        missing = [m for m in beat_models if m not in all_accs]
        if missing:
            skipped_no_pred += 1
            continue

        margins = {m: robo_acc - all_accs[m] for m in beat_models}
        min_margin = min(margins.values())

        if min_margin <= 0:
            skipped_not_beat += 1
            continue

        qualified.append((min_margin, robo_acc, robo_idx, global_id, all_accs))

    print(f"   {len(qualified)} videos qualify  |  "
          f"{skipped_no_pred} skipped (missing preds)  |  "
          f"{skipped_not_beat} skipped (robo doesn't beat baselines)")

    if not qualified:
        print("[find_best] No qualifying candidates. Try relaxing beat_models.")
        return None

    # Sort by min_margin descending (most clearly beating both baselines first)
    qualified.sort(key=lambda r: (r[0], r[1]), reverse=True)

    # Print full ranking table
    present_models = [m for m in ROW_ORDER if m != "RoboSubtaskNet"
                      and any(m in r[4] for r in qualified)]
    col_w = max(len(m) for m in present_models) + 1

    header = (f"{'rank':>4}  {'robo_idx':>10}  {'global':>7}  "
              f"{'Robo':>{col_w}}  " +
              "  ".join(f"{m:>{col_w}}" for m in present_models) +
              f"  {'min_margin':>10}")
    print("\n" + header)
    print("-" * len(header))
    for rank, (margin, racc, ridx, gid, oaccs) in enumerate(qualified[:max(topk, 10)], 1):
        row = (f"{rank:>4}  {ridx:>10}  {gid:>7}  "
               f"{racc:>{col_w-1}.1f}%  " +
               "  ".join(f"{oaccs.get(m, float('nan')):>{col_w-1}.1f}%" for m in present_models) +
               f"  {margin:>+9.1f}%")
        marker = "  ← chosen" if rank == min(len(qualified), max(1, topk // 2)) else ""
        print(row + marker)

    # Pick a solid representative: the (topk//2)-th best (not the extreme outlier)
    pick_rank = min(len(qualified), max(1, topk // 2))
    chosen = qualified[pick_rank - 1]
    print(f"\n[find_best] Chosen for {task}: robo_idx={chosen[2]}  "
          f"global={chosen[3]}  robo={chosen[1]:.1f}%  "
          f"min_margin={chosen[0]:+.1f}%  (rank {pick_rank}/{len(qualified)})\n")
    # Return BOTH robo_idx and the already-resolved global_id so build_figure
    # uses exactly the same chunk (avoids resolver giving a different result).
    return chosen[2], chosen[3]


# =====================================================================
# 6. BUILD ONE FIGURE
# =====================================================================

def build_figure(task, robo_idx, forced_global=None):
    robo_prefix = ROBO_TASK_PREFIX[task]
    robo_name = f"{robo_prefix}_{robo_idx}"
    print(f"\n=== {task}  |  RoboSubtaskNet video '{robo_name}' ===")

    # ---- RoboSubtaskNet ground truth (its own naming) ----
    robo_gt_file = find_pred_file(ROBO_GT, f"{robo_name}.txt", robo_name)
    if robo_gt_file is None:
        print(f"   !! RoboSubtaskNet GT not found for {robo_name} in {ROBO_GT}")
        return
    robo_gt = read_labels(robo_gt_file)

    # ---- map to the global chunk index used by the other models ----
    global_id = resolve_global_id(robo_gt, forced=forced_global, robo_name=robo_name)
    if global_id is None:
        print("   !! could not resolve global chunk id; "
              "pass --global_id <N> manually.")
        return

    # ---- the reference GT for the top row = global GT (same video) ----
    gt_file = find_pred_file(GLOBAL_GT, f"{global_id}.txt", str(global_id))
    GT = read_labels(gt_file)

    # ---- gather every available model row ----
    rows = [("GT", GT)]                       # top row is always ground truth
    accs = {}
    for model in ROW_ORDER:
        pdir = PRED_DIRS.get(model)
        if pdir is None:
            print(f"   - {model:14s}: skipped (no prediction directory configured)")
            continue
        if model == "RoboSubtaskNet":
            # Prediction files may have no extension (pick_place_483) or .txt
            pf = find_pred_file(pdir, f"{robo_name}.txt", robo_name, f"{robo_name}")
            ref = robo_gt
        else:
            # FACT's checkpoint internally names pick_pour_* videos
            # pick_pour_new_* -- see baselines/README.md.
            fact_alias = str(global_id).replace("pick_pour_", "pick_pour_new_", 1)
            pf = find_pred_file(pdir, f"{global_id}.txt", str(global_id), f"{fact_alias}.txt", fact_alias)
            ref = GT
        if pf is None:
            print(f"   - {model:14s}: skipped (prediction file missing)")
            continue
        pred = read_labels(pf)
        acc, n = frame_acc(ref, pred)
        accs[model] = acc
        rows.append((model, pred))
        print(f"   - {model:14s}: {pf.name:24s} frames={len(pred):4d}  acc={acc:5.1f}%")

    if len(rows) == 1:
        print("   !! no model rows available; nothing to plot.")
        return

    # ---- draw ----
    plot_rows(task, robo_name, global_id, rows, accs)


def plot_rows(task, robo_name, global_id, rows, accs):
    n_rows = len(rows)
    bar_h = 2
    gap = 2.25
    fig_h = 1.4 + 1.3 * n_rows  # taller rows so bar "breadth" matches GTEA/Breakfast figures
    fig, ax = plt.subplots(figsize=(15, fig_h))

    # rows drawn top -> bottom; y decreases
    y_positions = {}
    for i, (name, labels) in enumerate(rows):
        y = (n_rows - 1 - i) * gap
        y_positions[name] = y
        total = len(labels)
        for s, e, lab in segments(labels):
            ax.add_patch(patches.Rectangle(
                (s / total, y), (e - s) / total, bar_h,
                facecolor=COLORS.get(lab, "#cccccc"),
                edgecolor="black", linewidth=0.25))
        # left-hand row label (+ accuracy except for GT)
        label_txt = name if name == "GT" else f"{name}"
        ax.text(-0.012, y + bar_h / 2, label_txt,
                ha="right", va="center", fontsize=22, fontweight="bold")
        
        # NOTE: Side percentages have been removed per request.

    # ---- fine black vertical guide lines at each GT action boundary,
    # drawn across every model's row so misalignment vs. GT is visible ----
    gt_labels = rows[0][1]   # row 0 is always ("GT", GT_labels)
    gt_total = len(gt_labels)
    y_bottom = 0
    y_top = (n_rows - 1) * gap + bar_h
    for s, e, lab in segments(gt_labels)[:-1]:   # skip final boundary (video end)
        x = e / gt_total
        ax.vlines(x, y_bottom, y_top, colors="black", linewidth=1.0, zorder=10)


   # Shifted Timelines Left & Tightened Right Margin
    ax.set_xlim(-0.10, 1.02)
    ax.set_ylim(-bar_h * 1.3, (n_rows - 1) * gap + bar_h + 0.25)
    
    ax.set_yticks([]) 
    ax.set_xticks([])  # Removes the numbers

    # Hide all borders, including the bottom line
    ax.spines[["left", "top", "right", "bottom"]].set_visible(False)
    pretty = "Pick & Place" if task == "pick_place" else "Pick & Pour"
    
    # Title removed per request (frame collage now sits directly above the bars)

    legend_handles = [
        patches.Patch(facecolor=COLORS[l], edgecolor="black", linewidth=0.3,
                      label=DISPLAY[l])
        for l in LEGEND_ORDER[task] if l in COLORS
    ]
    
    # Legend Enhancements (Stretched, fonts bumped, shifted box)
    leg = ax.legend(handles=legend_handles, loc="upper left",
                    bbox_to_anchor=(0.05, -0.03, 0.90, 0.13), ncol=len(legend_handles),
                    mode="expand",
                    frameon=True, fontsize=17,
                    handleheight=1.3, handlelength=1.9, borderpad=1.0,
                    title="Color coding for sub-task representation",
                    title_fontsize=20, borderaxespad=0.0)
    leg.get_title().set_fontweight("bold")
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(1.0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"fig5_{task}_{robo_name}.png"
    
    # Global Layout Alignment
    plt.tight_layout(rect=[0.05, 0.20, 0.95, 1.0])
    
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   >> saved {out}")


# =====================================================================
# 7. CLI
# =====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["pick_place", "pick_pour", "both"],
                    default="both")
    ap.add_argument("--robo_idx", default="483",
                    help="RoboSubtaskNet per-task index (e.g. 483)")
    ap.add_argument("--global_id", default=None,
                    help="force the global chunk id and skip the resolver")
    ap.add_argument("--find_best", action="store_true",
                    help="scan all videos and auto-select the one where "
                         "RoboSubtaskNet has the highest advantage over other models")
    ap.add_argument("--topk", type=int, default=5,
                    help="number of top candidates to print when using --find_best (default 5)")
    args = ap.parse_args()

    tasks = ["pick_place", "pick_pour"] if args.task == "both" else [args.task]
    for t in tasks:
        if args.find_best:
            result = find_best_video(t, topk=args.topk)
            if result is None:
                print(f"[find_best] could not determine best video for {t}; skipping.")
                continue
            best_idx, best_global = result
            # Pass the resolved global_id directly so build_figure uses the
            # exact same chunk that find_best used when checking predictions.
            build_figure(t, best_idx, forced_global=best_global)
        else:
            build_figure(t, args.robo_idx, forced_global=args.global_id)


if __name__ == "__main__":
    main()


#To run this file : python visualize_fig5_all.py --task pick_place --robo_idx 490
#To run this file : python visualize_fig5_all.py --task pick_pour --robo_idx 498