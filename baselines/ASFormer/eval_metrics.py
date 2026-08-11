#!/usr/bin/env python3
"""
eval_metrics.py  (ASFormer)
=====================================================================
Evaluates the trained ASFormer checkpoint (epoch-50, split_1) on the
160 held-out test videos of our_new_dataset, saving everything into
THIS folder only:

    metrics_per_activity.csv   frame Acc / Edit / F1@{10,25,50} per
                                activity + overall
    flops_params.csv           MACs, FLOPs, #parameters (thop)
    results_table.png / .pdf   summary table image

Predictions (results/our_dataset/split_1/<id>, no extension,
'### ...' header + space-separated labels) and ground truth
(../../dataset/test/groundtruth/<id>.txt, the shared repo-root
dataset) are 1:1 frame-aligned.
=====================================================================
"""
import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from thop import profile

from model import MyTransformer
import dataset_paths

HERE = os.path.dirname(os.path.abspath(__file__))
# Groundtruth/mapping/splits are read directly from the shared repo-root
# dataset/ (see dataset_paths.py) -- this folder keeps no local data/ copy.
RESULTS_DIR = os.path.join(HERE, "results", "our_dataset", "split_1")
GT_DIR = os.path.join(dataset_paths.DATASET_ROOT, "test", "groundtruth")

THR = [0.10, 0.25, 0.50]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# our_dataset video ids are namespaced "<task>_<local_id>"
TASKS = ["mopping", "pick_place", "pick_give", "pick_pour"]


def activity_of(vid_id):
    for name in TASKS:
        if vid_id.startswith(name + "_"):
            return name
    return "unknown"


with open(dataset_paths.mapping_path()) as f:
    mapping = {}
    for line in f:
        parts = line.split()
        if len(parts) == 2:
            mapping[parts[1]] = int(parts[0])
num_classes = len(mapping)

with open(dataset_paths.split_bundle_path("test")) as f:
    test_ids = [l.strip()[:-4] for l in f if l.strip()]  # strip ".txt"


# ---------------------------------------------------------------------
# metric helpers (frame accuracy, segmental edit score, segmental F1)
# ---------------------------------------------------------------------
def segs(lab):
    if not lab:
        return []
    s, cur, st = [], lab[0], 0
    for i, l in enumerate(lab[1:], 1):
        if l != cur:
            s.append((cur, st, i)); cur, st = l, i
    s.append((cur, st, len(lab)))
    return s


def lev(a, b):
    n, m = len(a), len(b); dp = list(range(m + 1))
    for i in range(1, n + 1):
        nw = [i]
        for j in range(1, m + 1):
            nw.append(dp[j - 1] if a[i - 1] == b[j - 1] else 1 + min(dp[j], nw[-1], dp[j - 1]))
        dp = nw
    return dp[m]


def edit_score(gt, pr):
    g = [x[0] for x in segs(gt)]; p = [x[0] for x in segs(pr)]
    if not g and not p:
        return 100.0
    if not g or not p:
        return 0.0
    return max(0.0, 100.0 * (1 - lev(g, p) / max(len(g), len(p))))


def f1_counts(gt, pr, t):
    gs, ps = segs(gt), segs(pr); tp = 0; matched = set()
    for pl, s, e in ps:
        best, bi = 0.0, -1
        for idx, (gl, gs2, ge) in enumerate(gs):
            if idx in matched or gl != pl:
                continue
            inter = max(0, min(e, ge) - max(s, gs2)); uni = max(e, ge) - min(s, gs2)
            iou = inter / uni if uni > 0 else 0.0
            if iou > best:
                best, bi = iou, idx
        if best > t and bi >= 0:
            tp += 1; matched.add(bi)
    return tp, len(ps) - tp, len(gs) - len(matched)


def frame_acc(gt, pr):
    n = min(len(gt), len(pr))
    return 100.0 * sum(a == b for a, b in zip(gt[:n], pr[:n])) / n if n else 0.0


def read_pred(path):
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    toks = []
    for line in lines:
        toks.extend(line.split())
    return toks


def read_gt(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


# ---------------------------------------------------------------------
# collect (gt, pred) pairs
# ---------------------------------------------------------------------
per_activity_pairs = {}
missing = []
input_lengths = []
for vid_id in test_ids:
    pred_path = os.path.join(RESULTS_DIR, vid_id)  # no extension
    gt_path = os.path.join(GT_DIR, f"{vid_id}.txt")
    if not os.path.isfile(pred_path) or not os.path.isfile(gt_path):
        missing.append(vid_id)
        continue
    gt = read_gt(gt_path)
    pred = read_pred(pred_path)
    input_lengths.append(len(pred))
    act = activity_of(vid_id)
    per_activity_pairs.setdefault(act, []).append((gt, pred))

if missing:
    print(f"[WARN] {len(missing)} test videos missing prediction/GT: {missing[:5]}...")


def metrics_over(pairs):
    accs, edits = [], []
    tp = {t: 0 for t in THR}; fp = {t: 0 for t in THR}; fn = {t: 0 for t in THR}
    for gt, pr in pairs:
        accs.append(frame_acc(gt, pr)); edits.append(edit_score(gt, pr))
        for t in THR:
            a, b, c = f1_counts(gt, pr, t); tp[t] += a; fp[t] += b; fn[t] += c
    f1 = {}
    for t in THR:
        p_ = tp[t] / (tp[t] + fp[t]) if tp[t] + fp[t] else 0.0
        r_ = tp[t] / (tp[t] + fn[t]) if tp[t] + fn[t] else 0.0
        f1[t] = 100.0 * 2 * p_ * r_ / (p_ + r_) if (p_ + r_) else 0.0
    return {"n": len(pairs), "acc": np.mean(accs), "edit": np.mean(edits),
            "f1_10": f1[0.10], "f1_25": f1[0.25], "f1_50": f1[0.50]}


rows = []
all_pairs = []
for act in sorted(per_activity_pairs):
    pairs = per_activity_pairs[act]
    all_pairs += pairs
    m = metrics_over(pairs)
    rows.append({"activity": act, **m})

overall = metrics_over(all_pairs)
rows.append({"activity": "OVERALL", **overall})

metrics_df = pd.DataFrame(rows)[["activity", "n", "acc", "edit", "f1_10", "f1_25", "f1_50"]]
metrics_df.columns = ["Activity", "#Videos", "Acc(%)", "Edit(%)", "F1@10(%)", "F1@25(%)", "F1@50(%)"]
metrics_csv = os.path.join(HERE, "metrics_per_activity.csv")
metrics_df.to_csv(metrics_csv, index=False, float_format="%.2f")
print(metrics_df.to_string(index=False))
print(f"\nSaved: {metrics_csv}")

# ---------------------------------------------------------------------
# FLOPs / parameters (thop), at the mean test-time feature length
# (same hyperparameters as main.py: 3 decoders, 10 layers, r1=r2=2,
#  64 maps, 2048-dim input; channel_masking_rate=0 for inference)
# ---------------------------------------------------------------------
num_layers, num_f_maps, features_dim = 10, 64, 2048
model = MyTransformer(3, num_layers, 2, 2, num_f_maps, features_dim, num_classes,
                      channel_masking_rate=0.0).to(device)
model.eval()

mean_T = int(round(np.mean(input_lengths))) if input_lengths else 50
dummy_x = torch.randn(1, features_dim, mean_T).to(device)
dummy_mask = torch.ones(1, num_classes, mean_T).to(device)

macs, params = profile(model, inputs=(dummy_x, dummy_mask), verbose=False)
flops = 2 * macs  # 1 MAC = 2 FLOPs

flops_df = pd.DataFrame([{
    "Model": "ASFormer",
    "Input length T (mean test frames)": mean_T,
    "Params": int(params),
    "Params (M)": round(params / 1e6, 3),
    "MACs": int(macs),
    "GMACs": round(macs / 1e9, 3),
    "FLOPs": int(flops),
    "GFLOPs": round(flops / 1e9, 3),
}])
flops_csv = os.path.join(HERE, "flops_params.csv")
flops_df.to_csv(flops_csv, index=False)
print("\n" + flops_df.to_string(index=False))
print(f"Saved: {flops_csv}")

# ---------------------------------------------------------------------
# summary table image
# ---------------------------------------------------------------------
summary = {
    "Model": "ASFormer",
    "#Test videos": int(overall["n"]),
    "Acc (%)": f"{overall['acc']:.2f}",
    "Edit (%)": f"{overall['edit']:.2f}",
    "F1@10 (%)": f"{overall['f1_10']:.2f}",
    "F1@25 (%)": f"{overall['f1_25']:.2f}",
    "F1@50 (%)": f"{overall['f1_50']:.2f}",
    "Params (M)": f"{params / 1e6:.3f}",
    "GFLOPs": f"{flops / 1e9:.3f}",
}
cols = list(summary.keys())
row = [list(summary.values())]

fig, ax = plt.subplots(figsize=(11, 1.6)); ax.axis("off")
tbl = ax.table(cellText=row, colLabels=cols, loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 2.0)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#b0b0b0"); cell.set_linewidth(0.8)
    if r == 0:
        cell.set_facecolor("#2c3e50"); cell.set_text_props(color="white", fontweight="bold")
    else:
        cell.set_facecolor("#f4f6f7")
ax.set_title("ASFormer (official) on our_dataset (split_1, 1920 train / 160 test)",
             fontsize=13, fontweight="bold", pad=14)
plt.tight_layout()
table_png = os.path.join(HERE, "results_table.png")
table_pdf = os.path.join(HERE, "results_table.pdf")
fig.savefig(table_png, dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig(table_pdf, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {table_png}")
print(f"Saved: {table_pdf}")

print("\nDONE.")
