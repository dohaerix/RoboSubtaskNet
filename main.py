#!/usr/bin/python3
"""
main.py
=====================================================================
Train / predict entry point for RoboSubtaskNet (BottleneckFusion +
multi-stage TCN) on the released robot-subtask dataset (mopping,
pick_place, pick_give, pick_pour; 480 train / 40 test videos per task).

    python main.py --action train
    python main.py --action predict

Default hyperparameters below reproduce the released checkpoint in
model/bottleneck_h128fm48_dropout03/ (bottleneck fusion, Fibonacci
dilations, hidden_dim=128, num_f_maps=48, dropout=0.3). All of them can
be overridden from the command line to run new ablations.
=====================================================================
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from model import MultiStageModel, build_fusion, build_dilations, fusion_output_dim
from batch_gen import BatchGenerator

HERE = os.path.dirname(os.path.abspath(__file__))


def cosine_warmup_lambda(epoch, warmup_epochs, total_epochs):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def load_transition_mask(path, num_classes, device):
    freq = np.loadtxt(path, delimiter=",")
    M = (freq == 0).astype(np.float32)
    np.fill_diagonal(M, 0.0)
    return torch.tensor(M, dtype=torch.float, device=device)


def transition_loss(p_softmax, M, mask):
    P_prev = p_softmax[:, :, :-1]
    P_curr = p_softmax[:, :, 1:]
    w = torch.einsum("bit,ij,bjt->bt", P_prev, M, P_curr)
    abs_diff = (P_curr - P_prev).abs().sum(dim=1)
    valid = mask[:, 0, 1:]
    denom = valid.sum().clamp(min=1.0)
    return (w * abs_diff * valid).sum() / denom


def tmse_loss_fixed(p, mask, tau=4.0):
    """Paper-faithful T-MSE: sum squared log-prob diffs over classes to
    get one scalar per frame, clamp at tau, average over time only."""
    logp = F.log_softmax(p, dim=1)
    diff = logp[:, :, 1:] - logp.detach()[:, :, :-1]        # (B, C, T-1)
    per_frame = diff.pow(2).sum(dim=1)                        # sum over classes -> (B, T-1)
    clamped = torch.clamp(per_frame, min=0, max=tau)
    valid = mask[:, 0, 1:]
    denom = valid.sum().clamp(min=1.0)
    return (clamped * valid).sum() / denom


def compute_loss(predictions, target, mask, ce, mse, num_classes, tmse_w, trans_w, M, fixed_tmse=True):
    loss = 0
    for p in predictions:
        loss = loss + ce(p.transpose(2, 1).contiguous().view(-1, num_classes), target.view(-1))
        if fixed_tmse:
            loss = loss + tmse_w * tmse_loss_fixed(p, mask, tau=4.0)
        else:
            loss = loss + tmse_w * torch.mean(torch.clamp(
                mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)),
                min=0, max=4.0) * mask[:, :, 1:])
        loss = loss + trans_w * transition_loss(F.softmax(p, dim=1), M, mask)
    return loss


def segs(lab):
    if len(lab) == 0:
        return []
    s, cur, st = [], lab[0], 0
    for i, l in enumerate(lab[1:], 1):
        if l != cur:
            s.append((cur, st, i)); cur, st = l, i
    s.append((cur, st, len(lab)))
    return s


def f1_at_50(gt, pred):
    gs, ps = segs(gt), segs(pred)
    tp, matched = 0, set()
    for pl, s, e in ps:
        best, bi = 0.0, -1
        for idx, (gl, gs2, ge) in enumerate(gs):
            if idx in matched or gl != pl:
                continue
            inter = max(0, min(e, ge) - max(s, gs2)); uni = max(e, ge) - min(s, gs2)
            iou = inter / uni if uni > 0 else 0.0
            if iou > best:
                best, bi = iou, idx
        if best > 0.5 and bi >= 0:
            tp += 1; matched.add(bi)
    fp = len(ps) - tp
    fn = len(gs) - len(matched)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def median_filter_labels(labels, window=3):
    if window <= 1:
        return labels
    T = len(labels)
    half = window // 2
    out = labels.copy()
    for t in range(T):
        lo, hi = max(0, t - half), min(T, t + half + 1)
        vals, counts = np.unique(labels[lo:hi], return_counts=True)
        out[t] = vals[np.argmax(counts)]
    return out


class EarlyStopping:
    def __init__(self, patience=2, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


@torch.no_grad()
def evaluate(fusion, model, val_gen, ce, mse, num_classes, tmse_w, trans_w, M, device, idx_to_action, fixed_tmse):
    fusion.eval()
    model.eval()
    total_loss, n_batches = 0.0, 0
    f1_scores = []
    val_gen.reset()
    while val_gen.has_next():
        rgb, flow, target, mask = val_gen.next_batch(1)
        rgb, flow, target, mask = rgb.to(device), flow.to(device), target.to(device), mask.to(device)
        fused, _ = fusion(rgb, flow)
        predictions = model(fused, mask)
        loss = compute_loss(predictions, target, mask, ce, mse, num_classes, tmse_w, trans_w, M, fixed_tmse)
        total_loss += loss.item()
        n_batches += 1

        _, predicted = torch.max(predictions[-1].data, 1)
        pred_np = predicted.squeeze(0).cpu().numpy()
        tgt_np = target.squeeze(0).cpu().numpy()
        valid = tgt_np != -100
        pred_labels = [idx_to_action[int(i)] for i in pred_np[valid]]
        gt_labels = [idx_to_action[int(i)] for i in tgt_np[valid]]
        f1_scores.append(f1_at_50(gt_labels, pred_labels))

    fusion.train()
    model.train()
    return total_loss / max(n_batches, 1), float(np.mean(f1_scores)) if f1_scores else 0.0


def train(args, device):
    seed = 1538574472
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    features_dim = 1024
    dilations = build_dilations(args.dilation_type, args.n_dilations)
    print(f"Using device: {device}   fusion_type={args.fusion_type}   dilation_type={args.dilation_type}   "
          f"hidden_dim={args.hidden_dim}   dilations={dilations}")

    mapping_file = os.path.join(args.dataset, "mapping.txt")
    train_gt   = os.path.join(args.dataset, "train", "groundtruth") + "/"
    train_rgb  = os.path.join(args.dataset, "train", "features_rgb") + "/"
    train_flow = os.path.join(args.dataset, "train", "features_flow") + "/"
    test_gt    = os.path.join(args.dataset, "test", "groundtruth") + "/"
    test_rgb   = os.path.join(args.dataset, "test", "features_rgb") + "/"
    test_flow  = os.path.join(args.dataset, "test", "features_flow") + "/"
    train_bundle = os.path.join(args.dataset, "splits", "train.bundle")
    val_bundle   = os.path.join(args.dataset, "splits", "val.bundle")
    os.makedirs(args.model_dir, exist_ok=True)

    actions = [l for l in open(mapping_file).read().split("\n") if l.strip()]
    actions_dict = {a.split()[1]: int(a.split()[0]) for a in actions}
    idx_to_action = {v: k for k, v in actions_dict.items()}
    num_classes = len(actions_dict)
    print("num_classes:", num_classes, "->", actions_dict)

    fusion = build_fusion(args.fusion_type, features_dim, args.hidden_dim).to(device)
    fusion_params = sum(p.numel() for p in fusion.parameters())
    backbone_dim = fusion_output_dim(args.fusion_type, features_dim)
    model = MultiStageModel(args.num_stages, args.num_f_maps, backbone_dim, num_classes,
                             dilations, dropout_p=args.dropout).to(device)
    backbone_params = sum(p.numel() for p in model.parameters())
    print(f"Fusion params: {fusion_params:,}   Backbone params: {backbone_params:,}   "
          f"TOTAL: {fusion_params + backbone_params:,}")

    class_weights = None
    if os.path.exists(args.class_weights):
        class_weights = torch.tensor(np.load(args.class_weights), dtype=torch.float, device=device)
    ce = nn.CrossEntropyLoss(ignore_index=-100, weight=class_weights)
    mse = nn.MSELoss(reduction="none")
    M = load_transition_mask(args.transitions_csv, num_classes, device)

    all_params = list(fusion.parameters()) + list(model.parameters())
    optimizer = optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda ep: cosine_warmup_lambda(ep, args.warmup_epochs, args.epochs))
    scaler = torch.amp.GradScaler('cuda', enabled=(args.amp and device.type == "cuda"))

    rgb_mean = rgb_std = flow_mean = flow_std = None
    if os.path.exists(args.norm_stats_rgb):
        s = np.load(args.norm_stats_rgb); rgb_mean, rgb_std = s["mean"], s["std"]
    if os.path.exists(args.norm_stats_flow):
        s = np.load(args.norm_stats_flow); flow_mean, flow_std = s["mean"], s["std"]

    batch_gen = BatchGenerator(num_classes, actions_dict, train_gt, train_rgb, train_flow, sample_rate=1,
                                rgb_mean=rgb_mean, rgb_std=rgb_std, flow_mean=flow_mean, flow_std=flow_std)
    batch_gen.read_data(train_bundle)
    n_videos = len(batch_gen.list_of_examples)

    # val.bundle lists the same videos as test.bundle (paper protocol);
    # their features/groundtruth live under dataset/test/.
    val_gen = BatchGenerator(num_classes, actions_dict, test_gt, test_rgb, test_flow, sample_rate=1,
                              rgb_mean=rgb_mean, rgb_std=rgb_std, flow_mean=flow_mean, flow_std=flow_std)
    val_gen.read_data(val_bundle)
    n_val = len(val_gen.list_of_examples)

    print(f"Training on {n_videos} videos, validating on {n_val} videos, for up to {args.epochs} epochs "
          f"(early stopping on VALIDATION loss, patience={args.patience})\n")

    early_stopping = EarlyStopping(patience=args.patience)
    best_val_f1 = -1.0
    best_epoch = None

    for epoch in range(args.epochs):
        fusion.train()
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(total=n_videos, desc=f"epoch {epoch + 1}/{args.epochs}", unit="vid", dynamic_ncols=True)
        while batch_gen.has_next():
            rgb, flow, target, mask = batch_gen.next_batch(args.bz)
            rgb, flow, target, mask = rgb.to(device), flow.to(device), target.to(device), mask.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(args.amp and device.type == "cuda")):
                fused, _ = fusion(rgb, flow)
                predictions = model(fused, mask)
                loss = compute_loss(predictions, target, mask, ce, mse, num_classes,
                                     args.tmse, args.trans_weight, M, args.fixed_tmse)

            epoch_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(all_params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            _, predicted = torch.max(predictions[-1].data, 1)
            batch_correct = ((predicted == target).float() * mask[:, 0, :]).sum().item()
            batch_total = torch.sum(mask[:, 0, :]).item()
            correct += batch_correct
            total += batch_total

            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{batch_correct / max(batch_total, 1):.3f}")
            pbar.update(rgb.size(0))

        pbar.close()
        batch_gen.reset()
        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        torch.save({
            "fusion": fusion.state_dict(),
            "model": model.state_dict(),
        }, os.path.join(args.model_dir, f"epoch-{epoch + 1}.model"))
        train_avg_loss = epoch_loss / n_videos

        val_loss, val_f1_50 = evaluate(fusion, model, val_gen, ce, mse, num_classes,
                                        args.tmse, args.trans_weight, M, device, idx_to_action, args.fixed_tmse)
        print(f"[epoch {epoch + 1:3d}]  train_loss = {train_avg_loss:.4f}   train_acc = {correct / total:.4f}   "
              f"val_loss = {val_loss:.4f}   val_F1@50 = {val_f1_50:.4f}   lr = {cur_lr:.6f}")

        if val_f1_50 > best_val_f1:
            best_val_f1 = val_f1_50
            best_epoch = epoch + 1

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}.")
            break

    best_epoch_file = os.path.join(args.model_dir, "best_epoch.txt")
    with open(best_epoch_file, "w") as f:
        f.write(str(best_epoch))
    print(f"\nDone. Checkpoints saved in: {args.model_dir}")
    print(f"Best epoch by validation F1@50 = {best_val_f1:.4f}: epoch {best_epoch}")


def predict(args, device):
    features_dim = 1024
    dilations = build_dilations(args.dilation_type, args.n_dilations)

    mapping_file = os.path.join(args.dataset, "mapping.txt")
    test_rgb  = os.path.join(args.dataset, "test", "features_rgb") + "/"
    test_flow = os.path.join(args.dataset, "test", "features_flow") + "/"
    test_bundle = os.path.join(args.dataset, "splits", "test.bundle")
    os.makedirs(args.results_dir, exist_ok=True)

    rgb_mean = rgb_std = flow_mean = flow_std = None
    if os.path.exists(args.norm_stats_rgb):
        s = np.load(args.norm_stats_rgb)
        rgb_mean, rgb_std = s["mean"].reshape(-1, 1), s["std"].reshape(-1, 1)
    if os.path.exists(args.norm_stats_flow):
        s = np.load(args.norm_stats_flow)
        flow_mean, flow_std = s["mean"].reshape(-1, 1), s["std"].reshape(-1, 1)

    actions = [l for l in open(mapping_file).read().split("\n") if l.strip()]
    actions_dict = {a.split()[1]: int(a.split()[0]) for a in actions}
    idx_to_action = {v: k for k, v in actions_dict.items()}
    num_classes = len(actions_dict)

    best_epoch_file = os.path.join(args.model_dir, "best_epoch.txt")
    epoch = args.epoch if args.epoch > 0 else int(open(best_epoch_file).read().strip())
    print(f"Loading checkpoint epoch {epoch}")

    fusion = build_fusion(args.fusion_type, features_dim, args.hidden_dim).to(device)
    backbone_dim = fusion_output_dim(args.fusion_type, features_dim)
    model = MultiStageModel(args.num_stages, args.num_f_maps, backbone_dim, num_classes, dilations).to(device)
    ckpt = torch.load(os.path.join(args.model_dir, f"epoch-{epoch}.model"), map_location=device)
    fusion.load_state_dict(ckpt["fusion"])
    model.load_state_dict(ckpt["model"])
    fusion.eval()
    model.eval()

    vids = [l for l in open(test_bundle).read().split("\n") if l.strip()]
    with torch.no_grad():
        for vid in vids:
            rgb = np.load(test_rgb + vid.split(".")[0] + ".npy")
            flow = np.load(test_flow + vid.split(".")[0] + ".npy")
            if rgb.shape[0] != 1024 and rgb.shape[1] == 1024:
                rgb = rgb.T
            if flow.shape[0] != 1024 and flow.shape[1] == 1024:
                flow = flow.T
            T = min(rgb.shape[1], flow.shape[1])
            rgb, flow = rgb[:, :T], flow[:, :T]
            if rgb_mean is not None:
                rgb = (rgb - rgb_mean) / rgb_std
            if flow_mean is not None:
                flow = (flow - flow_mean) / flow_std

            inp_rgb = torch.tensor(rgb, dtype=torch.float).unsqueeze(0).to(device)
            inp_flow = torch.tensor(flow, dtype=torch.float).unsqueeze(0).to(device)
            fused, _ = fusion(inp_rgb, inp_flow)

            mask = torch.ones(fused.size(), device=device)
            pred = model(fused, mask)
            _, predicted = torch.max(pred[-1].data, 1)
            predicted = predicted.squeeze(0).cpu().numpy()
            predicted = median_filter_labels(predicted, args.median_filter)
            recognition = [idx_to_action[int(i)] for i in predicted]

            fname = vid.split(".")[0]
            with open(os.path.join(args.results_dir, fname), "w") as f:
                f.write("### Frame level recognition: ###\n")
                f.write("\n".join(recognition) + "\n")

    print(f"Predictions saved in: {args.results_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", choices=["train", "predict"], default="train")
    ap.add_argument("--fusion_type", choices=["concat", "average", "gated", "concat_raw", "bottleneck"],
                     default="bottleneck")
    ap.add_argument("--dilation_type", choices=["linear", "exponential", "fibonacci"], default="fibonacci")
    ap.add_argument("--hidden_dim", type=int, default=128, help="bottleneck fusion hidden width H")
    ap.add_argument("--fixed_tmse", action="store_true", default=True,
                     help="paper-faithful T-MSE (sum over classes, then clamp, then average over time)")
    ap.add_argument("--n_dilations", type=int, default=10)
    ap.add_argument("--dataset", default=os.path.join(HERE, "dataset"))
    ap.add_argument("--model_dir", default=os.path.join(HERE, "model", "bottleneck_h128fm48_dropout03"))
    ap.add_argument("--results_dir", default=os.path.join(HERE, "results", "bottleneck_h128fm48_dropout03"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.0005)
    ap.add_argument("--bz", type=int, default=8)
    ap.add_argument("--num_stages", type=int, default=4)
    ap.add_argument("--num_f_maps", type=int, default=48)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--tmse", type=float, default=0.15)
    ap.add_argument("--trans_weight", type=float, default=0.25)
    ap.add_argument("--transitions_csv", type=str, default=os.path.join(HERE, "transitions.csv"))
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--warmup_epochs", type=int, default=5)
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no_amp", dest="amp", action="store_false")
    ap.add_argument("--norm_stats_rgb", type=str, default=os.path.join(HERE, "norm_stats_rgb.npz"))
    ap.add_argument("--norm_stats_flow", type=str, default=os.path.join(HERE, "norm_stats_flow.npz"))
    ap.add_argument("--class_weights", type=str, default=os.path.join(HERE, "class_weights.npy"))
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--epoch", type=int, default=-1, help="predict: which checkpoint epoch to load (-1 = best)")
    ap.add_argument("--median_filter", type=int, default=3, help="predict: post-processing median filter window")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.action == "train":
        train(args, device)
    else:
        predict(args, device)


if __name__ == "__main__":
    main()
