#!/usr/bin/python3
"""
model.py
-----------------------------
RoboSubtaskNet: an MS-TCN-style multi-stage temporal convolutional
network (Fibonacci dilations, multi-stage refinement) for two-stream
(RGB + optical-flow I3D feature) action segmentation.

The two streams are fused by a learned front-end module before being
fed to the MS-TCN backbone. Several fusion strategies were compared
during development; BottleneckFusion (concat -> Linear(2D,H) -> ReLU
-> Linear(H,D) -> Sigmoid gate) was selected as the final architecture
for the released checkpoint (H=128, num_f_maps=48, dropout=0.3) based
on held-out test F1@50.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class AttentionFusion(nn.Module):
    """Adaptive modality gating: alpha = sigmoid(W_rgb.rgb + W_flow.flow),
    fused = alpha*rgb + (1-alpha)*flow."""
    def __init__(self, feature_dim):
        super(AttentionFusion, self).__init__()
        self.linear_rgb = nn.Linear(feature_dim, feature_dim)
        self.linear_flow = nn.Linear(feature_dim, feature_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, rgb_feat, flow_feat):
        # rgb_feat, flow_feat: (B, C, T)
        rgb_feat = rgb_feat.transpose(1, 2)   # (B, T, C)
        flow_feat = flow_feat.transpose(1, 2)
        alpha = self.sigmoid(self.linear_rgb(rgb_feat) + self.linear_flow(flow_feat))
        fused = alpha * rgb_feat + (1 - alpha) * flow_feat
        fused = fused.transpose(1, 2)         # (B, C, T)
        return fused, alpha.transpose(1, 2)   # fused: (B,C,T), alpha: (B,C,T)


class ConcatFusion(nn.Module):
    """Concatenation baseline: [rgb; flow] (2*feature_dim) reduced back to
    feature_dim by a trained linear projection."""
    def __init__(self, feature_dim):
        super(ConcatFusion, self).__init__()
        self.proj = nn.Linear(2 * feature_dim, feature_dim)

    def forward(self, rgb_feat, flow_feat):
        rgb_feat = rgb_feat.transpose(1, 2)    # (B, T, C)
        flow_feat = flow_feat.transpose(1, 2)
        cat = torch.cat([rgb_feat, flow_feat], dim=-1)   # (B, T, 2C)
        fused = self.proj(cat)                 # (B, T, C)
        fused = fused.transpose(1, 2)           # (B, C, T)
        return fused, None


class AverageFusion(nn.Module):
    """Simple (unweighted) averaging baseline: fused = 0.5*rgb + 0.5*flow.
    No learnable parameters."""
    def __init__(self, feature_dim):
        super(AverageFusion, self).__init__()

    def forward(self, rgb_feat, flow_feat):
        return 0.5 * rgb_feat + 0.5 * flow_feat, None


class ConcatRawFusion(nn.Module):
    """Concatenation WITHOUT the down-projection: raw [rgb; flow]
    (2*feature_dim) fed straight into the backbone."""
    def __init__(self, feature_dim=None):
        super(ConcatRawFusion, self).__init__()

    def forward(self, rgb_feat, flow_feat):
        return torch.cat([rgb_feat, flow_feat], dim=1), None  # (B, 2C, T)


class BottleneckFusion(nn.Module):
    """Final fusion module used by the released model: concat(rgb,flow)
    -> Linear(2D,H) -> ReLU -> Linear(H,D) -> Sigmoid gate. A genuine
    non-linear hidden bottleneck of width H (H=128 for the released
    checkpoint), unlike AttentionFusion/ConcatFusion which are single
    affine transforms."""
    def __init__(self, feature_dim, hidden_dim=256):
        super(BottleneckFusion, self).__init__()
        self.fc1 = nn.Linear(2 * feature_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, feature_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, rgb_feat, flow_feat):
        rgb_feat = rgb_feat.transpose(1, 2)    # (B, T, C)
        flow_feat = flow_feat.transpose(1, 2)
        cat = torch.cat([rgb_feat, flow_feat], dim=-1)   # (B, T, 2C)
        alpha = self.sigmoid(self.fc2(self.relu(self.fc1(cat))))
        fused = alpha * rgb_feat + (1 - alpha) * flow_feat
        return fused.transpose(1, 2), alpha.transpose(1, 2)


def build_fusion(fusion_type, feature_dim, hidden_dim=256):
    if fusion_type == "gated":
        return AttentionFusion(feature_dim)
    elif fusion_type == "concat":
        return ConcatFusion(feature_dim)
    elif fusion_type == "average":
        return AverageFusion(feature_dim)
    elif fusion_type == "concat_raw":
        return ConcatRawFusion(feature_dim)
    elif fusion_type == "bottleneck":
        return BottleneckFusion(feature_dim, hidden_dim)
    raise ValueError(f"Unknown fusion_type: {fusion_type}")


def fusion_output_dim(fusion_type, feature_dim):
    return 2 * feature_dim if fusion_type == "concat_raw" else feature_dim


def build_dilations(dilation_type, n=10):
    if dilation_type == "fibonacci":
        fib = [1, 1]
        while len(fib) < n + 1:
            fib.append(fib[-1] + fib[-2])
        return fib[1:n + 1]
    elif dilation_type == "linear":
        return list(range(1, n + 1))
    elif dilation_type == "exponential":
        return [2 ** i for i in range(n)]
    raise ValueError(f"Unknown dilation_type: {dilation_type}")


class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_f_maps, dim, num_classes, dilations, dropout_p=0.5):
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(num_f_maps, dim, num_classes, dilations, dropout_p)
        self.stages = nn.ModuleList(
            [copy.deepcopy(SingleStageModel(num_f_maps, num_classes, num_classes, dilations, dropout_p))
             for s in range(num_stages - 1)]
        )

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SingleStageModel(nn.Module):
    def __init__(self, num_f_maps, dim, num_classes, dilations, dropout_p=0.5):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [DilatedResidualLayer(d, num_f_maps, num_f_maps, dropout_p) for d in dilations]
        )
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, dropout_p=0.5):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class FullModel(nn.Module):
    """Wraps a fusion module + MultiStageModel as one module, so
    thop.profile can measure combined FLOPs/params in eval.py."""
    def __init__(self, fusion_type, num_stages, num_f_maps, feature_dim, num_classes,
                 dilations, dropout_p=0.5, hidden_dim=256):
        super(FullModel, self).__init__()
        self.fusion = build_fusion(fusion_type, feature_dim, hidden_dim)
        backbone_dim = fusion_output_dim(fusion_type, feature_dim)
        self.model = MultiStageModel(num_stages, num_f_maps, backbone_dim, num_classes, dilations, dropout_p)

    def forward(self, rgb, flow, mask):
        fused, _ = self.fusion(rgb, flow)
        return self.model(fused, mask)
