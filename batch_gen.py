#!/usr/bin/python3
"""
batch_gen.py
------------------------
Two-stream batch generator: loads RGB and flow I3D features separately
(each 1024-dim, (C, T) layout) so they can be fused by the fusion
module inside the training loop (see model.py / main.py), rather than
being pre-fused on disk.
"""
import os
import random
import numpy as np
import torch


class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, gt_path, features_rgb_path, features_flow_path,
                 sample_rate=1, rgb_mean=None, rgb_std=None, flow_mean=None, flow_std=None):
        self.list_of_examples = []
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_rgb_path = features_rgb_path
        self.features_flow_path = features_flow_path
        self.sample_rate = sample_rate
        self.rgb_mean = rgb_mean.reshape(-1, 1) if rgb_mean is not None else None
        self.rgb_std = rgb_std.reshape(-1, 1) if rgb_std is not None else None
        self.flow_mean = flow_mean.reshape(-1, 1) if flow_mean is not None else None
        self.flow_std = flow_std.reshape(-1, 1) if flow_std is not None else None

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        return self.index < len(self.list_of_examples)

    def read_data(self, vid_list_file):
        with open(vid_list_file, "r") as f:
            self.list_of_examples = [x for x in f.read().split("\n") if x.strip()]
        random.shuffle(self.list_of_examples)

    def _load(self, path, mean, std):
        a = np.load(path)
        if a.shape[0] != 1024 and a.shape[1] == 1024:   # ensure (C=1024, T)
            a = a.T
        if mean is not None:
            a = (a - mean) / std
        return a

    def next_batch(self, batch_size):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_rgb, batch_flow, batch_target = [], [], []
        for vid in batch:
            rgb = self._load(os.path.join(self.features_rgb_path, vid.split(".")[0] + ".npy"),
                              self.rgb_mean, self.rgb_std)[:, ::self.sample_rate]
            flow = self._load(os.path.join(self.features_flow_path, vid.split(".")[0] + ".npy"),
                               self.flow_mean, self.flow_std)[:, ::self.sample_rate]
            with open(os.path.join(self.gt_path, vid), "r") as f:
                content = [x for x in f.read().split("\n") if x.strip()]
            T = min(rgb.shape[1], flow.shape[1], len(content))
            rgb, flow = rgb[:, :T], flow[:, :T]
            classes = np.zeros(T, dtype=np.int64)
            for i in range(T):
                classes[i] = self.actions_dict[content[i]]
            batch_rgb.append(rgb)
            batch_flow.append(flow)
            batch_target.append(classes)

        max_len = max(f.shape[1] for f in batch_rgb)
        rgb_tensor = torch.zeros(len(batch_rgb), 1024, max_len, dtype=torch.float)
        flow_tensor = torch.zeros(len(batch_flow), 1024, max_len, dtype=torch.float)
        target_tensor = torch.ones(len(batch_rgb), max_len, dtype=torch.long) * -100
        mask = torch.zeros(len(batch_rgb), self.num_classes, max_len, dtype=torch.float)

        for i in range(len(batch_rgb)):
            t = batch_rgb[i].shape[1]
            rgb_tensor[i, :, :t] = torch.from_numpy(batch_rgb[i])
            flow_tensor[i, :, :t] = torch.from_numpy(batch_flow[i])
            target_tensor[i, :t] = torch.from_numpy(batch_target[i])
            mask[i, :, :t] = 1

        return rgb_tensor, flow_tensor, target_tensor, mask
