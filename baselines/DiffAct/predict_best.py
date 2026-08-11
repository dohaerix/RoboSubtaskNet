#!/usr/bin/env python3
"""
predict_best.py  (DiffAct)
=====================================================================
Regenerates test-set predictions from the BEST checkpoint (epoch-40,
chosen by decoder-agg-Test-Acc in logs/train.log: 94.697 at ep40 vs
94.29 at ep90) using DiffAct's own Trainer.test() so the DDIM sampling
protocol is identical to training-time evals.

Output: result/OurDataset-Trained-S1/prediction/  (one .txt per video)
Run:    python predict_best.py --config configs/our_dataset.json --device 0 --epoch 40

Features/groundtruth/mapping/splits are read directly from the shared
repo-root dataset/ (see dataset_paths.py) -- root_data_dir/dataset_name
from the config are unused for that purpose.
=====================================================================
"""
import os
import argparse
import numpy as np
import torch

from main import Trainer
from dataset import get_data_dict, VideoFeatureDataset
from utils import load_config_file
import dataset_paths

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/our_dataset.json')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--epoch', type=int, default=40)
args = parser.parse_args()

all_params = load_config_file(args.config)
globals().update(all_params)

if args.device != -1:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

feature_dir = None
label_dir = os.path.join(dataset_paths.DATASET_ROOT, "test", "groundtruth")
mapping_file = dataset_paths.mapping_path()

event_list = np.loadtxt(mapping_file, dtype=str)
event_list = [i[1] for i in event_list]
num_classes = len(event_list)

test_video_list = np.loadtxt(dataset_paths.split_bundle_path("test"), dtype=str)
test_video_list = [i.split('.')[0] for i in test_video_list]

test_data_dict = get_data_dict(
    feature_dir=feature_dir,
    label_dir=label_dir,
    video_list=test_video_list,
    event_list=event_list,
    sample_rate=sample_rate,
    temporal_aug=temporal_aug,
    boundary_smooth=boundary_smooth
)
test_test_dataset = VideoFeatureDataset(test_data_dict, num_classes, mode='test')

trainer = Trainer(dict(encoder_params), dict(decoder_params), dict(diffusion_params),
    event_list, sample_rate, temporal_aug, set_sampling_seed, postprocess,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

result_dir = os.path.join('result', naming)
model_path = os.path.join(result_dir, f'epoch-{args.epoch}.model')
print(f'Loading {model_path}')

result_dict = trainer.test(
    test_test_dataset, 'decoder-agg', trainer.device, label_dir,
    result_dir=result_dir, model_path=model_path)

print(f'\nEpoch {args.epoch} test results (func_eval):')
for k, v in result_dict.items():
    print(f'  {k}: {v}')
