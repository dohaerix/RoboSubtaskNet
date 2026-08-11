#!/usr/bin/env python3
"""
make_config.py
=====================================================================
Writes configs/our_new_dataset.json for this DiffAct copy. Based on
the paper's GTEA defaults (closest match: similar scale, sample_rate=1,
2048-d features) from default_configs.py, with num_epochs/log_freq
sized to this GPU's measured throughput (see logs/speed_probe.log)
instead of the paper's num_epochs=10001 default, which would take
weeks at batch_size=1-video-per-forward-pass on this hardware.
=====================================================================
"""
import os
import json
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NUM_EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
LOG_FREQ = int(sys.argv[2]) if len(sys.argv) > 2 else 50

params = {
   "naming": "OurNewDataset-Trained-S1",
   "root_data_dir": "./datasets",
   "dataset_name": "our_new_dataset",
   "split_id": 1,
   "sample_rate": 1,
   "temporal_aug": True,
   "encoder_params": {
      "use_instance_norm": False,
      "num_layers": 10,
      "num_f_maps": 64,
      "input_dim": 2048,
      "kernel_size": 5,
      "normal_dropout_rate": 0.5,
      "channel_dropout_rate": 0.5,
      "temporal_dropout_rate": 0.5,
      "feature_layer_indices": [5, 7, 9]
   },
   "decoder_params": {
      "num_layers": 8,
      "num_f_maps": 24,
      "time_emb_dim": 512,
      "kernel_size": 5,
      "dropout_rate": 0.1,
   },
   "diffusion_params": {
      "timesteps": 1000,
      "sampling_timesteps": 25,
      "ddim_sampling_eta": 1.0,
      "snr_scale": 0.5,
      "cond_types": ["full", "zero", "boundary03-", "segment=1", "segment=1"],
      "detach_decoder": False,
   },
   "loss_weights": {
      "encoder_ce_loss": 0.5,
      "encoder_mse_loss": 0.1,
      "encoder_boundary_loss": 0.0,
      "decoder_ce_loss": 0.5,
      "decoder_mse_loss": 0.1,
      "decoder_boundary_loss": 0.1
   },
   "batch_size": 4,
   "learning_rate": 0.0005,
   "weight_decay": 1e-6,
   "num_epochs": NUM_EPOCHS,
   "log_freq": LOG_FREQ,
   "class_weighting": True,
   "set_sampling_seed": True,
   "boundary_smooth": 1,
   "soft_label": 1.4,
   "log_train_results": False,
   "postprocess": {
      "type": "purge",
      "value": 3
   },
}

os.makedirs(os.path.join(HERE, "configs"), exist_ok=True)
out_path = os.path.join(HERE, "configs", "our_new_dataset.json")
with open(out_path, "w") as f:
    json.dump(params, f, indent=2)
print(f"Wrote {out_path} (num_epochs={NUM_EPOCHS}, log_freq={LOG_FREQ})")
