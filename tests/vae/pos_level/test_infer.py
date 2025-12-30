# Evaluate VAE with bottleneck
import os
import sys
sys.path.append('/home/longshen/work/AccGen/AccGen')
import torch
from models.phrase_vae import load_t5_model_from_lit_ckpt, S2SVAE
from piano_roll_utils import save_piano_roll
from models.vae_inference import MQVAE
from sonata_utils import jpath


model = MQVAE(bottleneck=True)

# Sample directly
from sonata_utils import create_dir_if_not_exist

z_random = torch.randn(10, 128)
z_original = z_random.clone()

dim_wise_mean = torch.load('/data1/longshen/Datasets/Piano/POP909/latents/position_level/latent_dim_mean.pt') # [128]
dim_wise_std = torch.load('/data1/longshen/Datasets/Piano/POP909/latents/position_level/latent_dim_std.pt') # [128]

save_dir = '/home/longshen/work/AccGen/test_outputs/vae/pos_level/sample_from_normal/scale_by_dim'
create_dir_if_not_exist(save_dir)
z_random = z_random * dim_wise_std + dim_wise_mean # [10, 128]
mts = model.decode_batch(z_random, return_mt=True)
for i, mt in enumerate(mts):
    save_fp = jpath(save_dir, f'bar_sample_{i}.mid')
    mt.to_midi(save_fp, tempo=90)

save_dir = '/home/longshen/work/AccGen/test_outputs/vae/pos_level/sample_from_normal/scale_by_raw'
create_dir_if_not_exist(save_dir)
raw_mean = -0.0008415346383117139
raw_std = 0.3544125258922577
z_random = z_original.clone()
z_random = z_random * raw_std + raw_mean
mts = model.decode_batch(z_random, return_mt=True)
for i, mt in enumerate(mts):
    save_fp = jpath(save_dir, f'bar_sample_{i}.mid')
    mt.to_midi(save_fp, tempo=90)