# Evaluate new VAE
import os
import sys
sys.path.append('/home/longshen/work/AccGen/AccGen')
import torch
from models.phrase_vae import load_t5_model_from_lit_ckpt, S2SVAE
from piano_roll_utils import save_piano_roll
from models.vae_inference import MQVAE
from sonata_utils import jpath


model = MQVAE()

# Sample directly
from sonata_utils import create_dir_if_not_exist
save_dir = '/home/longshen/work/AccGen/test_outputs/vae/bar_level/sample_from_normal/trained_with_phrase_data/scale_with'

intended_std = 0.5
z_random = torch.randn(10, 4, 512) * intended_std
z_orginal = z_random.clone()

# Scale
dim_wise_mean = torch.load('/data1/longshen/Datasets/Piano/POP909/latents/bar_level/latent_dim_mean.pt').reshape(4, -1)
dim_wise_std = torch.load('/data1/longshen/Datasets/Piano/POP909/latents/bar_level/latent_dim_std.pt').reshape(4, -1)
avg_std = 0.12887243926525116
raw_std = 0.306642085313797
raw_mean = -1.6620271708234213e-05

# # Naive scaling
z_random = z_random * raw_std
exp_dir = jpath(save_dir, 'raw_std')
create_dir_if_not_exist(exp_dir)
mts = model.decode_batch(z_random, return_mt=True)
for i, mt in enumerate(mts):
    proll = mt[0].to_piano_roll(pos_per_bar=48)
    save_fp = jpath(exp_dir, f'bar_sample_{i}.mid')
    mt.to_midi(save_fp, tempo=90)

# # Scale with raw mean and std
z_random = z_orginal.clone()
z_random = z_random * raw_std + raw_mean
exp_dir = jpath(save_dir, 'raw_mean_and_std')
create_dir_if_not_exist(exp_dir)
mts = model.decode_batch(z_random, return_mt=True)
for i, mt in enumerate(mts):
    proll = mt[0].to_piano_roll(pos_per_bar=48)
    save_fp = jpath(exp_dir, f'bar_sample_{i}.mid')
    mt.to_midi(save_fp, tempo=90)

# # Scale with avg wise std only
z_random = z_orginal.clone()
z_random = z_random * avg_std
exp_dir = jpath(save_dir, 'avg_std')
create_dir_if_not_exist(exp_dir)
mts = model.decode_batch(z_random, return_mt=True)
for i, mt in enumerate(mts):
    proll = mt[0].to_piano_roll(pos_per_bar=48)
    save_fp = jpath(exp_dir, f'bar_sample_{i}.mid')
    mt.to_midi(save_fp, tempo=90)

# Scale with dim-wise std only
z_random = z_orginal.clone()
z_random = z_random * dim_wise_std
exp_dir = jpath(save_dir, 'dim_wise_std')
create_dir_if_not_exist(exp_dir)
mts = model.decode_batch(z_random, return_mt=True)
for i, mt in enumerate(mts):
    proll = mt[0].to_piano_roll(pos_per_bar=48)
    save_fp = jpath(exp_dir, f'bar_sample_{i}.mid')
    mt.to_midi(save_fp, tempo=90)

# # Scale with dim-wise mean and std
z_random = z_orginal.clone()
z_random = z_random * dim_wise_std + dim_wise_mean
exp_dir = jpath(save_dir, 'dim_wise_mean_and_std')
create_dir_if_not_exist(exp_dir)
mts = model.decode_batch(z_random, return_mt=True)
for i, mt in enumerate(mts):
    proll = mt[0].to_piano_roll(pos_per_bar=48)
    save_fp = jpath(exp_dir, f'bar_sample_{i}.mid')
    mt.to_midi(save_fp, tempo=90)