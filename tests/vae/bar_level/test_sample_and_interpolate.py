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
save_dir = '/home/longshen/work/AccGen/AccGen/test_outputs/vae/sample_from_normal/trained_with_phrase_data/sample_and_interpolate'
create_dir_if_not_exist(save_dir)
z_random = torch.randn(2, 4, 512)

# Scale
dim_wise_mean = torch.load('/data1/longshen/Datasets/Piano/POP909/latents/latent_dim_mean.pt').reshape(4, -1)
dim_wise_std = torch.load('/data1/longshen/Datasets/Piano/POP909/latents/latent_dim_std.pt').reshape(4, -1)
scale_factor = 0.306642085313797
raw_mean = -1.6620271708234213e-05

# Naive scaling
# z_random = z_random * scale_factor

# # Scale with raw mean and std
# z_random = z_random * scale_factor + raw_mean

# # Scale with dim-wise std only
# z_random = z_random * dim_wise_std

# Scale with dim-wise mean and std
z_random = z_random * dim_wise_std + dim_wise_mean # [2, 4, 512]

z1 = z_random[0:1]  # [1, 4, 512]
z2 = z_random[1:2]  # [1, 4, 512]
# Interpolate
z_inter = []
num_inter = 8
for alpha in torch.linspace(0, 1, num_inter):
    z_alpha = (1 - alpha) * z1 + alpha * z2
    z_inter.append(z_alpha)
z_inter = torch.cat(z_inter, dim=0)  # [num_inter, 4, 512]

mts = model.decode_batch(z_random, return_mt=True, scale_factor=scale_factor)
for i, mt in enumerate(mts):
    # proll = mt[0].to_piano_roll(pos_per_bar=48)
    # save_piano_roll(proll, jpath(save_dir, f'bar_sample_{i}.png'))
    save_fp = jpath(save_dir, f'bar_sample_{i}.mid')
    mt.to_midi(save_fp, tempo=90)

mts = model.decode_batch(z_inter, return_mt=True, scale_factor=scale_factor)
for i, mt in enumerate(mts):
    # proll = mt[0].to_piano_roll(pos_per_bar=48)
    # save_piano_roll(proll, jpath(save_dir, f'bar_inter_{i}.png'))
    save_fp = jpath(save_dir, f'bar_inter_{i}.mid')
    mt.to_midi(save_fp, tempo=90)