# Evaluate new VAE
import os
import sys
sys.path.append('/home/longshen/work/AccGen/AccGen')
import torch
from models.phrase_vae import load_t5_model_from_lit_ckpt, S2SVAE
from piano_roll_utils import save_piano_roll
from models.vae_inference import MQVAE
from sonata_utils import jpath, create_dir_if_not_exist
from remi_z import MultiTrack


model = MQVAE(bottleneck=True)

# data = torch.load('/data1/longshen/Datasets/Piano/POP909/latents/position_level/val.pt')
# save_dir = '/home/longshen/work/AccGen/AccGen/test_outputs/vae/pos_level/test_recon'
# create_dir_if_not_exist(save_dir)

# # Decode the first 5 samples
# for i in range(12):
#     z = data[i:i+1]  # [1, 128]
#     mts = model.decode_batch(z, return_mt=True)
#     for j, mt in enumerate(mts):
#         save_fp = jpath(save_dir, f'val_{i}_recon_{j}.mid')
#         mt.to_midi(save_fp, tempo=90)


# Sample directly
save_dir = '/home/longshen/work/AccGen/test_outputs/vae/pos_level/sample_from_normal/sample_and_interpolate/scale_by_raw_std_2'
create_dir_if_not_exist(save_dir)
z_random = torch.randn(2, 128)

# Scale
# dim_wise_mean = torch.load('/data1/longshen/Datasets/Piano/POP909/latents/position_level/latent_dim_mean.pt') # [128]
# dim_wise_std = torch.load('/data1/longshen/Datasets/Piano/POP909/latents/position_level/latent_dim_std.pt') # [128]
# print("dim_wise_mean shape:", dim_wise_mean.shape)
# print("dim_wise_std shape:", dim_wise_std.shape)
# z_random = z_random * dim_wise_std + dim_wise_mean # [2, 128]

z_random = z_random * 0.3544125258922577

z1 = z_random[0:1]  # [1, 4, 512]
z2 = z_random[1:2]  # [1, 4, 512]
# Interpolate
z_inter = []
num_inter = 8
for alpha in torch.linspace(0, 1, num_inter):
    z_alpha = (1 - alpha) * z1 + alpha * z2
    z_inter.append(z_alpha)
z_inter = torch.cat(z_inter, dim=0)  # [num_inter, 4, 512]

bar_strs = model.decode_batch(z_random)
mts = [MultiTrack.from_remiz_str(bar_str) for bar_str in bar_strs]
for i, mt in enumerate(mts):
    save_fp = jpath(save_dir, f'bar_sample_{i}.mid')
    mt.to_midi(save_fp, tempo=90)
    str_save_fp = jpath(save_dir, f'bar_sample_{i}.txt')
    with open(str_save_fp, 'w') as f:
        f.write(bar_strs[i])

bar_strs = model.decode_batch(z_inter)
mts = [MultiTrack.from_remiz_str(bar_str) for bar_str in bar_strs]
for i, mt in enumerate(mts):
    save_fp = jpath(save_dir, f'bar_inter_{i}.mid')
    mt.to_midi(save_fp, tempo=90)
    str_save_fp = jpath(save_dir, f'bar_inter_{i}.txt')
    with open(str_save_fp, 'w') as f:
        f.write(bar_strs[i])