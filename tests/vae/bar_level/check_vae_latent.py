import torch

# 1. 收集所有training latents
all_latents = torch.load('/data1/longshen/Datasets/Piano/POP909/latents_2/pop909_piano_track_bars.pt')
# all_latents = torch.load('/data1/longshen/Datasets/Piano/POP909/latents_ae/pop909_piano_track_bars.pt')
print(f"All latents shape: {all_latents.shape}")  # (N, 4, 512)
n_sample = all_latents.shape[0]
all_latents = all_latents.view(n_sample, -1)  # 展平

# calculate mean
mean = all_latents.mean()
print(f"Mean shape: {mean.shape}")  # (2048,)
print(f'Mean: {mean}')

# 2. 计算std
std = all_latents.std(dim=-1)
std = std.mean()  # 平均std
print()
print(f"Original std: {std}")
scale_factor = 1.0 / std

# 2.5 Check direct std
std_direct = all_latents.std()
print(f"Direct std: {std_direct}")

# print(f"Scale factor: {scale_factor}")
# 如果std=2.5，scale_factor=0.4

# # 3. Scale latents
# z_scaled = all_latents * scale_factor
# print(f"Scaled std: {z_scaled.std()}")  # 应该≈1.0