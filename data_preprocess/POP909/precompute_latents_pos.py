"""
This file precompute and save bar-level latent representations

[2025-10-03] When using MQVAE,
- Need to remove the instrument token at the beginning of each bar
- Scale the latents by 1/0.3428 to make std=1.0 (original std=0.3428)
"""

import os
import sys

# Add project root to sys.path
dirof = os.path.dirname
try:
    dir_of_file = dirof(__file__)
except NameError:
    # .ipynb 文件中没有 __file__，使用当前工作目录
    dir_of_file = os.getcwd()
project_root = dirof(dirof(dir_of_file))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.vae_inference import MQVAE, MQVAE_Pos
from sonata_utils import read_jsonl, ls, create_dir_if_not_exist, jpath, save_json
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def main():
    # construct_bar_level_latents()
    get_scale_factor_2()


def procedures():
    compute_latents()
    get_scale_factor()
    construct_bar_level_latents()


def get_scale_factor_2():
    latent = torch.load('/data1/longshen/Datasets/Piano/POP909/latents/bar_level/vae3/train.pt') 
    # 64 dim PosVAE 1.6484638
    # 128dim w/o l2 reg: 2.4690728
    # 128dim bar level VAE3: 3.1647
    # Get 99.9% percentile of the absolute value of the latents
    abs_latent = latent.abs().view(-1).cpu().numpy()
    p999 = np.percentile(abs_latent, 99.9)
    print("99.9% percentile of absolute latent values:", p999)


def construct_bar_level_latents():
    """
    Construct bar-level latents from position-level latents by concatenation
    This will lead to uneven sequence length.
    Do zero-padding to the max length
    And note down the original lengths for each bar
    """
    pos_data_dir = "/data1/longshen/Datasets/Piano/POP909/jsonl/position_level"
    out_dir = "/data1/longshen/Datasets/Piano/POP909/latents/bar_level_pos_seq_64"
    create_dir_if_not_exist(out_dir)
    data_fns = ls(pos_data_dir, ext=".jsonl")
    model = MQVAE_Pos().cuda()
    for fn in data_fns:
        print("Processing", fn)
        jsonl_fp = os.path.join(pos_data_dir, fn)
        pos_data = read_jsonl(jsonl_fp)
        all_bars = []
        bar_seq = []
        cnt = 0
        for pos_str in pos_data:
            bar_seq.append(pos_str)
            if pos_str.startswith("b-"):
                if len(bar_seq) > 0:
                    all_bars.append(bar_seq)
                bar_seq = []

        if len(bar_seq) > 0:
            all_bars.append(bar_seq)

        # # Debug
        # all_bars = all_bars[0:2]

        # Pad with 'b-1' token to max length. Pad in the n_pos dimension
        # While count the original lengths
        lengths = []
        max_pos_len = max([len(bar) for bar in all_bars])
        print("Max position length per bar:", max_pos_len)
        for i in range(len(all_bars)):
            bar = all_bars[i]
            n_pos = len(bar)
            lengths.append(n_pos)
            if n_pos < max_pos_len:
                pad_len = max_pos_len - n_pos
                for _ in range(pad_len):
                    bar.append("b-1")
            all_bars[i] = bar
        lengths = torch.tensor(lengths, dtype=torch.long)

        # For each bar, encode all position sequence with VAE
        latents_list = []
        print("Encoding bars...")
        for bar in tqdm(all_bars):
            latents = model.encode_batch(bar, do_sample=False)  # (N_pos, D)
            latents_list.append(latents.cpu())
            n_pos = len(bar)
        latents = torch.stack(latents_list, dim=0)  # (N_bars, N_pos, D)
        print("Latents shape:", latents.shape)

        # Compute raw std
        raw_std = latents.std().item()
        raw_mean = latents.mean().item()
        raw_max = latents.max().item()
        raw_min = latents.min().item()
        save_json(
            {
                "raw_mean": raw_mean,
                "raw_std": raw_std,
                "raw_max": raw_max,
                "raw_min": raw_min,
            },
            jpath(out_dir, fn.replace(".jsonl", "_raw_stat.json")),
        )

        # Compute dim-wise mean and std
        dim_mean = latents.mean(dim=(0, 1))
        dim_std = latents.std(dim=(0, 1))
        torch.save(dim_mean, jpath(out_dir, fn.replace(".jsonl", "_dim_mean.pt")))
        torch.save(dim_std, jpath(out_dir, fn.replace(".jsonl", "_dim_std.pt")))

        # Save
        latents_fp = os.path.join(out_dir, fn.replace(".jsonl", "_latents.pt"))
        lengths_fp = os.path.join(out_dir, fn.replace(".jsonl", "_lengths.pt"))
        torch.save(latents, latents_fp)
        torch.save(lengths, lengths_fp)
        print("Saved to", latents_fp, lengths_fp)


def get_scale_factor():
    latent_dir = "/data1/longshen/Datasets/Piano/POP909/latents/position_level"
    data_fns = "train.pt"
    data_fp = jpath(latent_dir, data_fns)
    latents = torch.load(data_fp)
    print("Latents shape:", latents.shape)

    scale_factor_fp = jpath(latent_dir, "scale_factor.pt")

    # Compute mean and std
    raw_mean = latents.mean().item()
    raw_std = latents.std().item()
    dim_mean = latents.mean(dim=0).cpu().view(-1)
    dim_std = latents.std(dim=0).cpu().view(-1)
    avg_std = dim_std.mean().item()
    print(f"Latents mean: {raw_mean}, std: {raw_std}", f"avg dim-wise std: {avg_std}")
    print(f"dim-wise shape: {dim_mean.shape}")
    raw_stats = {"raw_mean": raw_mean, "raw_std": raw_std, "avg_dim_std": avg_std}
    save_json(raw_stats, jpath(latent_dir, "raw_latent_stats.json"))
    torch.save(dim_mean, jpath(latent_dir, "latent_dim_mean.pt"))
    torch.save(dim_std, jpath(latent_dir, "latent_dim_std.pt"))

    # Plot the dim-wise mean with plt. x:dim index, y: mean value
    plt.figure(figsize=(10, 4))
    plt.scatter(range(len(dim_mean)), dim_mean.numpy(), label="Mean", s=10)
    plt.xlabel("Latent Dimension")
    plt.ylabel("Value")
    plt.title("Latent Dimension-wise Mean")
    plt.legend()
    plt.grid()
    plt.savefig(jpath(latent_dir, "latent_dim_mean.png"), dpi=300)
    plt.close()

    # Plot the dim-wise std with plt. x:dim index, y: std value
    plt.figure(figsize=(10, 4))
    plt.scatter(
        range(len(dim_std)), dim_std.numpy(), label="Std Dev", s=10, color="orange"
    )
    plt.xlabel("Latent Dimension")
    plt.ylabel("Value")
    plt.title("Latent Dimension-wise Std Dev")
    plt.legend()
    plt.grid()
    plt.savefig(jpath(latent_dir, "latent_dim_std.png"), dpi=300)
    plt.close()

    # Calculate percentile for dim-wise std, save
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    percentile_values = torch.quantile(
        dim_std, torch.tensor([p / 100 for p in percentiles])
    ).numpy()
    percentile_dict = {
        f"{p}th_percentile": float(v) for p, v in zip(percentiles, percentile_values)
    }
    save_json(percentile_dict, jpath(latent_dir, "latent_dim_std_percentiles.json"))


def compute_latents():

    model = MQVAE(bottleneck=True).cuda()
    src_dir = "/data1/longshen/Datasets/Piano/POP909/jsonl/position_level"
    tgt_dir = "/data1/longshen/Datasets/Piano/POP909/latents/position_level"
    create_dir_if_not_exist(tgt_dir)

    data_fns = ls(src_dir, ext=".jsonl")
    for fn in data_fns:
        print("Processing", fn)
        jsonl_fp = os.path.join(src_dir, fn)
        data = read_jsonl(jsonl_fp)
        bar_strs = []
        for bar in data:
            bar_seq = bar.split()

            # NOTE: time signature, tempo, instrument tokens are pre-removed
            bar_str = " ".join(bar_seq)
            bar_strs.append(bar_str)

        # Encode with 128 batch size
        batch_size = 128
        bar_str_batches = [
            bar_strs[i : i + batch_size] for i in range(0, len(bar_strs), batch_size)
        ]
        latents_list = []
        for bar_str_batch in tqdm(bar_str_batches):
            latents = model.encode_batch(bar_str_batch, do_sample=False)
            latents_list.append(latents)
        latents = torch.cat(latents_list, dim=0)
        print("Latents shape:", latents.shape)

        tgt_fp = os.path.join(tgt_dir, fn.replace(".jsonl", ".pt"))
        torch.save(latents, tgt_fp)

        # Compute mean and std
        tot_mean = latents.mean().item()
        tot_std = latents.std().item()
        dim_mean = latents.mean(dim=0)
        dim_std = latents.std(dim=0)
        print(f"Latents mean: {tot_mean}, std: {tot_std}")
        print(f"dim-wise shape: {dim_mean.shape}, mean: {dim_mean}, std: {dim_std}")


if __name__ == "__main__":
    main()
