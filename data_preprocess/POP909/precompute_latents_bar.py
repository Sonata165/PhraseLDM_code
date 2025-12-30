'''
This file precompute and save bar-level latent representations

[2025-10-03] When using MQVAE, 
- Need to remove the instrument token at the beginning of each bar
- Scale the latents by 1/0.3428 to make std=1.0 (original std=0.3428)
'''

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

from models.vae_inference import MQVAE, BarVAE
from sonata_utils import read_jsonl, ls, create_dir_if_not_exist, jpath, save_json, read_json
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from remi_z import MultiTrack
import numpy as np
from datasets.dataset_utils import load_phrase_annot_cleaned


def main():
    get_evaluation_set()


def procedures():
    # Old implementation
    compute_bar_level_latents()

    # New implementation
    construct_song_level_latents()
    get_scale_factor()
    plot_latent_std_hist()
    prepare_latent_dataset_with_phrase_annotation()
    get_evaluation_set()


def get_evaluation_set():
    '''
    Hold out 100 songs from training set as evaluation set
    '''
    train_fp = '/data1/longshen/Datasets/Piano/POP909/latents/song_level_bar_seq_with_annot/dataset.pt'
    train_dataset = torch.load(train_fp)
    
    idx = np.arange(len(train_dataset))
    np.random.seed(42)
    eval_indices = np.random.choice(idx, size=100, replace=False).tolist()
    train_song_names = list(train_dataset.keys())
    eval_song_names = [train_song_names[i] for i in eval_indices]

    eval_dataset = {name: train_dataset[name] for name in eval_song_names}
    print("Evaluation songs:", len(eval_dataset))

    # Save eval dataset
    save_dir = '/data1/longshen/Datasets/Piano/POP909/latents/song_level_bar_seq_with_annot'
    eval_fp = jpath(save_dir, 'eval_data.pt')
    torch.save(eval_dataset, eval_fp)
    print("Saved eval dataset to", eval_fp)


def prepare_latent_dataset_with_phrase_annotation():
    all_song_latent_fp = '/data1/longshen/Datasets/Piano/POP909/latents/song_level_bar_seq/all_songs_latents.pt'
    all_song_latent_lengths_fp = '/data1/longshen/Datasets/Piano/POP909/latents/song_level_bar_seq/all_songs_latent_lengths.pt'
    latent = torch.load(all_song_latent_fp)  # (n_song, max_n_phrase, d_z)
    lengths = torch.load(all_song_latent_lengths_fp)  # (n_song,)
    dataset_dir = '/data1/longshen/Datasets/Piano/POP909/pop909_longshen/data_key_normed'
    song_names = ls(dataset_dir)

    dataset = {}
    for i, song_name in enumerate(song_names):
        dataset[song_name] = {'latent': latent[i], 'length': lengths[i].item()}
    print("Total songs:", len(dataset))

    # Remove songs with length > 128 bars
    max_n_bars = 128
    filtered_dataset = {}
    for song_name, data in dataset.items():
        if data['length'] <= max_n_bars:
            filtered_dataset[song_name] = data
    print("Filtered songs:", len(filtered_dataset))
    dataset = filtered_dataset

    # Remove songs with annotation error
    annot_error_ids = read_json('/data1/longshen/Datasets/Piano/POP909/pop909_longshen/phrase_annotation_errors.json')
    filtered_dataset = {}
    for song_name, data in dataset.items():
        if song_name not in annot_error_ids:
            filtered_dataset[song_name] = data
    print("After removing annotation error songs:", len(filtered_dataset))
    dataset = filtered_dataset

    # Read phrase annotation
    for song_name in dataset:
        song_data_dir = jpath(dataset_dir, song_name)
        annot_fp = jpath(song_data_dir, 'phrase_annot_cleaned.txt')
        assert os.path.exists(annot_fp), f"Annotation file does not exist for {song_name}!"
        annot = load_phrase_annot_cleaned(annot_fp)
        dataset[song_name]['phrase_annotation'] = annot

    # Unpad latents according to length
    for song_name in dataset:
        length = dataset[song_name]['length']
        dataset[song_name]['latent'] = dataset[song_name]['latent'][:length, :]

    # Save the dataset
    save_dir = '/data1/longshen/Datasets/Piano/POP909/latents/song_level_bar_seq_with_annot'
    save_fn = 'dataset.pt'
    create_dir_if_not_exist(save_dir)
    save_fp = jpath(save_dir, save_fn)
    torch.save(dataset, save_fp)
    print("Saved dataset with phrase annotation to", save_fp)

    # Calculate statistics (max, min, mean, std, 99.9% percentile of absolute values)
    all_latents = torch.cat([data['latent'] for data in dataset.values()], dim=0)
    print("All latents shape:", all_latents.shape)
    max_val = all_latents.max().item()
    min_val = all_latents.min().item()
    mean_val = all_latents.mean().item()
    std_val = all_latents.std().item()
    abs_latents = all_latents.abs().view(-1).cpu().numpy()
    p999 = np.percentile(abs_latents, 99.9).item()
    stats = {'max': max_val, 'min': min_val, 'mean': mean_val, 'std': std_val, '99.9_percentile_abs': p999}
    print("Latent statistics:", stats)
    stats_fp = jpath(save_dir, 'latent_stats.json')
    save_json(stats, stats_fp)
    print("Saved latent statistics to", stats_fp)


def plot_latent_std_hist():
    latent_fp = '/data1/longshen/Datasets/Piano/POP909/latents/song_level_bar_seq/latent_dim_std.pt'
    out_dir = os.path.dirname(latent_fp)
    dim_std = torch.load(latent_fp)

    from matplotlib.ticker import PercentFormatter

    # Plot histogram of dim-wise std, x: std value, y: percentage
    plt.figure(figsize=(8, 4))
    plt.hist(
        dim_std.numpy(),
        weights=np.ones(len(dim_std)) / len(dim_std),  # ✅ fixed parentheses placement
        bins=50,
        density=False,  # ✅ turn off density since we’re normalizing manually
        alpha=0.7,
        color="blue",
        edgecolor="black",
    )
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))  # ✅ 1 = 100%
    plt.xlabel("Dimension-wise Std Dev")
    plt.ylabel("Percentage")
    plt.title("Histogram of Dimension-wise Std Dev")
    plt.grid()

    plt.savefig(
        jpath(out_dir, "dim_std_histogram.png"), dpi=300
    )
    plt.close()


def get_scale_factor():
    latent_dir = '/data1/longshen/Datasets/Piano/POP909/latents/song_level_bar_seq'
    data_fns = 'all_songs_latents.pt'
    data_fp = jpath(latent_dir, data_fns)
    latents = torch.load(data_fp)
    length_fp = jpath(latent_dir, 'all_songs_latent_lengths.pt')
    lengths = torch.load(length_fp)
    
    # Remove padding according to lengths
    all_latents = []
    for i in range(latents.shape[0]):
        l = lengths[i]
        all_latents.append(latents[i, :l, :])
    latents = torch.cat(all_latents, dim=0) # (total_bars, d_z)
    print("Total latents shape after removing padding:", latents.shape)

    scale_factor_fp = jpath(latent_dir, 'scale_factor.pt')
    
    # Compute mean and std
    raw_mean = latents.mean().item()
    raw_std = latents.std().item()
    dim_mean = latents.mean(dim=0).cpu().view(-1)
    dim_std = latents.std(dim=0).cpu().view(-1)
    avg_std = dim_std.mean().item()
    print(f"Latents mean: {raw_mean}, std: {raw_std}", f"avg dim-wise std: {avg_std}")
    print(f'dim-wise shape: {dim_mean.shape}')
    raw_max = latents.max().item()
    raw_min = latents.min().item()
    print(f"Latents min: {raw_min}, max: {raw_max}")
    raw_stats = {
        'raw_mean': raw_mean,
        'raw_std': raw_std,
        'avg_dim_std': avg_std,
        'raw_min': raw_min,
        'raw_max': raw_max
    }
    save_json(raw_stats, jpath(latent_dir, 'raw_latent_stats.json'))
    torch.save(dim_mean, jpath(latent_dir, 'latent_dim_mean.pt'))
    torch.save(dim_std, jpath(latent_dir, 'latent_dim_std.pt'))

    # Plot the dim-wise mean with plt. x:dim index, y: mean value
    plt.figure(figsize=(10, 4))
    plt.scatter(range(len(dim_mean)), dim_mean.numpy(), label='Mean', s=10)
    plt.xlabel('Latent Dimension')
    plt.ylabel('Value')
    plt.title('Latent Dimension-wise Mean')
    plt.legend()
    plt.grid()
    plt.savefig(jpath(latent_dir, 'latent_dim_mean.png'), dpi=300)
    plt.close()

    # Plot the dim-wise std with plt. x:dim index, y: std value
    plt.figure(figsize=(10, 4))
    plt.scatter(range(len(dim_std)), dim_std.numpy(), label='Std Dev', s=10, color='orange')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Value')
    plt.title('Latent Dimension-wise Std Dev')
    plt.legend()
    plt.grid()
    plt.savefig(jpath(latent_dir, 'latent_dim_std.png'), dpi=300)
    plt.close()

    # Calculate percentile for dim-wise std, save
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    percentile_values = torch.quantile(dim_std, torch.tensor([p/100 for p in percentiles])).numpy()
    percentile_dict = {f'{p}th_percentile': float(v) for p, v in zip(percentiles, percentile_values)}
    save_json(percentile_dict, jpath(latent_dir, 'latent_dim_std_percentiles.json'))


def construct_song_level_latents():
    '''
    Construct song-level latents from MIDI files
    The result will be [n_song, max_n_bars, d_z]
    '''
    midi_dir = '/data1/longshen/Datasets/Piano/POP909/pop909_longshen/data_key_normed'
    song_fns = ls(midi_dir)
    midi_fps = [jpath(midi_dir, fn, f'{fn}.mid') for fn in song_fns]
    assert os.path.exists(midi_fps[0]), "MIDI file does not exist!"
    vae = BarVAE().cuda()

    all_latents = []
    for midi_fp in tqdm(midi_fps):
        mt = MultiTrack.from_midi(midi_fp)
        song_latents = vae.encode_mt(mt, do_sample=False)  # (n_bars, d_z)
        all_latents.append(song_latents.cpu())
        
    # Pad to max sequence length (max_n_bar)
    max_n_bar = max([latents.shape[0] for latents in all_latents])
    print('Max n_bars:', max_n_bar)
    padded_latents = []
    lengths = []
    for latents in all_latents:
        n_bar = latents.shape[0]
        lengths.append(n_bar)
        if n_bar < max_n_bar:
            pad_len = max_n_bar - n_bar
            pad_tensor = torch.zeros((pad_len, latents.shape[1]))
            latents = torch.cat([latents, pad_tensor], dim=0)
        padded_latents.append(latents.unsqueeze(0))
    padded_latents = torch.cat(padded_latents, dim=0) # (n_song, max_n_phrase, d_z)
    print("Padded latents shape:", padded_latents.shape)
    lengths = torch.tensor(lengths, dtype=torch.long)

    # Save
    out_dir = '/data1/longshen/Datasets/Piano/POP909/latents/song_level_bar_seq'
    create_dir_if_not_exist(out_dir)
    out_fn = 'all_songs_latents.pt'
    out_fp = jpath(out_dir, out_fn)
    torch.save(padded_latents, out_fp)
    lengths_fp = jpath(out_dir, 'all_songs_latent_lengths.pt')
    torch.save(lengths, lengths_fp)
    print("Saved to", out_fp, lengths_fp)


def compute_bar_level_latents():
    '''
    Convert the two splits (train, valid) of POP909 bar-level jsonl files
    '''
    model = BarVAE().cuda()
    src_dir = '/data1/longshen/Datasets/Piano/POP909/jsonl/bar_level'
    tgt_dir = '/data1/longshen/Datasets/Piano/POP909/latents/bar_level/multitrack'

    scale_factor_dir = jpath(tgt_dir, 'scale_factor.pt')

    create_dir_if_not_exist(tgt_dir)
    data_fns = ls(src_dir, ext='.jsonl')
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
        bar_str_batches = [bar_strs[i:i+batch_size] for i in range(0, len(bar_strs), batch_size)]
        latents_list = []
        for bar_str_batch in tqdm(bar_str_batches):
            latents = model.encode_batch(bar_str_batch, do_sample=False)
            latents_list.append(latents)
        latents = torch.cat(latents_list, dim=0)
        print("Latents shape:", latents.shape)

        tgt_fp = os.path.join(tgt_dir, fn.replace('.jsonl', '.pt'))
        torch.save(latents, tgt_fp)

        # Compute mean and std
        tot_mean = latents.mean().item()
        tot_std = latents.std().item()
        dim_mean = latents.mean(dim=0)
        dim_std = latents.std(dim=0)
        print(f"Latents mean: {tot_mean}, std: {tot_std}")
        print(f'dim-wise shape: {dim_mean.shape}, mean: {dim_mean}, std: {dim_std}')

        # # Scale the latents to make std=1.0
        # scale_factor = latents.std()
        # print(f"Scaling latents by factor {scale_factor} to make std=1.0")
        # latents = latents / scale_factor

        # # Debug: check std
        # latents_reshaped = latents.view(latents.shape[0], -1)
        # print(f"Latents after scaling, shape: {latents.shape}, std: {latents_reshaped.std()}")

        

if __name__ == "__main__":
    main()