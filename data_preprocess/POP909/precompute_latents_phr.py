"""
This file precompute and save phrase-level (track within a bar) latent representations

[2025-10-03] When using MQVAE,
- Need to remove the instrument token at the beginning of each bar
- Scale the latents by 1/0.3428 to make std=1.0 (original std=0.3428)
"""

import os
import sys

# Add project root to sys.path
cwd = os.getcwd()
dirof = os.path.dirname
project_root = dirof(dirof(cwd))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.vae_inference import MQVAE, MQVAE_Pos, PhraseVAE
from sonata_utils import read_jsonl, ls, create_dir_if_not_exist, jpath, save_json, read_json
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from remi_z import MultiTrack
from datasets.dataset_utils import load_phrase_annot_cleaned


def main():
    # get_evaluation_set()
    add_inst_config_info_to_latent_dataset()


def procedures():
    compute_latents()
    compute_latents_no_pad()
    # get_scale_factor()
    # construct_bar_level_latents()
    
    construct_song_level_latents()
    prepare_latent_dataset_with_phrase_annotation()
    split_latent_dataset_with_phrase_annotation()
    get_evaluation_set()
    add_inst_config_info_to_latent_dataset()


def add_inst_config_info_to_latent_dataset():
    '''
    Add instrument configuration info to each song in the latent dataset
    Instrument configuration is a sequence of tokens, indicating instruments used in each bar, and bar lines.
    '''
    dataset_dir = '/data1/longshen/Datasets/Piano/POP909/latents/song_level_phr_seq_with_annot'
    dataset_fns = ['train_data.pt', 'valid_data.pt']
    midi_dir = '/data1/longshen/Datasets/Piano/POP909/pop909_longshen/data_key_normed'
    new_dataset_dir = '/data1/longshen/Datasets/Piano/POP909/latents/song_level_phr_seq_with_annot_and_instconf'
    create_dir_if_not_exist(new_dataset_dir)

    for data_fn in dataset_fns:
        data = torch.load(jpath(dataset_dir, data_fn))
        new_data = {}
        pbar = tqdm(total=len(data), desc=f"Processing {data_fn}")
        for song_name, song_data in data.items():
            pbar.update(1)
            # Read instrument configuration from midi file
            midi_fp = jpath(midi_dir, song_name, f'{song_name}.mid')
            mt = MultiTrack.from_midi(midi_fp)
            inst_conf = []
            for bar in mt.bars:
                bar_inst_conf = []
                # Get instruments in this bar
                inst_of_the_bar = [f'i-{i}' for i in bar.get_unique_insts()]
                bar_inst_conf.extend(inst_of_the_bar)
                bar_inst_conf.append('b-1')
                inst_conf.append(bar_inst_conf)
            song_data['inst_conf'] = inst_conf
            new_data[song_name] = song_data

        # Save new dataset
        save_fp = jpath(new_dataset_dir, data_fn)
        torch.save(new_data, save_fp)
        print("Saved new dataset with inst conf to", save_fp)


def get_evaluation_set():
    '''
    Hold out 100 songs from training set as evaluation set
    '''
    train_fp = '/data1/longshen/Datasets/Piano/POP909/latents/song_level_phr_seq_with_annot/train_data.pt'
    train_dataset = torch.load(train_fp)
    
    idx = np.arange(len(train_dataset))
    np.random.seed(42)
    eval_indices = np.random.choice(idx, size=100, replace=False).tolist()
    train_song_names = list(train_dataset.keys())
    eval_song_names = [train_song_names[i] for i in eval_indices]

    eval_dataset = {name: train_dataset[name] for name in eval_song_names}
    print("Evaluation songs:", len(eval_dataset))

    # Save eval dataset
    save_dir = '/data1/longshen/Datasets/Piano/POP909/latents/song_level_phr_seq_with_annot'
    eval_fp = jpath(save_dir, 'eval_data.pt')
    torch.save(eval_dataset, eval_fp)
    print("Saved eval dataset to", eval_fp)


def split_latent_dataset_with_phrase_annotation():
    data_fp = '/data1/longshen/Datasets/Piano/POP909/latents/song_level_phr_seq_with_annot/song_level_latents_with_phrase_annot.pt'
    dataset = torch.load(data_fp)
    song_names = list(dataset.keys())
    n_songs = len(song_names)
    print("Total songs:", n_songs)

    # Randomly split into train and val set (10% for val)
    np.random.seed(42)
    validation_ratio = 0.05
    n_val = int(validation_ratio * n_songs)
    val_song_names = np.random.choice(song_names, size=n_val, replace=False).tolist()
    train_song_names = [name for name in song_names if name not in val_song_names]
    print("Train songs:", len(train_song_names))
    print("Val songs:", len(val_song_names))

    train_dataset = {name: dataset[name] for name in train_song_names}
    val_dataset = {name: dataset[name] for name in val_song_names}
    print(len(train_dataset), len(val_dataset))
    save_dir = '/data1/longshen/Datasets/Piano/POP909/latents/song_level_phr_seq_with_annot'
    assert os.path.exists(save_dir), "Save dir does not exist!"
    torch.save(train_dataset, jpath(save_dir, 'train_data.pt'))
    torch.save(val_dataset, jpath(save_dir, 'valid_data.pt'))


def prepare_latent_dataset_with_phrase_annotation():
    all_song_latent_fp = '/data1/longshen/Datasets/Piano/POP909/latents/song_level_phr_seq/all_songs_latents.pt'
    data = torch.load(all_song_latent_fp)  # (n_song, max_n_phrase, d_z)
    latents = data['latents']
    lengths = data['lengths']
    
    dataset_dir = '/data1/longshen/Datasets/Piano/POP909/pop909_longshen/data_key_normed'
    song_names = ls(dataset_dir)

    dataset = {}
    for i, song_name in enumerate(song_names):
        dataset[song_name] = {'latent': latents[i], 'length': lengths[i]}
    print("Total songs:", len(dataset))

    # Remove songs with length > 512 phrases
    max_n_phrase = 512
    filtered_dataset = {}
    for song_name, data in dataset.items():
        if data['length'] <= max_n_phrase:
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
    save_dir = '/data1/longshen/Datasets/Piano/POP909/latents/song_level_phr_seq_with_annot'
    save_fn = 'song_level_latents_with_phrase_annot.pt'
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


def get_new_scale_factor():
    latent = torch.load('/data1/longshen/Datasets/Piano/POP909/latents/song_level_phr_seq/all_songs_latents.pt') 
    # Get 99.9% percentile of the absolute value of the latents
    abs_latent = latent.abs().view(-1).cpu().numpy()
    p999 = np.percentile(abs_latent, 99.9)
    print("99.9% percentile of absolute latent values:", p999)


def filter_and_split_latents():
    data_fp = '/data1/longshen/Datasets/Piano/POP909/latents/song_level_phr_seq/all_songs_latents.pt'
    lengths_fp = '/data1/longshen/Datasets/Piano/POP909/latents/song_level_phr_seq/all_songs_latent_lengths.pt'
    latents = torch.load(data_fp)  # (n_song, max_n_phrase, d_z)
    lengths = torch.load(lengths_fp)  # (n_song,)

    # Note down songs with length > 128 * 4 = 512 phrases
    max_n_phrase = 512
    oversized_song_indices = []
    for i in range(len(lengths)):
        if lengths[i] > max_n_phrase:
            oversized_song_indices.append(i)
    print("Oversized song indices:", oversized_song_indices)

    # Get remaining indices
    remaining_indices = [i for i in range(len(lengths)) if i not in oversized_song_indices]
    
    # Random select 5% as validation set
    n_val = int(0.05 * len(remaining_indices))
    np.random.seed(42)
    val_indices = np.random.choice(remaining_indices, size=n_val, replace=False).tolist()
    train_indices = [i for i in remaining_indices if i not in val_indices]

    # Save train and val latents and lengths
    train_latents = latents[train_indices, :max_n_phrase, :]
    train_lengths = lengths[train_indices]
    val_latents = latents[val_indices, :max_n_phrase, :]
    val_lengths = lengths[val_indices]
    save_dir = '/data1/longshen/Datasets/Piano/POP909/latents/song_level_phr_seq'
    torch.save(train_latents, jpath(save_dir, 'train_latents.pt'))
    torch.save(train_lengths, jpath(save_dir, 'train_lengths.pt'))
    torch.save(val_latents, jpath(save_dir, 'val_latents.pt'))
    torch.save(val_lengths, jpath(save_dir, 'val_lengths.pt'))
    print("Saved train and val latents and lengths.")


def validate_song_level_latents():
    model = PhraseVAE().cuda()
    latent_fp = '/data1/longshen/Datasets/Piano/POP909/latents/song_level_phr_seq/all_songs_latents.pt'
    latents = torch.load(latent_fp).cuda()  # (n_song, max_n_phrase, d_z)
    print("Loaded latents shape:", latents.shape)
    lengths_fp = '/data1/longshen/Datasets/Piano/POP909/latents/song_level_phr_seq/all_songs_latent_lengths.pt'
    lengths = torch.load(lengths_fp)  # (n_song,)
    print("Loaded lengths shape:", lengths.shape)

    # Decode the first song
    song_latents = latents[0, :lengths[0], :].unsqueeze(0)  # (1, n_phrase, d_z)
    decoded_phrases = model.decode_batch(song_latents)
    song_str = decoded_phrases[0]
    print("Decoded phrases of first song:")
    save_dir = '/home/longshen/work/AccGen/AccGen/temp/song_level_phr_seq_latents'
    create_dir_if_not_exist(save_dir)
    out_fp = jpath(save_dir, 'first_song_decoded_phrases.txt')
    with open(out_fp, 'w') as f:
        f.write(song_str)
    print("Saved decoded phrases to", out_fp)


def construct_song_level_latents():
    '''
    Construct song-level latents.
    The result will be [n_song, max_n_bars * max_n_phrase=4, d_z]
    '''
    midi_dir = '/data1/longshen/Datasets/Piano/POP909/pop909_longshen/data_key_normed'
    song_fns = ls(midi_dir)
    midi_fps = [jpath(midi_dir, fn, f'{fn}.mid') for fn in song_fns]
    assert os.path.exists(midi_fps[0]), "MIDI file does not exist!"
    vae = PhraseVAE().cuda()

    all_latents = []
    n_phrase_of_songs = []
    cnt = 0
    for midi_fp in tqdm(midi_fps):
        mt = MultiTrack.from_midi(midi_fp)
        phrases_of_song = []
        song_latents = []
        for bar in mt.bars:
            phrase_strs = []
            for track_id, track in bar.tracks.items():
                phrase_seq = track.to_remiz_seq()
                phrase_strs.append(' '.join(phrase_seq))

            # Pad to 3 phrases
            while len(phrase_strs) < 3:
                phrase_strs.append('[INST]') # empty track token
            
            phrase_strs.append('b-1') # Add a bar line, 4 latents per bar

            phrases_of_song.append(phrase_strs)
            
            # Encode all phrases in the bar
            phrase_latents_of_bar = vae.encode_batch(phrase_strs, do_sample=False) # (n_phrase, d_z)
            song_latents.append(phrase_latents_of_bar.cpu())

        song_latent_tensor = torch.cat(song_latents, dim=0)  # (n_bars * 4, d_z)
        n_phrase_of_songs.append(song_latent_tensor.shape[0])
        all_latents.append(song_latent_tensor)

        # print(song_latent_tensor.shape)

        cnt += 1
        # if cnt == 10:
        #     break # Debug


    # # Pad to max sequence length (max_n_bar * 4)
    # max_n_phrase = max([latents.shape[0] for latents in all_latents])
    # print("Max n_phrase:", max_n_phrase)
    # print('Max n_bars:', max_n_phrase // 4)
    # padded_latents = []
    # lengths = []
    # for latents in all_latents:
    #     n_phrase = latents.shape[0]
    #     lengths.append(n_phrase)
    #     if n_phrase < max_n_phrase:
    #         pad_len = max_n_phrase - n_phrase
    #         pad_tensor = torch.zeros((pad_len, latents.shape[1]))
    #         latents = torch.cat([latents, pad_tensor], dim=0)
    #     padded_latents.append(latents.unsqueeze(0))
    # padded_latents = torch.cat(padded_latents, dim=0) # (n_song, max_n_phrase, d_z)
    # print("Padded latents shape:", padded_latents.shape)
    # lengths = torch.tensor(lengths, dtype=torch.long)

    # Save
    out_dir = '/data1/longshen/Datasets/Piano/POP909/latents/song_level_phr_seq'
    create_dir_if_not_exist(out_dir)
    out_fn = 'all_songs_latents.pt'

    res = {
        'latents': all_latents,
        'lengths': n_phrase_of_songs,
    }
    save_fp = jpath(out_dir, out_fn)
    torch.save(res, save_fp)
    print("Saved to", save_fp)

    # out_fp = jpath(out_dir, out_fn)
    # torch.save(padded_latents, out_fp)
    # lengths_fp = jpath(out_dir, 'all_songs_latent_lengths.pt')
    # torch.save(lengths, lengths_fp)
    # print("Saved to", out_fp, lengths_fp)
    



def plot_latent_std_hist():
    latent_fp = '/data1/longshen/Datasets/Piano/POP909/latents/bar_level_pos_seq_64/train_dim_std.pt'
    out_dir = os.path.dirname(latent_fp)
    dim_std = torch.load(latent_fp)

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
    phrase_jsonl_dir = "/data1/longshen/Datasets/Piano/POP909/jsonl/phrase_level"
    out_dir = "/data1/longshen/Datasets/Piano/POP909/latents/bar_level_phr_seq"
    create_dir_if_not_exist(out_dir)
    data_fns = ls(phrase_jsonl_dir, ext=".jsonl")
    model = PhraseVAE().cuda()
    for fn in data_fns:
        print("Processing", fn)
        jsonl_fp = os.path.join(phrase_jsonl_dir, fn)
        pos_data = read_jsonl(jsonl_fp)
        all_bars = []
        latent_seq_of_bar = []
        cnt = 0
        for phr_str in pos_data:
            latent_seq_of_bar.append(phr_str)
            if phr_str.startswith("b-"):
                if len(latent_seq_of_bar) > 0:
                    all_bars.append(latent_seq_of_bar)
                latent_seq_of_bar = []

        if len(latent_seq_of_bar) > 0:
            all_bars.append(latent_seq_of_bar)

        # # Debug
        # all_bars = all_bars[0:10]
        

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

        # Debug
        # print(all_bars)
        # print(all_bars[1]) # list[n_phrase (padded to max n_track)]
        debug_idx = 0
        bar_str = ' '.join(all_bars[debug_idx])
        l = model.encode_batch(all_bars[debug_idx], do_sample=False) # (n_track, d_z)
        t = model.decode_batch(l.cuda().unsqueeze(0))[0]
        print(bar_str)
        print(t)
        for i in range(len(bar_str.split())):
            if bar_str.split()[i] != t.split()[i]:
                print(f"Mismatch at position {i}: original {bar_str.split()[i]}, decoded {t.split()[i]}")
        # assert t == bar_str, "Decoded bar does not match original"
        # print('Successfully verified encoding and decoding for one bar!')

        # For each bar, encode all position sequence with VAE
        latents_list = []
        print("Encoding bars...")
        for bar in tqdm(all_bars):
            latents = model.encode_batch(bar, do_sample=False)  # (N_pos, D)
            # print("Encoded latents shape for one bar:", latents.shape)
            latents_list.append(latents.cpu())
            n_pos = len(bar)
        latents = torch.stack(latents_list, dim=0)  # (N_bars, N_pos, D)
        print("Latents shape:", latents.shape)

        # Compute raw std
        raw_std = latents.std().item()
        raw_mean = latents.mean().item()
        raw_max = latents.max().item()
        raw_min = latents.min().item()

        # Compute max and 99.9% percentile of absolute value
        abs_latents = latents.abs().view(-1).cpu().numpy()
        abs_max = np.max(abs_latents).item()
        abs_p999 = np.percentile(abs_latents, 99.9).item()
        print("Latents raw mean:", raw_mean, "raw std:", raw_std)
        print("Latents abs max:", abs_max, "99.9% percentile:", abs_p999)

        # Plot the first latent vector
        plt.figure(figsize=(10, 4))
        plt.plot(latents[0, 0, :].numpy(), label="Latent Vector", marker="o")
        plt.xlabel("Dimension Index")
        plt.ylabel("Value")
        plt.title("First Latent Vector of First Bar")
        plt.legend()
        plt.grid()
        plt.savefig(jpath(out_dir, fn.replace(".jsonl", "_first_latent_vec.png")), dpi=300)
        plt.close()

        save_json(
            {
                "raw_mean": raw_mean,
                "raw_std": raw_std,
                "raw_max": raw_max,
                "raw_min": raw_min,
                "abs_max": abs_max,
                "abs_99.9_percentile": abs_p999,
            },
            jpath(out_dir, fn.replace(".jsonl", "_raw_stat.json")),
        )

        # Compute dim-wise mean and std
        dim_mean = latents.mean(dim=(0, 1))
        dim_std = latents.std(dim=(0, 1))
        torch.save(dim_mean, jpath(out_dir, fn.replace(".jsonl", "_dim_mean.pt")))
        torch.save(dim_std, jpath(out_dir, fn.replace(".jsonl", "_dim_std.pt")))

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
            jpath(out_dir, fn.replace(".jsonl", "_dim_std_histogram.png")), dpi=300
        )
        plt.close()

        # Save
        latents_fp = os.path.join(out_dir, fn.replace(".jsonl", "_latents.pt"))
        lengths_fp = os.path.join(out_dir, fn.replace(".jsonl", "_lengths.pt"))
        torch.save(latents, latents_fp)
        torch.save(lengths, lengths_fp)
        print("Saved to", latents_fp, lengths_fp)

        # Decode the first 10 bars for sanity check
        decoded_bars = model.decode_batch(latents[0:10, :, :].cuda())
        # for i in range(min(10, latents.shape[0])):
        #     latent_bar = latents[i, :, :].cuda()
        #     decoded_bar_strs = model.decode_batch(latent_bar)
        #     decoded_bars.append(decoded_bar_strs)
        decoded_fp = os.path.join(
            out_dir, fn.replace(".jsonl", "_decoded_bars.txt")
        )
        with open(decoded_fp, "w") as f:
            for i, bar_str in enumerate(decoded_bars):
                f.write(f"Bar {i}:\n")
                f.write(bar_str + "\n")
        print("Decoded bars saved to", decoded_fp)  


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


def compute_latents_no_pad():
    model = PhraseVAE().cuda()
    src_dir = '/data1/longshen/Datasets/Piano/POP909/jsonl/phrase_level'
    tgt_dir = "/data1/longshen/Datasets/Piano/POP909/latents/phrase_level"
    create_dir_if_not_exist(tgt_dir)

    data_fns = ls(src_dir, ext=".jsonl")
    phrase_strs = []
    for fn in data_fns:
        print("Processing", fn)
        jsonl_fp = os.path.join(src_dir, fn)
        data = read_jsonl(jsonl_fp)
        data = [phr for phr in data if phr != "[INST]"]  # Remove empty phrases
        phrase_strs.extend(data)
    
    print("Total phrases:", len(phrase_strs))

    # Encode with 128 batch size
    batch_size = 128
    bar_str_batches = [
        phrase_strs[i : i + batch_size] for i in range(0, len(phrase_strs), batch_size)
    ]
    latents_list = []
    for phr_str_batch in tqdm(bar_str_batches):
        latents = model.encode_batch(phr_str_batch, do_sample=False)
        latents_list.append(latents)
    latents = torch.cat(latents_list, dim=0)
    print("Latents shape:", latents.shape)

    save_fn = 'all_phr_latents_no_pad.pt'
    tgt_fp = os.path.join(tgt_dir, save_fn)
    torch.save(latents, tgt_fp)

    # Compute mean and std
    tot_mean = latents.mean().item()
    tot_std = latents.std().item()
    print(f"Latents mean: {tot_mean}, std: {tot_std}")
    
    # Get 99.9% percentile of absolute value
    abs_latents = latents.abs().view(-1).cpu().numpy()
    abs_p999 = np.percentile(abs_latents, 99.9).item()
    
    stat_save_fp = os.path.join(tgt_dir, 'all_phr_latents_no_pad_stats.json')
    save_json({
        'count': latents.shape[0],
        'mean': tot_mean, 
        'std': tot_std, 
        '99.9_percentile_abs': abs_p999}, stat_save_fp)

    


def compute_latents():
    model = PhraseVAE().cuda()
    src_dir = '/data1/longshen/Datasets/Piano/POP909/jsonl/phrase_level'
    tgt_dir = "/data1/longshen/Datasets/Piano/POP909/latents/phrase_level"
    create_dir_if_not_exist(tgt_dir)

    data_fns = ls(src_dir, ext=".jsonl")
    phrase_strs = []
    for fn in data_fns:
        print("Processing", fn)
        jsonl_fp = os.path.join(src_dir, fn)
        data = read_jsonl(jsonl_fp)
        phrase_strs.extend(data)
    
    print("Total phrases:", len(phrase_strs))

    # Encode with 128 batch size
    batch_size = 128
    bar_str_batches = [
        phrase_strs[i : i + batch_size] for i in range(0, len(phrase_strs), batch_size)
    ]
    latents_list = []
    for phr_str_batch in tqdm(bar_str_batches):
        latents = model.encode_batch(phr_str_batch, do_sample=False)
        latents_list.append(latents)
    latents = torch.cat(latents_list, dim=0)
    print("Latents shape:", latents.shape)

    save_fn = 'all_phr_latents_with_pad.pt'
    tgt_fp = os.path.join(tgt_dir, save_fn)
    torch.save(latents, tgt_fp)

    # Compute mean and std
    tot_mean = latents.mean().item()
    tot_std = latents.std().item()
    print(f"Latents mean: {tot_mean}, std: {tot_std}")
    
    # Get 99.9% percentile of absolute value
    abs_latents = latents.abs().view(-1).cpu().numpy()
    abs_p999 = np.percentile(abs_latents, 99.9).item()
    
    stat_save_fp = os.path.join(tgt_dir, 'all_phr_latents_no_pad_stats.json')
    save_json({
        'count': latents.shape[0],
        'mean': tot_mean, 
        'std': tot_std, 
        '99.9_percentile_abs': abs_p999}, stat_save_fp)


if __name__ == "__main__":
    main()
