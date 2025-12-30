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

from evaluation.metrics import compute_fid, compute_ngram_fid, visualize_ssm_for_song_phrase, visualize_ssm_for_song_bar, bright_ratio, bar_recurrence_score_diag
import numpy as np
import matplotlib.pyplot as plt
from sonata_utils import jpath, read_jsonl, save_jsonl, ls, save_json, create_dir_if_not_exist
import re
import json
from remi_z import MultiTrack
from tqdm import tqdm
import torch
import random
from the_utils.latent_utils import augment_phrase_permute
from tqdm import tqdm


def main():
    check_n_token_per_song()


def procedures():
    check_note_duration_token_distribution()
    check_n_token_and_n_note_distribution()
    check_unique_piano_bars()
    check_n_bar()
    check_note_per_song()
    check_note_per_song_slack()
    check_position_per_song()
    check_pos_per_bar()
    check_unique_positions()
    check_unique_phrases()
    check_latent_length_distribution()
    check_splitting()
    check_real_vs_real_fid()
    check_real_diversity()

    visualize_ssm()
    visualize_ssm_permute_section()
    check_bright_ratio()
    check_brs_diag()
    check_srs_distribution()

    check_n_token_per_bar()
    check_n_phrase_per_bar()


def check_srs_distribution():
    # Check SSM of a song in the dataset
    eval_latent_fp = '/data1/longshen/Datasets/Piano/POP909/latents/song_level_phr_seq_with_annot/song_level_latents_with_phrase_annot.pt'
    eval_data = torch.load(eval_latent_fp, map_location='cpu') # list of latents
    
    srss = []
    for song_id in eval_data:
        latent = eval_data[song_id]['latent']
        srs = bar_recurrence_score_diag(latent, threshold=0.5, min_run_length=4)
        srss.append(srs)

    # Draw histogram
    plt.figure(figsize=(8, 4))
    plt.hist(srss, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel('BRS Diagonal Score')
    plt.ylabel('Count')
    plt.title('Distribution of BRS Diagonal Scores in POP909')
    plt.tight_layout()
    save_dir = '/home/longshen/work/AccGen/AccGen/data_preprocess/POP909/statistics/brs_diag_distribution'
    create_dir_if_not_exist(save_dir)
    save_fp = jpath(save_dir, 'brs_diag_distribution.png')
    plt.savefig(save_fp, dpi=150)
    plt.close()


def check_brs_diag():
    # eval_latent_fp = '/data1/longshen/Datasets/Piano/POP909/latents/song_level_phr_seq_with_annot/eval_data.pt'
    eval_latent_fp = '/data1/longshen/Datasets/Piano/POP909/latents/song_level_phr_seq_with_annot/train_data.pt'
    eval_data = torch.load(eval_latent_fp, map_location='cpu') # list of latents
    eval_latents = [eval_data[k]['latent'] for k in eval_data] # list of [seq_len, dim]

    brs_diag_scores = []
    cnt = 0
    for z in eval_latents:
        score = bar_recurrence_score_diag(z, threshold=0.5, min_run_length=4)
        brs_diag_scores.append(score)
        cnt += 1
        if cnt == 20:
            break
    avg_brs_diag = sum(brs_diag_scores) / len(brs_diag_scores)
    print(f"Average BRS diagonal over eval set: {avg_brs_diag:.4f}")


def check_bright_ratio():
    eval_latent_fp = '/data1/longshen/Datasets/Piano/POP909/latents/song_level_phr_seq_with_annot/eval_data.pt'
    eval_data = torch.load(eval_latent_fp, map_location='cpu') # list of latents
    eval_latents = [eval_data[k]['latent'] for k in eval_data] # list of [seq_len, dim]

    bright_ratios = []
    cnt = 0
    for z in eval_latents:
        ratio = bright_ratio(z, 0.5)
        bright_ratios.append(ratio)
        cnt += 1
        if cnt == 10:
            break
    avg_bright_ratio = sum(bright_ratios) / len(bright_ratios)
    print(f"Average bright ratio over eval set: {avg_bright_ratio:.4f}") # 0.1532


def visualize_ssm_permute_section():
    '''
    Add section permutation to a real song, observe self similarity matrix changes
    '''
    eval_latent_fp = '/data1/longshen/Datasets/Piano/POP909/latents/song_level_phr_seq_with_annot/eval_data.pt'
    eval_data = torch.load(eval_latent_fp, map_location='cpu') # list of latents
    eval_latents = [eval_data[k]['latent'] for k in eval_data] # list of [seq_len, dim]
    z = eval_latents[0]
    annot = eval_data[list(eval_data.keys())[0]]['phrase_annotation'] # list of section labels
    print(annot)

    save_dir = '/home/longshen/work/AccGen/AccGen/data_preprocess/POP909/statistics/smm_visualization'
    # visualize_ssm_for_song_phrase(eval_latents[0], save_dir)
    # visualize_ssm_for_song_bar(eval_latents[0], save_dir)
    z = augment_phrase_permute(z, max_n_bar=64, sec_annot=annot)
    visualize_ssm_for_song_bar(z, save_dir, filename='ssm_bar_section_permute.png')


def visualize_ssm():
    eval_latent_fp = '/data1/longshen/Datasets/Piano/POP909/latents/song_level_phr_seq_with_annot/eval_data.pt'
    eval_data = torch.load(eval_latent_fp, map_location='cpu') # list of latents
    eval_latents = [eval_data[k]['latent'] for k in eval_data] # list of [seq_len, dim]

    save_dir = '/home/longshen/work/AccGen/AccGen/data_preprocess/POP909/statistics/smm_visualization'
    visualize_ssm_for_song_phrase(eval_latents[0], save_dir)
    visualize_ssm_for_song_bar(eval_latents[0], save_dir)


def check_real_diversity():
    eval_latent_fp = '/data1/longshen/Datasets/Piano/POP909/latents/song_level_phr_seq_with_annot/eval_data.pt'
    eval_data = torch.load(eval_latent_fp, map_location='cpu') # list of latents
    eval_latents = [eval_data[k]['latent'] for k in eval_data] # list of [seq_len, dim]

    # Random choose 20 songs to compute diversity
    n = 20
    idx = list(range(len(eval_latents)))
    random.shuffle(idx)
    selected_idx = idx[:n]
    eval_latents = [eval_latents[i] for i in selected_idx]

    diversity = compute_real_diversity_all_pairs(eval_latents) # 7.2841
    print(f"Real-data diversity: {diversity:.4f}")


def compute_real_diversity_all_pairs(eval_set):
    """
    eval_set: list of tensors, each [seq_len_i, dim]
    Returns:
        avg_distance (float): average pairwise L2 distance over ALL pairs
    """

    # flatten
    all_latents = torch.cat(eval_set, dim=0).float()   # [N, dim]
    N = all_latents.shape[0]

    # pairwise distance matrix (N x N)
    # d[i,j] = ||x_i - x_j||
    # This constructs full matrix: O(N^2) memory and compute
    diff = all_latents.unsqueeze(1) - all_latents.unsqueeze(0)   # [N, N, dim]
    dist_mat = torch.norm(diff, p=2, dim=-1)                     # [N, N]

    # use only upper triangle (i < j)
    triu = torch.triu(dist_mat, diagonal=1)
    distances = triu[triu > 0]

    return distances.mean().item()


def compute_real_diversity(eval_set, n_pairs=5000):
    """
    eval_set: list of tensors, each [seq_len_i, dim]
    n_pairs: number of random pairs to sample

    Returns:
        avg_distance (float): estimated real-data diversity
    """

    # flatten all songs into one big latent matrix
    all_latents = torch.cat(eval_set, dim=0).float()   # [N, dim]
    N = all_latents.shape[0]

    distances = []
    for _ in range(n_pairs):
        i = random.randint(0, N-1)
        j = random.randint(0, N-1)
        if i == j:
            continue
        d = torch.norm(all_latents[i] - all_latents[j], p=2)
        distances.append(d.item())

    return sum(distances) / len(distances)


def check_real_vs_real_fid():
    eval_latent_fp = '/data1/longshen/Datasets/Piano/POP909/latents/song_level_phr_seq_with_annot/eval_data.pt'
    eval_data = torch.load(eval_latent_fp, map_location='cpu') # list of latents
    eval_latents = [eval_data[k]['latent'] for k in eval_data] # list of [seq_len, dim]
    fid = real_vs_real_fid_split_in_middle(eval_latents)
    print(f"Real vs Real FID, split from middle: {fid:.4f}") # 0.7830

    fid = real_vs_real_fid_split_by_song(eval_latents)
    print(f"Real vs Real FID, split by song: {fid:.4f}") # 1.53

    bigram_fid = real_vs_real_bigram_fid(eval_latents)
    print(f"Real vs Real Bigram FID: {bigram_fid:.4f}") # 3.8

    fourgram_fid = real_vs_real_4gram_fid(eval_latents)
    print(f"Real vs Real 4-gram FID: {fourgram_fid:.4f}") 

    eightgram_fid = real_vs_real_8gram_fid(eval_latents)
    print(f"Real vs Real 8-gram FID: {eightgram_fid:.4f}")

    sixteengram_fid = real_vs_real_16gram_fid(eval_latents)
    print(f"Real vs Real 16-gram FID: {sixteengram_fid:.4f}")

    thirtytwogram_fid = real_vs_real_32gram_fid(eval_latents)
    print(f"Real vs Real 32-gram FID: {thirtytwogram_fid:.4f}")





def real_vs_real_32gram_fid(eval_set):
    idx = list(range(len(eval_set)))
    random.shuffle(idx)

    mid = len(idx) // 2
    A = [eval_set[i] for i in idx[:mid]]
    B = [eval_set[i] for i in idx[mid:]]

    return compute_ngram_fid(A, B, n=32)


def real_vs_real_16gram_fid(eval_set):
    idx = list(range(len(eval_set)))
    random.shuffle(idx)

    mid = len(idx) // 2
    A = [eval_set[i] for i in idx[:mid]]
    B = [eval_set[i] for i in idx[mid:]]

    return compute_ngram_fid(A, B, n=16)

def real_vs_real_8gram_fid(eval_set):
    idx = list(range(len(eval_set)))
    random.shuffle(idx)

    mid = len(idx) // 2
    A = [eval_set[i] for i in idx[:mid]]
    B = [eval_set[i] for i in idx[mid:]]

    return compute_ngram_fid(A, B, n=8)


def real_vs_real_4gram_fid(eval_set):
    """
    eval_set: list of songs [seq_len_i, dim]
    """
    idx = list(range(len(eval_set)))
    random.shuffle(idx)
    mid = len(idx) // 2
    A = [eval_set[i] for i in idx[:mid]]
    B = [eval_set[i] for i in idx[mid:]]

    return compute_ngram_fid(A, B, n=4)


def real_vs_real_bigram_fid(eval_set, n_A=20):
    N = len(eval_set)
    assert n_A < N

    idx = list(range(N))
    random.shuffle(idx)

    idx_A = idx[:n_A]
    idx_B = idx[n_A:]

    A = [eval_set[i] for i in idx_A]
    B = [eval_set[i] for i in idx_B]

    return compute_ngram_fid(A, B, n=2)


def real_vs_real_fid_split_by_song(eval_set, n_A=20):
    """
    eval_set: list of tensors, each [seq_len_i, dim]
    n_A: number of songs in group A
    """

    N = len(eval_set)
    assert n_A < N, "n_A must be smaller than total number of songs."

    idx = list(range(N))
    random.shuffle(idx)

    idx_A = idx[:n_A]
    idx_B = idx[n_A:]

    A = [eval_set[i] for i in idx_A]
    B = [eval_set[i] for i in idx_B]

    # 1-gram FID (phrase-level FID)
    return compute_ngram_fid(A, B, n=1)


def real_vs_real_fid_split_in_middle(eval_set):
    """
    eval_set: list of tensors, each [seq_len_i, dim]
              trimmed (no padding)
    Returns:
        fid (float)
    """

    # ------------------------
    # 1) shuffle and split
    # ------------------------
    idx = list(range(len(eval_set)))
    random.shuffle(idx)

    mid = len(idx) // 2
    idx_a = idx[:mid]
    idx_b = idx[mid:]

    real_a = [eval_set[i] for i in idx_a]   # list of [L_i, dim]
    real_b = [eval_set[i] for i in idx_b]   # list of [L_j, dim]

    # ------------------------
    # 2) flatten each side
    # ------------------------
    z_a = torch.cat(real_a, dim=0).float()   # [N_a, dim]
    z_b = torch.cat(real_b, dim=0).float()   # [N_b, dim]

    # ------------------------
    # 3) compute FID using the generic function
    # ------------------------
    fid = compute_fid(z_a, z_b)

    return fid


def check_splitting():
    '''
    Check number of songs in train and valid splits
    '''
    train_fp = '/data1/longshen/Datasets/Piano/POP909/latents/song_level_phr_seq_with_annot/train_data.pt'
    valid_fp = '/data1/longshen/Datasets/Piano/POP909/latents/song_level_phr_seq_with_annot/valid_data.pt'
    train_data = torch.load(train_fp)
    valid_data = torch.load(valid_fp)
    print(f"Train songs: {len(train_data)}")
    print(f"Valid songs: {len(valid_data)}")


def check_latent_length_distribution():
    '''
    Check the distribution of latent lengths in POP909 dataset
    '''
    latent_fp = '/data1/longshen/Datasets/Piano/POP909/latents/song_level_phr_seq_with_annot/eval_data.pt'
    save_dir = '/home/longshen/work/AccGen/AccGen/data_preprocess/POP909/statistics/n_bar_of_song/eval_split'

    latents = torch.load(latent_fp)
    lengths = []
    for song_id in latents:
        song_lengths = latents[song_id]['length']
        lengths.append(song_lengths)
    lengths = np.array(lengths, dtype=np.int32)
    lengths = lengths // 4

    # Draw histogram of multiple bin_size
    bin_sizes = [1, 2, 5, 10, 20, 50]
    for bin_size in bin_sizes:
        n_bin = lengths.max() // bin_size
        plt.figure(figsize=(8, 4))
        plt.hist(lengths, bins=n_bin, color='skyblue', edgecolor='black', alpha=0.7)
        plt.xlabel('Latent Length')
        plt.ylabel('Count')
        plt.title(f'Distribution of Latent Lengths in POP909 (bins={n_bin})')
        plt.tight_layout()
        
        create_dir_if_not_exist(save_dir)
        save_fp = jpath(save_dir, f'latent_nbar_{n_bin}bins_binsize{bin_size}_distribution.png')
        plt.savefig(save_fp, dpi=150)
        plt.close()

    # plt.figure(figsize=(8, 4))
    # plt.hist(lengths, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    # plt.xlabel('Latent Length')
    # plt.ylabel('Count')
    # plt.title('Distribution of Latent Lengths in POP909')
    # plt.tight_layout()
    # save_dir = '/home/longshen/work/AccGen/AccGen/data_preprocess/POP909/statistics/n_bar_of_song'
    # create_dir_if_not_exist(save_dir)
    # save_fp = jpath(save_dir, 'latent_length_distribution.png')
    # plt.savefig(save_fp, dpi=150)
    # plt.close()


def check_unique_phrases():
    # 路径设置
    DATASET_DIR = '/data1/longshen/Datasets/Piano/POP909/jsonl/phrase_level'
    train_fn = 'train.jsonl'
    val_fn = 'val.jsonl'
    train_data = read_jsonl(jpath(DATASET_DIR, train_fn))
    val_data = read_jsonl(jpath(DATASET_DIR, val_fn))
    data = train_data + val_data
    bars = data

    # # Remove first two tokens (tempo and time signature)
    # bars = [' '.join(bar.split()[2:]) for bar in bars]
    print(bars[:5])

    unique_bars = set(bars)

    print(f"Total phrases: {len(bars)}")
    print(f"Unique phrases: {len(unique_bars)}")
    print(f"Uniqueness ratio: {len(unique_bars) / len(bars):.4f}")


def check_n_token_per_song():
    '''
    How many tokens are in a song?
    '''
    data_dir = '/data1/longshen/Datasets/Piano/POP909/pop909_longshen/data_key_normed'
    song_ids = ls(data_dir)

    n_tokens_per_song = []
    for song_id in song_ids:
        song_fp = jpath(data_dir, song_id)
        midi_fp = jpath(song_fp, f'{song_id}.mid')
        mt = MultiTrack.from_midi(midi_fp)
        tokens = mt.to_remiz_seq()
        n_tokens_per_song.append(len(tokens))
    print(f"Average tokens per song: {np.mean(n_tokens_per_song)}")
    print(f"Max tokens per song: {np.max(n_tokens_per_song)}")
    print(f"Min tokens per song: {np.min(n_tokens_per_song)}")
    # Median, 95%, 99% percentiles
    print(f"Median tokens per song: {np.median(n_tokens_per_song)}")
    print(f"95% tokens per song: {np.percentile(n_tokens_per_song, 95)}")
    print(f"99% tokens per song: {np.percentile(n_tokens_per_song, 99)}")
    save_dir = '/home/longshen/work/AccGen/AccGen/data_preprocess/POP909/statistics/n_token_per_song'
    # draw histogram
    plt.figure(figsize=(8, 4))
    plt.hist(n_tokens_per_song, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Number of Tokens per Song')
    plt.ylabel('Count')
    plt.title('Distribution of Number of Tokens per Song in POP909')
    plt.tight_layout()
    save_fp = jpath(save_dir, 'n_token_per_song_dist.png')
    plt.savefig(save_fp, dpi=150)
    plt.close()



def check_n_phrase_per_bar():
    '''
    How many unique phrases are there in each bar in POP909 dataset?
    '''
    data_dir = '/data1/longshen/Datasets/Piano/POP909/jsonl/bar_level'
    unique_phrases_per_bar = []
    train_data = read_jsonl(jpath(data_dir, 'train.jsonl'))
    valid_data = read_jsonl(jpath(data_dir, 'val.jsonl'))
    data = train_data + valid_data
    for bar in data:
        phrases = set()
        tokens = bar.split()
        for token in tokens:
            if token.startswith('i-'):
                phrases.add(token)
        unique_phrases_per_bar.append(len(phrases))
    print(f"Average unique phrases per bar: {np.mean(unique_phrases_per_bar)}")
    print(f"Max unique phrases per bar: {np.max(unique_phrases_per_bar)}")
    print(f"Min unique phrases per bar: {np.min(unique_phrases_per_bar)}")
    # Median, 95%, 99% percentiles
    print(f"Median unique phrases per bar: {np.median(unique_phrases_per_bar)}")
    print(f"95% unique phrases per bar: {np.percentile(unique_phrases_per_bar, 95)}")
    print(f"99% unique phrases per bar: {np.percentile(unique_phrases_per_bar, 99)}")

    save_dir = '/home/longshen/work/AccGen/AccGen/data_preprocess/POP909/statistics/n_pos_per_bar'
    # draw histogram
    plt.figure(figsize=(8, 4))
    plt.hist(unique_phrases_per_bar, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Number of Unique phrases per Bar')
    plt.ylabel('Count')
    plt.title('Distribution of Number of Unique phrases per Bar in POP909')
    plt.tight_layout()
    save_fp = jpath(save_dir, 'n_unique_pos_per_bar_dist.png')
    plt.savefig(save_fp, dpi=150)
    plt.close()

def check_pos_per_bar():
    '''
    How many unique positions are there in each bar in POP909 dataset?
    '''
    data_dir = '/data1/longshen/Datasets/Piano/POP909/jsonl/bar_level'
    unique_positions_per_bar = []
    train_data = read_jsonl(jpath(data_dir, 'train.jsonl'))
    valid_data = read_jsonl(jpath(data_dir, 'val.jsonl'))
    data = train_data + valid_data
    for bar in data:
        positions = set()
        tokens = bar.split()
        for token in tokens:
            if token.startswith('o-'):
                positions.add(token)
        unique_positions_per_bar.append(len(positions))
    print(f"Average unique positions per bar: {np.mean(unique_positions_per_bar)}")
    print(f"Max unique positions per bar: {np.max(unique_positions_per_bar)}")
    print(f"Min unique positions per bar: {np.min(unique_positions_per_bar)}")
    # Median, 95%, 99% percentiles
    print(f"Median unique positions per bar: {np.median(unique_positions_per_bar)}")
    print(f"95% unique positions per bar: {np.percentile(unique_positions_per_bar, 95)}")
    print(f"99% unique positions per bar: {np.percentile(unique_positions_per_bar, 99)}")

    save_dir = '/home/longshen/work/AccGen/AccGen/data_preprocess/POP909/statistics/n_pos_per_bar'
    # draw histogram
    plt.figure(figsize=(8, 4))
    plt.hist(unique_positions_per_bar, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Number of Unique Positions per Bar')
    plt.ylabel('Count')
    plt.title('Distribution of Number of Unique Positions per Bar in POP909')
    plt.tight_layout()
    save_fp = jpath(save_dir, 'n_unique_pos_per_bar_dist.png')
    plt.savefig(save_fp, dpi=150)
    plt.close()


def check_unique_positions():
    '''
    How many unique positions are there in POP909 dataset?
    '''
    data_dir = '/data1/longshen/Datasets/Piano/POP909/jsonl/position_level'
    unique_positions = set()
    train_data = read_jsonl(jpath(data_dir, 'train.jsonl'))
    valid_data = read_jsonl(jpath(data_dir, 'val.jsonl'))
    data = train_data + valid_data
    print(len(set(data))) # total unique position: 168678

    # Remove the first token
    data_no_pos = [' '.join(pos.split()[1:]) if len(pos.split()) > 1 else pos for pos in data ]
    unique_positions = set(data_no_pos)
    print(len(unique_positions)) # unique position without tempo token: 112177

    # Remove duration token (d-XX)
    data_no_dur = []
    for pos in data:
        tokens = pos.split()
        tokens_no_dur = [t for t in tokens if not t.startswith('d-')]
        data_no_dur.append(' '.join(tokens_no_dur))
    unique_positions_no_dur = set(data_no_dur)
    print(len(unique_positions_no_dur)) # unique position without tempo and duration token:

    # Remove both first token and duration token
    data_no_pos_dur = []
    for pos in data:
        tokens = pos.split()
        tokens_no_dur = [t for t in tokens[1:] if not t.startswith('d-')]
        data_no_pos_dur.append(' '.join(tokens_no_dur))
    unique_positions_no_pos_dur = set(data_no_pos_dur)
    print(len(unique_positions_no_pos_dur)) # unique position without tempo and duration token:


def check_position_per_song():
    dataset_dir = '/data2/longshen/Datasets/Piano/POP909/quantized/midi'
    song_fns = ls(dataset_dir)
    lines = []

    # Randomly select 10% songs as test set
    song_fns = sorted(song_fns)
    np.random.seed(42)

    n_pos = []
    for song_fn in tqdm(song_fns):
        song_fp = jpath(dataset_dir, song_fn)
        mt = MultiTrack.from_midi(song_fp)
        pos_per_song = 0
        for bar in mt.bars:
            notes = bar.get_all_notes()
            unique_pos = set(note.onset for note in notes)
            pos_per_song += len(unique_pos)
        n_pos.append(pos_per_song)

    save_dir = '/home/longshen/work/AccGen/AccGen/data_preprocess/POP909/statistics/n_pos_per_song'
    save_fn = 'n_pos_of_song.json'
    # Calculate percentiles [0, 10, ..., 90, 95, 98, 99, 100]
    percentiles = {f"p{p}": float(np.percentile(n_pos, p)) for p in range(0, 100, 10)}
    for p in [95, 98, 99, 100]:
        percentiles[f"p{p}"] = float(np.percentile(n_pos, p))
    save_fp = jpath(save_dir, save_fn)
    save_json(percentiles, save_fp)

    # Draw histogram
    plt.figure(figsize=(8, 4))
    plt.hist(n_pos, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Number of positions per Song')
    plt.ylabel('Count')
    plt.title('Distribution of Number of positions per Song in POP909')
    plt.tight_layout()
    save_fp = jpath(save_dir, 'n_pos_per_song_dist.png')
    plt.savefig(save_fp, dpi=150)
    plt.close()


def check_note_per_song_slack():
    '''
    Check number of notes of each song in Slach2100 dataset test split
    '''
    dataset_dir = '/data1/longshen/Datasets/Multitrack/slakh2100_flac_redux/test'
    
    # get all midi fps
    song_fns = ls(dataset_dir)
    midi_fps = [jpath(dataset_dir, fn, 'all_src.mid') for fn in song_fns]

    np.random.seed(42)

    n_notes = []
    for song_fp in tqdm(midi_fps):
        mt = MultiTrack.from_midi(song_fp)
        notes = mt.get_all_notes(include_drum=True)
        n_notes.append(len(notes))

    save_dir = '/home/longshen/work/AccGen/AccGen/data_preprocess/POP909/statistics'
    save_fn = 'slack_withdrum_n_note_of_song.json'
    # Calculate percentiles [0, 10, ..., 90, 95, 98, 99, 100]
    percentiles = {f"p{p}": float(np.percentile(n_notes, p)) for p in range(0, 100, 10)}
    for p in [95, 98, 99, 100]:
        percentiles[f"p{p}"] = float(np.percentile(n_notes, p))
    save_fp = jpath(save_dir, save_fn)
    save_json(percentiles, save_fp)

    # Draw histogram
    plt.figure(figsize=(8, 4))
    plt.hist(n_notes, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Number of Notes per Song')
    plt.ylabel('Count')
    plt.title('Distribution of Number of Notes per Song in POP909')
    plt.tight_layout()
    save_fp = jpath(save_dir, 'slack_withdrum_n_note_distribution.png')
    plt.savefig(save_fp, dpi=150)
    plt.close()


def check_note_per_song():
    '''
    Check number of notes of each song in POP909 dataset
    '''
    dataset_dir = '/data1/longshen/Datasets/Piano/POP909/quantized/midi'
    song_fns = ls(dataset_dir)
    lines = []

    # Randomly select 10% songs as test set
    song_fns = sorted(song_fns)
    np.random.seed(42)

    n_notes = []
    for song_fn in tqdm(song_fns):
        song_fp = jpath(dataset_dir, song_fn)
        mt = MultiTrack.from_midi(song_fp)
        notes = mt.get_all_notes(of_insts=[0])
        n_notes.append(len(notes))

    save_dir = '/home/longshen/work/AccGen/AccGen/data_preprocess/POP909/statistics'
    save_fn = 'n_note_of_song.json'
    # Calculate percentiles [0, 10, ..., 90, 95, 98, 99, 100]
    percentiles = {f"p{p}": float(np.percentile(n_notes, p)) for p in range(0, 100, 10)}
    for p in [95, 98, 99, 100]:
        percentiles[f"p{p}"] = float(np.percentile(n_notes, p))
    save_fp = jpath(save_dir, save_fn)
    save_json(percentiles, save_fp)

    # Draw histogram
    plt.figure(figsize=(8, 4))
    plt.hist(n_notes, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Number of Notes per Song')
    plt.ylabel('Count')
    plt.title('Distribution of Number of Notes per Song in POP909')
    plt.tight_layout()
    save_fp = jpath(save_dir, 'n_note_distribution.png')
    plt.savefig(save_fp, dpi=150)
    plt.close()


def check_n_bar():
    '''
    Check number of bars of each song in POP909 dataset
    '''
    dataset_dir = '/data1/longshen/Datasets/Piano/POP909/quantized/midi'
    song_fns = ls(dataset_dir)
    lines = []

    # Randomly select 10% songs as test set
    song_fns = sorted(song_fns)
    np.random.seed(42)

    n_bars = []
    for song_fn in tqdm(song_fns):
        song_fp = jpath(dataset_dir, song_fn)
        mt = MultiTrack.from_midi(song_fp)
        n_bars.append(len(mt.bars))


    save_dir = '/home/longshen/work/AccGen/AccGen/data_preprocess/POP909/statistics'
    save_fn = 'n_bar_of_song.json'
    # Calculate percentiles [0, 10, ..., 90, 95, 98, 99, 100]
    percentiles = {f"p{p}": float(np.percentile(n_bars, p)) for p in range(0, 100, 10)}
    for p in [95, 98, 99, 100]:
        percentiles[f"p{p}"] = float(np.percentile(n_bars, p))
    save_fp = jpath(save_dir, save_fn)
    save_json(percentiles, save_fp)

    # Draw histogram
    plt.figure(figsize=(8, 4))
    plt.hist(n_bars, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Number of Bars per Song')
    plt.ylabel('Count')
    plt.title('Distribution of Number of Bars per Song in POP909')
    plt.tight_layout()
    save_fp = jpath(save_dir, 'n_bar_distribution.png')
    plt.savefig(save_fp, dpi=150)
    plt.close()


def check_unique_piano_bars():
    # 路径设置
    DATASET_DIR = '/data1/longshen/Datasets/Piano/POP909/jsonl/bar_level'
    train_fn = 'train.jsonl'
    val_fn = 'val.jsonl'
    train_data = read_jsonl(jpath(DATASET_DIR, train_fn))
    val_data = read_jsonl(jpath(DATASET_DIR, val_fn))
    data = train_data + val_data
    bars = data

    # # Remove first two tokens (tempo and time signature)
    # bars = [' '.join(bar.split()[2:]) for bar in bars]
    print(bars[:5])

    unique_bars = set(bars)

    print(f"Total bars: {len(bars)}")
    print(f"Unique bars: {len(unique_bars)}")
    print(f"Uniqueness ratio: {len(unique_bars) / len(bars):.4f}")

    # # Save unique bars to a file
    # bars = list(unique_bars)
    # bars.sort()
    # save_dir = '/home/longshen/work/AccGen/AccGen/data_preprocess/POP909/statistics'
    # save_fp = jpath(save_dir, 'unique_piano_bars.jsonl')
    # save_jsonl(bars, save_fp)
    # print(f"Unique bars saved to {save_fp}")

    # # Save number of unique bars
    # save_fp = jpath(save_dir, 'n_unique_piano_bars.txt')
    # with open(save_fp, 'w') as f:
    #     f.write(str(len(unique_bars)) + '\n')
    # print(f"Number of unique bars saved to {save_fp}")


def check_n_token_per_bar():
    # 路径设置
    data_fp1 = '/data1/longshen/Datasets/Piano/POP909/jsonl/bar_level/train.jsonl'
    data_fp2 = '/data1/longshen/Datasets/Piano/POP909/jsonl/bar_level/val.jsonl'
    data1 = read_jsonl(data_fp1)
    data2 = read_jsonl(data_fp2)
    data = data1 + data2

    # Remove lines that is [SEP]
    data = [line for line in data if line.strip() != '[SEP]']
    
    save_dir = '/home/longshen/work/AccGen/AccGen/data_preprocess/POP909/statistics/n_token_per_bar_multitrack'
    create_dir_if_not_exist(save_dir)

    n_tokens = []
    for bar_str in data:
        tokens = bar_str.split()
        n_tokens.append(len(tokens))

    n_tokens = np.array(n_tokens)

    # 画分布图
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(n_tokens, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Number of Tokens per Bar')
    plt.ylabel('Count')
    plt.title('Distribution of Number of Tokens per Bar in POP909')

    plt.tight_layout()
    save_fp = jpath(save_dir, 'n_token_n_note_distribution.png')
    plt.savefig(save_fp, dpi=150)
    plt.close()

    # 输出统计信息
    print(f"Total bars: {len(data)}")
    print(f"Tokens - Mean: {n_tokens.mean():.2f}, Median: {np.median(n_tokens):.2f}, Max: {n_tokens.max()}, Min: {n_tokens.min()}")

    # 计算每隔10%的分位数，并额外存p95, p98, p99
    percentiles = {f"p{p}": float(np.percentile(n_tokens, p)) for p in range(0, 101, 10)}
    for p in [95, 98, 99]:
        percentiles[f"p{p}"] = float(np.percentile(n_tokens, p))
    save_fp = jpath(save_dir, 'n_token_percentiles.json')
    with open(save_fp, 'w') as f:
        json.dump(percentiles, f, indent=2)
    print("Token percentiles saved to statistics/n_token_percentiles.json")

    print(max(n_tokens))



def check_n_token_and_n_note_distribution():
    # 路径设置
    DATASET_DIR = '/data1/longshen/Datasets/Piano/POP909/jsonl'
    JSONL_FP = jpath(DATASET_DIR, 'pop909_piano_track_bars.jsonl')
    save_dir = '/home/longshen/work/AccGen/AccGen/data_preprocess/POP909/statistics'

    bars = read_jsonl(JSONL_FP)
    n_tokens = []
    n_notes = []

    for bar_str in bars:
        tokens = bar_str.split()
        n_tokens.append(len(tokens))
        n_notes.append(sum(1 for t in tokens if t.startswith('o-')))

    n_tokens = np.array(n_tokens)
    n_notes = np.array(n_notes)

    # 画分布图
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(n_tokens, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Number of Tokens per Bar')
    plt.ylabel('Count')
    plt.title('Distribution of Number of Tokens per Bar in POP909')

    plt.subplot(1, 2, 2)
    plt.hist(n_notes, bins=30, color='salmon', edgecolor='black', alpha=0.7)
    plt.xlabel('Number of Notes per Bar')
    plt.ylabel('Count')
    plt.title('Distribution of Number of Notes per Bar in POP909')

    plt.tight_layout()
    save_fp = jpath(save_dir, 'n_token_n_note_distribution.png')
    plt.savefig(save_fp, dpi=150)
    plt.close()

    # 输出统计信息
    print(f"Total bars: {len(bars)}")
    print(f"Tokens - Mean: {n_tokens.mean():.2f}, Median: {np.median(n_tokens):.2f}, Max: {n_tokens.max()}, Min: {n_tokens.min()}")
    print(f"Notes - Mean: {n_notes.mean():.2f}, Median: {np.median(n_notes):.2f}, Max: {n_notes.max()}, Min: {n_notes.min()}")

    # 计算每隔10%的分位数，并额外存p95, p98, p99
    percentiles = {f"p{p}": float(np.percentile(n_tokens, p)) for p in range(0, 101, 10)}
    for p in [95, 98, 99]:
        percentiles[f"p{p}"] = float(np.percentile(n_tokens, p))
    save_fp = jpath(save_dir, 'n_token_percentiles.json')
    with open(save_fp, 'w') as f:
        json.dump(percentiles, f, indent=2)
    print("Token percentiles saved to statistics/n_token_percentiles.json")

    # 同理存n_notes的分位数
    percentiles = {f"p{p}": float(np.percentile(n_notes, p)) for p in range(0, 101, 10)}
    for p in [95, 98, 99]:
        percentiles[f"p{p}"] = float(np.percentile(n_notes, p))
    save_fp = jpath(save_dir, 'n_note_percentiles.json')
    with open(save_fp, 'w') as f:
        json.dump(percentiles, f, indent=2)
    print("Note percentiles saved to statistics/n_note_percentiles.json")


def check_note_duration_token_distribution():
    # 路径设置
    DATASET_DIR = '/data1/longshen/Datasets/Piano/POP909/jsonl'
    JSONL_FP = jpath(DATASET_DIR, 'pop909_piano_track_bars.jsonl')

    bars = read_jsonl(JSONL_FP)
    lengths = []

    # 正则表达式匹配 d-XX token
    pattern = re.compile(r'd-(\d+)')

    for bar_str in bars:
        # 查找所有 d-XX token
        matches = pattern.findall(bar_str)
        lengths.extend([int(x) for x in matches])

    lengths = np.array(lengths)

    # 画分布图
    plt.figure(figsize=(8, 4))
    plt.hist(lengths, bins=range(1, lengths.max()+2), color='royalblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Note Duration Token (XX in d-XX)')
    plt.ylabel('Count')
    plt.title('Distribution of Note Duration Tokens in POP909 (bar-level)')
    plt.tight_layout()
    plt.savefig('statistics/note_duration_token_distribution.png', dpi=150)
    plt.close()

    # 输出统计信息
    print(f"Total note tokens: {len(lengths)}")
    print(f"Mean: {lengths.mean():.2f}, Median: {np.median(lengths):.2f}, Max: {lengths.max()}, Min: {lengths.min()}")

    # 计算每隔10%的分位数，并额外存p95, p98, p99
    percentiles = {f"p{p}": float(np.percentile(lengths, p)) for p in range(0, 101, 10)}
    for p in [95, 98, 99]:
        percentiles[f"p{p}"] = float(np.percentile(lengths, p))
    with open('statistics/note_duration_token_percentiles.json', 'w') as f:
        json.dump(percentiles, f, indent=2)
    print("Percentiles saved to statistics/note_duration_token_percentiles.json")


if __name__ == '__main__':
    main()