import os
import sys

# Add project root to sys.path
dirof = os.path.dirname
try:
    dir_of_file = dirof(__file__)
except NameError:
    # .ipynb 文件中没有 __file__，使用当前工作目录
    dir_of_file = os.getcwd()
project_root = dirof(dir_of_file)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn.functional as F
from datasets.dataset_utils import calculate_bar_id_of_phrases
import random


def augment_phrase_permute(latent, max_n_bar, sec_annot):
    '''
    Duplicate to right below max_len, then permute phrases
    '''
    bar_id_of_sections = calculate_bar_id_of_phrases(sec_annot)
    
    # Copy sections
    new_sections = bar_id_of_sections.copy()

    # Duplication
    current_n_bar = sum([x[2]-x[1] for x in bar_id_of_sections])
    while current_n_bar < max_n_bar:
        # Randomly select a phrase to duplicate (not the first or last)
        phrase_to_dup_idx = random.randint(0, len(bar_id_of_sections)-1)
        phrase_to_dup = bar_id_of_sections[phrase_to_dup_idx]
        
        len_after_dup = current_n_bar + (phrase_to_dup[2]-phrase_to_dup[1])
        if len_after_dup > max_n_bar:
            break

        new_sections.append(phrase_to_dup)
        current_n_bar += (phrase_to_dup[2]-phrase_to_dup[1])

    random.shuffle(new_sections)

    # Permute the latents according to new_sections
    new_latent = []
    for i in range(len(new_sections)):
        start_bar = new_sections[i][1]
        end_bar = new_sections[i][2]
        start_idx = start_bar * 4
        end_idx = end_bar * 4
        # print(f'start_bar: {start_bar}, end_bar: {end_bar}, start_idx: {start_idx}, end_idx: {end_idx}')
        new_latent.append(latent[start_idx:end_idx])
    new_latent = torch.cat(new_latent, dim=0)
    # print("Original latent shape:", latent.shape)
    # print("New latent shape:", new_latent.shape)

    return new_latent


def truncate_by_self_similarity(z, phrase_per_bar=4, pad_threshold=0.95, min_real_bars=4):
    """
    z: Tensor [T, dim]  (phrase-level latent for one song, with padding at end)
    phrase_per_bar: usually 4
    pad_threshold: similarity threshold to detect padding bars
    min_real_bars: do not truncate below this (safety)

    Returns:
        z_truncated: truncated latent [real_T, dim]
        real_bars:   number of detected real bars
    """

    T, D = z.shape
    assert T % phrase_per_bar == 0, f"seq_len {T} not divisible by {phrase_per_bar}"

    # ------------------------
    # 1. reshape to bars
    # ------------------------
    num_bars = T // phrase_per_bar
    z_bar = z.reshape(num_bars, phrase_per_bar, D).mean(dim=1)   # [B, D]

    # ------------------------
    # 2. compute bar-level self similarity (cosine)
    # ------------------------
    z_norm = F.normalize(z_bar, dim=-1)
    ssm = z_norm @ z_norm.T      # [B, B]

    # ------------------------
    # 3. detect padding bars
    # idea:
    #   - padding bars are nearly identical
    #   - their last row has similarity ~1 with many trailing bars
    # ------------------------

    # look at similarity of last bar with all bars
    last_row = ssm[-1]        # [B]

    # find the earliest bar index where similarity with last bar is high
    # (indicating padding region)
    pad_start = None
    for i in range(B:=num_bars):
        if last_row[i] > pad_threshold:
            pad_start = i
            break

    # fallback if no padding detected
    if pad_start is None:
        pad_start = num_bars

    # safety: avoid truncating too aggressively
    pad_start = max(pad_start, min_real_bars)

    # ------------------------
    # 4. truncate phrase-level latent
    # ------------------------
    real_bars = pad_start
    real_T = real_bars * phrase_per_bar
    z_truncated = z[:real_T].clone()

    return z_truncated, real_bars