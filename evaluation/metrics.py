import os
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from typing import Dict
from tqdm import tqdm
from jiwer import wer as jiwer_wer
from sonata_utils import read_json
from typing import Tuple
import numpy as np
from jiwer import wer as jiwer_wer
from tqdm import tqdm
from typing import List, Dict


class Metric:
    def __init__(self):
        mel_dict_fp = (
            "/data1/longshen/Datasets/Piano/POP909/remi_z/melody_remiz_dict.json"
        )
        self.mel_dict = read_json(mel_dict_fp)

        eval_latent_fp = "/data1/longshen/Datasets/Piano/POP909/latents/song_level_phr_seq_with_annot/eval_data.pt"
        eval_data = torch.load(eval_latent_fp, map_location="cpu")
        eval_latents = [
            eval_data[k]["latent"] for k in eval_data
        ]  # list of [seq_len, dim]
        self.eval_latents = eval_latents

    def compute(self, *args, **kwargs):
        raise NotImplementedError

    def memorization_rate_mt_batch(self, out_mt_list, return_list=False):
        return memorization_rate_batch_mt(
            out_mt_list, self.mel_dict, return_list=return_list
        )

    def memorization_rate_mt(self, out_mt):
        out_mt.filter_tracks(insts=[13])  # melody track
        out_str = out_mt.to_remiz_str()
        return memorization_rate(out_str, self.mel_dict)

    def memorization_rate_batch(self, out_str_list):
        return memorization_rate_batch(out_str_list, self.mel_dict)

    def memorization_rate(self, out_str):
        return memorization_rate(out_str, self.mel_dict)

    def new_sample_rate_batch(
        self, out_mt_list, threshold=0.333333
    ) -> Tuple[float, float]:
        top_two_ratios = self.top_two_ratio_batch(out_mt_list, return_list=True)
        new_sample_flags = [
            1.0 if ratio > threshold else 0.0 for ratio in top_two_ratios
        ]
        new_sample_rate = (
            sum(new_sample_flags) / len(new_sample_flags)
            if len(new_sample_flags) > 0
            else 0.0
        )

        avg_top_two_ratio = (
            sum(top_two_ratios) / len(top_two_ratios)
            if len(top_two_ratios) > 0
            else 0.0
        )

        return new_sample_rate, avg_top_two_ratio, top_two_ratios

    def top_two_ratio_batch(self, out_mt_list, return_list=False) -> float:
        """
        scores:
        find most similar two samples in the training set
        Get WER of top1 (lowest) and top2 (second lowest)
        Compute ratio of top1 / top2

        Returns:
            ratio of (top1 - top2) / top1
        """
        ratios = []
        for out_mt in tqdm(out_mt_list, desc="Computing top-two ratios"):
            ratio = self.top_two_ratio(out_mt)
            ratios.append(ratio)

        if return_list:
            return ratios

        mean_ratio = sum(ratios) / len(ratios) if len(ratios) > 0 else 0.0

        return mean_ratio

    def top_two_ratio(self, out_mt) -> float:
        """
        scores:
        find most similar two samples in the training set
        Get WER of top1 (lowest) and top2 (second lowest)
        Compute ratio of top1 / top2

        Returns:
            ratio of (top1 - top2) / top1
        """
        out_mt.filter_tracks(insts=[13])  # melody track
        out_str = out_mt.to_remiz_str()

        wers = []
        for song_id in self.mel_dict:
            ref_remiz = self.mel_dict[song_id]
            error_rate = jiwer_wer(ref_remiz, out_str)
            wers.append(error_rate)

        wers.sort()
        top1 = wers[0]
        top2 = wers[1] if len(wers) > 1 else None
        if top2 is None:
            return 0.0
        ratio = top1 / top2 if top2 > 0 else 0.0
        return ratio

    def compute_srs(self, latent_tensor, out_n_bars) -> float:
        """
        x: [n_samples, seq_len, dim]
        sample_lens: list[int], len = n_bars
        """

        n_samples, seq_len, dim = latent_tensor.shape

        # Bright ratio
        bright_ratio_list = []
        for i in range(n_samples):
            z = latent_tensor[i, : out_n_bars[i] * 4, :]  # [valid_len, dim]
            br = bar_recurrence_score_diag(z, threshold=0.5, min_run_length=4)
            # br = bright_ratio(z, 0.5)
            bright_ratio_list.append(br)
        # print(f"Bright ratio of sample {i}:", br)
        srs = sum(bright_ratio_list) / len(bright_ratio_list)

        return srs

    def compute_fid(self, out_tensor, out_n_bars) -> float:
        """
        gt_latent_list: list of tensors [seq_len_i, dim]
        out_tensor: [n_samples, seq_len, dim]
        sample_lens: list[int], len = n_samples
                     number of valid timesteps for each sample
        """
        fid = compute_fid_gt_vs_out(
            eval_set=self.eval_latents,
            out_set=out_tensor.cpu(),
            out_lens=[n_bars * 4 for n_bars in out_n_bars],
        )
        return fid

    def calculate_memorization_metrics_single(
        self, mt, threshold=1 / 3, mel_program_id=13
    ):
        return calculate_memorization_metrics_single(
            mt, self.mel_dict, threshold=threshold, mel_program_id=mel_program_id
        )

    def calculate_memorization_metrics_batch(
        self, mt_list, threshold=1 / 3, mel_program_id=13
    ):
        return calculate_memorization_metrics_batch(
            mt_list, self.mel_dict, threshold=threshold, mel_program_id=mel_program_id
        )


def calculate_memorization_metrics_single(
    mt, mel_dict: Dict[str, str], threshold=1 / 3, mel_program_id=13
):
    """
    mt: MultiTrack (generated song)
    mel_dict: {song_id -> melody_string} for the training set

    Returns dict:
    {
        "mem_rate": float,          # 1 - min(WER)
        "top2_ratio": float,        # top1 / top2
        "new_sample_flag": int      # 1 if top2_ratio > threshold
    }
    """

    # 1. extract melody from mt
    mt.filter_tracks(insts=[mel_program_id])  # melody track
    out_str = mt.to_remiz_str()

    # 2. compute WER with all reference melodies (ONE PASS)
    wers = []
    for song_id, ref_str in mel_dict.items():
        wers.append(jiwer_wer(ref_str, out_str))

    wers = np.array(wers)
    wers_sorted = np.sort(wers)

    # 3. memorization rate
    min_wer = wers_sorted[0]
    mem_rate = 1.0 - min_wer

    # 4. top-2 ratio
    if len(wers_sorted) >= 2:
        top1 = wers_sorted[0]
        top2 = wers_sorted[1]
        top2_ratio = top1 / top2 if top2 > 0 else 0.0
    else:
        top2_ratio = 0.0

    # 5. new sample flag
    new_sample_flag = 1 if top2_ratio > threshold else 0

    return {
        "mem_rate": float(mem_rate),
        "top2_ratio": float(top2_ratio),
        "new_sample_flag": int(new_sample_flag),
    }


def calculate_memorization_metrics_batch(
    mt_list: List, mel_dict: Dict[str, str], threshold=1 / 3, mel_program_id=13
):
    """
    Batch version: compute sample-wise and mean metrics.

    Returns:
    {
        "mem_rates": [...],            # per-sample
        "top2_ratios": [...],
        "new_sample_flags": [...],
        "avg_mem_rate": float,
        "avg_top2_ratio": float,
        "avg_new_sample_rate": float
    }
    """

    mem_rates = []
    top2_ratios = []
    new_flags = []

    for mt in tqdm(mt_list, desc="Computing memorization metrics"):
        res = calculate_memorization_metrics_single(
            mt, mel_dict, threshold, mel_program_id
        )

        mem_rates.append(res["mem_rate"])
        top2_ratios.append(res["top2_ratio"])
        new_flags.append(res["new_sample_flag"])

    avg_mem_rate = float(np.mean(mem_rates)) if len(mem_rates) else 0.0
    avg_top2_ratio = float(np.mean(top2_ratios)) if len(top2_ratios) else 0.0
    avg_new_flag = float(np.mean(new_flags)) if len(new_flags) else 0.0

    return {
        "mem_rates": mem_rates,
        "top2_ratios": top2_ratios,
        "new_sample_flags": new_flags,
        "avg_mem_rate": avg_mem_rate,
        "avg_top2_ratio": avg_top2_ratio,
        "avg_new_sample_rate": avg_new_flag,
    }


def compute_fid_gt_vs_out(eval_set, out_set, out_lens):
    """
    eval_set: list of tensors, each [seq_len_i, dim]   (trimmed)
    out_set:  Tensor [N2, seq_len, dim]                (with padding)
    out_lens: list[int], valid lengths per generated sample

    Returns:
        fid (float)
    """

    # ------------------------
    # 1) flatten eval_set → [N_real, dim]
    # ------------------------
    # print('Eval set length:', len(eval_set))
    # print('Eval set first sample shape:', eval_set[0].shape)
    z_real = torch.cat(eval_set, dim=0).float()

    # ------------------------
    # 2) flatten out_set using out_lens → [N_gen, dim]
    # ------------------------
    z_list = []
    for i, L in enumerate(out_lens):
        if L > 0:
            z_list.append(out_set[i, :L])
    z_gen = torch.cat(z_list, dim=0).float()

    # ------------------------
    # 3) use generic compute_fid
    # ------------------------
    # print(z_real.shape, z_gen.shape)
    # exit(10)
    fid = compute_fid(z_real, z_gen)

    return fid


def compute_all_fids_gt_vs_out(eval_set, out_set, out_lens):
    """
    Compute:
        - phrase-FID (1-gram)
        - bigram-FID (2-gram)
        - 4-gram-FID
        - 8-gram-FID

    eval_set: list[tensor], each [L_i, dim], trimmed
    out_set:  tensor [N, T, dim], contains padding
    out_lens: list[int], valid timesteps for each sample

    Returns:
        dict with keys:
            {
             "fid_1gram": float,
             "fid_2gram": float,
             "fid_4gram": float,
             "fid_8gram": float
            }
    """

    # ------------------------
    # 1) flatten eval_set → [N_real, dim]
    # ------------------------
    z_real = torch.cat(eval_set, dim=0).float()

    # ------------------------
    # 2) flatten out_set using out_lens → [N_gen, dim]
    # ------------------------
    z_gen_list = []
    out_song_list = []  # for n-grams

    for i, L in enumerate(out_lens):
        if L > 0:
            seq = out_set[i, :L].float()
            z_gen_list.append(seq)
            out_song_list.append(seq)

    z_gen = torch.cat(z_gen_list, dim=0).float()

    # phrase-level FID
    fid_1 = compute_fid(z_real, z_gen)

    # ------------------------
    # 3) prepare for n-gram
    # ------------------------
    eval_song_list = eval_set  # already trimmed

    # helper: n-gram fid
    def _ngram_fid(n):
        return compute_ngram_fid(eval_song_list, out_song_list, n=n)

    # ------------------------
    # 4) compute multi-scale FIDs
    # ------------------------
    results = {
        "fid_1gram": fid_1,
        "fid_2gram": _ngram_fid(2),
        "fid_4gram": _ngram_fid(4),
        "fid_8gram": _ngram_fid(8),
        "fid_16gram": _ngram_fid(16),
        "fid_32gram": _ngram_fid(32),
    }

    return results


def memorization_rate_batch_mt(
    out_mt_list, mel_dict: Dict[str, str], return_list=False
) -> float:
    """
    out_mt_list: list of MultiTrack generated melodies
    mel_dict: melody of all songs in training set

    Returns:
        list of memorization rates (float), calculated by
        1 - min(WER(out_str, mel[i])) over all i in training set
    """
    out_str_list = []
    for mt in out_mt_list:
        mt.filter_tracks(insts=[13])  # melody track
        out_str = mt.to_remiz_str()
        out_str_list.append(out_str)

    mem_rates = memorization_rate_batch(out_str_list, mel_dict, return_list=return_list)

    return mem_rates


def memorization_rate_batch(
    out_str_list, mel_dict: Dict[str, str], return_list=False
) -> float:
    """
    out_str_list: list of generated melody strings
    mel_dict: melody of all songs in training set

    Returns:
        list of memorization rates (float), calculated by
        1 - min(WER(out_str, mel[i])) over all i in training set
    """
    mem_rates = []
    for out_str in tqdm(out_str_list, desc="Computing memorization rates"):
        mem_rate = memorization_rate(out_str, mel_dict)
        mem_rates.append(mem_rate)

    if return_list:
        return mem_rates

    mean_mem_rate = sum(mem_rates) / len(mem_rates) if len(mem_rates) > 0 else 0.0

    return mean_mem_rate


def memorization_rate(out_str: str, mel_dict: Dict[str, str]):
    """
    out_str: generated melody string
    mel_dict: melody of all songs in training set

    Returns:
        memorization rate (float), calculated by
        1 - min(WER(out_str, mel[i])) over all i in training set
    """
    wers = {}
    for song_id in mel_dict:
        ref_remiz = mel_dict[song_id]
        error_rate = jiwer_wer(ref_remiz, out_str)
        wers[song_id] = error_rate

    min_wer = min(wers.values())
    mem_rate = 1.0 - min_wer

    return mem_rate


def bar_recurrence_score_diag(z, threshold=0.6, min_run_length=4):
    """
    z: Tensor [T, D]  phrase-level latent sequence
       (T should be 4 * num_bars)
    threshold: similarity threshold
    max_offset: how far we check diagonals (None = B-1)

    Returns:
        bar recurrence score (float)
    """

    z = z.float()
    T = z.shape[0]

    assert T % 4 == 0, f"Total phrases ({T}) is not a multiple of 4."

    ssm = compute_bar_ssm_from_phrase_latents(z)  # [num_bars, num_bars]
    return bar_recurrence_run_length_from_ssm(
        ssm, threshold=threshold, min_run_length=min_run_length
    )


def bar_recurrence_run_length_from_ssm(ssm, threshold=0.6, min_run_length=2):
    """
    ssm: [B, B] bar-level SSM
    threshold: similarity threshold
    min_run_length: minimum consecutive bright run to be counted

    New rule:
      A run on diagonal k is INVALID if the diagonal (k+1) has ANY bright point
      in the same index range (i .. i+run_len-1).
    """

    ssm = ssm.cpu()
    B = ssm.shape[0]

    total_run_length = 0

    # ------- Precompute all diagonals as binarized -------
    diags = []
    for k in range(1, B):  # skip main diag
        diag = torch.diagonal(ssm, offset=k)  # length B-k
        diag_bin = (diag > threshold).float()  # convert to 0/1
        diags.append(diag_bin)
    # diags[k-1] is diag k

    # ------- Process each diagonal -------
    for k in range(1, B):  # diag offset k
        diag_bin = diags[k - 1]  # length = B-k
        L = diag_bin.shape[0]

        i = 0
        while i < L:
            if diag_bin[i] == 1:
                # start of a run
                start = i
                while i < L and diag_bin[i] == 1:
                    i += 1
                run_len = i - start

                if run_len >= min_run_length:
                    # check the diagonal above (k+1)
                    if k < B - 1:  # diagonal k+1 exists
                        diag_above = diags[k]  # diag k+1, length B-(k+1)
                        # overlapping range in diag_above is:
                        # indices [start, start+run_len)
                        end = start + run_len
                        end = min(end, diag_above.shape[0])  # avoid overflow

                        above_has_bright = (diag_above[start:end] > 0).any().item()

                        if not above_has_bright:
                            total_run_length += run_len
                    else:
                        # last diagonal, no diagonal above → always valid
                        total_run_length += run_len
            else:
                i += 1

    score = total_run_length / B
    return score


def bar_recurrence_score_diag_from_ssm(ssm, threshold=0.6, max_offset=None):
    """
    ssm: [B, B] bar-level self-similarity matrix
    threshold: similarity threshold
    max_offset: how far we check diagonals (None = B-1)
    """

    B = ssm.shape[0]
    if max_offset is None:
        max_offset = B - 1

    ssm = ssm.cpu()

    total = 0.0
    for k in range(1, max_offset + 1):  # skip main diagonal
        diag = torch.diagonal(ssm, offset=k)
        if len(diag) == 0:
            continue
        bright = (diag > threshold).float().mean().item()
        total += bright

    return total


def bright_ratio(z, threshold=0.8):
    """
    z: Tensor [T, D]  phrase-level latent sequence
    threshold: float, cosine similarity threshold

    Returns:
        bright_ratio (float)
    """

    z = z.float()
    T = z.shape[0]

    ssm = compute_bar_ssm_from_phrase_latents(z)  # [num_bars, num_bars]
    n_bar = ssm.shape[0]

    # -------- 3. upper triangle mask (exclude diagonal) --------
    mask = torch.triu(torch.ones_like(ssm), diagonal=1).bool()

    # -------- 4. count bright points --------
    bright_count = (ssm[mask] > threshold).sum().item()

    # -------- 5. divide by number of bars --------
    bright_ratio = bright_count / n_bar

    return bright_ratio


def compute_bar_ssm_from_phrase_latents(z):
    """
    z: Tensor [T, D], phrase-level latent for one song
       (T should be 4 * num_bars)

    Returns:
        smm: Tensor [num_bars, num_bars], bar-level self-similarity matrix
    """

    z = z.float()  # [T, D]
    T, D = z.shape

    assert T % 4 == 0, f"Total phrases ({T}) is not a multiple of 4."

    # ------------------------
    # 1) reshape to bars and average 4 phrases per bar
    #    new shape → [num_bars, 4, D] → mean over dim 1 → [num_bars, D]
    # ------------------------
    num_bars = T // 4
    z_bar = z.reshape(num_bars, 4, D).mean(dim=1)  # [num_bars, D]

    # ------------------------
    # 2) compute SSM (cosine similarity)
    # ------------------------
    z_norm = F.normalize(z_bar, dim=-1)  # [num_bars, D]
    smm = z_norm @ z_norm.T  # [num_bars, num_bars]

    return smm


def visualize_ssm_for_song_bar(
    z, save_dir=None, filename="smm_of_song_bar.png", show=False
):
    """
    z: Tensor [T, dim], phrase-level latent for one song
       (T should be 4 * num_bars)
    save_dir: output directory
    filename: output image name
    """

    # ------------------------
    # 1) ensure directory exists
    # ------------------------
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # ------------------------
    # 2) convert to float
    # ------------------------
    z = z.float()  # [T, D]
    T, D = z.shape

    assert T % 4 == 0, f"Total phrases ({T}) is not a multiple of 4."

    smm = compute_bar_ssm_from_phrase_latents(z)  # [num_bars, num_bars]

    # ------------------------
    # 5) visualize
    # ------------------------
    plt.figure(figsize=(6, 6))
    plt.imshow(smm.cpu().numpy(), cmap="viridis", origin="lower")
    plt.colorbar(label="cosine similarity")
    plt.title("Self-Similarity Matrix (bar-level)")
    plt.xlabel("Bar index")
    plt.ylabel("Bar index")

    # ------------------------
    # 6) save
    # ------------------------

    if not show:
        out_path = os.path.join(save_dir, filename)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        # print(f"[Bar-level SMM] saved to: {out_path}")
    else:
        plt.show()

    plt.close()


def visualize_ssm_for_song_phrase(z, save_dir, filename="smm_of_song.png"):
    """
    eval_set: list of tensors, each [seq_len, dim]
    save_dir: directory to save the output plot
    filename: name of the output image file
    """

    # ------------------------
    # 1) ensure directory exists
    # ------------------------
    os.makedirs(save_dir, exist_ok=True)

    # ------------------------
    # 2) take first song latent
    # ------------------------
    z = z.float()  # [T, D]
    T = z.shape[0]

    # ------------------------
    # 3) compute SSM (cosine similarity)
    # ------------------------
    # normalize each vector
    z_norm = F.normalize(z, dim=-1)  # [T, D]
    # cosine similarity = z z^T
    smm = z_norm @ z_norm.T  # [T, T], values in [-1, 1]

    # ------------------------
    # 4) visualize
    # ------------------------
    plt.figure(figsize=(6, 6))
    plt.imshow(smm.cpu().numpy(), cmap="viridis", origin="lower")
    plt.colorbar(label="cosine similarity")
    plt.title("Self-Similarity Matrix (first song)")
    plt.xlabel("Time index")
    plt.ylabel("Time index")

    # ------------------------
    # 5) save
    # ------------------------
    out_path = os.path.join(save_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[SMM] saved to: {out_path}")


def compute_diversity(x, sample_lens, chunk=1024):
    """
    x: [n_samples, seq_len, dim]
    sample_lens: list[int], len = n_samples
                 number of valid timesteps for each sample
    Returns: scalar diversity score
    """
    n_samples, seq_len, dim = x.shape

    # ---- 1) collect only valid latent vectors ----
    z_list = []
    for i in range(n_samples):
        L = sample_lens[i]
        if L > 0:
            z_list.append(x[i, :L])  # shape [L, dim]

    # concat → [N, dim]
    z = torch.cat(z_list, dim=0)
    z = z.reshape(-1, dim).float()  # ensure float
    N = z.size(0)

    if N <= 1:
        return 0.0

    # ---- 2) compute mean pairwise L2 distance ----
    total = 0.0
    count = 0

    for i in range(0, N, chunk):
        zi = z[i : i + chunk]  # [c, dim]
        dist = torch.cdist(zi, z)  # [c, N]

        for ii in range(zi.size(0)):
            row_idx = i + ii
            start_j = row_idx + 1
            if start_j < N:
                d = dist[ii, start_j:]
                total += d.sum().item()
                count += d.numel()

    return total / count if count > 0 else 0.0


def compute_ngram_fid(song_list_A, song_list_B, n=2):
    z_A = compute_ngram_vectors(song_list_A, n=n)
    z_B = compute_ngram_vectors(song_list_B, n=n)
    return compute_fid(z_A, z_B)


def compute_ngram_vectors(song_list, n=2):
    """
    song_list: list of tensors [T_i, dim]
    n: n-gram length (e.g., 2→bigram, 3→trigram, 4→4-gram)

    return:
        [M, n*dim] tensor of concatenated n-gram latents
    """
    vecs = []

    for song in song_list:
        T = song.shape[0]
        if T < n:
            continue

        # e.g., for n=3: [(z0,z1,z2), (z1,z2,z3), ...]
        grams = []
        for offset in range(n):
            grams.append(song[offset : T - n + offset + 1])  # [T-n+1, dim]

        ngram = torch.cat(grams, dim=-1)  # [T-n+1, n*dim]
        vecs.append(ngram)

    return torch.cat(vecs, dim=0).float()


def compute_bigram_vectors(song_list):
    """
    song_list: list of tensors [seq_len_i, dim]
    return:
        bigram_latents: [M, 2*dim] (concatenated bigrams)
    """
    bigrams = []

    for song in song_list:
        T = song.shape[0]
        if T < 2:
            continue

        z1 = song[:-1]  # [T-1, dim]
        z2 = song[1:]  # [T-1, dim]
        bg = torch.cat([z1, z2], dim=-1)  # [T-1, 2*dim]
        bigrams.append(bg)

    return torch.cat(bigrams, dim=0).float()


def compute_bigram_fid(song_list_A, song_list_B):
    """
    song_list_A: list of tensors [seq_len_i, dim]
    song_list_B: same format

    returns: bigram FID
    """
    z_A = compute_bigram_vectors(song_list_A)
    z_B = compute_bigram_vectors(song_list_B)
    return compute_fid(z_A, z_B)


def compute_fid(z_A, z_B):
    """
    z_A: [N1, D] tensor
    z_B: [N2, D] tensor
    Returns: float (FID)
    """
    # means
    mu_A = z_A.mean(dim=0)
    mu_B = z_B.mean(dim=0)

    # centered
    xc_A = z_A - mu_A
    xc_B = z_B - mu_B

    # covariances
    cov_A = (xc_A.T @ xc_A) / (z_A.shape[0] - 1)
    cov_B = (xc_B.T @ xc_B) / (z_B.shape[0] - 1)

    # sqrt term
    sqrt_term = matrix_sqrt(cov_A @ cov_B)

    # FID formula
    fid = torch.sum((mu_A - mu_B) ** 2).item()
    fid += torch.trace(cov_A + cov_B - 2 * sqrt_term).item()
    return fid


# def matrix_sqrt(mat):
#     """
#     Stable symmetric matrix square root using eigen decomposition (PSD).
#     """
#     vals, vecs = torch.linalg.eigh(mat)
#     vals = torch.clamp(vals, min=1e-12)
#     sqrt_vals = torch.sqrt(vals)
#     return (vecs * sqrt_vals.unsqueeze(0)) @ vecs.T

def matrix_sqrt(mat):
    """
    Symmetric matrix square root that works under bf16, fp16, fp32.
    Automatically upcasts to float32 for eigh.
    """
    orig_dtype = mat.dtype
    mat32 = mat.float()  # upcast to fp32

    # eigen-decomposition on fp32
    vals, vecs = torch.linalg.eigh(mat32)

    vals = torch.clamp(vals, min=1e-12)
    sqrt_vals = torch.sqrt(vals)

    out32 = (vecs * sqrt_vals.unsqueeze(0)) @ vecs.T

    return out32.to(orig_dtype)
