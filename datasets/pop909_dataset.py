import os
import sys

dirof = os.path.dirname
sys.path.insert(0, dirof(dirof(__file__)))

import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from remi_z import MultiTrack
from torch.utils.data import Dataset
from sonata_utils import jpath, read_jsonl, create_dir_if_not_exist
from torch.utils.data import DataLoader
from models.pianotree_vae_utils import pianoroll_to_pianotree_input
from transformers import PreTrainedTokenizerFast
from datasets.dataset_utils import calculate_bar_id_of_phrases


def main():
    dataset_dir = "/data1/longshen/Datasets/Piano/POP909/jsonl"
    jsonl_fp = jpath(dataset_dir, "pop909_piano_track_bars.jsonl")
    dataset = POP909PianoTreeDataset(jsonl_fp)
    print(len(dataset))
    print(dataset[1])


class RepeatLoader:
    '''
    This dataset class loop the real dataset infinitely
    And create pseudo-epoch with fixed number of steps
    '''
    def __init__(self, loader, steps_per_epoch):
        self.loader = loader
        self.steps_per_epoch = steps_per_epoch

    def __iter__(self):
        it = iter(self.loader)
        for _ in range(self.steps_per_epoch):
            try:
                yield next(it)
            except StopIteration:
                it = iter(self.loader)
                yield next(it)

    def __len__(self):
        return self.steps_per_epoch


class POP909Dataset(Dataset):
    """
    This is a bar-level dataset for POP909
    """

    def __init__(
        self,
        jsonl_fp,
        tokenizer_path,
        max_length,
        span_mask=False,
        n_query_tokens=0,
        split=None,
    ):
        self.jsonl_fp = jsonl_fp
        self.data = read_jsonl(jsonl_fp)

        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        self.max_length = max_length

        self.span_mask = span_mask

        assert 0 <= n_query_tokens <= 9
        self.n_query_tokens = n_query_tokens

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ret = self.data[idx]
        tgt_seq = ret.split(" ")

        # NOTE: s-XX t-YY (and i-ZZ, if needed) are pre-removed in preprocessing

        assert len(tgt_seq) > 0, f"Empty token sequence after removing s-XX t-YY: {ret}"

        """ Span masking """
        # copy original tok_seq
        inp_seq = tgt_seq.copy()

        if self.span_mask:
            # Target mask token: 30%
            tgt_mask_ratio = 0.3
            n_toks = len(inp_seq)
            n_tgt_mask_toks = n_toks * tgt_mask_ratio

            # Sample span length
            masked_n_tokens = 0
            # span_lens = []
            while masked_n_tokens < n_tgt_mask_toks:
                # Sample from lampda=3 poisson distribution
                span_len = np.random.poisson(3)
                masked_n_tokens += span_len

                # Randomly select start position of the span from [0, n_toks - 1]
                start_pos = np.random.randint(0, n_toks)
                end_pos = min(start_pos + span_len, n_toks)
                # Replace the span with [MASK]
                inp_seq = inp_seq[:start_pos] + ["[MASK]"] + inp_seq[end_pos:]

                # Update n_toks
                n_toks = len(inp_seq)

        # Add special tokens
        # Input sequence: [BOS] ... [EOS]
        inp_seq = ["[BOS]"] + inp_seq + ["[EOS]"]
        # Target sequence: ... [EOS]
        tgt_seq = tgt_seq + ["[EOS]"]

        # Add query tokens to the front
        if self.n_query_tokens > 0:
            inp_seq = [f"[C{i}]" for i in range(self.n_query_tokens)] + inp_seq

        inp_str = " ".join(inp_seq)
        tgt_str = " ".join(tgt_seq)

        return (inp_str, tgt_str)

    def collate_fn(self, batch):
        # batch: list of remiz_str (raw sequence)
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be provided to collate_fn.")
        input_strs = [item[0] for item in batch]
        tgt_strs = [item[1] for item in batch]

        # Tokenize input sequences
        tokenized = self.tokenizer(
            input_strs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        # Tokenize target sequences
        tokenized_tgt = self.tokenizer(
            tgt_strs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        labels = tokenized_tgt["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100  # ignore padding

        ret = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "remiz_seq": tgt_strs,
            "masked_seq": input_strs,
        }
        return ret


class POP909FullSongBarLatentDataset(Dataset):
    """
    This dataset class loads precomputed bar latents for each song in POP909.
    True song-level, no bar-level splitting.
    """

    def __init__(
        self,
        latent_fp,
        scaling_factor=1.0,
        debug_temp_dir=None,
        augmentation_type="none",
        pad_type="max_len",
        max_len=512,
        split=None,
        augment_in_val=False,
        n_bars=None,
        n_songs=None
    ):
        '''
        n_bars: only use first n_bars bars of each song
        n_songs: only use first n_songs songs in the dataset
        '''
        assert split in ["train", "valid"]
        self.split = split
        self.augment_in_val = augment_in_val

        assert augmentation_type in [
            "none",
            "phrase_permute",  # A(n, k) with duplicate
            "phrase_subset",  # C(n-2, k), keep first and last phrase unchanged
            "bar_permute", # permute bars within the song (duplicate to max length)
        ]
        self.augmentation_type = augmentation_type

        assert pad_type in [
            "max_len",  # pad to max length in the dataset
            "max_batch",  # pad to max length in the batch
        ]
        self.pad_type = pad_type
        self.max_len = max_len # maximum number of phrases (for pop909, 4 * max_bars)

        # Load the file
        self.latent_fp = latent_fp
        self.data = torch.load(latent_fp, map_location="cpu")  # [N, max_len=17, 128]
        self.scaling_factor = scaling_factor

        print("Constant scaling factor:", scaling_factor)

        

        # Re-index data, integer as key; move current key as 'song_id' value
        data_new = {}
        for i, (key, value) in enumerate(self.data.items()):
            data_new[i] = value
            data_new[i]["song_id"] = key
        self.data = data_new

        if n_songs is not None:
            # take first n_songs
            data_list = list(self.data.items())[:n_songs]

            # expand to ≥512 entries
            expanded_values = []
            repeat = (512 // len(data_list)) + 1

            for _ in range(repeat):
                for key, val in data_list:
                    expanded_values.append(val)

            # trim to 512
            expanded_values = expanded_values[:512]

            # re-assign integer keys
            self.data = {i: v for i, v in enumerate(expanded_values)}

        if n_bars is not None:
            # only use first n_bars bars of each song
            for key in self.data.keys():
                latent = self.data[key]['latent']
                length = self.data[key]['length']
                sec_annot = self.data[key]['phrase_annotation']

                max_bars = n_bars
                max_len = max_bars

                if length > max_len:
                    # trim
                    self.data[key]['latent'] = latent[:max_len, :]
                    self.data[key]['length'] = max_len

                    # also trim phrase_annotation
                    bar_id_of_sections = calculate_bar_id_of_phrases(sec_annot)
                    new_sec_annot = []
                    for (sec_type, start_bar, end_bar) in bar_id_of_sections:
                        if start_bar >= max_bars:
                            break
                        if end_bar > max_bars:
                            end_bar = max_bars
                        new_sec_annot.append((sec_type, start_bar, end_bar))
                    self.data[key]['phrase_annotation'] = new_sec_annot

        # print stats
        print("Dataset size:", len(self.data))
        print('First latent shape:', self.data[0]['latent'].shape)
        


        """ --- DEBUG ONLY --- """

        # Save the first sample for debugging
        self.debug_temp_dir = debug_temp_dir
        if self.debug_temp_dir is not None and self.split == 'train':
            latent_save_fn = "debug_first_sample_latent_scaled.pt"
            latent_save_fp = jpath(self.debug_temp_dir, latent_save_fn)
            create_dir_if_not_exist(self.debug_temp_dir)
            gt_latent = self.data[0]['latent']
            print('GT latent shape when saving:')
            print(gt_latent.shape)
            torch.save(gt_latent, latent_save_fp)

            # Save the first bar's latent diagram
            # Save the defussed latent
            save_dir = self.debug_temp_dir
            save_fp = jpath(save_dir, f"gt_latent.png")

            plt.figure(figsize=(10, 4))
            plt.scatter(range(gt_latent.shape[1]), gt_latent[0, :].cpu().numpy(), s=10)
            plt.title("First bar latent of first sample")
            plt.xlabel("Latent Dimension")
            plt.ylabel("Latent Value")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.savefig(save_fp)

        # # Decode the first and last sample
        # ae = PhraseVAE().cuda()
        # first_sample = self.data[0:1, :, :].cuda() * scaling_factor  # [1, max_len, 128]
        # recon = ae.decode_batch(first_sample)
        # save_dir = '/home/longshen/work/AccGen/AccGen/temp/song_level_phr_seq_latents'
        # save_fn = "debug_first_sample_decoded_phrases.txt"
        # save_fp = jpath(save_dir, save_fn)
        # with open(save_fp, 'w') as f:
        #     f.write(recon[0])
        # last_sample = self.data[-1:, :, :].cuda() * scaling_factor  # [1, max_len, 128]
        # recon2 = ae.decode_batch(last_sample)
        # save_fn2 = "debug_last_sample_decoded_phrases.txt"
        # save_fp2 = jpath(save_dir, save_fn2)
        # with open(save_fp2, 'w') as f:
        #     f.write(recon2[0])
        # exit(10)

        """ --- END DEBUG ONLY --- """

    def augment_phrase_subset(self, latent, annot):
        bar_id_of_sections = calculate_bar_id_of_phrases(annot)
        body = bar_id_of_sections[1:-1]
        n = len(body)
        if n == 0:
            return latent

        # Random select k from [1, n]
        try:
            k = random.randint(1, n)
        except ValueError:
            print("Error in random selection of k:")
            print("n:", n)
            print("bar_id_of_phrase:", bar_id_of_sections)
            print("annot:", annot)
            raise ValueError("Error in random selection of k.")

        ids = list(range(n))
        selected_ids = random.sample(ids, k)
        selected_ids.sort()
        selected = [body[i] for i in selected_ids]
        selected = [bar_id_of_sections[0]] + selected + [bar_id_of_sections[-1]]

        # Permute the latents according to selected phrase bar ids
        new_latent = []
        for i in range(len(selected)):
            start_bar = selected[i][1]
            end_bar = selected[i][2]
            start_idx = start_bar * 4
            end_idx = end_bar * 4
            # print(f'start_bar: {start_bar}, end_bar: {end_bar}, start_idx: {start_idx}, end_idx: {end_idx}')
            new_latent.append(latent[start_idx:end_idx])
        new_latent = torch.cat(new_latent, dim=0)
        # print("Original latent shape:", latent.shape)
        # print("New latent shape:", new_latent.shape)

        return new_latent
    
    def augment_bar_permute(self, latent, max_len):
        cur_len = latent.shape[0] # [n_phrase, dim]

        # Duplicate but not pad to max_len
        if cur_len < max_len:
            repeat_times = (max_len // cur_len) + 1
            latent = latent.repeat(repeat_times, 1)[:max_len, :]
        
        # Permute the bars
        n_bars = latent.shape[0] // 4
        bar_indices = list(range(n_bars))
        random.shuffle(bar_indices)
        new_latent = []
        for bar_idx in bar_indices:
            start_idx = bar_idx * 4
            end_idx = start_idx + 4
            new_latent.append(latent[start_idx:end_idx])
        new_latent = torch.cat(new_latent, dim=0)
        return new_latent
    
    def augment_phrase_permute(self, latent, max_n_bar, sec_annot):
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        latent = self.data[idx]["latent"]  # [n_phrase, dim]
        length = self.data[idx]["length"]  # int
        sec_annot = self.data[idx]["phrase_annotation"]  # list of (type, n_bar)

        """ Data Augmentation """
        if self.split == 'train' or (self.split == 'valid' and self.augment_in_val):
            if self.augmentation_type == "none":
                pass
            elif self.augmentation_type == "phrase_permute":  
                # A(n, k) with duplicate
                latent = self.augment_phrase_permute(latent, self.max_len // 4, sec_annot)
            elif self.augmentation_type == "phrase_subset":  
                # C(n-2, k), keep first and last phrase unchanged
                latent = self.augment_phrase_subset(latent, sec_annot)
                length = latent.shape[0]
            elif self.augmentation_type == 'bar_permute':
                latent = self.augment_bar_permute(latent, self.max_len)
                length = latent.shape[0]

        # Scale the latent
        latent = latent / self.scaling_factor

        return latent, length

    def collate_fn(self, batch):
        # batch: list of (data, length)
        data_list = [item[0] for item in batch]
        length_list = [item[1] for item in batch]

        if self.pad_type == "max_batch":
            raise NotImplementedError("max_batch padding not implemented yet.")
        elif self.pad_type == "max_len":
            # Pad to max_len
            for i in range(len(data_list)):
                cur_len = data_list[i].shape[0]
                if cur_len < self.max_len:
                    pad_len = self.max_len - cur_len
                    pad_tensor = torch.zeros(pad_len, data_list[i].shape[1])
                    data_list[i] = torch.cat([data_list[i], pad_tensor], dim=0)

        ret_data = torch.stack(data_list, dim=0)
        ret_lengths = torch.tensor(length_list)

        return ret_data, ret_lengths


class POP909FullSongPhrLatentDataset(Dataset):
    """
    This dataset class loads precomputed phrase latents for each song in POP909.
    True song-level, no bar-level splitting.

    Might apply phrase augmentation, e.g., phrase permutation.
    """

    def __init__(
        self,
        latent_fp,
        scaling_factor=1.0,
        debug_temp_dir=None,
        augmentation_type="none",
        pad_type="max_len",
        max_len=512,
        split=None,
        augment_in_val=False,
        n_songs=None,
        n_bars=None,
    ):
        assert split in ["train", "valid"]
        self.split = split
        self.augment_in_val = augment_in_val

        assert augmentation_type in [
            "none",
            "phrase_permute",  # A(n, k) with duplicate
            "phrase_subset",  # C(n-2, k), keep first and last phrase unchanged
            "bar_permute", # permute bars within the song (duplicate to max length)
        ]
        self.augmentation_type = augmentation_type

        assert pad_type in [
            "max_len",  # pad to max length in the dataset
            "max_batch",  # pad to max length in the batch
        ]
        self.pad_type = pad_type
        self.max_len = max_len # maximum number of phrases (for pop909, 4 * max_bars)

        # Load the file
        self.latent_fp = latent_fp
        self.data = torch.load(latent_fp, map_location="cpu")  # [N, max_len=17, 128]
        self.scaling_factor = scaling_factor

        print("Constant scaling factor:", scaling_factor)

        # Print some stats
        print("Dataset size:", len(self.data))

        # Re-index data, integer as key; move current key as 'song_id' value
        data_new = {}
        for i, (key, value) in enumerate(self.data.items()):
            data_new[i] = value
            data_new[i]["song_id"] = key
        self.data = data_new

        if n_songs is not None:
            # take first n_songs
            data_list = list(self.data.items())[:n_songs]

            # expand to ≥512 entries
            expanded_values = []
            repeat = (512 // len(data_list)) + 1

            for _ in range(repeat):
                for key, val in data_list:
                    expanded_values.append(val)

            # trim to 512
            expanded_values = expanded_values[:512]

            # re-assign integer keys
            self.data = {i: v for i, v in enumerate(expanded_values)}

        if n_bars is not None:
            # only use first n_bars bars of each song
            for key in self.data.keys():
                latent = self.data[key]['latent']
                length = self.data[key]['length']
                sec_annot = self.data[key]['phrase_annotation']

                max_bars = n_bars
                max_len = max_bars

                if length > max_len:
                    # trim
                    self.data[key]['latent'] = latent[:max_len, :]
                    self.data[key]['length'] = max_len

                    # also trim phrase_annotation
                    bar_id_of_sections = calculate_bar_id_of_phrases(sec_annot)
                    new_sec_annot = []
                    for (sec_type, start_bar, end_bar) in bar_id_of_sections:
                        if start_bar >= max_bars:
                            break
                        if end_bar > max_bars:
                            end_bar = max_bars
                        new_sec_annot.append((sec_type, start_bar, end_bar))
                    self.data[key]['phrase_annotation'] = new_sec_annot

        """ --- DEBUG ONLY --- """

        # # Save the first sample for debugging
        # self.debug_temp_dir = debug_temp_dir
        # if self.debug_temp_dir is not None:
        #     latent_save_fn = "debug_first_sample_latent_scaled.pt"
        #     latent_save_fp = jpath(self.debug_temp_dir, latent_save_fn)
        #     create_dir_if_not_exist(self.debug_temp_dir)
        #     torch.save(self.data[0:1, :, :], latent_save_fp)

        self.debug_temp_dir = debug_temp_dir
        if self.debug_temp_dir is not None and self.split == 'train':
            latent_save_fn = "debug_first_sample_latent_scaled.pt"
            latent_save_fp = jpath(self.debug_temp_dir, latent_save_fn)
            create_dir_if_not_exist(self.debug_temp_dir)
            gt_latent = self.data[0]['latent']
            print('GT latent shape when saving:')
            print(gt_latent.shape)
            torch.save(gt_latent, latent_save_fp)

            # Save the first bar's latent diagram
            # Save the defussed latent
            save_dir = self.debug_temp_dir
            save_fp = jpath(save_dir, f"gt_latent.png")

            plt.figure(figsize=(10, 4))
            plt.scatter(range(gt_latent.shape[1]), gt_latent[0, :].cpu().numpy(), s=10)
            plt.title("First bar latent of first sample")
            plt.xlabel("Latent Dimension")
            plt.ylabel("Latent Value")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.savefig(save_fp)

        # # Decode the first and last sample
        # ae = PhraseVAE().cuda()
        # first_sample = self.data[0:1, :, :].cuda() * scaling_factor  # [1, max_len, 128]
        # recon = ae.decode_batch(first_sample)
        # save_dir = '/home/longshen/work/AccGen/AccGen/temp/song_level_phr_seq_latents'
        # save_fn = "debug_first_sample_decoded_phrases.txt"
        # save_fp = jpath(save_dir, save_fn)
        # with open(save_fp, 'w') as f:
        #     f.write(recon[0])
        # last_sample = self.data[-1:, :, :].cuda() * scaling_factor  # [1, max_len, 128]
        # recon2 = ae.decode_batch(last_sample)
        # save_fn2 = "debug_last_sample_decoded_phrases.txt"
        # save_fp2 = jpath(save_dir, save_fn2)
        # with open(save_fp2, 'w') as f:
        #     f.write(recon2[0])
        # exit(10)

        """ --- END DEBUG ONLY --- """

        # Load special latent
        special_latent_fp = '/home/longshen/work/AccGen/AccGen/models/phr_vae_special_latents.pt'
        self.special_latents = torch.load(special_latent_fp, map_location='cpu')




    def augment_phrase_subset(self, latent, annot):
        bar_id_of_sections = calculate_bar_id_of_phrases(annot)
        body = bar_id_of_sections[1:-1]
        n = len(body)
        if n == 0:
            return latent

        # Random select k from [1, n]
        try:
            k = random.randint(1, n)
        except ValueError:
            print("Error in random selection of k:")
            print("n:", n)
            print("bar_id_of_phrase:", bar_id_of_sections)
            print("annot:", annot)
            raise ValueError("Error in random selection of k.")

        ids = list(range(n))
        selected_ids = random.sample(ids, k)
        selected_ids.sort()
        selected = [body[i] for i in selected_ids]
        selected = [bar_id_of_sections[0]] + selected + [bar_id_of_sections[-1]]

        # Permute the latents according to selected phrase bar ids
        new_latent = []
        for i in range(len(selected)):
            start_bar = selected[i][1]
            end_bar = selected[i][2]
            start_idx = start_bar * 4
            end_idx = end_bar * 4
            # print(f'start_bar: {start_bar}, end_bar: {end_bar}, start_idx: {start_idx}, end_idx: {end_idx}')
            new_latent.append(latent[start_idx:end_idx])
        new_latent = torch.cat(new_latent, dim=0)
        # print("Original latent shape:", latent.shape)
        # print("New latent shape:", new_latent.shape)

        return new_latent
    
    def augment_bar_permute(self, latent, max_len):
        cur_len = latent.shape[0] # [n_phrase, dim]

        # Duplicate but not pad to max_len
        if cur_len < max_len:
            repeat_times = (max_len // cur_len) + 1
            latent = latent.repeat(repeat_times, 1)[:max_len, :]
        
        # Permute the bars
        n_bars = latent.shape[0] // 4
        bar_indices = list(range(n_bars))
        random.shuffle(bar_indices)
        new_latent = []
        for bar_idx in bar_indices:
            start_idx = bar_idx * 4
            end_idx = start_idx + 4
            new_latent.append(latent[start_idx:end_idx])
        new_latent = torch.cat(new_latent, dim=0)
        return new_latent
    
    def augment_phrase_permute(self, latent, max_n_bar, sec_annot, inst_conf=None):
        '''
        Duplicate to right below max_len, then permute phrases
        
        latent: [n_phrase, dim]
        inst_conf: instrument configuration, also need to be permuted accordingly. Looks like [[inst1, inst2, b-1], [inst1, inst3, b-1], ...]
            Indicating instrument and voice relationship within each bar
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
        new_inst_conf = []
        new_latent = []
        for i in range(len(new_sections)):
            start_bar = new_sections[i][1]
            end_bar = new_sections[i][2]
            start_idx = start_bar * 4
            end_idx = end_bar * 4
            # print(f'start_bar: {start_bar}, end_bar: {end_bar}, start_idx: {start_idx}, end_idx: {end_idx}')
            new_latent.append(latent[start_idx:end_idx])
            
            # Also permute instrument configuration accordingly
            ori_inst_conf = inst_conf[start_bar:end_bar]
            new_inst_conf.extend(ori_inst_conf)

        new_latent = torch.cat(new_latent, dim=0)

        res = {
            'new_latent': new_latent,
            'new_inst_conf': new_inst_conf
        }

        return res

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        latents = self.data[idx]["latent"]  # [n_phrase, dim]
        n_phrase = self.data[idx]["length"]  # int
        sec_annot = self.data[idx]["phrase_annotation"]  # list of (type, n_bar)
        inst_conf = self.data[idx]["inst_conf"]  # instrument configuration

        """ Data Augmentation """
        if self.split == 'train' or (self.split == 'valid' and self.augment_in_val):
            if self.augmentation_type == "none":
                pass
            elif self.augmentation_type == "phrase_permute":  
                # A(n, k) with duplicate
                res = self.augment_phrase_permute(latents, self.max_len // 4, sec_annot, inst_conf=inst_conf)
                latents = res['new_latent']
                inst_conf = res['new_inst_conf']
                n_phrase = latents.shape[0]
            elif self.augmentation_type == "phrase_subset":  
                # C(n-2, k), keep first and last phrase unchanged
                latents = self.augment_phrase_subset(latents, sec_annot)
                n_phrase = latents.shape[0]
            elif self.augmentation_type == 'bar_permute':
                latents = self.augment_bar_permute(latents, self.max_len)
                n_phrase = latents.shape[0]

        # Flatten the instrument configuration to a single sequence
        inst_conf_flat = []
        for inst in inst_conf:
            inst_conf_flat.extend(inst)

        # Flatten the section annotation to a single sequence
        # Original: [('i', 8), ('A', 8), ('A', 8), ('B', 8) ... ]
        # Target: 'i-8 A-8 A-8 B-8 ...'
        sec_annot_flat = []
        for (sec_type, sec_n_bar) in sec_annot:
            sec_annot_flat.append(f"{sec_type}-{sec_n_bar}")
        sec_annot = " ".join(sec_annot_flat)

        # Scale the latent
        latents = latents / self.scaling_factor

        return latents, n_phrase, inst_conf_flat, sec_annot

    def collate_fn(self, batch):
        # batch: list of (data, length)
        data_list = [item[0] for item in batch]
        length_list = [item[1] for item in batch]
        inst_conf_list = [item[2] for item in batch]
        sec_annot_list = [item[3] for item in batch]

        if self.pad_type == "max_batch":
            raise NotImplementedError("max_batch padding not implemented yet.")
        elif self.pad_type == "max_len":
            # Pad to max_len
            for i in range(len(data_list)):
                cur_len = data_list[i].shape[0]
                if cur_len < self.max_len:
                    pad_len = self.max_len - cur_len
                    # pad_tensor = torch.zeros(pad_len, data_list[i].shape[1])
                    pad_tensor = self.special_latents['eos_pad_latent'].repeat(pad_len, 1)
                    data_list[i] = torch.cat([data_list[i], pad_tensor], dim=0)

        ret_data = torch.stack(data_list, dim=0)
        ret_lengths = torch.tensor(length_list)

        return ret_data, ret_lengths, inst_conf_list, sec_annot_list


class POP909PhrLatentFixedLenDataset(Dataset):
    """
    This dataset class loads precomputed phrase latents for each song in POP909.
    Will use "first n_bars" and "last n_bars" of each song as data.
    """

    def __init__(
        self,
        latent_fp,
        scaling_factor=1.0,
        n_songs=-1,
        n_bars=1,
        debug_temp_dir=None,
        batch_size=1024,
        split=None,
    ):

        self.latent_fp = latent_fp
        self.data = torch.load(latent_fp, map_location="cpu")  # [N, max_len=17, 128]
        len_desc_fp = latent_fp.replace("latents.pt", "lengths.pt")
        self.lengths = torch.load(
            len_desc_fp, map_location="cpu"
        )  # [N], actual lengths of each sample

        # Use subset of data if n_sample > 0
        if n_songs > batch_size:
            self.data = self.data[:n_songs, :, :]
            self.lengths = self.lengths[:n_songs]
        elif n_songs > 0:
            self.data = self.data[:n_songs, :, :]
            self.lengths = self.lengths[:n_songs]

            # duplicate to batch_size samples
            self.data = self.data.repeat(batch_size // n_songs + 1, 1, 1)[
                :batch_size, :, :
            ]
            self.lengths = self.lengths.repeat(batch_size // n_songs + 1)[:batch_size]

        elif n_songs == 0:
            raise ValueError("n_song cannot be zero.")
        else:  # -1, use all data
            print("Using all songs:", self.data.shape[0])
            pass
        # exit(10)

        self.n_bars = n_bars
        self.n_phrase = n_bars * 4
        """ Construct two part of data: first n bars and last n bars of each song """
        if debug_temp_dir is not None:
            print("Debug mode, only use first n_bars of each song")
            first_n_bar_data = self.data[:, : self.n_phrase, :]  # [N, n_bars, 128]
            self.data = first_n_bar_data
        else:
            first_n_bar_data = self.data[:, : self.n_phrase, :]  # [N, n_bars, 128]

            # Get song_id for song have more than n_bars
            long_song_ids = torch.where(self.lengths >= self.n_phrase)[0]
            long_song_lengths = self.lengths[long_song_ids]
            last_n_bar_data = []
            for i, song_id in enumerate(long_song_ids):
                song_len = long_song_lengths[i]
                last_n_bars = self.data[song_id, song_len - self.n_phrase : song_len, :]
                last_n_bar_data.append(last_n_bars.unsqueeze(0))
            if len(last_n_bar_data) == 0:
                last_n_bar_data = torch.empty(0, self.n_phrase, 128)
            else:
                last_n_bar_data = torch.cat(
                    last_n_bar_data, dim=0
                )  # [N_long, n_bars, 128]
            self.data = torch.cat(
                [first_n_bar_data, last_n_bar_data], dim=0
            )  # [N + N_long, n_bars, 128]

        # # Print some stats
        # print(self.data.shape)
        # print(self.data.max(), self.data.min(), self.data.mean(), self.data.std())

        print("Constant scaling factor:", scaling_factor)
        self.data = self.data / scaling_factor

        print("Scaled data shape:")
        print(self.data.shape)

        """ --- DEBUG ONLY --- """

        # Save the first sample for debugging
        self.debug_temp_dir = debug_temp_dir
        if self.debug_temp_dir is not None:
            latent_save_fn = "debug_first_sample_latent_scaled.pt"
            latent_save_fp = jpath(self.debug_temp_dir, latent_save_fn)
            create_dir_if_not_exist(self.debug_temp_dir)
            torch.save(self.data[0:1, :, :], latent_save_fp)

        # # Decode the first and last sample
        # ae = PhraseVAE().cuda()
        # first_sample = self.data[0:1, :, :].cuda() * scaling_factor  # [1, max_len, 128]
        # recon = ae.decode_batch(first_sample)
        # save_dir = '/home/longshen/work/AccGen/AccGen/temp/song_level_phr_seq_latents'
        # save_fn = "debug_first_sample_decoded_phrases.txt"
        # save_fp = jpath(save_dir, save_fn)
        # with open(save_fp, 'w') as f:
        #     f.write(recon[0])
        # last_sample = self.data[-1:, :, :].cuda() * scaling_factor  # [1, max_len, 128]
        # recon2 = ae.decode_batch(last_sample)
        # save_fn2 = "debug_last_sample_decoded_phrases.txt"
        # save_fp2 = jpath(save_dir, save_fn2)
        # with open(save_fp2, 'w') as f:
        #     f.write(recon2[0])
        # exit(10)

        """ --- END DEBUG ONLY --- """

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], torch.tensor(self.n_phrase)

    def collate_fn(self, batch):
        # batch: list of (data, length)
        data_list = [item[0] for item in batch]
        length_list = [item[1] for item in batch]

        ret_data = torch.stack(data_list, dim=0)
        ret_lengths = torch.stack(length_list, dim=0)

        return ret_data, ret_lengths


class POP909OneBarPosLatentDataset(Dataset):
    """
    This dataset class loads precomputed position-level latents for each bar in POP909.
    """

    def __init__(
        self,
        latent_fp,
        scaling_factor=1.0,
        scale="none",
        n_bars=1,
        n_sample=-1,
        split=None,
    ):
        """
        scaling_factor: original std of the latents. Will use 1/scaling_factor to scale the latents to N(0,1)
        """

        assert scale in [
            "none",
            "raw",
            "dim",
            "constant",
        ], f"Unsupported scale type: {scale}"
        if scale == "constant":
            assert (
                scaling_factor != 1.0
            ), "For 'constant' scale type, scaling_factor must be provided and not equal to 1.0"

        self.latent_fp = latent_fp
        self.data = torch.load(latent_fp, map_location="cpu")  # [N, max_len=17, 128]

        # Concat every n_bars adjacent bars
        assert n_bars >= 1
        if n_bars > 1:
            data_list = [self.data]
            for i in range(1, n_bars):
                data_shifted = torch.roll(self.data, shifts=-i, dims=0)
                data_list.append(data_shifted)
            self.data = torch.cat(data_list, dim=1)[
                : -n_bars + 1
            ]  # [N-n_bars+1, n_bars*17, 512]
        elif n_bars == 1:
            pass  # [N, max_len=17, 128]

        # Use subset of data if n_sample > 0
        if n_sample > 1024:
            self.data = self.data[:n_sample, :, :]
        elif n_sample > 0:
            self.data = self.data[:n_sample, :, :]
            # duplicate to 1024 samples
            self.data = self.data.repeat(1024 // n_sample + 1, 1, 1)[:1024, :, :]
        elif n_sample == 0:
            raise ValueError("n_sample cannot be zero.")
        else:  # -1, use all data
            pass

        """ --- DEBUG ONLY --- """
        # # Select only the first sample
        # self.data = self.data[0:1, :, :]  # [1, seq_len=17, dim=128]

        # # Duplicate to 1024 samples for debugging
        # self.data = self.data.repeat(1024, 1, 1) # [N=1024, seq_len=13, dim=128]

        # Select only the last position's latents
        # self.data = self.data[:, -1:, :]
        # Select only the first latent
        # self.data = self.data[:, 0:1, :]

        # # Duplicate along seq_len dimension to 13
        # self.data = self.data.repeat(1, 13, 1)  # [N=1024, seq_len=13, dim=128]

        """ --- END DEBUG ONLY --- """

        # Print some stats
        print(self.data.shape)
        print(self.data.max(), self.data.min(), self.data.mean(), self.data.std())

        # Scale the latents
        if scale == "dim":
            dim_mean_fp = latent_fp.replace("latents.pt", "dim_mean.pt")
            dim_std_fp = latent_fp.replace("latents.pt", "dim_std.pt")
            self.dim_mean = torch.load(dim_mean_fp, map_location="cpu")  # [128]
            self.dim_std = torch.load(dim_std_fp, map_location="cpu")  # [128]

            # Normalize to zero mean and unit std
            self.data = (
                self.data - self.dim_mean.unsqueeze(0).unsqueeze(0)
            ) / self.dim_std.unsqueeze(0).unsqueeze(0)
        elif scale == "raw":
            scaling_factor = torch.std(self.data)
            print("Raw scaling factor:", scaling_factor)
            self.data = self.data / scaling_factor
        elif scale == "constant":
            print("Constant scaling factor:", scaling_factor)
            self.data = self.data / scaling_factor
        elif scale == "none":
            pass

        # Load the length description file
        len_desc_fp = latent_fp.replace("latents.pt", "lengths.pt")
        self.lengths = torch.load(
            len_desc_fp, map_location="cpu"
        )  # [N], actual lengths of each sample

        """ --- DEBUG ONLY --- """

        # Save the first sample for debugging
        save_dir = "/home/longshen/work/AccGen/AccGen/temp/pos_latent_debug"
        latent_save_fn = "debug_first_sample_latent_scaled.pt"
        latent_save_fp = jpath(save_dir, latent_save_fn)
        torch.save(self.data[0:1, :, :], latent_save_fp)

        # Save the defussed latent
        img_fn = "debug_first_sample_latent_scaled.png"
        img_fp = jpath(save_dir, img_fn)
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4))
        plt.scatter(range(self.data.shape[2]), self.data[0, 0, :].cpu().numpy(), s=10)
        plt.title("First pos latent of first sample (scaled)")
        plt.xlabel("Latent Dimension")
        plt.ylabel("Latent Value")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.savefig(img_fp)

        # # Duplicate lengths to 1024 samples for debugging
        # self.lengths = self.lengths.repeat(1024)

        # # Decode the first sample for debugging
        # from models.vae_inference import MQVAE_Pos
        # ae = MQVAE_Pos().cuda()
        # first_sample = self.data[0, :, :].cuda() * scaling_factor  # [1, max_len, 128]
        # recon = ae.decode_batch(first_sample)
        # print(' '.join(recon))
        # exit(10)

        """ --- END DEBUG ONLY --- """

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.lengths[idx]

    def collate_fn(self, batch):
        # batch: list of (data, length)
        data_list = [item[0] for item in batch]
        length_list = [item[1] for item in batch]

        ret_data = torch.stack(data_list, dim=0)
        ret_lengths = torch.stack(length_list, dim=0)

        return ret_data, ret_lengths


class POP909PianoRollDataset(Dataset):
    """
    This is a bar-level dataset for POP909, return 2D piano roll matrix 
    """

    def __init__(self, jsonl_fp):
        self.jsonl_fp = jsonl_fp
        self.data = read_jsonl(jsonl_fp)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        bar = MultiTrack.from_remiz_str(self.data[idx])[0]
        ret = bar.to_piano_roll(pos_per_bar=48)
        ret = np.clip(ret, 0, 48)  # clip duration to at most 48 (one whole note)
        ret = ret / 48.0  # 归一化到0~1，最大时值为1
        return torch.from_numpy(ret).float()  # [48, 128]


class POP909PianoTreeDataset(Dataset):
    """
    Bar-level dataset for POP909, outputs pianotree input dict for PianoTreeVAE
    """

    def __init__(self, jsonl_fp):
        self.jsonl_fp = jsonl_fp
        self.data = read_jsonl(jsonl_fp)
        # pianotree_converter: function or object to convert piano roll to pianotree input dict
        # If None, will use default from piano_roll_utils

        self.pianotree_converter = pianoroll_to_pianotree_input

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        bar = MultiTrack.from_remiz_str(self.data[idx])[0]
        piano_roll = bar.to_piano_roll(pos_per_bar=48)  # [48, 128]
        piano_roll = np.clip(piano_roll, 0, 48)
        # Optionally clip/normalize as needed for pianotree conversion
        pianotree_input = self.pianotree_converter(
            piano_roll
        )  # [n_pos, max_simu_note, 1+dur_width]

        ret = {
            "piano_roll": torch.from_numpy(piano_roll).float(),  # [48, 128]
            "pianotree_input": pianotree_input,  # [n_pos, max_simu_note, 1+dur_width]
        }

        return ret

    @staticmethod
    def collate_fn(batch):
        # Batch a list of dicts with keys: 'pitch', 'duration', ...
        # Pad sequences to max length in batch for each key
        from torch.nn.utils.rnn import pad_sequence

        batch_out = {}
        for key in batch[0].keys():
            seqs = [torch.as_tensor(item[key]) for item in batch]
            # Pad with 0 (or -1 if needed for your model)
            batch_out[key] = pad_sequence(seqs, batch_first=True, padding_value=0)
        return batch_out


class POP909DataModule(pl.LightningDataModule):
    def __init__(
        self,
        jsonl_fp_train,
        jsonl_fp_val,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        val_split=0.1,
        dataset_class="POP909PianoTreeDataset",  # NEW: dataset class name as string
        dataset_kwargs={},  # NEW: allow extra kwargs for dataset
        val_every_n_step=250,
    ):
        super().__init__()
        self.jsonl_fp_train = jsonl_fp_train
        self.jsonl_fp_val = jsonl_fp_val
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_split = val_split
        self.dataset_class = dataset_class
        self.dataset_kwargs = dataset_kwargs

        print("Dataset Kwargs:", dataset_kwargs)

        self.dataset_cls = getattr(sys.modules[__name__], self.dataset_class)
        self.val_every_n_step = val_every_n_step

    def setup(self, stage=None):
        self.train_set = self.dataset_cls(
            self.jsonl_fp_train, split="train", **self.dataset_kwargs
        )
        self.val_set = self.dataset_cls(
            self.jsonl_fp_val, split="valid", **self.dataset_kwargs
        )

    def train_dataloader(self):
        real_loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            collate_fn=self.train_set.collate_fn,
        )
        return RepeatLoader(real_loader, steps_per_epoch=self.val_every_n_step)

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=self.val_set.collate_fn,
        )

    def test_dataloader(self):
        # By default, use the validation set as the test set. Modify as needed.
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=self.val_set.collate_fn,
        )


if __name__ == "__main__":
    main()
