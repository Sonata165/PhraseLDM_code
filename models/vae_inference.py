"""
Inference class for the MQAE model (Multi-Query Autoencoder)
"""

import torch.nn as nn
import torch
from models.phrase_vae import S2SVQAE, load_t5_model_from_lit_ckpt, S2SVAE, S2SVAE2, S2SVAE3
from transformers import PreTrainedTokenizerFast
from remi_z import MultiTrack
from typing import List
from huggingface_hub import PyTorchModelHubMixin


class MQAE(nn.Module):
    """
    This is a wrapper class of S2SVQAE model,
    providing easy-to-use inference API.
    """

    def __init__(self):
        super().__init__()

        # Initialize model parameters based on config

        ckpt_fp = "/data1/longshen/Results/AccGenResults/bar_compression/s2s/mqcomp_ft/q4_d512_l3_lr1e-4/tb_logs/version_1/checkpoints/epoch=129_step=69420_val_loss=0.0077.ckpt"
        t5 = load_t5_model_from_lit_ckpt(ckpt_fp)

        model = S2SVQAE(
            tokenizer_path="LongshenOu/phrase-vae-tokenizer",
            t5_config={
                "d_model": 512,
                "d_ff": 1024,
                "num_layers": 3,
                "num_heads": 6,
                "vocab_size": 1000,
                "decoder_start_token_id": 1,
            },
            t5_model_name=ckpt_fp,
            lit_ckpt=True,
            compress_style="first_n_tokens",
            n_compress_tokens=4,
        )
        model.t5 = t5
        model.eval()
        self.model = model

        # Prepare tokenizer
        tokenizer_path = "LongshenOu/phrase-vae-tokenizer"
        self.tk = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    def from_pretrained(self, model_path):
        # Load model parameters from the specified path
        pass

    def encode_batch(self, remiz_strs):
        """
        Batch version of encode()
        """
        # Add special tokens (and remove instrument token for now)
        bar_strs = []
        for bar_str in remiz_strs:
            inp_seq = bar_str.split()
            inp_seq = ["[BOS]"] + inp_seq + ["[EOS]"]
            inp_seq = [f"[C{i}]" for i in range(4)] + inp_seq
            bar_str = " ".join(inp_seq)
            bar_strs.append(bar_str)

        # Tokenize
        inputs = self.tk(
            bar_strs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        input_ids = inputs.input_ids
        attn_mask = inputs.attention_mask

        # # Debug: detokenize
        # print(self.tk.batch_decode(input_ids, skip_special_tokens=False))

        # Encode
        with torch.no_grad():
            encoder_out = self.model.encode(input_ids, attn_mask)
            latents = encoder_out["quantized"]

        return latents

    def decode_batch(self, latents, return_mt=False, return_proll=False):
        """
        Decode from latent representation to remiz string
        If return_mt is True, also return the MultiTrack object
        """
        assert not (return_mt and return_proll), "Cannot return both MultiTrack and piano roll."

        # Decode
        out = self.model.decode(
            {"quantized": latents},
            max_length=128,
            pad_token_id=self.tk.pad_token_id,
            eos_token_id=self.tk.eos_token_id,
            do_sample=False,
            num_beams=1,
        )

        # Detokenize
        bar_strs = self.tk.batch_decode(out, skip_special_tokens=True)
        # print(bar_strs)

        if return_mt:
            # Convert to MultiTrack
            mts = [MultiTrack.from_remiz_str(bar_str) for bar_str in bar_strs]
            return mts
        elif return_proll:
            # Convert to piano roll
            mts = [MultiTrack.from_remiz_str(bar_str)[0] for bar_str in bar_strs]
            prolls = [mt.to_piano_roll(pos_per_bar=48) for mt in mts]
            return prolls
        else:
            return bar_strs


class MQVAE(nn.Module):
    """
    This is a wrapper class of S2SVAE model,
    providing easy-to-use inference API.
    Deprecated. Please use PhraseVAE class instead
    """

    def __init__(self, ckpt_fp=None, bottleneck=False):
        super().__init__()

        assert bottleneck is False, "Bottleneck MQVAE is not supported yet. Please use MQVAE_Pos."

        # Initialize model parameters based on config
        if ckpt_fp is None:
            ckpt_fp = '/data1/longshen/Results/AccGenResults/aes/bar_level/pop909_phrase/s2s/mqcomp_vae/aeft_lr1e-4_klw1.0/tb_logs/version_0/checkpoints/epoch=203_step=98736_val_loss=0.1258.ckpt'
        model = S2SVAE(
            tokenizer_path="LongshenOu/phrase-vae-tokenizer",
            t5_config={
                "d_model": 512,
                "d_ff": 1024,
                "num_layers": 3,
                "num_heads": 6,
                "vocab_size": 1000,
                "decoder_start_token_id": 1,
            },
            compress_style="first_n_tokens",
            n_compress_tokens=4,
        )

        # Load state dict
        # Remove "model." prefix from state dict keys
        state_dict_new = {}
        for k, v in torch.load(ckpt_fp)["state_dict"].items():
            assert k.startswith("model.")
            k_new = k[len("model.") :]
            state_dict_new[k_new] = v
        model.load_state_dict(state_dict_new, strict=True)

        model.eval()
        self.model = model

        # Prepare tokenizer
        tokenizer_path = "LongshenOu/phrase-vae-tokenizer"
        self.tk = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    def from_pretrained(self, model_path):
        # Load model parameters from the specified path
        pass

    def encode_batch(self, remiz_strs, do_sample=False):
        """
        Batch version of encode()
        """
        # Add special tokens (and remove instrument token for now)
        bar_strs = []
        for bar_str in remiz_strs:
            inp_seq = bar_str.split()

            # IMPORTANT! Remove instrument token if present
            if inp_seq[0].startswith("i-"):
                inp_seq = inp_seq[1:]

            inp_seq = ["[BOS]"] + inp_seq + ["[EOS]"]
            inp_seq = [f"[C{i}]" for i in range(4)] + inp_seq
            bar_str = " ".join(inp_seq)
            bar_strs.append(bar_str)

        # Tokenize
        inputs = self.tk(
            bar_strs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        input_ids = inputs.input_ids
        attn_mask = inputs.attention_mask

        # # Debug: detokenize
        # print(self.tk.batch_decode(input_ids, skip_special_tokens=False))

        # Encode
        with torch.no_grad():
            encoder_out = self.model.encode(input_ids, attn_mask, sample=do_sample)
            latents = encoder_out["latent"]

        return latents

    def decode_batch(self, latents, return_mt=False, return_proll=False, scale_factor=1.0, n_compress_tokens=4):
        """
        Decode from latent representation to remiz string
        If return_mt is True, also return the MultiTrack object

        latents: (batch, n_compress_tokens, d_model), or (batch, n_compress_tokens * n_bars, d_model)
        """
        assert not (return_mt and return_proll), "Cannot return both MultiTrack and piano roll."

        # Scale the latents
        latents = latents * scale_factor

        # If input latents contain multiple bars, reshape to (batch * n_bars, n_compress_tokens, d_model)
        flatten = False
        if latents.shape[1] > n_compress_tokens:
            # only do when latent is 3d
            if latents.ndim != 3:
                pass
            else:
                flatten = True
                assert latents.shape[1] % n_compress_tokens == 0, "Latent length must be multiple of n_compress_tokens."
                n_bars = latents.shape[1] // n_compress_tokens
                latents = latents.view(-1, n_compress_tokens, latents.shape[2])

        # Decode
        out = self.model.decode(
            latents,
            max_length=128,
            pad_token_id=self.tk.pad_token_id,
            eos_token_id=self.tk.eos_token_id,
            do_sample=False,
            num_beams=1,
        )

        # Detokenize
        bar_strs = self.tk.batch_decode(out, skip_special_tokens=True) # list[Str], (batch * n_bars,)

        # Reshape back to (batch, n_bars)
        if flatten:
            bar_strs_new = []
            for i in range(0, len(bar_strs), n_bars):
                bar_strs_new.append(" ".join(bar_strs[i:i+n_bars]))
            bar_strs = bar_strs_new

        if return_mt:
            # Convert to MultiTrack
            mts = [MultiTrack.from_remiz_str(bar_str) for bar_str in bar_strs]
            return mts
        elif return_proll:
            # Convert to piano roll
            mts = [MultiTrack.from_remiz_str(bar_str)[0] for bar_str in bar_strs]
            prolls = [mt.to_piano_roll(pos_per_bar=48) for mt in mts]
            return prolls
        else:
            return bar_strs
        
    
class BarVAE(nn.Module):
    """
    This is a wrapper class of S2SVAE model,
    providing easy-to-use inference API.
    """

    def __init__(self, ckpt_fp=None):
        super().__init__()

        # Initialize model parameters based on config
        if ckpt_fp is None:
            # New multitrack BarVAE
            ckpt_fp = '/data1/longshen/Results/AccGenResults/aes/pretrained/vae/bar_vae_mt/step=63210_val_f1_iopd=0.9865.ckpt'
            # Old flattened BarVAE
            # ckpt_fp = '/data1/longshen/Results/AccGenResults/aes/bar_level/pop909_phrase_flatten/s2s/mqcomp_vae_bottleneck/aeft_lr1e-4_klw0.1_b512_4dectok_ctn/tb_logs/version_0/checkpoints/epoch=215_step=104544_val_loss=0.0704.ckpt'
        model = S2SVAE3(
            tokenizer_path="LongshenOu/phrase-vae-tokenizer",
            t5_config={
                "d_model": 512,
                "d_ff": 1024,
                "num_layers": 3,
                "num_heads": 6,
                "vocab_size": 1000,
                "decoder_start_token_id": 1,
            },
            compress_style="first_n_tokens",
            n_compress_tokens=4,
        )

        # Load state dict
        # Remove "model." prefix from state dict keys
        state_dict_new = {}
        for k, v in torch.load(ckpt_fp)["state_dict"].items():
            assert k.startswith("model.")
            k_new = k[len("model.") :]
            state_dict_new[k_new] = v
        model.load_state_dict(state_dict_new, strict=True)

        model.eval()
        self.model = model

        # Prepare tokenizer
        tokenizer_path = "LongshenOu/phrase-vae-tokenizer"
        self.tk = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    def from_pretrained(self, model_path):
        # Load model parameters from the specified path
        pass

    def encode_mt(self, mt:MultiTrack, do_sample=False):
        """
        Encode a MultiTrack object to latent representation

        Return: latents: (n_bars, d_z)
        """
        remiz_strs = []
        for bar in mt:
            remiz_seq = bar.to_remiz_seq()
            remiz_str = ' '.join(remiz_seq)
            remiz_strs.append(remiz_str)
        remiz_strs.append('[SEP]')

        latents = self.encode_batch(remiz_strs, do_sample=do_sample)

        return latents
    
    def decode_song(self, latents, early_stop=True, return_mt=False, bar_sep=' '):
        """
        Decode from latent representation to remiz string
        If return_mt is True, also return the MultiTrack object

        NOTE: ensure the latents are already in VAE scale

        latents: (n_bars, d_z)
        early_stop: If True, stop decoding when [SEP] is generated
        """
        assert latents.ndim == 2, "Latents should be 2D tensor of shape (n_bars, d_model)."
        n_bars, d_latent = latents.shape

        # Decode
        out = self.model.decode(
            latents, # [bs, seq_len, d_model]
            max_length=128,
            pad_token_id=self.tk.pad_token_id,
            eos_token_id=self.tk.eos_token_id,
            do_sample=False,
            num_beams=1,
        )

        # Detokenize
        bar_strs = self.tk.batch_decode(out, skip_special_tokens=True) # list[Str], (n_bars,)

        # Ensure bar line token for any bar that is not skipped
        bar_strs = [bar_str + ' b-1' if bar_str != '' and 'b-1' not in bar_str else bar_str for bar_str in bar_strs]

        # If any bar is '' (skipped in decoding), convert it to [SEP]
        if '' in bar_strs:
            print("There are empty bars decoded, converting them to [SEP] tokens.")
        for i in range(len(bar_strs)):
            if bar_strs[i] == '':
                print(f"Bar {i} is empty.")
            
        bar_strs = [bar_str if bar_str != '' else '[SEP]' for bar_str in bar_strs]

        if early_stop:
            # Stop decoding when [SEP] is generated
            new_bar_strs = []
            for bar_str in bar_strs:
                if bar_str == '[SEP]':
                    print('Early stopping at [SEP] token.')
                    break
                new_bar_strs.append(bar_str)
            bar_strs = new_bar_strs

        # Merge bars into one song string
        bar_strs = bar_sep.join(bar_strs)

        if return_mt:
            # Convert to MultiTrack
            mts = [MultiTrack.from_remiz_str(bar_str) for bar_str in bar_strs]
            return mts
        
        return bar_strs

    def encode_batch(self, remiz_strs, do_sample=False):
        """
        Batch version of encode()
        Input: remiz_strs: list of remiz strings, each string corresponds to one bar
        """
        # Add special tokens (and remove instrument token for now)
        bar_strs = []
        for bar_str in remiz_strs:
            inp_seq = bar_str.split()

            # NOTE: IMPORTANT! Do not remove any token here, e.g., instrument token 

            inp_seq = ["[BOS]"] + inp_seq + ["[EOS]"]
            inp_seq = [f"[C{i}]" for i in range(4)] + inp_seq
            bar_str = " ".join(inp_seq)
            bar_strs.append(bar_str)

        # Tokenize
        inputs = self.tk(
            bar_strs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        input_ids = inputs.input_ids
        attn_mask = inputs.attention_mask

        # Encode
        with torch.no_grad():
            encoder_out = self.model.encode(input_ids, attn_mask, sample=do_sample)
            latents = encoder_out["latent"]

        return latents

    def decode_batch(self, latents, return_mt=False, return_proll=False, scale_factor=1.0, n_compress_tokens=4, bar_sep=None):
        """
        Decode from latent representation to remiz string
        If return_mt is True, also return the MultiTrack object

        latents: (batch, n_bars, d_model)
        """
        assert not (return_mt and return_proll), "Cannot return both MultiTrack and piano roll."

        assert latents.ndim == 3, "Latents should be 3D tensor of shape (batch, n_bars, d_model)."

        # Scale the latents
        latents = latents * scale_factor

        # If input latents contain multiple bars, reshape to (batch * n_bars, n_compress_tokens, d_model)
        n_bars = latents.shape[1]
        d_latent = latents.shape[2]
        latents = latents.view(-1, d_latent)

        # Decode
        out = self.model.decode(
            latents, # [bs, seq_len, d_model]
            max_length=128,
            pad_token_id=self.tk.pad_token_id,
            eos_token_id=self.tk.eos_token_id,
            do_sample=False,
            num_beams=1,
        )

        # Detokenize
        bar_strs = self.tk.batch_decode(out, skip_special_tokens=True) # list[Str], (batch * n_bars,)

        # Reshape back to (batch, n_bars)
        bar_strs_new = []
        for i in range(0, len(bar_strs), n_bars):
            bar_strs_new.append(" ".join(bar_strs[i:i+n_bars]))
        bar_strs = bar_strs_new

        return bar_strs
        

class MQVAE_Pos(nn.Module):
    """
    This is a wrapper class of S2SVAE model,
    providing easy-to-use inference API.
    To decode position-level latents
    """

    def __init__(self, ckpt_fp=None):
        super().__init__()

        # Initialize model parameters based on config
        # Default: position-level VAE with 128-dim bottleneck
        if ckpt_fp is None:
            # ckpt_fp = '/data1/longshen/Results/AccGenResults/aes/pos_level/pop909_phrase_flatten/mqcomp_vae_bottleneck/b128_withbarline/tb_logs/version_0/checkpoints/epoch=209_step=1086540_val_loss=0.1615.ckpt'
            ckpt_fp = '/data1/longshen/Results/AccGenResults/aes/pos_level/pop909_phrase_flatten/mqcomp_vae_bottleneck/b64_l2w1/tb_logs/version_0/checkpoints/epoch=16_step=87958_val_loss=0.4785.ckpt'
        model = S2SVAE2(
            tokenizer_path="LongshenOu/phrase-vae-tokenizer",
            t5_config={
                "d_model": 512,
                "d_ff": 1024,
                "num_layers": 3,
                "num_heads": 6,
                "vocab_size": 1000,
                "decoder_start_token_id": 1,
            },
            compress_style="first_n_tokens",
            n_compress_tokens=4,
            bottleneck_dim=64, # 128
        )
        self.bottleneck = True

        # Load state dict
        # Remove "model." prefix from state dict keys
        state_dict_new = {}
        for k, v in torch.load(ckpt_fp)["state_dict"].items():
            assert k.startswith("model.")
            k_new = k[len("model.") :]
            state_dict_new[k_new] = v
        model.load_state_dict(state_dict_new, strict=True)

        model.eval()
        self.model = model

        # Prepare tokenizer
        tokenizer_path = "LongshenOu/phrase-vae-tokenizer"
        self.tk = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    def from_pretrained(self, model_path):
        # Load model parameters from the specified path
        pass

    def encode_batch(self, remiz_strs, do_sample=False):
        """
        Batch version of encode()
        """
        # Add special tokens (and remove instrument token for now)
        bar_strs = []
        for bar_str in remiz_strs:
            inp_seq = bar_str.split()

            # IMPORTANT! Remove instrument token if present
            if inp_seq[0].startswith("i-"):
                inp_seq = inp_seq[1:]

            inp_seq = ["[BOS]"] + inp_seq + ["[EOS]"]
            inp_seq = [f"[C{i}]" for i in range(4)] + inp_seq
            bar_str = " ".join(inp_seq)
            bar_strs.append(bar_str)

        # Tokenize
        inputs = self.tk(
            bar_strs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        input_ids = inputs.input_ids
        attn_mask = inputs.attention_mask

        # # Debug: detokenize
        # print(self.tk.batch_decode(input_ids, skip_special_tokens=False))

        # Encode
        with torch.no_grad():
            encoder_out = self.model.encode(input_ids, attn_mask, sample=do_sample)
            latents = encoder_out["latent"]

        return latents

    def decode_batch(self, latents, return_mt=False, return_proll=False, scale_factor=1.0):
        """
        Decode from latent representation to remiz string
        If return_mt is True, also return the MultiTrack object

        latents: (batch, max_seq_len, d_model)
        """
        assert not (return_mt and return_proll), "Cannot return both MultiTrack and piano roll."

        # Scale the latents
        latents = latents * scale_factor

        # Flatten
        # latents = latents.view(-1, 128)
        latents = latents.view(-1, 64)
        # print("Latents shape after flatten:", latents.shape)

        # TODO: need to take care when batch decode multiple bars

        # Decode
        out = self.model.decode(
            latents,
            max_length=128,
            pad_token_id=self.tk.pad_token_id,
            eos_token_id=self.tk.eos_token_id,
            do_sample=False,
            num_beams=1,
        )

        # Detokenize
        bar_strs = self.tk.batch_decode(out, skip_special_tokens=True) # list[Str], (batch * n_bars,)

        if return_mt:
            # Convert to MultiTrack
            mts = [MultiTrack.from_remiz_str(bar_str) for bar_str in bar_strs]
            return mts
        elif return_proll:
            # Convert to piano roll
            mts = [MultiTrack.from_remiz_str(bar_str)[0] for bar_str in bar_strs]
            prolls = [mt.to_piano_roll(pos_per_bar=48) for mt in mts]
            return prolls
        else:
            return bar_strs
        
class PhraseVAE(
        nn.Module, 
        PyTorchModelHubMixin,
    ):
    """
    This is a wrapper class of S2SVAE3 model,
    providing easy-to-use inference API for Phrase-level VAE.
    """

    def __init__(self, ckpt_fp=None):
        super().__init__()

        # Initialize model parameters based on config
        if ckpt_fp is None:
            ckpt_fp = '/data1/longshen/Results/AccGenResults/aes/pretrained/vae/phrase_vae/epoch=75_step=171684_val_loss=0.0461.ckpt'

        model = S2SVAE3(
            tokenizer_path="LongshenOu/phrase-vae-tokenizer",
            t5_config={
                "d_model": 512,
                "d_ff": 1024,
                "num_layers": 3,
                "num_heads": 6,
                "vocab_size": 1000,
                "decoder_start_token_id": 1,
            },
            compress_style="first_n_tokens",
            n_compress_tokens=4,
            bottleneck_dim=64,
        )

        # Load state dict
        # Remove "model." prefix from state dict keys
        state_dict_new = {}
        for k, v in torch.load(ckpt_fp)["state_dict"].items():
            assert k.startswith("model.")
            k_new = k[len("model.") :]
            state_dict_new[k_new] = v
        model.load_state_dict(state_dict_new, strict=True)

        model.eval()
        self.model = model

        # Prepare tokenizer
        tokenizer_path = "LongshenOu/phrase-vae-tokenizer"
        self.tk = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    # def from_pretrained(self, model_path):
    #     # Load model parameters from the specified path
    #     pass

    def encode_batch(self, remiz_strs, do_sample=False):
        """
        Batch version of encode()

        remiz_strs: list of remiz strings, each string corresponds to one phrase (multiple bars)
        return: latents: (batch, d_model)
        """
        # Add special tokens (and remove instrument token for now)
        bar_strs = []
        for bar_str in remiz_strs:
            inp_seq = bar_str.split()

            inp_seq = ["[BOS]"] + inp_seq + ["[EOS]"]
            inp_seq = [f"[C{i}]" for i in range(4)] + inp_seq
            bar_str = " ".join(inp_seq)
            bar_strs.append(bar_str)

        # Tokenize
        inputs = self.tk(
            bar_strs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        input_ids = inputs.input_ids
        attn_mask = inputs.attention_mask

        # # Debug: detokenize
        # print(self.tk.batch_decode(input_ids, skip_special_tokens=False))

        # Encode
        with torch.no_grad():
            encoder_out = self.model.encode(input_ids, attn_mask, sample=do_sample)
            latents = encoder_out["latent"]

        return latents

    def decode_song(self, latents, early_stop=True):
        """
        Decode from latent representation to remiz string
        If return_mt is True, also return the MultiTrack object

        NOTE: ensure the latents are already in VAE scale

        latents: (n_phrase, d_model)
        early_stop: If True, stop decoding when [SEP] is generated
        """
        assert latents.ndim == 2, "Latents should be 1D tensor of shape (d_model)."

        # Decode
        out = self.model.decode(
            latents,
            max_length=128,
            pad_token_id=self.tk.pad_token_id,
            eos_token_id=self.tk.eos_token_id,
            do_sample=False,
            num_beams=1,
        )

        # Detokenize
        phr_strs = self.tk.batch_decode(out, skip_special_tokens=False)

        song_seq = []
        eos_found = False
        for phr_str in phr_strs:
            phr_seq = phr_str.split()
            for token in phr_seq:
                # skip '[BOS]', '[EOS]', '[PAD]',
                if token == '[BOS]' or token == '[EOS]' or token == '[PAD]':
                    continue
                if token == '[SEP]' and early_stop:
                    # print('Early stopping at [SEP] token.')
                    eos_found = True
                    break
                song_seq.append(token)
            if eos_found:
                break
        song_str = ' '.join(song_seq)
        
        return song_str

    def decode_batch(self, latents, scale_factor=1.0, bar_sep=' ') -> List[str]:
        """
        Decode from latent representation to remiz string
        If return_mt is True, also return the MultiTrack object

        latents: (n_song, n_phrase, d_model)
        """
        assert latents.ndim == 3, "Latents should be 3D tensor of shape (batch, n_phrase, d_model), but got {}".format(latents.shape)

        # Scale the latents
        latents = latents * scale_factor

        song_strs = []
        for i in range(latents.shape[0]):
            song_str = self.decode_song(latents[i], early_stop=True)
            song_strs.append(song_str)
        return song_strs

    
def remove_extra_bar_line(remiz_str):
    """
    Merge repeated 'b-1' tokens into a single 'b-1' token.
    """
    tokens = remiz_str.split()
    new_tokens = []
    prev_token = None
    for token in tokens:
        if token == 'b-1' and prev_token == 'b-1':
            continue
        new_tokens.append(token)
        prev_token = token
    new_remiz_str = ' '.join(new_tokens)    
    return new_remiz_str