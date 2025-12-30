"""
This file contains inference APIs for latent diffusion models.
"""

import os
import sys

# Add project root to sys.path
dirof = os.path.dirname
project_root = dirof(dirof(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from models.diffusion_prior_onebar import UnconditionalDiT, LengthEncoder, SecEncoder
from models.vae_inference import PhraseVAE, BarVAE
from diffusers import DDPMScheduler
from huggingface_hub import PyTorchModelHubMixin


def main():
    model = PhraseLDM().cuda()
    a = model.generate(n_sample=1)
    out_1 = a[0]
    print(out_1[:50])
    print(len(out_1))


class BarLDM(nn.Module):
    def __init__(self, ckpt_fp=None, length_control=False):
        super().__init__()

        # Load LDM checkpoint
        assert ckpt_fp is not None, "Please provide checkpoint file path."
        ckpt = torch.load(ckpt_fp, map_location="cpu")
        config = ckpt["hyper_parameters"]["model_config"]
        state_dict = ckpt["state_dict"]

        # Only keep state dict keys start with model., and delete this prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_k = k[len("model.") :]
                new_state_dict[new_k] = v
        model_state_dict = new_state_dict

        # Initialize LDM model
        ldm = UnconditionalDiT(**config)
        ldm.load_state_dict(model_state_dict, strict=True)
        self.ldm = ldm

        # Initialize VAE
        self.scale_factor = ckpt["hyper_parameters"]["scale_factor"]
        # print(f"VAE scale factor: {self.scale_factor}")
        self.vae = BarVAE()

        self.length_control = length_control
        if length_control:
            # Initialize length encoder
            self.length_encoder = LengthEncoder(
                max_bar=128,
                len_embed_dim=ckpt["hyper_parameters"]["model_config"]["time_proj_dim"],
                length_bucket_size=ckpt["hyper_parameters"]["length_bucket_size"],
            )
            new_state_dict = {}
            # Find the self.length_embedding parameters in state_dict
            for k, v in state_dict.items():
                if k.startswith("length_embedding."):
                    new_k = k[len("length_embedding.") :]
                    new_state_dict[new_k] = v
            self.length_encoder.length_embedding.load_state_dict(
                new_state_dict, strict=True
            )

        # Scheduler
        diffusion_steps = ckpt["hyper_parameters"]["diffusion_steps"]
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=diffusion_steps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            clip_sample_range=3.241875410079956,
        )
        self.noise_scheduler = noise_scheduler

    def generate(self, n_sample, bar_sep=" ", n_bars=None, verbose=True):
        """
        Generate music, n_sample songs in total
        """
        # length embedding
        if self.length_control:
            length_embeds = self.length_encoder(
                torch.tensor([n_bars] * n_sample).cuda()
            )  # (bs, 1, dim)
            # print(f'Length Embeds Shape: {length_embeds.shape}')
            latent = self.ldm.generate(
                n_sample,
                self.noise_scheduler,
                verbose=verbose,
                length_embeds=length_embeds,
            )
        else:
            latent = self.ldm.generate(n_sample, self.noise_scheduler, verbose=verbose)

        latent = latent * self.scale_factor  # [bs, len, d_z]

        decoded = []
        for i in range(latent.size(0)):
            t = self.vae.decode_song(latent[i], bar_sep=bar_sep)
            decoded.append(t)

        return decoded, latent


class PhraseLDM(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self, ckpt_fp=None, length_control=False, sec_control=False
    ):
        super().__init__()

        # Load LDM checkpoint
        assert ckpt_fp is not None, "Please provide checkpoint file path."
        ckpt = torch.load(ckpt_fp, map_location="cpu")
        config = ckpt["hyper_parameters"]["model_config"]
        state_dict = ckpt["state_dict"]

        # Only keep state dict keys start with model., and delete this prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_k = k[len("model.") :]
                new_state_dict[new_k] = v
        model_state_dict = new_state_dict

        # Initialize LDM model
        ldm = UnconditionalDiT(**config)
        ldm.load_state_dict(model_state_dict, strict=True)
        self.ldm = ldm

        # Initialize VAE
        self.scale_factor = ckpt["hyper_parameters"]["scale_factor"]
        print(f"VAE scale factor: {self.scale_factor}")

        self.length_control = length_control
        if length_control:
            # Initialize length encoder
            self.length_encoder = LengthEncoder(
                max_bar=128,
                len_embed_dim=ckpt["hyper_parameters"]["model_config"]["time_proj_dim"],
                length_bucket_size=ckpt["hyper_parameters"]["length_bucket_size"],
            )
            new_state_dict = {}
            # Find the self.length_embedding parameters in state_dict
            for k, v in state_dict.items():
                if k.startswith("length_embedding."):
                    new_k = k[len("length_embedding.") :]
                    new_state_dict[new_k] = v
            self.length_encoder.length_embedding.load_state_dict(
                new_state_dict, strict=True
            )

        self.sec_control = sec_control
        if sec_control:
            # Initialize section encoder
            self.sec_encoder = SecEncoder(
                conf_embed_dim=ckpt["hyper_parameters"]["model_config"][
                    "cross_attention_input_dim"
                ]
            )
            old_state_dict = self.sec_encoder.state_dict()
            new_state_dict = {}
            # Find the sec_encoder parameters in state_dict
            for k, v in state_dict.items():
                if k.startswith("sec_encoder."):
                    new_k = k[len("sec_encoder.") :]
                    new_state_dict[new_k] = v
            self.sec_encoder.load_state_dict(new_state_dict, strict=True)

        # Scheduler
        diffusion_steps = ckpt["hyper_parameters"]["diffusion_steps"]
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=diffusion_steps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            clip_sample_range=3.064450740814209,
        )
        self.noise_scheduler = noise_scheduler

    def generate(
        self,
        n_sample,
        bar_sep=" ",
        n_bars=None,
        sec_cond="",
        decode=True,
        verbose=True,
        vae: PhraseVAE = None,
    ):
        """
        Generate music, n_sample songs in total
        """
        # length embedding
        if self.length_control:
            length_embeds = self.length_encoder(
                n_bar=torch.tensor([n_bars] * n_sample).cuda()
            )  # (bs, 1, dim)
        else:
            length_embeds = None

        # section embedding
        if self.sec_control:
            sec_cond_batch = [sec_cond] * n_sample
            sec_embeds, attn_mask = self.sec_encoder(
                sec_cond_batch
            )  # (bs, max_sec_cond_len, dim)
        else:
            sec_embeds = None
            attn_mask = None

        latent = self.ldm.generate(
            n_sample,
            self.noise_scheduler,
            verbose=verbose,
            length_embeds=length_embeds,
            sec_embeds=sec_embeds,
            sec_attn_mask=attn_mask,
        )  # [n_sample, max_pos, d_z]

        latent = latent * self.scale_factor

        if vae is None:
            print('Warning: Using default PhraseVAE for decoding. To use your own VAE, please provide it as an argument.')
            vae = PhraseVAE()

        decoded = vae.decode_batch(latent, bar_sep=bar_sep)

        return decoded, latent


if __name__ == "__main__":
    main()
