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
from models.diffusion_prior import UnconditionalDiT, LengthEncoder, SecEncoder
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


# class PhraseLDM(nn.Module, PyTorchModelHubMixin):
#     def __init__(
#         self, 
#         ckpt_fp=None, 
#         length_control=False, 
#         sec_control=False
#     ):
#         super().__init__()

#         # Load LDM checkpoint
#         assert ckpt_fp is not None, "Please provide checkpoint file path."
#         ckpt_fp = '/data1/longshen/Results/PhraseLDMResults/diffusion_prior/phr_latent/full_song/64dim_latent/unconditional/tb_logs/version_2/checkpoints/step_step=200000.ckpt'
#         ckpt = torch.load(ckpt_fp, map_location="cpu")
#         config = ckpt["hyper_parameters"]["model_config"]
#         state_dict = ckpt["state_dict"]

#         print(ckpt["hyper_parameters"].keys())
#         print(ckpt["hyper_parameters"]["model_config"])
#         print({k: ckpt["hyper_parameters"][k] for k in ["scale_factor","diffusion_steps","length_bucket_size"] if k in ckpt["hyper_parameters"]})
#         exit()

#         # Only keep state dict keys start with model., and delete this prefix
#         new_state_dict = {}
#         for k, v in state_dict.items():
#             if k.startswith("model."):
#                 new_k = k[len("model.") :]
#                 new_state_dict[new_k] = v
#         model_state_dict = new_state_dict

#         # Initialize LDM model
#         ldm = UnconditionalDiT(**config)
#         ldm.load_state_dict(model_state_dict, strict=True)
#         self.ldm = ldm

#         # Initialize VAE
#         self.scale_factor = ckpt["hyper_parameters"]["scale_factor"]
#         print(f"VAE scale factor: {self.scale_factor}")

#         self.length_control = length_control
#         if length_control:
#             # Initialize length encoder
#             self.length_encoder = LengthEncoder(
#                 max_bar=128,
#                 len_embed_dim=ckpt["hyper_parameters"]["model_config"]["time_proj_dim"],
#                 length_bucket_size=ckpt["hyper_parameters"]["length_bucket_size"],
#             )
#             new_state_dict = {}
#             # Find the self.length_embedding parameters in state_dict
#             for k, v in state_dict.items():
#                 if k.startswith("length_embedding."):
#                     new_k = k[len("length_embedding.") :]
#                     new_state_dict[new_k] = v
#             self.length_encoder.length_embedding.load_state_dict(
#                 new_state_dict, strict=True
#             )

#         self.sec_control = sec_control
#         if sec_control:
#             # Initialize section encoder
#             self.sec_encoder = SecEncoder(
#                 conf_embed_dim=ckpt["hyper_parameters"]["model_config"][
#                     "cross_attention_input_dim"
#                 ]
#             )
#             old_state_dict = self.sec_encoder.state_dict()
#             new_state_dict = {}
#             # Find the sec_encoder parameters in state_dict
#             for k, v in state_dict.items():
#                 if k.startswith("sec_encoder."):
#                     new_k = k[len("sec_encoder.") :]
#                     new_state_dict[new_k] = v
#             self.sec_encoder.load_state_dict(new_state_dict, strict=True)

#         # Scheduler
#         diffusion_steps = ckpt["hyper_parameters"]["diffusion_steps"]
#         noise_scheduler = DDPMScheduler(
#             num_train_timesteps=diffusion_steps,
#             beta_schedule="squaredcos_cap_v2",
#             clip_sample=True,
#             clip_sample_range=3.064450740814209,
#         )
#         self.noise_scheduler = noise_scheduler


class PhraseLDM(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        model_config: dict = None,
        scale_factor: float = 0.7590118646621704,
        diffusion_steps: int = 1000,
        clip_range: float = 3.064450740814209,
        length_control: bool = False,
        length_bucket_size: int = 10,
        sec_control: bool = False,
    ):
        super().__init__()

        if model_config is None:
            cross_attn_dim = 256 if sec_control else 128
            model_config = {
                "attention_head_dim": 32,
                "cross_attention_dim": cross_attn_dim,
                "cross_attention_input_dim": cross_attn_dim,
                "global_states_input_dim": 128,
                "in_channels": 64,
                "num_attention_heads": 16,
                "num_key_value_attention_heads": 8,
                "num_layers": 6,
                "out_channels": 64,
                "sample_size": 512,
                "time_proj_dim": 128,
            }
            
        self.model_config = model_config
        self.scale_factor = float(scale_factor)
        self.diffusion_steps = int(diffusion_steps)
        self.clip_range = float(clip_range)

        self.length_control = bool(length_control)
        self.length_bucket_size = int(length_bucket_size)

        self.sec_control = bool(sec_control)

        # ---- core DiT ----
        self.ldm = UnconditionalDiT(**self.model_config)

        # ---- length conditioning ----
        if self.length_control:
            self.length_encoder = LengthEncoder(
                max_bar=128,
                len_embed_dim=self.model_config["time_proj_dim"],
                length_bucket_size=self.length_bucket_size,
            )

        # ---- optional section conditioning ----
        if self.sec_control:
            self.sec_encoder = SecEncoder(
                conf_embed_dim=self.model_config["cross_attention_input_dim"]
            )

        # ---- scheduler ----
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.diffusion_steps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            clip_sample_range=self.clip_range,
        )

        self.eval()

    @classmethod
    def from_lightning_ckpt(cls, ckpt_fp: str, device="cuda", torch_dtype=None):
        ckpt = torch.load(ckpt_fp, map_location="cpu")
        hp = ckpt["hyper_parameters"]
        sd = ckpt["state_dict"]
        print(hp)

        sec_control = any(k.startswith("sec_encoder.") for k in sd.keys())

        model = cls(
            model_config=hp["model_config"],
            scale_factor=hp["scale_factor"],
            diffusion_steps=hp["diffusion_steps"],
            clip_range=hp['clip_range'],
            length_control=hp['length_condition'],
            length_bucket_size=hp['length_bucket_size'],
            sec_control=sec_control,
        )

        # ---- load ldm (strip "model.") ----
        ldm_sd = {}
        for k, v in sd.items():
            if k.startswith("model."):
                ldm_sd[k[len("model."):]] = v
        model.ldm.load_state_dict(ldm_sd, strict=True)

        # ---- load length encoder ----
        if model.length_control:
            len_sd = {}
            for k, v in sd.items():
                if k.startswith("length_embedding."):
                    len_sd[k[len("length_embedding."):]] = v
            model.length_encoder.length_embedding.load_state_dict(len_sd, strict=True)

        # ---- load sec encoder ----
        if model.sec_control:
            sec_sd = {}
            for k, v in sd.items():
                if k.startswith("sec_encoder."):
                    sec_sd[k[len("sec_encoder."):]] = v
            model.sec_encoder.load_state_dict(sec_sd, strict=True)

        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)
        model = model.to(device)
        model.eval()
        return model


    def generate(
        self,
        n_sample,
        vae: PhraseVAE,
        n_bars=None,
        sec_cond="",
        verbose=True,
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

        decoded = vae.decode_batch(latent)

        return decoded, latent


if __name__ == "__main__":
    main()
