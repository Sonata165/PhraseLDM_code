"""
Train diffusion prior model for generating one-bar latent representations
With lightning
"""
import os
import sys

# Add project root to sys.path
dirof = os.path.dirname
project_root = dirof(dirof(dirof(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import pytorch_lightning as pl
from torch import nn
from models.diffusion_prior_onebar import (
    UnconditionedTransformerEnc,
    UnconditionalDiT,
    InstConfEncoder,
    SecEncoder,
)
from diffusers import DDPMScheduler
from models.vae_inference import MQVAE_Pos, BarVAE, PhraseVAE
from tqdm import tqdm
from diffusers.models.transformers.stable_audio_transformer import StableAudioDiTModel
from sonata_utils import jpath, read_json, create_dir_if_not_exist
from evaluation.metrics import (
    compute_fid,
    compute_diversity,
    compute_ngram_fid,
    bright_ratio,
    compute_bar_ssm_from_phrase_latents,
    bar_recurrence_score_diag,
    memorization_rate_batch,
    Metric,
    compute_fid_gt_vs_out,
)
import io
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from remi_z import MultiTrack
from the_utils.midi_utils import save_midi


class LitPhraseLDM(pl.LightningModule):
    """
    A 1-D Diffusion Transformer model that generates variable-length latent sequences
    """

    def __init__(
        self,
        model_config,
        scale_factor=1.0,
        mode=None,
        lr=1e-4,
        eps=1e-8,
        diffusion_steps=1000,
        overfit_debug=False,
        debug_temp_dir=None,
        from_pretrained_ldm_fp=None,
        length_condition=False,
        length_bucket_size=10,
        eval_latent_fp=None,
        clip_range=1.0,
        n_gen_in_valid=20,
        inst_conf_condition=False,
        sec_condition=False,
    ):
        super().__init__()
        self.save_hyperparameters()

        assert mode in ["pos", "bar", "phr"], "mode should be 'pos' or 'bar' or 'phr'"
        self.latent_mode = mode
        print("Initializing LitPhraseLDM with mode:", mode)

        # Set up model
        self.model = UnconditionalDiT(**model_config)
        input_dim = model_config["in_channels"]
        max_pos = model_config["sample_size"]
        self.max_pos = max_pos
        self.input_dim = input_dim
        self.lr = lr
        self.eps = eps

        # Load pretrained weights if specified
        if from_pretrained_ldm_fp is not None:
            self.load_ldm_weights(from_pretrained_ldm_fp)

        # Noise scheduler
        self.diffusion_steps = diffusion_steps
        self.clip_range = clip_range
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=diffusion_steps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            clip_sample_range=self.clip_range,
        )
        self.noise_scheduler = noise_scheduler

        self.loss_fn = nn.MSELoss()

        dim_mean_fp = "/data1/longshen/Datasets/Piano/POP909/latents/bar_level_pos_seq/train_dim_mean.pt"
        dim_std_fp = "/data1/longshen/Datasets/Piano/POP909/latents/bar_level_pos_seq/train_dim_std.pt"
        self.dim_mean = torch.load(dim_mean_fp, map_location="cpu")  # [128]
        self.dim_std = torch.load(dim_std_fp, map_location="cpu")  # [128]

        # Set up VAE
        if self.latent_mode == "pos":
            self.vae = MQVAE_Pos()
        elif self.latent_mode == "bar":
            self.vae = BarVAE()
        elif self.latent_mode == "phr":
            self.vae = PhraseVAE()
        self.scale_factor = scale_factor
        print("Model scale factor:", self.scale_factor)

        # For debug
        self.overfit_debug = overfit_debug
        if overfit_debug:
            assert debug_temp_dir is not None, "debug_temp_dir cannot be None"
            self.debug_temp_dir = debug_temp_dir
            self.gt_latent_fp = jpath(
                self.debug_temp_dir, "debug_first_sample_latent_scaled.pt"
            )

        # Length encoder
        if length_condition:
            self.length_embedding = nn.Embedding(
                128, model_config["global_states_input_dim"]
            )
            self.length_bucket_size = length_bucket_size
        else:
            self.length_embedding = None
            self.length_bucket_size = None

        # Instrumentation configuration encoder
        if inst_conf_condition:
            self.inst_conf_encoder = InstConfEncoder(
                conf_embed_dim=model_config["cross_attention_input_dim"]
            )
        else:
            self.inst_conf_encoder = None

        # Section condition encoder
        self.sec_condition = sec_condition
        if sec_condition:
            self.sec_encoder = SecEncoder(
                conf_embed_dim=model_config["cross_attention_input_dim"]
            )
        else:
            self.sec_encoder = None

        mel_dict_fp = (
            "/data1/longshen/Datasets/Piano/POP909/remi_z/melody_remiz_dict.json"
        )
        self.mel_dict = read_json(mel_dict_fp)

        self.metric = Metric()

        assert eval_latent_fp is not None, "eval_latent_fp cannot be None"
        self.eval_latent_fp = eval_latent_fp
        # phrase LDM: "/data1/longshen/Datasets/Piano/POP909/latents/song_level_phr_seq_with_annot/eval_data.pt"

        self.n_gen_in_valid = n_gen_in_valid

        # Read instrumentation layout reference for 64-bar song
        inst_cond_ref_fp = (
            "/home/longshen/work/AccGen/AccGen/models/083_inst_layout_64bar.json"
        )
        inst_conf_ref = read_json(inst_cond_ref_fp)
        inst_conf_ref_flatten = []
        for bar in inst_conf_ref:
            inst_conf_ref_flatten.extend(bar)
        self.inst_conf_ref_flatten = inst_conf_ref_flatten  # list of str

        # Read section annotation reference for 64-bar song
        sec_annot_ref_fp = '/home/longshen/work/AccGen/AccGen/models/083_sec_annot_64bar.json'
        sec_annot_ref_list = read_json(sec_annot_ref_fp)
        self.sec_annot_ref = ' '.join(sec_annot_ref_list)


        self.model_config = model_config

        self.save_hyperparameters()

    def forward(self, x, t):
        # x: (batch, seq_len, dim)
        return self.model(x, t)

    def training_step(self, batch, batch_idx):
        latent, seq_len, inst_conf, sec_annot = batch  # [bs, 4, 512]

        # print(latent.shape)
        # print(seq_len)

        bs, l, dim = latent.shape
        noise = torch.randn_like(latent)
        # timesteps = torch.randint(0, self.diffusion_steps-1, (latent.shape[0],)).long().to(latent.device)
        timesteps = (
            torch.randint(0, self.diffusion_steps, (latent.shape[0],))
            .long()
            .to(latent.device)
        )
        noisy_x = self.noise_scheduler.add_noise(latent, noise, timesteps)

        # Length embedding
        if self.length_embedding is not None:
            if self.length_bucket_size is not None:  # Length bucketing
                n_bar = seq_len // 4
                length_buckets = (n_bar / self.length_bucket_size).long()
            length_embeds = self.length_embedding(length_buckets).unsqueeze(
                1
            )  # (bs, 1, dim)
        else:
            length_embeds = torch.zeros(bs, 1, 128).to(noisy_x.device)  # dummy variable

        # Instrumentation configuration embedding
        if self.inst_conf_encoder is not None:
            structure_embeds, attn_mask = self.inst_conf_encoder(
                inst_conf
            )  # (bs, max_len, dim)
        elif self.sec_encoder is not None:
            structure_embeds, attn_mask = self.sec_encoder(
                sec_annot
            )  # (bs, max_len, dim)
        else:
            structure_embeds = torch.zeros(
                bs, 7, self.model_config["cross_attention_dim"]
            ).to(noisy_x.device)
            attn_mask = None

        # Get the model prediction
        pred = self.model(
            noisy_x.transpose(1, 2),
            timesteps,
            encoder_hidden_states=structure_embeds,  # inst conf condition
            global_hidden_states=length_embeds,  # length condition
            encoder_attention_mask=attn_mask,
        )["sample"].transpose(1, 2)
        # Note that we pass in the labels y

        # Calculate the loss, but only for valid positions
        # mask = torch.arange(latent.size(1))[None, :].to(seq_len.device) < seq_len[:, None]
        # pred = pred[mask]
        # noise = noise[mask]

        loss = self.loss_fn(pred, noise)  # How close is the output to the noise

        self.log("train_loss", loss, batch_size=bs, prog_bar=True)

        # # Debug: note down first 10 samples
        # latents = latent[0:10].detach() # [10, seq_len, dim]
        # out = self.vae.decode_batch(latents)
        # save_dir = '/data1/longshen/Results/AccGenResults/test_outputs/dataset/phrldm_dataset'
        # from sonata_utils import create_dir_if_not_exist
        # from remi_z import MultiTrack
        # create_dir_if_not_exist(save_dir)

        # out = self.vae.decode_batch(latents, bar_sep='\n')
        # with open(jpath(save_dir, f"train_step_decoded_samples.txt"), "w") as f:
        #     for i, s in enumerate(out):
        #         f.write(f"=== Sample {i} ===\n")
        #         f.write(s + "\n\n")

        # out = self.vae.decode_batch(latents)
        # for i in range(10):
        #     mt = MultiTrack.from_remiz_str(out[i], remove_repeated_eob=True)
        #     midi_fp = jpath(save_dir, f"train_step_decoded_sample_{i}.mid")
        #     mt.to_midi(midi_fp)
        # exit(10)

        # exit(10)

        return loss

    def validation_step(self, batch, batch_idx):
        self.eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self.eval_step(batch, batch_idx, "test")

    def eval_step(self, batch, batch_idx, prefix):
        latent, seq_len, inst_conf, sec_annot = batch

        bs, l, dim = latent.shape
        noise = torch.randn_like(latent)
        # timesteps = torch.randint(0, self.diffusion_steps-1, (latent.shape[0],)).long().to(latent.device)
        timesteps = (
            torch.randint(0, self.diffusion_steps, (latent.shape[0],))
            .long()
            .to(latent.device)
        )
        noisy_x = self.noise_scheduler.add_noise(latent, noise, timesteps)

        # Length embedding
        if self.length_embedding is not None:
            if self.length_bucket_size is not None:  # Length bucketing
                n_bar = seq_len // 4
                length_buckets = (n_bar / self.length_bucket_size).long()
            length_embeds = self.length_embedding(length_buckets).unsqueeze(
                1
            )  # (bs, 1, dim)
        else:
            length_embeds = torch.zeros(bs, 1, 128).to(noisy_x.device)  # dummy variable

        # Instrumentation configuration embedding
        if self.inst_conf_encoder is not None:
            structure_embeds, attn_mask = self.inst_conf_encoder(
                inst_conf
            )  # (bs, max_len, dim)
        elif self.sec_encoder is not None:
            structure_embeds, attn_mask = self.sec_encoder(
                sec_annot
            )  # (bs, max_len, dim)
        else:
            structure_embeds = torch.zeros(
                bs, 7, self.model_config["cross_attention_dim"]
            ).to(noisy_x.device)
            attn_mask = None

        # Forward pass. NOTE: Transpose to (bs, dim, seq_len) for StableAudioDiTModel
        pred = self.model(
            noisy_x.transpose(1, 2),
            timesteps,
            encoder_hidden_states=structure_embeds,  # inst conf condition
            global_hidden_states=length_embeds,  # length condition
            encoder_attention_mask=attn_mask,
        )["sample"].transpose(1, 2)

        # # Add mask for loss
        # mask = torch.arange(latent.size(1))[None, :].to(seq_len.device) < seq_len[:, None]
        # pred = pred[mask]
        # noise = noise[mask]

        loss = self.loss_fn(pred, noise)
        self.log(f"{prefix}_loss", loss, batch_size=bs)

        # Generate some samples for FID and diversity evaluation
        debug_n_samples = self.n_gen_in_valid
        if batch_idx == 0:
            max_pos = self.max_pos
            n_samples = debug_n_samples
            x = torch.randn(n_samples, max_pos, self.input_dim).to(latent.device)
            ae: PhraseVAE = self.vae
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.diffusion_steps,
                beta_schedule="squaredcos_cap_v2",
                clip_sample=True,
                clip_sample_range=self.clip_range,
            )

            # Length embedding for sampling
            if self.length_embedding is not None:

                # Fixed length condition, 64 bars
                len_cond = torch.full(
                    size=[
                        n_samples,
                    ],
                    fill_value=64 * 4,
                )

                sample_len = torch.tensor(len_cond).to(latent.device)  # e.g., 64 bars
                if self.length_bucket_size is not None:  # Length bucketing
                    n_bar = sample_len // 4
                    length_buckets = (n_bar / self.length_bucket_size).long()
                length_embeds = self.length_embedding(length_buckets).unsqueeze(
                    1
                )  # (n_samples, 1, dim)
            else:
                length_embeds = torch.zeros(n_samples, 1, 128).to(x.device)

            # Instrumentation layout condition
            if self.inst_conf_encoder is not None:
                inst_conf_cond = self.inst_conf_ref_flatten  # list of str
                batch_inst_conf_cond = [inst_conf_cond for _ in range(n_samples)]
                structure_embeds, attn_mask = self.inst_conf_encoder(
                    batch_inst_conf_cond
                )
            elif self.sec_encoder is not None:
                sec_ref = self.sec_annot_ref  # str
                batch_sec_cond = [sec_ref for _ in range(n_samples)]
                structure_embeds, attn_mask = self.sec_encoder(
                    batch_sec_cond
                )
            else:
                structure_embeds = torch.zeros(
                    n_samples, 7, self.model_config["cross_attention_dim"]
                ).to(
                    x.device
                )  # dummy variable, use for conditioning
                attn_mask = None

            # Sampling loop
            # for i, t in enumerate(tqdm(noise_scheduler.timesteps)):
            for i, t in enumerate(noise_scheduler.timesteps):
                x = x.to(latent.device)
                t = t.to(latent.device)
                # broadcast t to (bs,)
                t = t.expand(x.size(0)).to(x.device)
                with torch.no_grad():
                    residual = self.model(
                        x.transpose(1, 2),
                        t,
                        encoder_hidden_states=structure_embeds,  # inst conf condition
                        global_hidden_states=length_embeds,  # length condition
                        encoder_attention_mask=attn_mask,
                    )["sample"].transpose(1, 2)

                # Update sample with step
                x = noise_scheduler.step(residual, t[0], x).prev_sample
            # x shape: (bs=3, max_pos=17, dim)

            # Scale to VAE latent space
            x = x * self.scale_factor  # [n_samples, max_pos, dim]

            """ --- Debugging --- """
            if self.overfit_debug:
                # Calculate mse between x and gt_latent
                gt_latent = torch.load(self.gt_latent_fp).to(x.device)
                # print('GT latent ft', self.gt_latent_fp)
                # print('GT latent shape', gt_latent.shape )
                x_0 = x[0]
                tgt_len = gt_latent.shape[0]
                x_0 = x_0[:tgt_len, :]
                # print('x_0 shape:', x_0.shape)
                # exit(0)
                mse = nn.MSELoss()(x_0, gt_latent)
                self.log(f"{prefix}_sample_mse", mse, batch_size=1)

                # Save the defussed latent
                save_dir = self.debug_temp_dir
                save_fp = jpath(save_dir, f"model_out.png")
                import matplotlib.pyplot as plt

                plt.figure(figsize=(10, 4))
                plt.scatter(range(x.shape[2]), x[0, 0, :].cpu().numpy(), s=10)
                plt.title("First bar latent of first sample")
                plt.xlabel("Latent Dimension")
                plt.ylabel("Latent Value")
                plt.grid(True, linestyle="--", alpha=0.5)
                plt.savefig(save_fp)

                torch.save(x, jpath(save_dir, f"model_out.pt"))

                # Save output midi
                out_str = self.vae.decode_song(x_0)
                out_mt = MultiTrack.from_remiz_str(out_str, verbose=False)
                midi_fp = jpath(save_dir, f"model_out.mid")
                save_midi(out_mt, midi_fp)

                # Save output str
                out_str = self.vae.decode_song(x_0, early_stop=False)
                with open(jpath(save_dir, f"model_out.txt"), "w") as f:
                    f.write(out_str)

            """ --- End of Debugging --- """

            # Decode the generated latents
            out_str_list = ae.decode_batch(x)  # [n_samples,]

            # Get n_bar of each generated sample
            n_bars = []
            sample_lens = []

            out_mts = []
            for i in range(n_samples):
                mt = MultiTrack.from_remiz_str(out_str_list[i], verbose=False)
                n_bars.append(len(mt))
                sample_lens.append(len(mt) * 4)  # in position latent length
                out_mts.append(mt)

            # Calculate FID against eval set
            eval_latent_fp = self.eval_latent_fp
            eval_data = torch.load(
                eval_latent_fp, map_location=latent.device
            )  # list of latents
            eval_latents = [
                eval_data[k]["latent"] for k in eval_data
            ]  # list of [seq_len, dim]
            # print(eval_latents[0].shape)
            fid = compute_fid_gt_vs_out(
                eval_set=eval_latents,
                out_set=x,
                out_lens=sample_lens,
            )
            # print('FID of generated samples against eval set:', fid)
            self.log(f"{prefix}_sample_fid", fid, batch_size=n_samples)

            # Bright ratio
            bright_ratio_list = []
            for i in range(n_samples):
                z = x[i, : sample_lens[i], :]  # [valid_len, dim]
                br = bar_recurrence_score_diag(z, threshold=0.5, min_run_length=4)
                # br = bright_ratio(z, 0.5)
                bright_ratio_list.append(br)
            # print(f"Bright ratio of sample {i}:", br)
            srs = sum(bright_ratio_list) / len(bright_ratio_list)
            # print(f'Average bright ratio of generated samples:', bright_ratio_avg)
            self.log(f"{prefix}_sample_bright_ratio", srs, batch_size=n_samples)

            self.log_ssm_matplotlib(x[0, : sample_lens[0], :], prefix, self.global_step)

            # # Melody memorization rate
            # mem_rate = self.metric.memorization_rate_mt_batch(out_mts)
            # self.log(f"{prefix}_mel_memorization_rate", mem_rate, batch_size=n_samples)

            # # New sample rate and average top-2 ratio
            # new_sample_rate, top2_ratio, _ = self.metric.new_sample_rate_batch(out_mts)
            # self.log(f"{prefix}_mel_top2_ratio", top2_ratio, batch_size=n_samples)
            # self.log(f"{prefix}_mel_new_sample_rate", new_sample_rate, batch_size=n_samples)

            # 3 memorization metrics: Mel mem rate, top2 ratio, new sample rate
            mem_rate_metrics = self.metric.calculate_memorization_metrics_batch(out_mts)
            for k in mem_rate_metrics:
                if "avg" in k:
                    self.log(f"{prefix}_{k}", mem_rate_metrics[k], batch_size=n_samples)

            # Redo the MIDI conversion (previously only melody track was kept)
            out_mts = [
                MultiTrack.from_remiz_str(s, verbose=False) for s in out_str_list
            ]

            # Save MIDI to output dir
            midi_save_dir = jpath(
                self.logger.log_dir, f"{prefix}_gen_midis", f"step_{self.global_step}"
            )
            out_str_dir = jpath(midi_save_dir, "out_strs")
            create_dir_if_not_exist(midi_save_dir)
            create_dir_if_not_exist(out_str_dir)
            for i in range(n_samples):
                out_mt = out_mts[i]
                n_bar = len(out_mt)
                midi_fp = jpath(
                    midi_save_dir,
                    f"sample_{i}_bars_{n_bar}.mid",
                )
                save_midi(out_mt, midi_fp)

                # Save the output str as well
                out_str = out_str_list[i]
                out_seq = out_str.split()
                # Find all 'b-1' location
                bar_line_idx = [
                    idx for idx, token in enumerate(out_seq) if token == "b-1"
                ]
                # get bar seq
                bar_seqs = []
                prev_idx = 0
                for bar_idx in bar_line_idx:
                    bar_seq = out_seq[prev_idx : bar_idx + 1]
                    bar_seqs.append(" ".join(bar_seq))
                    prev_idx = bar_idx + 1
                out_str = "\n".join(bar_seqs)

                out_str_save_fp = jpath(
                    out_str_dir,
                    f"sample_{i}_bars_{n_bar}.txt",
                )
                with open(out_str_save_fp, "w") as f:
                    f.write(out_str)

        return loss

    def log_ssm_matplotlib(self, z, prefix: str, global_step: int):
        """
        z: Tensor [T, D], phrase-level latents (single song)
        prefix: "val" or "test"
        """

        # ----- compute SSM -----
        ssm = compute_bar_ssm_from_phrase_latents(z)  # [B, B]
        ssm_np = ssm.float().detach().cpu().numpy()

        # ----- draw with matplotlib -----
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(ssm_np, cmap="viridis", origin="lower")
        fig.colorbar(im, ax=ax, label="cosine similarity")

        ax.set_title(f"SSM ({prefix})")
        ax.set_xlabel("Bar index")
        ax.set_ylabel("Bar index")

        # ----- convert matplotlib figure → image array -----
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=200, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)

        # load as Pillow image → numpy array
        img = Image.open(buf)
        img = np.array(img)  # shape [H, W, 3]

        # ----- log to TensorBoard -----
        # TensorBoard expects CHW
        img_tensor = torch.tensor(img).permute(2, 0, 1) / 255.0
        writer = self.logger.experiment
        writer.add_image(f"{prefix}/SSM_matplotlib", img_tensor, global_step)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, eps=self.eps)
        return optimizer

    def load_ldm_weights(self, lit_ckpt_fp):
        """
        Load weights from a Lightning checkpoint file
        """
        ckpt = torch.load(lit_ckpt_fp, map_location="cpu")
        state_dict = ckpt["state_dict"]
        model_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_k = k[len("model.") :]
                model_state_dict[new_k] = v
        self.model.load_state_dict(model_state_dict, strict=True)
        print(f"Loaded LDM weights from {lit_ckpt_fp}")


class LitBarLDM(pl.LightningModule):
    """
    A 1-D Diffusion Transformer model that generates variable-length latent sequences
    """

    def __init__(
        self,
        model_config,
        scale_factor=1.0,
        mode=None,
        lr=1e-4,
        diffusion_steps=1000,
        overfit_debug=False,
        debug_temp_dir=None,
        from_pretrained_ldm_fp=None,
        length_condition=False,
        length_bucket_size=10,
        eval_latent_fp=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        assert mode in ["pos", "bar", "phr"], "mode should be 'pos' or 'bar' or 'phr'"
        self.latent_mode = mode
        print("Initializing LitPhraseLDM with mode:", mode)

        # Set up model
        self.model = UnconditionalDiT(**model_config)
        input_dim = model_config["in_channels"]
        max_pos = model_config["sample_size"]
        self.max_pos = max_pos
        self.input_dim = input_dim
        self.lr = lr

        # Load pretrained weights if specified
        if from_pretrained_ldm_fp is not None:
            self.load_ldm_weights(from_pretrained_ldm_fp)

        # Noise scheduler
        self.diffusion_steps = diffusion_steps
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=diffusion_steps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            clip_sample_range=3.241875410079956,
        )
        # print(noise_scheduler.config)
        self.noise_scheduler = noise_scheduler

        # print(noise_scheduler.config.prediction_type)
        # exit()

        self.loss_fn = nn.MSELoss()

        dim_mean_fp = "/data1/longshen/Datasets/Piano/POP909/latents/bar_level_pos_seq/train_dim_mean.pt"
        dim_std_fp = "/data1/longshen/Datasets/Piano/POP909/latents/bar_level_pos_seq/train_dim_std.pt"
        self.dim_mean = torch.load(dim_mean_fp, map_location="cpu")  # [128]
        self.dim_std = torch.load(dim_std_fp, map_location="cpu")  # [128]

        # Set up VAE
        if self.latent_mode == "pos":
            self.vae = MQVAE_Pos()
        elif self.latent_mode == "bar":
            self.vae = BarVAE()
        elif self.latent_mode == "phr":
            self.vae = PhraseVAE()
        self.scale_factor = scale_factor
        print("Model scale factor:", self.scale_factor)

        # For debug
        self.overfit_debug = overfit_debug
        if overfit_debug:
            assert debug_temp_dir is not None, "debug_temp_dir cannot be None"
            self.debug_temp_dir = debug_temp_dir
            self.gt_latent_fp = jpath(
                self.debug_temp_dir, "debug_first_sample_latent_scaled.pt"
            )

        # Length encoder
        if length_condition:
            self.length_embedding = nn.Embedding(
                128, model_config["global_states_input_dim"]
            )
            self.length_bucket_size = length_bucket_size
        else:
            self.length_embedding = None
            self.length_bucket_size = None

        mel_dict_fp = (
            "/data1/longshen/Datasets/Piano/POP909/remi_z/melody_remiz_dict.json"
        )
        self.mel_dict = read_json(mel_dict_fp)

        self.metric = Metric()

        assert eval_latent_fp is not None, "eval_latent_fp cannot be None"
        self.eval_latent_fp = eval_latent_fp
        # phrase LDM: "/data1/longshen/Datasets/Piano/POP909/latents/song_level_phr_seq_with_annot/eval_data.pt"

        self.save_hyperparameters()

    def forward(self, x, t):
        # x: (batch, seq_len, dim)
        return self.model(x, t)

    def training_step(self, batch, batch_idx):
        latent, seq_len = batch  # [bs, 4, 512]

        bs, l, dim = latent.shape
        noise = torch.randn_like(latent)
        # timesteps = torch.randint(0, self.diffusion_steps-1, (latent.shape[0],)).long().to(latent.device)
        timesteps = (
            torch.randint(0, self.diffusion_steps, (latent.shape[0],))
            .long()
            .to(latent.device)
        )
        noisy_x = self.noise_scheduler.add_noise(latent, noise, timesteps)
        # print('shapes:', noisy_x.shape, timesteps.shape)
        # exit(10)

        # Length embedding
        if self.length_embedding is not None:
            if self.length_bucket_size is not None:  # Length bucketing
                n_bar = seq_len
                length_buckets = (n_bar / self.length_bucket_size).long()
            length_embeds = self.length_embedding(length_buckets).unsqueeze(
                1
            )  # (bs, 1, dim)
        else:
            length_embeds = torch.zeros(bs, 1, 128).to(noisy_x.device)  # dummy variable

        # Get the model prediction
        pred = self.model(
            noisy_x.transpose(1, 2),
            timesteps,
            encoder_hidden_states=torch.zeros(bs, 7, 128).to(
                noisy_x.device
            ),  # dummy variable, use for conditioning
            global_hidden_states=length_embeds,  # length condition
        )["sample"].transpose(1, 2)
        # Note that we pass in the labels y

        # print("model_out max/min:", pred.max(), pred.min())

        # Calculate the loss, but only for valid positions
        # mask = torch.arange(latent.size(1))[None, :].to(seq_len.device) < seq_len[:, None]
        # pred = pred[mask]
        # noise = noise[mask]

        loss = self.loss_fn(pred, noise)  # How close is the output to the noise

        self.log("train_loss", loss, batch_size=bs, prog_bar=True)

        # # Debug: note down first 10 samples
        # latents = latent[0:10].detach()
        # out = []
        # for i in range(10):
        #     out_str = self.vae.decode_song(latents[i])
        #     out.append(out_str)
        # save_dir = '/data1/longshen/Results/AccGenResults/test_outputs/dataset/barldm_dataset'
        # from sonata_utils import create_dir_if_not_exist
        # from remi_z import MultiTrack
        # create_dir_if_not_exist(save_dir)
        # with open(jpath(save_dir, f"train_step_decoded_samples.txt"), "w") as f:
        #     for i, s in enumerate(out):
        #         f.write(f"=== Sample {i} ===\n")
        #         f.write(s + "\n\n")
        # for i in range(10):
        #     out_str = out[i]

        #     mt = MultiTrack.from_remiz_str(out[i], remove_repeated_eob=True)
        #     midi_fp = jpath(save_dir, f"train_step_decoded_sample_{i}.mid")
        #     mt.to_midi(midi_fp)
        # exit(10)

        return loss

    def validation_step(self, batch, batch_idx):
        self.eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self.eval_step(batch, batch_idx, "test")

    def eval_step(self, batch, batch_idx, prefix):
        latent, seq_len = batch

        bs, l, dim = latent.shape
        noise = torch.randn_like(latent)
        # timesteps = torch.randint(0, self.diffusion_steps-1, (latent.shape[0],)).long().to(latent.device)
        timesteps = (
            torch.randint(0, self.diffusion_steps, (latent.shape[0],))
            .long()
            .to(latent.device)
        )
        noisy_x = self.noise_scheduler.add_noise(latent, noise, timesteps)

        # Length embedding
        if self.length_embedding is not None:
            if self.length_bucket_size is not None:  # Length bucketing
                n_bar = seq_len // 4
                length_buckets = (n_bar / self.length_bucket_size).long()
            length_embeds = self.length_embedding(length_buckets).unsqueeze(
                1
            )  # (bs, 1, dim)
        else:
            length_embeds = torch.zeros(bs, 1, 128).to(noisy_x.device)  # dummy variable

        # transpose to (bs, dim, seq_len) for StableAudioDiTModel
        # print(noisy_x.shape)
        pred = self.model(
            noisy_x.transpose(1, 2),
            timesteps,
            encoder_hidden_states=torch.zeros(bs, 7, 128).to(
                noisy_x.device
            ),  # dummy variable, use for conditioning
            global_hidden_states=length_embeds,  # length condition
        )["sample"].transpose(1, 2)

        # # Add mask
        # mask = torch.arange(latent.size(1))[None, :].to(seq_len.device) < seq_len[:, None]
        # pred = pred[mask]
        # noise = noise[mask]

        loss = self.loss_fn(pred, noise)
        self.log(f"{prefix}_loss", loss, batch_size=bs)

        # # Generate 1 samples for visualization
        # if batch_idx == 0:
        #     max_pos = self.max_pos
        #     x = torch.randn(1, max_pos, self.input_dim).to(latent.device)
        #     ae = self.vae
        #     noise_scheduler = DDPMScheduler(
        #         num_train_timesteps=self.diffusion_steps,
        #         beta_schedule="squaredcos_cap_v2",
        #     )

        #     # Length embedding for sampling
        #     if self.length_embedding is not None:
        #         sample_len = torch.tensor(64 * 4).to(latent.device)  # e.g., 64 bars
        #         if self.length_bucket_size is not None: # Length bucketing
        #             n_bar = sample_len // 4
        #             length_buckets = (n_bar / self.length_bucket_size).long()
        #         length_embeds = self.length_embedding(length_buckets).unsqueeze(0)  # (1, 1, dim)
        #     else:
        #         length_embeds = torch.zeros(1, 1, 128).to(x.device)

        #     # Sampling loop
        #     for i, t in enumerate(noise_scheduler.timesteps):
        #         x = x.to(latent.device)
        #         t = t.to(latent.device)
        #         # broadcast t to (bs,)
        #         t = t.expand(x.size(0)).to(x.device)
        #         with torch.no_grad():
        #             residual = self.model(
        #                 x.transpose(1, 2),
        #                 t,
        #                 encoder_hidden_states=torch.zeros(1, 7, 128).to(
        #                     x.device
        #                 ),  # dummy variable, use for conditioning
        #                 global_hidden_states=length_embeds,  # length condition
        #             )["sample"].transpose(1, 2)

        #         # Update sample with step
        #         x = noise_scheduler.step(residual, t[0], x).prev_sample
        #     # x shape: (bs=3, max_pos=17, dim)

        #     """ --- End of Debugging --- """

        #     # Scale
        #     x = x * self.scale_factor

        #     # Debug for decomposed model
        #     if self.latent_mode == "bar":
        #         n_bars = max_pos // 4
        #         x = x.reshape(1, n_bars, -1)
        #         # x = x.reshape(1, 4, -1)

        #     out_str_list = ae.decode_batch(x)
        #     out_str = " <SEP> ".join(out_str_list)

        #     # log to tensorboard
        #     self.logger.experiment.add_text(
        #         f"{prefix}_sample", out_str, self.global_step
        #     )

        # Generate 100 samples for FID and diversity evaluation
        debug_n_samples = 1
        if batch_idx == 0:
            max_pos = self.max_pos
            n_samples = debug_n_samples
            x = torch.randn(n_samples, max_pos, self.input_dim).to(latent.device)
            ae = self.vae
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.diffusion_steps,
                beta_schedule="squaredcos_cap_v2",
                clip_sample=True,
                clip_sample_range=3.241875410079956,
            )

            # Length embedding for sampling
            if self.length_embedding is not None:

                # # Random length condition [1, 512]
                # min_n_bar = 16
                # max_n_bar = 128
                # min_seq_len = min_n_bar * 4
                # max_seq_len = max_n_bar * 4
                # len_cond = torch.randint(
                #     low=min_seq_len,
                #     high=max_seq_len + 1,
                #     size=[
                #         n_samples,
                #     ],
                # )  # [n_samples,]

                # Fixed length condition, 64 bars
                len_cond = torch.full(
                    size=[
                        n_samples,
                    ],
                    fill_value=64 * 4,
                )

                sample_len = torch.tensor(len_cond).to(latent.device)  # e.g., 64 bars
                if self.length_bucket_size is not None:  # Length bucketing
                    n_bar = sample_len // 4
                    length_buckets = (n_bar / self.length_bucket_size).long()
                length_embeds = self.length_embedding(length_buckets).unsqueeze(
                    1
                )  # (n_samples, 1, dim)
                # print("Length condition for FID samples:", len_cond)
            else:
                length_embeds = torch.zeros(n_samples, 1, 128).to(x.device)

            # Sampling loop
            # for i, t in enumerate(tqdm(noise_scheduler.timesteps)):
            for i, t in enumerate(noise_scheduler.timesteps):
                x = x.to(latent.device)
                t = t.to(latent.device)
                # broadcast t to (bs,)
                t = t.expand(x.size(0)).to(x.device)
                with torch.no_grad():
                    residual = self.model(
                        x.transpose(1, 2),
                        t,
                        encoder_hidden_states=torch.zeros(n_samples, 7, 128).to(
                            x.device
                        ),  # dummy variable, use for conditioning
                        global_hidden_states=length_embeds,  # length condition
                    )["sample"].transpose(1, 2)

                # Update sample with step
                x = noise_scheduler.step(residual, t[0], x).prev_sample
            # x shape: (bs=3, max_pos=17, dim)

            """ --- Debugging --- """
            if self.overfit_debug:
                # Calculate mse between x and gt_latent
                gt_latent = torch.load(self.gt_latent_fp).to(x.device)
                # print('GT latent ft', self.gt_latent_fp)
                # print('GT latent shape', gt_latent.shape )
                x_0 = x[0]
                tgt_len = gt_latent.shape[0]
                x_0 = x_0[:tgt_len, :]
                # print('x_0 shape:', x_0.shape)
                # exit(0)
                mse = nn.MSELoss()(x_0, gt_latent)
                self.log(f"{prefix}_sample_mse", mse, batch_size=1)

                # Save the defussed latent
                save_dir = self.debug_temp_dir
                save_fp = jpath(save_dir, f"model_out.png")
                import matplotlib.pyplot as plt

                plt.figure(figsize=(10, 4))
                plt.scatter(range(x.shape[2]), x[0, 0, :].cpu().numpy(), s=10)
                plt.title("First bar latent of first sample")
                plt.xlabel("Latent Dimension")
                plt.ylabel("Latent Value")
                plt.grid(True, linestyle="--", alpha=0.5)
                plt.savefig(save_fp)

                torch.save(x, jpath(save_dir, f"model_out.pt"))

                # Save output midi
                out_str = self.vae.decode_song(x_0)
                out_mt = MultiTrack.from_remiz_str(out_str, verbose=False)
                midi_fp = jpath(save_dir, f"model_out.mid")
                save_midi(out_mt, midi_fp)

                # Save output str
                out_str = self.vae.decode_song(x_0, bar_sep="\n", early_stop=False)
                with open(jpath(save_dir, f"model_out.txt"), "w") as f:
                    f.write(out_str)

            """ --- End of Debugging --- """

            # Scale to VAE latent space
            x = x * self.scale_factor  # [n_samples, max_pos, dim]
            out_str_list = ae.decode_batch(x)  # [n_samples,]

            # Get n_bar of each generated sample
            n_bars = []
            sample_lens = []

            out_mts = []
            for i in range(n_samples):
                mt = MultiTrack.from_remiz_str(
                    out_str_list[i], remove_repeated_eob=True, verbose=False
                )
                n_bars.append(len(mt))
                sample_lens.append(len(mt) * 4)  # in position latent length
                out_mts.append(mt)

            # # # Debug for decomposed model
            # # if self.latent_mode == "bar":
            # #     n_bars = max_pos // 4
            # #     x = x.reshape(1, n_bars, -1)
            #     # x = x.reshape(1, 4, -1)

            # Calculate diversity: pairwise MSE
            # x: [n_samples, seq_len, dim]
            # print(sample_lens)
            # div = compute_diversity(x, sample_lens)
            # print('Diversity of generated samples:', div)
            # self.log(f"{prefix}_sample_diversity", div, batch_size=n_samples)

            # Calculate FID against eval set
            eval_latent_fp = self.eval_latent_fp
            eval_data = torch.load(
                eval_latent_fp, map_location=latent.device
            )  # list of latents
            eval_latents = [
                eval_data[k]["latent"] for k in eval_data
            ]  # list of [seq_len, dim]
            # print(eval_latents[0].shape)

            # To be debug
            # fid = compute_fid_gt_vs_out(
            #     eval_set=eval_latents,
            #     out_set=x,
            #     out_lens=sample_lens,
            # )
            # # print('FID of generated samples against eval set:', fid)
            # self.log(f"{prefix}_sample_fid", fid, batch_size=n_samples)

            # all_fid = compute_all_fids_gt_vs_out(
            #     eval_set=eval_latents,
            #     out_set=x,
            #     out_lens=sample_lens,
            # )
            # for k in all_fid:
            #     # print(f"{k} of generated samples against eval set:", all_fid[k])
            #     self.log(f"{prefix}_sample_{k}", all_fid[k], batch_size=n_samples)

            # # Bright ratio
            # bright_ratio_list = []
            # for i in range(n_samples):
            #     z = x[i, : sample_lens[i], :]  # [valid_len, dim]
            #     br = bar_recurrence_score_diag(z, threshold=0.5, min_run_length=4)
            #     # br = bright_ratio(z, 0.5)
            #     bright_ratio_list.append(br)
            # # print(f"Bright ratio of sample {i}:", br)
            # srs = sum(bright_ratio_list) / len(bright_ratio_list)
            # # print(f'Average bright ratio of generated samples:', bright_ratio_avg)
            # self.log(
            #     f"{prefix}_sample_bright_ratio", srs, batch_size=n_samples
            # )

            # self.log_ssm_matplotlib(x[0, : sample_lens[0], :], prefix, self.global_step)

            # Melody memorization rate
            mem_rate = self.metric.memorization_rate_mt_batch(out_mts)
            self.log(f"{prefix}_mel_memorization_rate", mem_rate, batch_size=n_samples)

            # New sample rate and average top-2 ratio
            new_sample_rate, top2_ratio, _ = self.metric.new_sample_rate_batch(out_mts)
            self.log(f"{prefix}_mel_top2_ratio", top2_ratio, batch_size=n_samples)
            self.log(
                f"{prefix}_mel_new_sample_rate", new_sample_rate, batch_size=n_samples
            )

        return loss

    def log_ssm_matplotlib(self, z, prefix: str, global_step: int):
        """
        z: Tensor [T, D], phrase-level latents (single song)
        prefix: "val" or "test"
        """

        # ----- compute SSM -----
        ssm = compute_bar_ssm_from_phrase_latents(z)  # [B, B]
        ssm_np = ssm.detach().cpu().numpy()

        # ----- draw with matplotlib -----
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(ssm_np, cmap="viridis", origin="lower")
        fig.colorbar(im, ax=ax, label="cosine similarity")

        ax.set_title(f"SSM ({prefix})")
        ax.set_xlabel("Bar index")
        ax.set_ylabel("Bar index")

        # ----- convert matplotlib figure → image array -----
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=200, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)

        # load as Pillow image → numpy array
        img = Image.open(buf)
        img = np.array(img)  # shape [H, W, 3]

        # ----- log to TensorBoard -----
        # TensorBoard expects CHW
        img_tensor = torch.tensor(img).permute(2, 0, 1) / 255.0
        writer = self.logger.experiment
        writer.add_image(f"{prefix}/SSM_matplotlib", img_tensor, global_step)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        return optimizer

    def load_ldm_weights(self, lit_ckpt_fp):
        """
        Load weights from a Lightning checkpoint file
        """
        ckpt = torch.load(lit_ckpt_fp, map_location="cpu")
        state_dict = ckpt["state_dict"]
        model_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_k = k[len("model.") :]
                model_state_dict[new_k] = v
        self.model.load_state_dict(model_state_dict, strict=True)
        print(f"Loaded LDM weights from {lit_ckpt_fp}")
