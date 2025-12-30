import random
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sonata_utils import jpath
import matplotlib.pyplot as plt
import io
from PIL import Image
from torchvision.transforms import ToTensor
from models.phrase_vae import S2SVAE3
from remi_z import MultiTrack
from experiments.autoencoders.lit_aes import (
    calculate_metrics_for_prolls,
    log_piano_roll,
    calculate_f1_ipod,
    compute_barvae_f1_metrics,
)


class LitPhraseVAE(pl.LightningModule):
    def __init__(
        self,
        t5_model_name=None,
        tokenizer_path=None,
        recon_weight=1.0,
        kld_weight=1.0,
        l2_reg_weight=0.0,
        t5_config=None,
        lr=1e-4,
        weight_decay=1e-3,
        compress_style="full_sequence",  # 'full_sequence' or 'first_token'
        n_compress_tokens=0,
        lit_ckpt=False,
        adaptive_scale=False,
        bottleneck_dim=None,
        vae_type=None,
    ):
        super().__init__()

        # if vae_type is None:
        #     vae_type = "S2SVAE2" if bottleneck_dim is not None else "S2SVAE"
        assert vae_type in [
            "S2SVAE",
            "S2SVAE2",
            "S2SVAE3",
        ], f"Unsupported VAE type: {vae_type}"
        print(f"Initializing LitPhraseVAE with VAE type: {vae_type}")

        if vae_type == "S2SVAE":
            raise NotImplementedError("S2SVAE is deprecated. Please use S2SVAE3.")
        elif vae_type == "S2SVAE2":
            raise NotImplementedError("S2SVAE2 is deprecated. Please use S2SVAE3.")
        elif vae_type == "S2SVAE3":
            self.model = S2SVAE3(
                t5_model_name=t5_model_name,
                tokenizer_path=tokenizer_path,
                recon_weight=recon_weight,
                l2_reg_weight=l2_reg_weight,
                kld_weight=kld_weight,
                t5_config=t5_config,
                compress_style=compress_style,  # 'full_sequence' or 'first_token'
                n_compress_tokens=n_compress_tokens,
                lit_ckpt=lit_ckpt,
                adaptive_scale=adaptive_scale,
                bottleneck_dim=bottleneck_dim,
            )
        else:
            raise ValueError(f"Unsupported VAE type: {vae_type}")

        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        labels=None,
        **kwargs,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            **kwargs,
        )

    def training_step(self, batch, batch_idx):
        # # Debug
        # for k in batch:
        #     for i in range(3):
        #         print(k, batch[k][i])
        # exit(10)

        input_ids = batch["input_ids"]
        bs = input_ids.size(0)
        attention_mask = batch.get("attention_mask", None)
        labels = batch.get("labels", None)
        outputs = self(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

        loss = outputs.loss
        self.log("train_loss", loss, batch_size=bs)

        recon_loss = outputs.recon_loss
        self.log("train_recon_loss", recon_loss, prog_bar=True, batch_size=bs)

        kld_loss = outputs.kld_loss
        self.log("train_kld_loss", kld_loss, prog_bar=True, batch_size=bs)

        latent_mean = outputs.latent_mean
        latent_std = outputs.latent_std
        self.log("train_latent_mean", latent_mean, batch_size=bs)
        self.log("train_latent_std", latent_std, batch_size=bs)

        # Log the largest absolute latent value in latent_
        max_abs_latent = torch.max(torch.abs(outputs.encoded_latent))
        self.log("train_max_abs_latent", max_abs_latent, batch_size=bs)

        # log adaptive scale factor
        if self.model.adaptive_scale:
            self.log("train_scale", self.model.scale, batch_size=bs)

        return loss

    def on_validation_start(self):
        self.val_latents = []

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)
        labels = batch.get("labels", None)
        bs = input_ids.size(0)

        # Validation loss
        outputs = self(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss
        self.log("val_loss", loss, batch_size=bs)

        recon_loss = outputs.recon_loss
        self.log("val_recon_loss", recon_loss, batch_size=bs)

        kld_loss = outputs.kld_loss
        self.log("val_kld_loss", kld_loss, batch_size=bs)

        latent_mean = outputs.latent_mean
        latent_std = outputs.latent_std
        self.log("val_latent_mean", latent_mean, batch_size=bs)
        self.log("val_latent_std", latent_std, batch_size=bs)

        # Autoregressive generation (greedy)
        samples_to_validate_per_batch = 8
        input_ids = input_ids[:samples_to_validate_per_batch]
        attention_mask = (
            attention_mask[:samples_to_validate_per_batch]
            if attention_mask is not None
            else None
        )

        encode_out = self.model.encode(
            input_ids, attention_mask=attention_mask, sample=False
        )
        max_length = input_ids.shape[1]
        generated_ids = self.model.decode(
            encode_out["latent"],
            max_length=max_length,
            pad_token_id=self.model.tokenizer.pad_token_id,
            eos_token_id=self.model.tokenizer.eos_token_id,
            do_sample=False,
            num_beams=1,
        )

        # Detokenize to REMI-z strs
        out_texts = self.model.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=False
        )
        labels_for_decode = labels.clone()
        labels_for_decode[labels_for_decode == -100] = self.model.tokenizer.pad_token_id
        tgt_texts = self.model.tokenizer.batch_decode(
            labels_for_decode, skip_special_tokens=False
        )
        input_texts = batch["masked_seq"]

        # Validate F1 metrics
        samples_to_validate = bs
        metrics = compute_barvae_f1_metrics(
            out_texts,
            tgt_texts,
            samples_to_validate_per_batch=samples_to_validate,
            pos_per_bar=48,
            verbose=False,
        )
        self.log("val_f1_op", metrics['f1_op'], prog_bar=False, batch_size=samples_to_validate)
        self.log("val_f1_opd", metrics['f1_opd'], prog_bar=True, batch_size=samples_to_validate)
        self.log("val_f1_iopd", metrics['f1_iopd'], prog_bar=False, batch_size=samples_to_validate)

        # Log to TensorBoard. Only log for the first batch to avoid flooding
        if batch_idx == 0:
            for i in range(min(3, len(out_texts))):
                self.logger.experiment.add_text(
                    f"val/generated_{i}",
                    f"Input: {input_texts[i]}\nTarget: {tgt_texts[i]}\nOutput: {out_texts[i]}",
                    self.global_step,
                )

            # Get piano roll
            tgt_mt = MultiTrack.from_remiz_str(tgt_texts[0], verbose=False)[0]
            tgt_proll = tgt_mt.to_piano_roll(pos_per_bar=48)
            try:
                out_mt = MultiTrack.from_remiz_str(out_texts[0], verbose=False)[0]
                out_proll = out_mt.to_piano_roll(pos_per_bar=48)
            except Exception as e:
                out_proll = np.zeros_like(tgt_proll)

            # Use reusable log_piano_roll function
            log_piano_roll(self.logger, tgt_proll, out_proll, self.global_step)

        # Store latents for epoch-end analysis
        self.val_latents.append(encode_out["latent"].detach().cpu())

    def on_validation_end(self):
        # Concatenate all latents
        all_latents = torch.cat(self.val_latents, dim=0)

        # Compute dim-wise mean and std
        dim_means = torch.mean(all_latents, dim=0)
        dim_stds = torch.std(all_latents, dim=0)

        # --- 3. Plot histogram (same style as your function) ---
        fig, ax = plt.subplots(figsize=(8, 4))

        values = dim_stds.detach().cpu().numpy()
        N = len(values)

        ax.hist(
            values,
            weights=np.ones(N) / N,        # normalized to percentage
            bins=50,
            density=False,
            alpha=0.7,
            color="blue",
            edgecolor="black",
        )

        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlabel("Dimension-wise Std Dev")
        ax.set_ylabel("Percentage")
        ax.set_title("Histogram of Dimension-wise Std Dev")
        ax.grid(True)

        # --- Fix X-axis range ---
        ax.set_xlim(0, 2)

        # --- 4. Add to TensorBoard ---
        self.logger.experiment.add_figure(
            "latent_dim_std_hist",
            fig,
            global_step=self.global_step,
        )

        plt.close(fig)

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        bs = input_ids.size(0)
        attention_mask = batch.get("attention_mask", None)
        labels = batch.get("labels", None)
        outputs = self(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss
        kld_loss = outputs.kld_loss
        recon_loss = outputs.recon_loss

        self.log("test_loss", loss, batch_size=bs)
        self.log("test_recon_loss", recon_loss, batch_size=bs)
        self.log("test_kld_loss", kld_loss, batch_size=bs)

        latent_mean = outputs.latent_mean
        latent_std = outputs.latent_std
        self.log("test_latent_mean", latent_mean, batch_size=bs)
        self.log("test_latent_std", latent_std, batch_size=bs)

        # Autoregressive generation (greedy)
        encode_out = self.model.encode(
            input_ids, attention_mask=attention_mask, sample=False
        )
        max_length = input_ids.shape[1]
        generated_ids = self.model.decode(
            encode_out["latent"],
            max_length=max_length,
            pad_token_id=self.model.tokenizer.pad_token_id,
            eos_token_id=self.model.tokenizer.eos_token_id,
            do_sample=False,
            num_beams=1,
        )

        # Detokenize to REMI-z strs
        out_texts = self.model.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=False
        )
        labels_for_decode = labels.clone()
        labels_for_decode[labels_for_decode == -100] = self.model.tokenizer.pad_token_id
        tgt_texts = self.model.tokenizer.batch_decode(
            labels_for_decode, skip_special_tokens=False
        )
        input_texts = batch["masked_seq"]

        # F1 metrics
        samples_to_validate = bs
        metrics = compute_barvae_f1_metrics(
            out_texts,
            tgt_texts,
            samples_to_validate_per_batch=samples_to_validate,
            pos_per_bar=48,
            verbose=False,
        )
        self.log("test_f1_op", metrics['f1_op'], prog_bar=False, batch_size=samples_to_validate)
        self.log("test_f1_opd", metrics['f1_opd'], prog_bar=True, batch_size=samples_to_validate)
        self.log('test_f1_iopd', metrics['f1_iopd'], prog_bar=False, batch_size=samples_to_validate)

        # Log to TensorBoard. Only log for the first batch to avoid flooding
        if batch_idx == 0:
            for i in range(min(3, len(out_texts))):
                self.logger.experiment.add_text(
                    f"val/generated_{i}",
                    f"Input: {input_texts[i]}\nTarget: {tgt_texts[i]}\nOutput: {out_texts[i]}",
                    self.global_step,
                )

            # Get piano roll
            tgt_mt = MultiTrack.from_remiz_str(tgt_texts[0], verbose=False)[0]
            tgt_proll = tgt_mt.to_piano_roll(pos_per_bar=48)
            try:
                out_mt = MultiTrack.from_remiz_str(out_texts[0], verbose=False)[0]
                out_proll = out_mt.to_piano_roll(pos_per_bar=48)
            except Exception as e:
                out_proll = np.zeros_like(tgt_proll)

            # Use reusable log_piano_roll function
            log_piano_roll(self.logger, tgt_proll, out_proll, self.global_step)

    def configure_optimizers(self):
        # Single optimizer, AdamW
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        return opt
