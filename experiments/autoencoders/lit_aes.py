import random
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from sonata_utils import jpath
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import ToTensor
from models.phrase_vae import S2SVQAE, S2SReVQAE
from remi_z import MultiTrack, Bar


def log_piano_roll(logger, tgt_proll, out_proll, global_step, tag_prefix="val_"):
    """
    Log two piano roll images to TensorBoard: one for target, one for output.
    Duration is visualized as bar length (not color intensity).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from io import BytesIO
    from PIL import Image
    from torchvision.transforms import ToTensor

    def expand_piano_roll(pr):
        time, pitch = pr.shape
        expanded = np.zeros((time, pitch), dtype=np.float32)
        for t in range(time):
            for p in range(pitch):
                dur = int(pr[t, p])
                if dur > 0:
                    for dt in range(dur):
                        tt = t + dt
                        if tt < time:
                            expanded[tt, p] = 1.0
        return expanded.T  # (pitch, time)

    # Target piano roll
    tgt_expanded = expand_piano_roll(tgt_proll)
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.imshow(tgt_expanded, aspect='auto', origin='lower', cmap='gray_r', interpolation='nearest')
    ax.set_title("Target Piano Roll")
    ax.set_xlabel("Time step")
    ax.set_ylabel("MIDI Pitch")
    ax.grid(color='lightgray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_ylim(20, 100)
    ax.set_yticks(np.arange(20, 101, 12))
    ax.axis("on")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=300)
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf)
    image_tensor = ToTensor()(image)
    logger.experiment.add_image(f"piano_rolls/{tag_prefix}tgt_proll", image_tensor, global_step)

    # Output piano roll
    out_expanded = expand_piano_roll(out_proll)
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.imshow(out_expanded, aspect='auto', origin='lower', cmap='gray_r', interpolation='nearest')
    ax.set_title("Generated Piano Roll")
    ax.set_xlabel("Time step")
    ax.set_ylabel("MIDI Pitch")
    ax.grid(color='lightgray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_ylim(20, 100)
    ax.set_yticks(np.arange(20, 101, 12))
    ax.axis("on")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=300)
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf)
    image_tensor = ToTensor()(image)
    logger.experiment.add_image(f"piano_rolls/{tag_prefix}out_proll", image_tensor, global_step)


def compute_barvae_f1_metrics(
        out_texts,
        tgt_texts,
        samples_to_validate_per_batch=8,
        pos_per_bar=48,
        verbose=False,
):
    """
    Convert out_texts and tgt_texts into piano rolls and compute F1 metrics.
    Returns a dictionary of metrics, e.g., {'f1_op': ..., 'f1_opd': ...}
    """
    out_prolls = []
    tgt_prolls = []
    cnt = 0

    f1_iopd_list = []

    for out_text, tgt_text in zip(out_texts, tgt_texts):

        # ----- Parse out_text -----
        out_bar = None
        try:
            out_bar = MultiTrack.from_remiz_str(out_text, verbose=verbose)[0] # Bar object
            out_proll = out_bar.to_piano_roll(pos_per_bar=pos_per_bar)
        except Exception:
            out_proll = np.zeros((pos_per_bar, 128), dtype=np.int32)

        out_prolls.append(out_proll)

        # ----- Parse tgt_text -----
        tgt_bar = None
        try:
            tgt_bar = MultiTrack.from_remiz_str(tgt_text, verbose=verbose)[0] # Bar object
            tgt_proll = tgt_bar.to_piano_roll(pos_per_bar=pos_per_bar)
        except Exception:
            tgt_proll = np.zeros((pos_per_bar, 128), dtype=np.int32)

        tgt_prolls.append(tgt_proll)

        if out_bar is not None and tgt_bar is not None:
            f1_iopd = calculate_f1_ipod(out_bar, tgt_bar, pos_per_bar=pos_per_bar)
            f1_iopd_list.append(f1_iopd)

        cnt += 1
        if cnt >= samples_to_validate_per_batch:
            break

    # ----- Calculate metrics -----
    metrics = calculate_metrics_for_prolls(out_prolls, tgt_prolls)

    if len(f1_iopd_list) > 0:
        metrics['f1_iopd'] = float(np.mean(f1_iopd_list))
    else:
        metrics['f1_iopd'] = 0.0

    return metrics


def calculate_f1_ipod(out_bar:Bar, tgt_bar:Bar, pos_per_bar=48):
    """
    Compute macro-F1 across instruments (instrument-preserved onset+duration).

    Special cases:
        - If tgt has no instruments:
            - If out also has no instruments: F1 = 1
            - Otherwise: F1 = 0
    """
    # ----- Get instrument lists -----
    tgt_insts = tgt_bar.get_unique_insts(sort_by_voice=True, include_drum=True)
    out_insts = out_bar.get_unique_insts(sort_by_voice=True, include_drum=True)

    # ============================================================
    #  Case A: tgt_mt has no instruments
    # ============================================================
    if len(tgt_insts) == 0:
        if len(out_insts) == 0:
            return 1.0   # both empty → perfect match
        else:
            return 0.0   # tgt empty but out has tracks → mismatch

    # ============================================================
    #  Case B: tgt_mt has instruments → normal macro-F1 logic
    # ============================================================
    f1_list = []

    for inst in tgt_insts:

        # ---- Target roll ----
        try:
            tgt_roll = tgt_bar.to_piano_roll(of_insts=[inst], pos_per_bar=pos_per_bar)
        except Exception:
            tgt_roll = np.zeros((pos_per_bar, 128), dtype=np.int32)

        # ---- Output roll ----
        try:
            out_roll = out_bar.to_piano_roll(of_insts=[inst], pos_per_bar=pos_per_bar)
        except Exception:
            out_roll = np.zeros((pos_per_bar, 128), dtype=np.int32)

        tgt_mask = tgt_roll > 0
        out_mask = out_roll > 0

        correct = ((tgt_mask) & (out_mask) & (out_roll == tgt_roll)).sum()
        pred_nonzero = out_mask.sum()
        tgt_nonzero = tgt_mask.sum()

        precision = correct / pred_nonzero if pred_nonzero > 0 else 0.0
        recall = correct / tgt_nonzero if tgt_nonzero > 0 else 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            # tgt_track Guaranteed to have notes → so this means mismatch
            f1 = 0.0

        f1_list.append(f1)

    return float(np.mean(f1_list))


def calculate_metrics_for_prolls(out_prolls, tgt_prolls):
    """
    proll_outs, proll_tgts: (batch, num_step, 128) or (batch, T, 128) torch.Tensor or np.ndarray
    Returns dict with keys 'f1_op' and 'f1_opd' (averaged over batch).
    """
    if isinstance(out_prolls, torch.Tensor):
        out_prolls = out_prolls.cpu().numpy()
    if isinstance(tgt_prolls, torch.Tensor):
        tgt_prolls = tgt_prolls.cpu().numpy()

    bs = len(out_prolls)
    f1_op_list = []
    f1_opd_list = []
    for i in range(bs):
        tgt = tgt_prolls[i]
        out = out_prolls[i]
        tgt_onset_mask = tgt > 0
        pred_onset_mask = out > 0
        correct = ((tgt_onset_mask) & (pred_onset_mask) & (out == tgt)).sum()
        pred_nonzero = pred_onset_mask.sum()
        tgt_nonzero = tgt_onset_mask.sum()
        precision = correct / pred_nonzero if pred_nonzero > 0 else 0.0
        recall = correct / tgt_nonzero if tgt_nonzero > 0 else 0.0
        if precision + recall > 0:
            f1_opd = 2 * precision * recall / (precision + recall)
        else:
            f1_opd = 0.0 if (tgt_nonzero > 0) else 1.0

        # F1 ignoring duration
        correct_simple = ((tgt_onset_mask) & (pred_onset_mask)).sum()
        precision_simple = correct_simple / pred_nonzero if pred_nonzero > 0 else 0.0
        recall_simple = correct_simple / tgt_nonzero if tgt_nonzero > 0 else 0.0
        if precision_simple + recall_simple > 0:
            f1_op = 2 * precision_simple * recall_simple / (precision_simple + recall_simple)
        else:
            f1_op = 0.0 if (tgt_nonzero > 0) else 1.0

        f1_op_list.append(f1_op)
        f1_opd_list.append(f1_opd)
    return {'f1_op': float(np.mean(f1_op_list)), 'f1_opd': float(np.mean(f1_opd_list))}


class MinExponentialLR(torch.optim.lr_scheduler.ExponentialLR):
    def __init__(self, optimizer, gamma, minimum, last_epoch=-1):
        self.min = minimum
        super(MinExponentialLR, self).__init__(optimizer, gamma, last_epoch=last_epoch)

    def get_lr(self):
        return [
            max(base_lr * self.gamma ** self.last_epoch, self.min)
            for base_lr in self.base_lrs
        ]

class LitPhraseAE(pl.LightningModule):
    '''
    This is the class used for 1st and 2nd stage training of PhraseVAE.
    '''
    def __init__(
        self,
        t5_model_name=None,
        tokenizer_path=None,
        vq_dim=512,
        vq_codebook_size=512,
        vq_loss_weight=1.0,
        commitment_cost=0.25,
        apply_pre_vq_proj=False,
        t5_config=None,
        lr=1e-4,
        lr_vq=1e-3,
        compress_style='full_sequence',  # 'full_sequence' or 'first_token'
        n_compress_tokens=0,
        quantize_enc_out=False,
        lit_ckpt=False,
    ):
        super().__init__()
        
        self.model = S2SVQAE(
            t5_model_name=t5_model_name,
            tokenizer_path=tokenizer_path,
            vq_dim=vq_dim,
            vq_codebook_size=vq_codebook_size,
            vq_loss_weight=vq_loss_weight,
            commitment_cost=commitment_cost,
            apply_pre_vq_proj=apply_pre_vq_proj,
            t5_config=t5_config,
            compress_style=compress_style,  # 'full_sequence' or 'first_token'
            n_compress_tokens=n_compress_tokens,
            quantize_enc_out=quantize_enc_out,
            lit_ckpt=lit_ckpt,
        )
        self.lr = lr
        self.lr_vq = lr_vq
        self.save_hyperparameters()
        self.automatic_optimization = False

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, labels=None, **kwargs):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            **kwargs
        )

    def training_step(self, batch, batch_idx):
        opt, opt_vq = self.optimizers()
        opt.zero_grad()
        opt_vq.zero_grad()
        

        input_ids = batch["input_ids"]
        bs = input_ids.size(0)
        attention_mask = batch.get("attention_mask", None)
        labels = batch.get("labels", None)
        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, batch_size=bs)

        vq_loss = outputs.vq_loss
        self.log("train_vq_loss", vq_loss, prog_bar=False, batch_size=bs)

        # Log reconstruction loss, vq loss, perplexity
        recon_loss = outputs.recon_loss
        self.log("train_recon_loss", recon_loss, prog_bar=False, batch_size=bs)

        if hasattr(outputs, 'vq_perplexity'):
            perplexity = outputs.vq_perplexity
            self.log("train_vq_perplexity", perplexity, prog_bar=False, batch_size=bs)

        # Backprop and optimize
        self.manual_backward(loss)
        opt_vq.step() 
        opt.step()


        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)
        labels = batch.get("labels", None)
        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        bs = input_ids.size(0)

        loss = outputs.loss
        self.log("val_loss", loss, prog_bar=True, batch_size=bs)

        recon_loss = outputs.recon_loss
        self.log("val_recon_loss", recon_loss, prog_bar=False, batch_size=bs)

        vq_loss = outputs.vq_loss
        self.log("val_vq_loss", vq_loss, prog_bar=False, batch_size=bs)

        # Autoregressive generation (greedy)
        encode_out = self.model.encode(input_ids, attention_mask=attention_mask)
        max_length = input_ids.shape[1]
        generated_ids = self.model.decode(
            encode_out, 
            max_length=max_length,
            pad_token_id=self.model.tokenizer.pad_token_id,
            eos_token_id=self.model.tokenizer.eos_token_id,
            do_sample=False,
            num_beams=1,
        )
        
        # Detokenize to REMI-z strs
        out_texts = self.model.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        labels_for_decode = labels.clone()
        labels_for_decode[labels_for_decode == -100] = self.model.tokenizer.pad_token_id
        tgt_texts = self.model.tokenizer.batch_decode(labels_for_decode, skip_special_tokens=False)
        input_texts = batch['masked_seq']

        # Validate F1 metrics
        samples_to_validate = min(8, bs)
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
                    self.global_step
                )

            # Get piano roll
            tgt_mt = MultiTrack.from_remiz_str(tgt_texts[0])[0]
            tgt_proll = tgt_mt.to_piano_roll(pos_per_bar=48)
            try:
                out_mt = MultiTrack.from_remiz_str(out_texts[0])[0]
                out_proll = out_mt.to_piano_roll(pos_per_bar=48)
            except Exception as e:
                out_proll = np.zeros_like(tgt_proll)

            # Use reusable log_piano_roll function
            log_piano_roll(self.logger, tgt_proll, out_proll, self.global_step)
        
    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)
        labels = batch.get("labels", None)
        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        vq_loss = outputs.vq_loss
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_vq_loss", vq_loss, prog_bar=True)
        bs = input_ids.size(0)

        # Autoregressive generation (greedy)
        encode_out = self.model.encode(input_ids, attention_mask=attention_mask)
        max_length = input_ids.shape[1]
        generated_ids = self.model.decode(
            encode_out, 
            max_length=max_length,
            pad_token_id=self.model.tokenizer.pad_token_id,
            eos_token_id=self.model.tokenizer.eos_token_id,
            do_sample=False,
            num_beams=1,
        )
        
        # Detokenize to REMI-z strs
        out_texts = self.model.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        labels_for_decode = labels.clone()
        labels_for_decode[labels_for_decode == -100] = self.model.tokenizer.pad_token_id
        tgt_texts = self.model.tokenizer.batch_decode(labels_for_decode, skip_special_tokens=False)
        input_texts = batch['masked_seq']

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
                    self.global_step
                )

            # Get piano roll
            tgt_mt = MultiTrack.from_remiz_str(tgt_texts[0])[0]
            tgt_proll = tgt_mt.to_piano_roll(pos_per_bar=48)
            try:
                out_mt = MultiTrack.from_remiz_str(out_texts[0])[0]
                out_proll = out_mt.to_piano_roll(pos_per_bar=48)
            except Exception as e:
                out_proll = np.zeros_like(tgt_proll)

            # Use reusable log_piano_roll function
            log_piano_roll(self.logger, tgt_proll, out_proll, self.global_step)

    def configure_optimizers(self):
        # Get VQEmbedding parameters
        vq_params = []
        if hasattr(self.model, 'vq'):
            vq_params = list(self.model.vq.parameters())
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'vq'):
            vq_params = list(self.model.model.vq.parameters())
        vq_param_ids = set(id(p) for p in vq_params)
        
        # Other parameters: not in vq_param_ids
        other_params = [p for p in self.parameters() if id(p) not in vq_param_ids]
        optimizer_vq = torch.optim.Adam(vq_params, lr=self.lr_vq)
        optimizer_main = torch.optim.Adam(other_params, lr=self.lr)
        print(f"Number of main optimizer params: {len(other_params)}")
        print(f"Number of VQ optimizer params: {len(vq_params)}")
        print('VQ lr:', self.lr_vq, 'Main lr:', self.lr)
        return [optimizer_main, optimizer_vq]

