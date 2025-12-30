"""
Sequence to Sequence VQ-AE Model
Compress a bar of music into a discrete latent vector
and reconstruct the bar from the latent vector

Author: Longshen Ou
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, PreTrainedTokenizerFast, T5Config
from transformers.modeling_outputs import BaseModelOutput
from huggingface_hub import PyTorchModelHubMixin


class VQEmbedding(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=512, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(
            -1 / self.num_embeddings, 1 / self.num_embeddings
        )

    def forward(self, z):
        # z: (batch, z_seq_len, embedding_dim)
        orig_shape = z.shape
        batch, z_seq_len, dim = z.shape
        z_flattened = z.reshape(-1, self.embedding_dim)  # (batch * z_seq_len, dim)
        # Compute distances
        distances = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )  # (batch * z_seq_len, num_embeddings)
        encoding_indices = torch.argmin(distances, dim=1)  # (batch * z_seq_len,)
        z_q = self.embedding(encoding_indices)  # (batch * z_seq_len, dim)
        z_q = z_q.view(batch, z_seq_len, dim)

        # Commitment loss
        loss = F.mse_loss(z_q, z.detach()) + self.commitment_cost * F.mse_loss(
            z_q.detach(), z
        )
        # Straight-through estimator
        z_q = z + (z_q - z).detach()
        encoding_indices = encoding_indices.view(batch, z_seq_len)

        return z_q, loss, encoding_indices


class VQEmbeddingEMA(nn.Module):
    def __init__(
        self,
        num_embeddings=512,
        embedding_dim=512,
        commitment_cost=0.25,
        decay=0.99,
        epsilon=1e-2,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(
            -1 / self.num_embeddings, 1 / self.num_embeddings
        )

        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_weight", self.embedding.weight.data.clone())

    def forward(self, z):
        # z: (batch, z_seq_len, embedding_dim)
        orig_shape = z.shape
        batch, z_seq_len, dim = z.shape
        z_flattened = z.reshape(-1, self.embedding_dim)  # (batch * z_seq_len, dim)

        # Compute distances
        distances = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )  # (batch * z_seq_len, num_embeddings)
        encoding_indices = torch.argmin(distances, dim=1)  # (batch * z_seq_len,)
        z_q = self.embedding(encoding_indices)  # (batch * z_seq_len, dim)
        z_q = z_q.view(batch, z_seq_len, dim)

        # Commitment loss
        loss = F.mse_loss(z_q, z.detach()) + self.commitment_cost * F.mse_loss(
            z_q.detach(), z
        )

        # EMA update
        if self.training:
            encoding_one_hot = F.one_hot(encoding_indices, self.num_embeddings).type(
                z_flattened.dtype
            )
            cluster_size = encoding_one_hot.sum(0)
            # EMA cluster size
            self.ema_cluster_size.mul_(self.decay).add_(
                cluster_size, alpha=1 - self.decay
            )
            # EMA embedding
            embed_sum = torch.matmul(encoding_one_hot.t(), z_flattened)
            self.ema_weight.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            # Laplace smoothing of the cluster size
            n = self.ema_cluster_size.sum()
            cluster_size = (
                (self.ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon)
                * n
            )
            # Normalize embedding
            embed_normalized = self.ema_weight / cluster_size.unsqueeze(1)
            self.embedding.weight.data.copy_(embed_normalized)

        # Straight-through estimator
        z_q = z + (z_q - z).detach()
        encoding_indices = encoding_indices.view(batch, z_seq_len)

        return z_q, loss, encoding_indices


def load_t5_model_from_lit_ckpt(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]

    # Remove 'model.t5.' prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model.t5."):
            new_key = k[len("model.t5.") :]
            new_state_dict[new_key] = v
        else:
            print(f"Skipping key {k} as it does not start with 'model.t5.'")
    state_dict = new_state_dict

    config = T5Config(**checkpoint["hyper_parameters"]["t5_config"])

    model = T5ForConditionalGeneration(config)
    model.load_state_dict(state_dict, strict=True)

    return model


def load_lit_t5_model(ckpt_path, config_or_model_name):
    """
    Load only the state_dict from a lightning checkpoint and initialize a T5ForConditionalGeneration model.
    Args:
        ckpt_path: Path to the lightning .ckpt file (should contain 'state_dict').
        config_or_model_name: Either a T5Config instance or a model name/path for config loading.
    Returns:
        model: T5ForConditionalGeneration with loaded weights.
    """
    import torch
    from transformers import T5ForConditionalGeneration, T5Config

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    # print('Loaded state_dict keys:', state_dict.keys())

    # Remove 'model.t5.' prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model.t5."):
            new_key = k[len("model.t5.") :]
            new_state_dict[new_key] = v
        else:
            print(f"Skipping key {k} as it does not start with 'model.t5.'")
    state_dict = new_state_dict

    # exit()
    if isinstance(config_or_model_name, T5Config):
        config = config_or_model_name
    else:
        config = T5Config.from_pretrained(config_or_model_name)
    model = T5ForConditionalGeneration(config)
    model.load_state_dict(state_dict, strict=True)
    return model


class S2SVAE3(nn.Module):
    '''
    S2SVAE that has smaller bottleneck
    Decoder receive 4 tokens as input
    '''
    def __init__(
        self,
        t5_model_name: str = None,
        tokenizer_path: str = None,
        recon_weight: float = 1.0,
        kld_weight: float = 1.0,
        l2_reg_weight: float = 0.0,
        t5_config: dict = None,
        compress_style: str = "full_sequence",
        n_compress_tokens: int = 0,  # used if compress_style == 'first_n_tokens'
        lit_ckpt: bool = False,
        adaptive_scale: bool = False,
        bottleneck_dim = None,
    ):
        super().__init__()

        # Initialize tokenizer
        if tokenizer_path is not None:
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        else:
            self.tokenizer = None

        # Always initialize T5 config first
        if isinstance(t5_config, T5Config):
            config = t5_config
        else:
            config = T5Config(**(t5_config or {}))

        # T5 initialization
        if lit_ckpt:
            assert (
                t5_model_name is not None
            ), "t5_model_name must be provided when lit_ckpt is True"
            self.t5 = load_lit_t5_model(t5_model_name, config)
        else:
            if t5_model_name:
                self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name)
            else:
                self.t5 = T5ForConditionalGeneration(config)

        # Resize token embeddings to match tokenizer
        self.t5.resize_token_embeddings(len(self.tokenizer))

        self.d_model = t5_config.get("d_model", 512)
        self.bottleneck_dim = bottleneck_dim if bottleneck_dim is not None else self.d_model

        assert compress_style in [
            "full_sequence",
            "first_token",
            "pooling",
            "first_n_tokens",
        ]
        self.compress_style = compress_style
        self.n_compress_tokens = n_compress_tokens
        if self.compress_style == "first_n_tokens":
            assert (
                1 <= self.n_compress_tokens <= 10
            ), "n_compress_tokens must be between 1 and 10"
        # print(f"Using compress_style: {self.compress_style}")

        self.proj = nn.Linear(self.d_model*4, self.bottleneck_dim * 2)

        # self.proj1 = nn.Linear(self.d_model*4, 1024)
        # self.proj2 = nn.Linear(1024, self.bottleneck_dim * 2)

        self.post_proj = nn.Linear(self.bottleneck_dim, self.d_model*4)

        self.adaptive_scale = adaptive_scale
        if adaptive_scale:
            self.scale = nn.Parameter(torch.tensor(1.0))
        # print(f"Adaptive scale: {self.adaptive_scale}")

        self.recon_weight = recon_weight
        self.kld_weight = kld_weight
        self.l2_reg_weight = l2_reg_weight

        self.layer_norm = nn.LayerNorm(self.d_model * 4)

    def encode(self, input_ids, attention_mask=None, sample=True):
        # input_ids: (batch, seq_len)

        # Obtain encoder hidden states
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        input_ids = input_ids.to(self.t5.device)
        attention_mask = attention_mask.to(self.t5.device)
        encoder_outputs = self.t5.encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        encoder_hidden_states = (
            encoder_outputs.last_hidden_state
        )  # (batch, seq_len, hidden)

        # Compress
        if self.compress_style == "first_token":
            # Compress
            compress_vecs = encoder_hidden_states[:, 0, :].unsqueeze(
                1
            )  # (batch, 1, hidden)
            attention_mask = attention_mask[:, :1]  # (batch, 1)
        elif self.compress_style == "pooling":
            # Mean pooling over the sequence length dimension
            compress_vecs = torch.mean(
                encoder_hidden_states, dim=1, keepdim=True
            )  # (batch, 1, hidden)
            attention_mask = attention_mask[:, :1]  # (batch, 1)
        elif self.compress_style == "full_sequence":
            compress_vecs = encoder_hidden_states
            # attention_mask remains the same
        elif self.compress_style == "first_n_tokens":
            compress_vecs = encoder_hidden_states[
                :, : self.n_compress_tokens, :
            ]  # (batch, n_compress_tokens, hidden)
            # Full attention mask for the first n tokens
            attention_mask = torch.ones(
                (input_ids.size(0), self.n_compress_tokens),
                dtype=torch.long,
                device=input_ids.device,
            )
        else:
            raise ValueError(f"Invalid compress_style: {self.compress_style}")

        assert compress_vecs.shape[1] == 4 # Does not support mq

        
        # Flatten the n_compress_tokens dimension
        bs = compress_vecs.size(0)
        compress_vecs = compress_vecs.reshape(bs, -1) # (batch, n_compress_tokens * hidden)

        # Layer norm
        compress_vecs = self.layer_norm(compress_vecs) # (batch, n_compress_tokens, hidden)


        # # Project to bottleneck
        # compress_vecs = self.proj1(compress_vecs) # (batch, 1024)
        # compress_vecs = F.gelu(compress_vecs)
        # compress_vecs = self.proj2(compress_vecs) # (batch, bottleneck_dim * 2)

        # Project to 2 * d_model for mean and logvar
        compress_vecs = self.proj(
            compress_vecs
        )  # (batch, n_compress_tokens, hidden * 2)
        mean = compress_vecs[:, : self.bottleneck_dim]  # (batch, n_compress_tokens, hidden)
        logvar = compress_vecs[
            :, self.bottleneck_dim :
        ]  # (batch, n_compress_tokens, hidden)

        # Sample to get latent vector
        if sample:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            latent = mean + eps * std  # (batch, n_compress_tokens, hidden)
        else:
            # print(f"Using mean as latent, no sampling")
            latent = mean # (batch, n_compress_tokens, hidden)

        # KLD loss (sample level)
        kld_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=-1) # (batch, n_compress_tokens)
        
        # Sum over tokens (treat 4 tokens as one latent), mean over batch
        # That is the correct per sample KLD loss
        kld_loss = kld_loss.mean(dim=0)  # scalar

        # Per-query KLD loss
        # kld_loss = torch.mean(kld_loss) # average over batch and n_query

        # Dim-level KLD normalize
        total_dim = mean.shape[1]
        kld_loss = kld_loss / total_dim

        # L2 regularization on mean
        l2_reg = mean.pow(2).mean()

        return {
            "latent": latent,  # (batch, seq_len, hidden)
            'clean_latent': mean,  # (batch, seq_len, hidden)
            "kld_loss": kld_loss,
            "attention_mask": attention_mask,
            "encoder_hidden_states": encoder_hidden_states,
            "l2_reg": l2_reg,
        }

    def decode(self, latent, attention_mask=None, max_length=128, **generate_kwargs):
        '''
        Autoregressive decode from the latent

        latent: [bs, bottleneck_dim]
        '''
        # latent = encoder_out["quantized"]  # (batch, seq_len, hidden)
        bs = latent.size(0)

        # Scale the latent
        if self.adaptive_scale:
            latent = latent * self.scale


        if attention_mask is None:
            assert (
                self.compress_style != "full_sequence"
            ), "attention_mask must be provided for full_sequence compress_style"

            # Create attention_mask based on compress_style
            if self.compress_style == "first_token":
                attention_mask = torch.ones(
                    (bs, 1), dtype=torch.long, device=latent.device
                )
            elif self.compress_style == "pooling":
                attention_mask = torch.ones(
                    (bs, 1), dtype=torch.long, device=latent.device
                )
            elif self.compress_style == "first_n_tokens":
                attention_mask = torch.ones(
                    (bs, self.n_compress_tokens),
                    dtype=torch.long,
                    device=latent.device,
                )

        # Wrap in BaseModelOutput for compatibility with T5 generate
        encoder_outputs = BaseModelOutput(last_hidden_state=latent)

        # Post projection to d_model
        latent = encoder_outputs.last_hidden_state
        latent = self.post_proj(latent)
        latent = latent.reshape(bs, 4, self.d_model)
        encoder_outputs.last_hidden_state = latent
        # encoder_outputs.last_hidden_state = self.post_proj(encoder_outputs.last_hidden_state).unsqueeze(1) # [bs, 1, d_model]
        # encoder_outputs.last_hidden_state = self.post_proj(encoder_outputs.last_hidden_state).reshape(bs, 4, self.d_model) # [bs, 4, d_model]
        attention_mask = torch.ones((latent.size(0), 1), dtype=torch.long, device=latent.device)

        generated_ids = self.t5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            max_length=max_length,
            decoder_start_token_id=self.tokenizer.bos_token_id,
            **generate_kwargs,
        )
        return generated_ids

    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        labels=None,
        **kwargs,
    ):
        '''
        Forward pass in training 
        '''
        # Use encode to get quantized and vq_loss
        encode_out = self.encode(input_ids, attention_mask=attention_mask)
        latent = encode_out["latent"]  # [bs, enc_out_len， hidden]
        kld_loss = encode_out["kld_loss"]
        attention_mask = encode_out["attention_mask"]

        # Scale the latent
        if self.adaptive_scale:
            latent = latent * self.scale

        # Post projection to d_model
        latent = self.post_proj(latent)
        latent = latent.reshape(latent.size(0), 4, self.d_model)
        # print(latent.shape)
        attention_mask = torch.ones((latent.size(0), 1), dtype=torch.long, device=latent.device)
        # print(attention_mask.shape) 

        # Decoder forward (teacher forcing)
        outputs = self.t5(
            inputs_embeds=None,
            encoder_outputs=(latent,),
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            return_dict=True,
            **kwargs,
        )
        # Add different loss to the output
        outputs.kld_loss = kld_loss
        outputs.recon_loss = outputs.loss
        outputs.encoded_latent = latent
        outputs.l2_reg = encode_out['l2_reg']

        # Note down mean and std of latents
        clean_latent = encode_out['clean_latent']
        latent_mean = torch.mean(clean_latent)
        latent_std = torch.std(clean_latent)
        outputs.latent_mean = latent_mean
        outputs.latent_std = latent_std
        outputs.encoded_latent = clean_latent

        # Combine reconstruction loss and vq_loss
        recon_loss = outputs.recon_loss
        l2_reg = outputs.l2_reg
        total_loss = recon_loss * self.recon_weight + kld_loss * self.kld_weight + l2_reg * self.l2_reg_weight
        outputs.loss = total_loss

        return outputs

    @classmethod
    def from_lit_ckpt(cls, ckpt_path, **kwargs):
        """
        Instantiate the model from a lightning checkpoint.
        Args:
            ckpt_path: Path to the lightning .ckpt file (should contain 'state_dict' and 'hyper_parameters').
            kwargs: Additional arguments to override hyper_parameters.
        Returns:
            model: S2SVQAE instance with loaded weights.
        """
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        hyper_params = checkpoint.get("hyper_parameters", {})
        hyper_params.update(kwargs)




class S2SVQAE(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        t5_model_name: str = None,
        tokenizer_path: str = None,
        vq_dim: int = 512,
        vq_codebook_size: int = 512,
        commitment_cost: float = 0.25,
        apply_pre_vq_proj: bool = False,
        vq_loss_weight: float = 1.0,
        t5_config: dict = None,
        compress_style: str = "full_sequence",  # 'full_sequence' or 'first_token'
        n_compress_tokens: int = 0,  # used if compress_style == 'first_n_tokens'
        quantize_enc_out: bool = False,
        lit_ckpt: bool = False,
    ):
        super().__init__()

        # Initialize tokenizer
        if tokenizer_path is not None:
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        else:
            self.tokenizer = None

        # Always initialize T5 config first
        if isinstance(t5_config, T5Config):
            config = t5_config
        else:
            config = T5Config(**(t5_config or {}))

        # T5 initialization
        if lit_ckpt:
            assert (
                t5_model_name is not None
            ), "t5_model_name must be provided when lit_ckpt is True"
            self.t5 = load_lit_t5_model(t5_model_name, config)
        else:
            if t5_model_name:
                self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name)
            else:
                self.t5 = T5ForConditionalGeneration(config)

        # Resize token embeddings to match tokenizer
        self.t5.resize_token_embeddings(len(self.tokenizer))

        d_model = t5_config.get("d_model", 512)

        # VQ module
        self.vq = VQEmbedding(
            num_embeddings=vq_codebook_size,
            embedding_dim=vq_dim,
            commitment_cost=commitment_cost,
        )

        assert compress_style in [
            "full_sequence",
            "first_token",
            "pooling",
            "first_n_tokens",
        ]
        self.compress_style = compress_style
        self.n_compress_tokens = n_compress_tokens
        if self.compress_style == "first_n_tokens":
            assert (
                1 <= self.n_compress_tokens <= 10
            ), "n_compress_tokens must be between 1 and 10"
        print(f"Using compress_style: {self.compress_style}")

        self.quantize_enc_out = quantize_enc_out
        self.vq_loss_weight = vq_loss_weight
        self.apply_pre_vq_proj = apply_pre_vq_proj

        # Projection layer
        if apply_pre_vq_proj:
            self.pre_vq_proj = nn.Linear(d_model, vq_dim)

    def encode(self, input_ids, attention_mask=None):
        # input_ids: (batch, seq_len)

        # Obtain encoder hidden states
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        input_ids = input_ids.to(self.t5.device)
        attention_mask = attention_mask.to(self.t5.device)
        encoder_outputs = self.t5.encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        encoder_hidden_states = (
            encoder_outputs.last_hidden_state
        )  # (batch, seq_len, hidden)

        # Compress
        if self.compress_style == "first_token":
            # Compress
            compress_vecs = encoder_hidden_states[:, 0, :].unsqueeze(
                1
            )  # (batch, 1, hidden)
            attention_mask = attention_mask[:, :1]  # (batch, 1)
        elif self.compress_style == "pooling":
            # Mean pooling over the sequence length dimension
            compress_vecs = torch.mean(
                encoder_hidden_states, dim=1, keepdim=True
            )  # (batch, 1, hidden)
            attention_mask = attention_mask[:, :1]  # (batch, 1)
        elif self.compress_style == "full_sequence":
            compress_vecs = encoder_hidden_states
            # attention_mask remains the same
        elif self.compress_style == "first_n_tokens":
            compress_vecs = encoder_hidden_states[
                :, : self.n_compress_tokens, :
            ]  # (batch, n_compress_tokens, hidden)
            # Full attention mask for the first n tokens
            attention_mask = torch.ones(
                (input_ids.size(0), self.n_compress_tokens),
                dtype=torch.long,
                device=input_ids.device,
            )
        else:
            raise ValueError(f"Invalid compress_style: {self.compress_style}")

        # Quantization
        if self.quantize_enc_out:
            if self.apply_pre_vq_proj:
                compress_vecs = self.pre_vq_proj(compress_vecs)
            quantized, vq_loss, vq_info = self.vq(
                compress_vecs
            )  # (batch, z_seq_len, hidden)
        else:
            quantized = compress_vecs  # (batch, 1 or n_compress_tokens, hidden)
            vq_loss = torch.tensor(0.0, device=input_ids.device)
            vq_info = None

        return {
            "quantized": quantized,  # (batch, seq_len, hidden)
            "vq_loss": vq_loss,
            "vq_ids": vq_info,
            "attention_mask": attention_mask,
            "encoder_hidden_states": encoder_hidden_states,
        }

    def decode(
        self, encoder_out, attention_mask=None, max_length=128, **generate_kwargs
    ):
        quantized = encoder_out["quantized"]  # (batch, seq_len, hidden)
        batch_size = quantized.size(0)

        if attention_mask is None:
            if self.compress_style == "first_token":
                attention_mask = torch.ones(
                    (batch_size, 1), dtype=torch.long, device=quantized.device
                )
            elif self.compress_style == "full_sequence":
                attention_mask = encoder_out["attention_mask"]
            elif self.compress_style == "pooling":
                attention_mask = torch.ones(
                    (batch_size, 1), dtype=torch.long, device=quantized.device
                )

        # Wrap in BaseModelOutput for compatibility with T5 generate
        encoder_outputs = BaseModelOutput(last_hidden_state=quantized)
        generated_ids = self.t5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            max_length=max_length,
            decoder_start_token_id=self.tokenizer.bos_token_id,
            **generate_kwargs,
        )
        return generated_ids

    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        labels=None,
        **kwargs,
    ):
        # Use encode to get quantized and vq_loss
        encode_out = self.encode(input_ids, attention_mask=attention_mask)
        quantized = encode_out["quantized"]  # [bs, enc_out_len， hidden]
        vq_loss = encode_out["vq_loss"]
        vq_ids = encode_out["vq_ids"]
        attention_mask = encode_out["attention_mask"]

        # Decoder forward
        outputs = self.t5(
            inputs_embeds=None,
            encoder_outputs=(quantized,),
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            return_dict=True,
            **kwargs,
        )
        # Add vq_loss to the output dict for downstream use
        outputs.vq_loss = vq_loss
        outputs.vq_ids = vq_ids

        # Calculate perplexity for VQ
        # vq_ids: (B, seq_len)
        if vq_ids is None:
            vq_ids = torch.zeros(
                (input_ids.size(0), 1), dtype=torch.long, device=input_ids.device
            )
        codebook_size = self.vq.num_embeddings  # 自己模型里的codebook大小
        one_hot = F.one_hot(vq_ids, num_classes=codebook_size).float()
        avg_probs = one_hot.mean(dim=(0, 1))  # 平均使用概率
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        outputs.vq_perplexity = perplexity

        # Combine reconstruction loss and vq_loss
        recon_loss = outputs.loss
        total_loss = recon_loss + vq_loss * self.vq_loss_weight
        outputs.loss = total_loss
        outputs.recon_loss = recon_loss

        # print(recon_loss.item(), vq_loss.item(), total_loss.item())

        return outputs

    @classmethod
    def from_lit_ckpt(cls, ckpt_path, **kwargs):
        """
        Instantiate the model from a lightning checkpoint.
        Args:
            ckpt_path: Path to the lightning .ckpt file (should contain 'state_dict' and 'hyper_parameters').
            kwargs: Additional arguments to override hyper_parameters.
        Returns:
            model: S2SVQAE instance with loaded weights.
        """
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        hyper_params = checkpoint.get("hyper_parameters", {})
        hyper_params.update(kwargs)
        print(hyper_params.keys())

        # Remove '_XXX' from the hyperparameter entries
        unwanted_keys = ['lr', 'lr_vq']
        cleaned_hyper_params = {}
        for key, value in hyper_params.items():
            if not key.startswith("_") and key not in unwanted_keys:
                cleaned_hyper_params[key] = value
        hyper_params = cleaned_hyper_params

        model = cls(**hyper_params)

        # Remove 'model.' prefix from state_dict keys
        unwanted_keys = ['model.pre_vq_proj.weight', 'model.pre_vq_proj.bias']
        state_dict = checkpoint["state_dict"]
        new_state_dict = {}
        for key, value in state_dict.items():
            if key in unwanted_keys:
                continue
            if key.startswith("model."):
                new_key = key[len("model.") :]
            else:
                new_key = key
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict, strict=True)

        return model

