"""
One-bar diffusion on latent space, for sanity check
"""

import torch
import torch.nn as nn
from diffusers import UNet2DModel
from sonata_utils import check_model_param
from diffusers.models.transformers.stable_audio_transformer import StableAudioDiTModel
from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
from typing import Optional, Union
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from tqdm import tqdm


def main():
    # test_transformer()
    # test_unet()
    test_stable_dit()


def test_transformer():
    # Example usage of UnconditionedTransformerEnc
    model = UnconditionedTransformerEnc()
    check_model_param(model)

    # Dummy input: batch of 8 sequences, each with seq_len=1 and feature_dim=2048
    x = torch.randn(8, 4, 512)

    # Forward pass
    output = model(x)
    print("Output shape:", output.shape)  # Should be (8, 4, 512)


def test_unet():
    # Example usage of ClassConditionedUnet
    model = ClassConditionedUnet(num_classes=10, class_emb_size=4)

    # Dummy input: batch of 8 grayscale images of size 28x28
    x = torch.randn(8, 1, 28, 28)

    # Dummy timestep tensor (for diffusion models)
    t = torch.randint(0, 1000, (8,))

    # Dummy class labels (integers from 0 to 9)
    class_labels = torch.randint(0, 10, (8,))

    # Forward pass
    output = model(x, t, class_labels)
    print("Output shape:", output.shape)  # Should be (8, 1, 28, 28)

def test_stable_dit():
    # Example usage of UnconditionalDiT
    model = UnconditionalDiT(
        sample_size=16,
        in_channels=128,
        num_layers=6,
        attention_head_dim=32,
        num_attention_heads=8,
        num_key_value_attention_heads=8,
        out_channels=128,
        cross_attention_dim=128,
        time_proj_dim=64,
        global_states_input_dim=768,
        cross_attention_input_dim=768,
    )
    check_model_param(model)

    # Dummy input: batch of 8 sequences, each with in_channels=128 and sequence_len=16
    hidden_states = torch.randn(8, 128, 16)

    # Dummy timestep tensor
    timestep = torch.randint(0, 1000, (8,))

    # Dummy encoder and global hidden states
    encoder_hidden_states = torch.randn(8, 16, 768)
    global_hidden_states = torch.randn(8, 1, 768)

    # Forward pass
    output = model(
        hidden_states,
        timestep,
        encoder_hidden_states=encoder_hidden_states,
        global_hidden_states=global_hidden_states
    )
    print("Output shape:", output.sample.shape)  # Should be (8, 128, 16)


class SecEncoder(nn.Module):
    '''
    Encoder that deal with section input that looks like
    [
        'i-8 A-8 A-16 B-8 ...',  # song 1's section annotation
        'i-16 A-8 B-8 A-8 ...',  # song 2's section annotation
        ...
    ]
    '''
    def __init__(self, conf_embed_dim=128, n_head=8, n_layer=3, max_len=30):
        '''
        max_len: maximum length of instrument configuration sequence, 4 * max_n_bar
        '''
        super().__init__()
        out_dim = conf_embed_dim
        embed_dim = out_dim // 2
        
        # Define vocab
        self.sec_type_vocab = {'o', 'b', 'A', 'i', 'B', 'x', 'E', 'c', 'X', 'D', 'C'}
        self.sec_len_vocab = {str(i) for i in range(1, 33)}  # lengths from 1 to 32 bars

        self.sec_type_to_id = {sec_type: idx + 1 for idx, sec_type in enumerate(self.sec_type_vocab)}
        self.sec_len_to_id = {sec_len: idx + 1 for idx, sec_len in enumerate(self.sec_len_vocab)}
        self.pad_id = 0

        type_vocab_size = len(self.sec_type_vocab) + 1  # +1 for padding
        length_vocab_size = len(self.sec_len_vocab) + 1  # +1 for padding
        self.type_embed = nn.Embedding(type_vocab_size, embed_dim)
        self.len_embed = nn.Embedding(length_vocab_size, embed_dim)
        
        self.sec_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=conf_embed_dim,
                nhead=n_head,
                dim_feedforward=conf_embed_dim * 4,
                batch_first=True,
            ),
            num_layers=n_layer,
        )
        self.max_len = max_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, sec_annot, n_batch=None):
        '''
        Section annot is a batch of section annotation strings:
        [
            'i-8 A-8 A-16 B-8 ...',  # song 1's section annotation
            'i-16 A-8 B-8 A-8 ...',  # song 2's section annotation
            ...
        ]
        '''
        # Get sec type matrix
        sec_type_seqs = []
        sec_len_seqs = []
        for annot in sec_annot:
            tokens = annot.split(' ')
            type_seq = []
            len_seq = []
            for token in tokens:
                sec_type, sec_len = token.split('-')
                type_seq.append(sec_type)
                len_seq.append(sec_len)
            sec_type_seqs.append(type_seq)
            sec_len_seqs.append(len_seq)

        # Convert to ids, pad to max_len
        sec_type_ids = []
        sec_len_ids = []
        for type_seq, len_seq in zip(sec_type_seqs, sec_len_seqs):
            type_ids = [self.sec_type_to_id[sec_type] for sec_type in type_seq]
            len_ids = [self.sec_len_to_id[sec_len] for sec_len in len_seq]
            # Pad
            if len(type_ids) < self.max_len:
                pad_length = self.max_len - len(type_ids)
                type_ids += [self.pad_id] * pad_length
                len_ids += [self.pad_id] * pad_length
            sec_type_ids.append(type_ids)
            sec_len_ids.append(len_ids)

        # Convert to tensor, embedding, and encode
        sec_type_tensor = torch.tensor(sec_type_ids, dtype=torch.long).to(self.device)
        sec_len_tensor = torch.tensor(sec_len_ids, dtype=torch.long).to(self.device)
        type_emb = self.type_embed(sec_type_tensor)
        len_emb = self.len_embed(sec_len_tensor)
        sec_emb = torch.cat([type_emb, len_emb], dim=-1)  # (bs, max_len, dim)
        sec_encoded = self.sec_encoder(sec_emb)  # (bs, max_len, dim)

        # Attention mask
        n_secs = [len(annot.split(' ')) for annot in sec_annot]
        attn_mask = torch.zeros(len(sec_annot), self.max_len).to(self.device)
        for i, n_sec in enumerate(n_secs):
            attn_mask[i, :n_sec] = 1

        return sec_encoded, attn_mask


class InstConfEncoder(nn.Module):
    '''
    Encoder that deal with instrument configuration input
    '''
    def __init__(self, vocab_size=5, conf_embed_dim=128, n_head=8, n_layer=3, max_len=512):
        '''
        max_len: maximum length of instrument configuration sequence, 4 * max_n_bar
        '''
        super().__init__()
        self.inst_embed = nn.Embedding(vocab_size, conf_embed_dim)
        self.inst_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=conf_embed_dim,
                nhead=n_head,
                dim_feedforward=conf_embed_dim * 4,
                batch_first=True,
            ),
            num_layers=n_layer,
        )
        self.max_len = max_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, inst_conf, n_batch=None):
        '''
        inst config is a batch of instrument configuration token sequences: 
        [
            [i-0, i-13, b-1, i-25, i-13, i-0, b-1, ...],  # song 1's instrumentation config
            [i-0, i-0, i-0, b-1, i-13, i-25, b-1, ...],  # song 2's instrumentation config
            ...
        ]
        '''
        # Convert to ids
        inst_conf_ids, attn_mask = self.tokenize(inst_conf) # (bs, max_len)
        inst_conf_emb = self.inst_embed(inst_conf_ids)  # (bs, max_len, dim)
        inst_conf_encoded = self.inst_encoder(inst_conf_emb)  # (bs, max_len, dim)

        return inst_conf_encoded, attn_mask

    def tokenize(self, inst_conf):
        '''
        Convert instrument configuration to ids
        '''
        # Simple mapping for demonstration purposes
        token_to_id = {
            'i-0': 2,
            'i-13': 1,
            'i-25': 3,
            'b-1': 0,
            'pad': 4,
        }
        inst_conf_ids = []
        for seq in inst_conf:
            seq_ids = [token_to_id[token] for token in seq]

            # Pad to max_len
            if len(seq_ids) < self.max_len:
                seq_ids += [4] * (self.max_len - len(seq_ids))

            inst_conf_ids.append(seq_ids)

        attention_mask = torch.zeros(len(inst_conf), self.max_len).to(self.device)
        for i, seq in enumerate(inst_conf):
            attention_mask[i, :len(seq)] = 1

        inst_conf_tensor = torch.tensor(inst_conf_ids, dtype=torch.long).to(self.device) # (bs, max_len)

        return inst_conf_tensor, attention_mask

class LengthEncoder(nn.Module):
    def __init__(self, max_bar=128, len_embed_dim=128, length_bucket_size=10):
        super().__init__()
        self.length_embedding = nn.Embedding(max_bar, len_embed_dim)
        self.length_bucket_size = length_bucket_size

    def forward(self, n_bar, n_batch=None):
        '''
        Forward pass for length encoding
        
        n_bar: (bs,) tensor of number of bars
        '''
        length_buckets = (n_bar / self.length_bucket_size).long() 
        print(f'Length Buckets: {length_buckets}')
        length_embeds = self.length_embedding(length_buckets).unsqueeze(1)  # (bs, 1, dim)

        return length_embeds


class UnconditionalDiT(StableAudioDiTModel):
    def __init__(
        self,
        sample_size: int = 1024,
        in_channels: int = 64,
        num_layers: int = 24,
        attention_head_dim: int = 64,
        num_attention_heads: int = 24,
        num_key_value_attention_heads: int = 12,
        out_channels: int = 64,
        cross_attention_dim: int = 768,
        time_proj_dim: int = 256,
        global_states_input_dim: int = 1536,
        cross_attention_input_dim: int = 768,
    ):
        self.input_dim = in_channels
        self.max_pos = sample_size
        super().__init__(
            sample_size=sample_size,
            in_channels=in_channels,
            num_layers=num_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            num_key_value_attention_heads=num_key_value_attention_heads,
            out_channels=out_channels,
            cross_attention_dim=cross_attention_dim,
            time_proj_dim=time_proj_dim,
            global_states_input_dim=global_states_input_dim,
            cross_attention_input_dim=cross_attention_input_dim,
        )
        

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        timestep: torch.LongTensor = None,
        encoder_hidden_states: torch.FloatTensor = None,
        global_hidden_states: torch.FloatTensor = None,
        rotary_embedding: torch.FloatTensor = None,
        return_dict: bool = True,
        attention_mask: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`StableAudioDiTModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, in_channels, sequence_len)`):
                Input `hidden_states`.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, encoder_sequence_len, cross_attention_input_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            global_hidden_states (`torch.FloatTensor` of shape `(batch size, global_sequence_len, global_states_input_dim)`):
               Global embeddings that will be prepended to the hidden states.
            rotary_embedding (`torch.Tensor`):
                The rotary embeddings to apply on query and key tensors during attention calculation.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_len)`, *optional*):
                Mask to avoid performing attention on padding token indices, formed by concatenating the attention
                masks
                    for the two text encoders together. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            encoder_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_len)`, *optional*):
                Mask to avoid performing attention on padding token cross-attention indices, formed by concatenating
                the attention masks
                    for the two text encoders together. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        cross_attention_hidden_states = self.cross_attention_proj(encoder_hidden_states)
        global_hidden_states = self.global_proj(global_hidden_states)
        time_hidden_states = self.timestep_proj(self.time_proj(timestep.to(self.dtype)))

        global_hidden_states = global_hidden_states + time_hidden_states.unsqueeze(1)

        hidden_states = self.preprocess_conv(hidden_states) + hidden_states
        # (batch_size, dim, sequence_length) -> (batch_size, sequence_length, dim)
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.proj_in(hidden_states)

        # prepend global states to hidden states
        hidden_states = torch.cat([global_hidden_states, hidden_states], dim=-2)
        if attention_mask is not None:
            prepend_mask = torch.ones((hidden_states.shape[0], 1), device=hidden_states.device, dtype=torch.bool)
            attention_mask = torch.cat([prepend_mask, attention_mask], dim=-1)

        # Prepare rotary embeddings
        rotary_embed_dim = self.attention_head_dim // 2
        rotary_embedding = get_1d_rotary_pos_embed(
            rotary_embed_dim,
            hidden_states.shape[1],
            use_real=True,
            repeat_interleave_real=False,
        )

        # # Add learnable positional embedding
        # pos_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0).expand(hidden_states.shape[0], -1)
        # pos_emb = self.pos_emb(pos_ids)
        # hidden_states = hidden_states + pos_emb

        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    attention_mask,
                    cross_attention_hidden_states,
                    encoder_attention_mask,
                    rotary_embedding,
                )

            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=cross_attention_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    rotary_embedding=rotary_embedding,
                )

        hidden_states = self.proj_out(hidden_states)

        # (batch_size, sequence_length, dim) -> (batch_size, dim, sequence_length)
        # remove prepend length that has been added by global hidden states
        hidden_states = hidden_states.transpose(1, 2)[:, :, 1:]
        hidden_states = self.postprocess_conv(hidden_states) + hidden_states

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)
    
    def generate(self, n_sample, noise_scheduler, verbose=False, length_embeds=None, sec_embeds=None, sec_attn_mask=None):
        '''
        Generate n latent sequences
        '''
        max_pos = self.max_pos
        x = torch.randn(n_sample, max_pos, self.input_dim).to(self.device)

        if length_embeds is None:
            length_embeds = torch.zeros(n_sample, 1, 128).to(x.device) # dummy variable

        if sec_embeds is None:
            sec_embeds = torch.zeros(n_sample, 7, 128).to(x.device) # dummy variable

        # Sampling loop
        if verbose is False:
            loop = enumerate(noise_scheduler.timesteps)
            # for i, t in loop:
            #     x = x.to(self.device)
            #     t = t.to(self.device)
            #     # broadcast t to (bs,)
            #     t = t.expand(x.size(0)).to(x.device)

            #     with torch.no_grad():
            #         residual = self(
            #             x.transpose(1, 2),
            #             t,
            #             encoder_hidden_states=sec_embeds,  # use section embedding as encoder states
            #             global_hidden_states=length_embeds,  # use length embedding as global states
            #         )["sample"].transpose(1, 2)

            #     # Update sample with step
            #     x = noise_scheduler.step(residual, t[0], x).prev_sample
        else: # use tqdm
            loop = enumerate(tqdm(noise_scheduler.timesteps))
            
        for i, t in loop:
            x = x.to(self.device)
            t = t.to(self.device)
            # broadcast t to (bs,)
            t = t.expand(x.size(0)).to(x.device)
            with torch.no_grad():
                residual = self(
                    x.transpose(1, 2),
                    t,
                    encoder_hidden_states=sec_embeds,  # use section embedding as encoder states
                    global_hidden_states=length_embeds,  # use length embedding as global states
                    encoder_attention_mask=sec_attn_mask
                )["sample"].transpose(1, 2)

            # Update sample with step
            x = noise_scheduler.step(residual, t[0], x).prev_sample

        return x


class UnconditionedTransformerEnc(nn.Module):
    """
    A simple transformer encoder without any conditioning
    1. Input: (batch, seq_len=4, feature_dim=512)
    2. Output: (batch, seq_len=4, feature_dim=512)
    """

    def __init__(self, d_model=512, d_ffn=2048, n_head=8, n_layer=6, max_time_step=1000):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, batch_first=True, dim_feedforward=d_ffn,
        )
        self.model = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

        self.timestep_emb = nn.Embedding(max_time_step, d_model)

        MAX_BAR = 128
        self.pos_emb = nn.Embedding(MAX_BAR, d_model)

        # sinusoidal positional encoding
        self.pos_enc = nn.Parameter(self._generate_sinusoidal_encoding(MAX_BAR, d_model), requires_grad=False)

    def _generate_sinusoidal_encoding(self, max_len, d_model):
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)

    def forward(self, x, timestep):
        # x: (batch, seq_len=4, feature_dim=512)
        # timestep: (batch,)

        # If timestep is a single integer or scalar tensor, broadcast to batch size
        if isinstance(timestep, int) or timestep.ndim == 0:
            timestep = torch.full((x.size(0),), timestep, dtype=torch.long, device=x.device)

        # Time embedding
        time_emb = self.timestep_emb(timestep).unsqueeze(1)  # (batch, 1, d_model)
        x = x + time_emb  # Broadcast addition

        # Get positional embedding
        seq_len = x.size(1)
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.size(0), -1) # (batch, seq_len)
        pos_emb = self.pos_emb(pos_ids)  # (batch, seq_len, d_model)
        x = x + pos_emb

        x = self.model(x)

        return x
    
    @classmethod
    def from_lit_ckpt(cls, ckpt_path):
        # Load model parameters from the specified path of a LightningModule checkpoint
        lit_ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = lit_ckpt['state_dict']
        config = lit_ckpt['hyper_parameters']['model_config']
        
        model = cls(**config)

        # Remove the 'model.' prefix if it exists
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_key = k[len('model.'):]
            else:
                new_key = k
            new_state_dict[new_key] = v

        model.load_state_dict(new_state_dict, strict=True)
        model.eval()
        return model
        

if __name__ == "__main__":
    main()
