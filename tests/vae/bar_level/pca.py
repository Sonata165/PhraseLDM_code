import os
import sys
dirof = os.path.dirname
sys.path.append('/home/longshen/work/AccGen/AccGen')

import torch
from models.phrase_vae import load_t5_model_from_lit_ckpt, S2SVQAE

ckpt_fp = '/data1/longshen/Results/AccGenResults/bar_compression/s2s/mqcomp_ft/q4_d512_l3_lr1e-4/tb_logs/version_1/checkpoints/epoch=129_step=69420_val_loss=0.0077.ckpt'
t5 = load_t5_model_from_lit_ckpt(ckpt_fp)

model = S2SVQAE(
    tokenizer_path='LongshenOu/phrase-vae-tokenizer',
    t5_config={
        'd_model': 512,
        'd_ff': 1024,
        'num_layers': 3,
        'num_heads': 6,
        'vocab_size': 1000,
        'decoder_start_token_id': 1
    },
    t5_model_name=ckpt_fp,
    lit_ckpt=True,
    compress_style='first_n_tokens',
    n_compress_tokens=4,
)
model.t5 = t5
model.eval()

# Prepare tokenizer
from transformers import PreTrainedTokenizerFast
tokenizer_path = 'LongshenOu/phrase-vae-tokenizer'
tok = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

# Prepare tokenizer
from transformers import PreTrainedTokenizerFast
tokenizer_path = 'LongshenOu/phrase-vae-tokenizer'
tok = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

# Prepare data
from sonata_utils import read_jsonl
jsonl_fp = '/home/longshen/work/AccGen/AccGen/data_preprocess/POP909/statistics/unique_piano_bars.jsonl'
data = read_jsonl(jsonl_fp)
bar1 = data[8]
bar2 = data[108]
print(bar1)
print(bar2)

from piano_roll_utils import save_piano_roll
from remi_z import MultiTrack

# Get latents of the two bars
def get_latents(model, tok, bar_str):
    # Add special tokens
    inp_seq = bar_str.split()
    inp_seq = ['[BOS]'] + inp_seq + ['[EOS]']
    inp_seq = [f'[C{i}]' for i in range(4)] + inp_seq
    bar_str = ' '.join(inp_seq)

    inputs = tok(bar_str, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
    input_ids = inputs.input_ids
    # Debug: detokenize
    # print(tok.batch_decode(input_ids, skip_special_tokens=False))
    attention_mask = inputs.attention_mask
    with torch.no_grad():
        latents = model.encode(input_ids, attention_mask)
    return latents

# Get latents of 100 random bars
latents_list = []
import random
random.seed(42)
from tqdm import tqdm
bar_indices = random.sample(range(len(data)), 1000)

for idx in tqdm(bar_indices):
    bar = data[idx]
    latents = get_latents(model, tok, bar)
    latents_list.append(latents['quantized'][0].numpy())
print(latents_list[0].shape)  # (4, 512)

def get_piano_roll_from_str(bar_str):
    bar_mt = MultiTrack.from_remiz_str(bar_str)[0]
    return bar_mt

from sonata_utils import create_dir_if_not_exist
save_dir = '/home/longshen/work/AccGen/AccGen/tests/check_latents'
create_dir_if_not_exist(save_dir)

for bar_id in bar_indices:
    bar_str = data[bar_id]
    bar_mt = get_piano_roll_from_str(bar_str)
    save_fp = os.path.join(save_dir, f'orig_bar_{bar_id}.mid')
    bar_mt.to_midi(save_fp)

exit(10)

def get_piano_roll_from_latents(model, tok, latents):
    out = model.decode(
            {'quantized':latents.unsqueeze(0)},
            max_length=128,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
            do_sample=False,
            num_beams=1,
        )
    # print(out)
    bar_str = tok.decode(out[0], skip_special_tokens=True)
    # print(bar_str)
    bar_mt = MultiTrack.from_remiz_str(bar_str)[0]
    proll = bar_mt.to_piano_roll(pos_per_bar=48)
    return proll, bar_mt


# PCA to 2D
from sklearn.decomposition import PCA
import numpy as np
latents_array = np.array(latents_list)  # (100, 4, 512)
latents_array_2d = latents_array.reshape(latents_array.shape[0], -1)  # (100, 2048)
pca = PCA(n_components=2)
latents_pca = pca.fit_transform(latents_array_2d)  # (100, 2)
print(latents_pca.shape)
print('Explained variance ratio:', pca.explained_variance_ratio_)
print('Cumulative explained variance ratio:', np.cumsum(pca.explained_variance_ratio_))
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
plt.scatter(latents_pca[:, 0], latents_pca[:, 1])
# for i, idx in enumerate(bar_indices):
#     plt.annotate(str(idx), (latents_pca[i, 0], latents_pca[i, 1]))
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('PCA of Latent Vectors of 100 Random Bars')
plt.grid()
plt.axis('equal')
plt.show()


# 3D PCA
from mpl_toolkits.mplot3d import Axes3D
pca_3d = PCA(n_components=3)
latents_pca_3d = pca_3d.fit_transform(latents_array_2d)  # (100, 3)
print(latents_pca_3d.shape)
print('Explained variance ratio:', pca_3d.explained_variance_ratio_)
print('Cumulative explained variance ratio:', np.cumsum(pca_3d.explained_variance_ratio_))
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(latents_pca_3d[:, 0], latents_pca_3d[:, 1], latents_pca_3d[:, 2], alpha=0.3)
for i, idx in enumerate(bar_indices):
    if i % 500 == 0:  # 每隔 500 个标注一个
        ax.text(latents_pca_3d[i, 0], latents_pca_3d[i, 1], latents_pca_3d[i, 2], str(idx), fontsize=8)
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
ax.set_title('3D PCA of Latent Vectors of 100 Random Bars')
plt.show()