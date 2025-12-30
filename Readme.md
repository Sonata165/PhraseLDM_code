# PhraseVAE and PhraseLDM

[![Paper](https://img.shields.io/badge/📄_Paper-arXiv-red.svg)](https://www.oulongshen.xyz/assets/pdf/papers/PhraseLDM_Report.pdf)
[![Demo](https://img.shields.io/badge/🎼_Demo-Listen-blue.svg)](https://www.oulongshen.xyz/midi_ldm)
[![Code](https://img.shields.io/badge/💻_Code-GitHub-black.svg)](https://github.com/Sonata165/PhraseLDM_code)
[![Models](https://img.shields.io/badge/🤗_Models-HuggingFace-yellow.svg)](https://huggingface.co/collections/LongshenOu/phrasevae-and-phraseldm)

[![Author](https://img.shields.io/badge/👤_Author-Longshen%20Ou-purple.svg)](https://www.oulongshen.xyz/)
[![REMI-z](https://img.shields.io/badge/🎹_REMI--z-Tokenization-green.svg)](https://github.com/Sonata165/REMI-z)
[![LinkedIn Post](https://img.shields.io/badge/📝_LinkedIn_Post-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/feed/update/urn:li:activity:7408804594631270400/)

This is the official implementation of the [PhraseVAE and PhraseLDM: Latent Diffusion for Full-Song Multitrack Symbolic Music Generation](https://www.oulongshen.xyz/midi_ldm).


## Overview
This project presents a hierachical model for full song multitrack symbolic music modeling. It consists of two components: (1) PhraseVAE, which compress each musical phrase to a 64-D latent representation with high reconstruction quality, and (2) PhraseLDM, which is a latent diffusion model that directly generate a full song with the help of phrase-level latent representation.

## Environment
    conda create -n phrase_ldm python=3.12
    conda activate phrase_ldm

    # Create a project dir
    mkdir PhraseLDM

    # Install pytorch
    pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

    # Install REMI-z
    git clone https://github.com/Sonata165/REMI-z.git
    cd REMI-z
    pip install -r Requirements.txt
    pip install -e .
    cd ..

    # Install SonataUtils
    git clone https://github.com/Sonata165/SonataUtil.git
    cd SonataUtil
    pip install -e .
    cd ..

    pip install tqdm==4.67.1
    pip install lightning==2.5.5
    pip install -U 'jsonargparse[signatures]>=4.27.7'
    pip install -U tensorboard
    pip install transformers==4.56.1 
    pip install diffusers==0.35.1 
    pip install accelerate==1.10.1
    pip install scikit-learn==1.7.2
    pip install plotly==6.3.0
    pip install pandas==2.3.2
    pip install jiwer==4.0.0

## What did I do
1. Convert POP909 dataset to midi version (`data_preprocess/preprocess.py`)
2. Prepare jsonl dataset at bar-level, position-level, phrase-level (`data_preprocess/preprocess.py`). Jsonl dataset is used for the VAE training. Note: bar-level jsonl is also needed since **the pretrain is done with the bar-level data**.
3. Train VAE model (3 procedures). The configs used are:
    - Stage 1: `hparams/aes/bar_level/pretrain/s2s_span_infill.yaml`
    - Stage 2: `hparams/aes/phrase_level/mqcomp_ft.yaml`
    - Stage 3: `hparams/aes/phrase_level/vae_aeft.yaml`
4. Generate latent for the entire dataset (`data_preprocess/precompute_latents_phr.py`)
5. Train the LDM.

## Inference with the model

### Inference with Models You Trained
PhraseVAE: see `tests/vae/phrase/interpolate_vae3.ipynb`

PhraseLDM: see `tests/ldm/phr_latent/check_infer.ipynb` or `infer.py`.

### Inference with Our Checkpoints
We release Hugging Face version of checkpoints for your convenience. See `tutorial.ipynb` for details.

## How to train with new data
The current training pipeline is implemented with `Lightning`. To train a model (both for VAE and LDM), run below command:
    
    CUDA_VISIBLE_DEVICES=1 python train.py --config [hparams/path_to_your_config_file]
The above script will do both training and testing (after all training epochs). Validation will be done during the training after a specified stpes. 

If you want to evaluate a specific checkpoint, run the command below:

    CUDA_VISIBLE_DEVICES=2 python train.py \
        --config [hparams/path_to_your_config_file] \
        --test \
        --ckpt_path [model_log_dir]/tb_logs/version_2/checkpoints/last.ckpt

### PhraseVAE
If you continue to train LDM models on key-normalized POP909 and continue to use the phrase-level latents, training another PhraseVAE is not necessary. 

The current VAE only operates on songs in the key of C major or A minor (see [Key Normalization](#key-normalization) for details). And we can shift the key to any one that user desired, just by changing the pitch value of all notes. If you want to support generating music in all keys natively, retraining the PhraseVAE is necessary, meanwhile using key shifting data augmentation. Just follow the 3-stage method in [the previous section](#what-did-i-do), and prepare proper jsonl dataset beforehand.

### PhraseLDM
To train new models with existing dataset (POP909), just modify the configuration file and rerun the training command. You may want to change the dataset path, model class name (if you modified the architecture), and other hyperparameters like learning rate, in the configuration yaml file.

## Some important notes
### VAE latent need to be scaled
VAE latent need to be scaled to standard normal distribution in LDM training. This is done by 
1. Check the average dimension-wise standard deviation value of all latents in the training set. In our case, this is a value around 0.75901.
2. Devide the VAE latent with this std value and then feed to the LDM.
3. In inference, LDM output need to be multiplied by the same std value and then send to VAE for decoding.

### Set a proper clipping range
The standard DDPM scheduler has a default cliping range of 1.0 since it is designed for image. In our case, we want to set it to a larger value to support the latents in a standard normal distribution. We observe the 99.9% percentile of the absolute value of the latent values (in our case, 3.06445) and set this as the clipping value for the noise scheduler. (**This is actually a little buggy**, since the LDM actually received scaled latents. A better clipping value would be 3.06445 / 0.75901). May need fix in future experiments.

### Sanity check
After VAE training, we need to encode and decode existing REMI-z strings with it to see if it can do a perfect reconstruction. If not, it is buggy.

### **PhraseVAE special token vocab**
- **[BOS]**: Begin of token sequence *for PhraseVAE*
- **[EOS]**: End of token sequence *for PhraseVAE*
- **[INST]**: Instrument padding token *for PhraseLDM*
- **[SEP]**: End of latent sequence *for PhraseLDM*
### Key normalization
Our experiments are done with the assumption that key shifting do not negatively impact the playability of music. This is not strictly true, but is usually valid in majority of cases. Under such assumption, generating music that are in all 24 keys can be reduced to generating music in a single specified key (C major / A minor) and then shifting it to the desired key by the user. This is done with `Multitrack.normalize_pitch` in `REMI-z`. **Note: this operation must be conducted in song level!**

### Half Precision Not Working
The current LDM does not support BF16 or FP16 training/inference yet. BF16 cause LDM loss value decrease lower, and diverge in latter half of training. FP16 converge well but the generation output is not of high quality (maybe somehow relates to the VAE implementation). With the current implementation, please use FP32 training. To support half precision training, you may want to check implementation and ensure both VAE and LDM works in a lower precision.

## Some Useful Commands
    
    Run the training
    CUDA_VISIBLE_DEVICES=0 python train.py --config hparams/bar_comp_vqvae.yaml

    CUDA_VISIBLE_DEVICES=1 python train.py --config hparams/bar_comp_pianotree.yaml

    CUDA_VISIBLE_DEVICES=1 python train.py --config hparams/bar_comp_s2svqae.yaml

    # Testing
    CUDA_VISIBLE_DEVICES=2 python train.py \
        --config /home/longshen/work/AccGen/AccGen/hparams/aes/bar_level/s2s/vae_bottleneck_multitrack/aeft_klw0.01_b256_4dec.yaml \
        --test \
        --ckpt_path /data1/longshen/Results/AccGenResults/aes/bar_level/pop909_multitrack/mqcomp_vae_bottleneck/aeft_lr1e-4_klw0.01_b256/tb_logs/version_2/checkpoints/last.ckpt