import os
import sys

sys.path.append("/home/longshen/work/AccGen/AccGen")

from models.diffusion_prior_onebar import UnconditionedTransformerEnc
import torch
from tqdm import tqdm
from diffusers import DDPMScheduler
from models.vae_inference import MQVAE_Pos, MQVAE
from sonata_utils import jpath, check_model_param, create_dir_if_not_exist
from remi_z import MultiTrack


def main():
    latent_fp = '/data1/longshen/Datasets/Piano/POP909/latents/bar_level_pos_seq/train_latents.pt'
    latent = torch.load(latent_fp)  # (num_samples, max_pos, dim)
    print("Loaded latents:", latent.shape)

    # reshape latent
    latent = latent.reshape(-1, 128)  # (num_samples * max_pos, 128)

    sample = latent.cuda()  # (num_samples, max_pos, 128)

    ae = MQVAE_Pos().cuda()
    # ae = MQVAE(bottleneck=True).cuda()
    scale_factor = 0.3544125258922577
    with torch.no_grad():
        out_str = ae.decode_batch(sample[:100])  # scale back

    # Print first 20 rows
    print("output strings:")
    # print(out_str)
    for s in out_str:
        print(s)
    exit(10)

    out = [MultiTrack.from_remiz_str(bar_str) for bar_str in out]

    out_dir = "/home/longshen/work/AccGen/test_outputs/diffuse_prior/pos_latent/1bar/initial"
    create_dir_if_not_exist(out_dir)
    for i, mt in enumerate(out):
        save_path = os.path.join(out_dir, f"sample_{i}.mid")
        mt.to_midi(save_path, verbose=False)

 
if __name__ == "__main__":
    main()
