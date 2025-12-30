import os
import sys

sys.path.append("/home/longshen/work/AccGen/AccGen")

from models.diffusion_prior_onebar import UnconditionedTransformerEnc
import torch
from tqdm import tqdm
from diffusers import DDPMScheduler
from models.vae_inference import MQVAE
from sonata_utils import jpath, check_model_param, create_dir_if_not_exist
from remi_z import MultiTrack


def main():
    ckpt_fp = '/data1/longshen/Results/AccGenResults/diffusion_prior/4bar/6layer_vae_scaled/bs1280_lr1e-3_ep1000_new/tb_logs/version_0/checkpoints/epoch=419_step=20160_val_loss=0.0890.ckpt'
    model = UnconditionedTransformerEnc.from_lit_ckpt(
        ckpt_fp
    ).cuda()
    device = "cuda"

    check_model_param(model)

    n_bars = 4
    x = torch.randn(10, 4 * n_bars, 512).to(device)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2"
    )

    # Sampling loop
    for i, t in enumerate(tqdm(noise_scheduler.timesteps)):

        x = x.to(device)
        t = t.to(device)

        # print(x.device, t.device)

        # Get model pred
        with torch.no_grad():
            residual = model(x, t) 

        # Update sample with step
        x = noise_scheduler.step(residual, t, x).prev_sample

    # check std
    print("Final std:", x.std().item())

    ae = MQVAE().to(device)
    scale_factor = 0.306642085313797
    with torch.no_grad():
        out = ae.decode_batch(x, return_mt=True, scale_factor=scale_factor)  # scale back

    out_dir = "/home/longshen/work/AccGen/test_outputs/diffuse_prior/4bars_posemb/ep419"
    create_dir_if_not_exist(out_dir)
    for i, mt in enumerate(out):
        save_path = os.path.join(out_dir, f"sample_{i}.mid")
        mt.to_midi(save_path)

 
if __name__ == "__main__":
    main()
