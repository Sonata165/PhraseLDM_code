import os
import sys

sys.path.append("/home/longshen/work/AccGen/AccGen")

from PhraseLDM_code.models.diffusion_prior import UnconditionedTransformerEnc
import torch
from tqdm import tqdm
from diffusers import DDPMScheduler
from models.vae_inference import MQVAE_Pos, MQVAE
from sonata_utils import jpath, check_model_param, create_dir_if_not_exist
from remi_z import MultiTrack


def main():
    ckpt_fp = '/data1/longshen/Results/AccGenResults/diffusion_prior/pos_latent/1024sample/d128_ff512_l12_lr1e-4_dimscale/tb_logs/version_1/checkpoints/last.ckpt'
    model = UnconditionedTransformerEnc.from_lit_ckpt(
        ckpt_fp
    ).cuda()
    device = "cuda"

    check_model_param(model)
    
    max_pos = 17
    x = torch.randn(1, max_pos, 128).to(device)

    dim_wise_mean = torch.load('/data1/longshen/Datasets/Piano/POP909/latents/position_level/latent_dim_mean.pt') # [128]
    dim_wise_std = torch.load('/data1/longshen/Datasets/Piano/POP909/latents/position_level/latent_dim_std.pt') # [128]
    print("dim_wise_mean shape:", dim_wise_mean.shape)
    print("dim_wise_std shape:", dim_wise_std.shape)

    # Try decode
    ae = MQVAE_Pos().to(device)
    # ae = MQVAE(bottleneck=True).to(device)
    # scale_factor = 0.3209168016910553
    scaled_input = x[0] * dim_wise_std.to(device) + dim_wise_mean.to(device)
    out_str = ae.decode_batch(scaled_input)
    print("Decoded string:", out_str)

    
    # z_random = z_random * dim_wise_std + dim_wise_mean # [2, 128]


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

    # x shape: (bs, max_pos, dim)
    scaled_input = x[0] * dim_wise_std.to(device) + dim_wise_mean.to(device)

    with torch.no_grad():
        out_str = ae.decode_batch(scaled_input)

    # Print first 20 rows
    print("First 20 decoded strings:")
    for s in out_str:
        print(s)
    # exit(10)

    out = [MultiTrack.from_remiz_str(bar_str) for bar_str in out_str]

    # concat 
    out_bar_str = ' '.join(out_str)
    out_mt = MultiTrack.from_remiz_str(out_bar_str)
    out_dir = "/home/longshen/work/AccGen/test_outputs/diffuse_prior/pos_latent/1bar/l12"
    create_dir_if_not_exist(out_dir)
    out_mt.to_midi(jpath(out_dir, "sample_concat.mid"), tempo=90)

    # for i, mt in enumerate(out):
    #     save_path = os.path.join(out_dir, f"sample_{i}.mid")
    #     mt.to_midi(save_path, verbose=False)

 
if __name__ == "__main__":
    main()
