import os
import sys
sys.path.append('/home/longshen/work/AccGen/AccGen')

from models.ldm_inference import PhraseLDM
from sonata_utils import create_dir_if_not_exist, read_json, save_json
from remi_z import MultiTrack
from evaluation.metrics import Metric
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# model_fp = '/data1/longshen/Results/AccGenResults/diffusion_prior/phr_latent/full_song/baseline_lencond/lr5e-4_bs32/tb_logs/version_1/checkpoints/step_step=10000.ckpt'

# metric = Metric()
# model = PhraseLDM(length_control=True, ckpt_fp=model_fp)
# model = model.cuda()

# n_sample = 20
# n_bar = 32
# sample_id = 1
# save_dir = f'/data1/longshen/Results/AccGenResults/test_outputs/ldm/PhrLDM/test_metric/'
# create_dir_if_not_exist(os.path.dirname(save_dir))

def main():
    save_dir = '/data1/longshen/Results/AccGenResults/diffusion_prior/phr_latent/full_song/baseline_lencond/tb_logs/version_2'
    ckpt_dir = '/data1/longshen/Results/AccGenResults/diffusion_prior/phr_latent/full_song/baseline_lencond/tb_logs/version_2/checkpoints'
    calculate_mem_rate_for_ckpts(ckpt_dir=ckpt_dir, save_dir=save_dir)

def calculate_mem_rate_for_ckpts(ckpt_dir, save_dir):
    ckpt_fps = [os.path.join(ckpt_dir, fp) for fp in os.listdir(ckpt_dir) if fp.endswith('.ckpt')]
    # Remove last.ckpt if exists
    if os.path.exists(os.path.join(ckpt_dir, 'last.ckpt')):
        ckpt_fps.remove(os.path.join(ckpt_dir, 'last.ckpt'))
    # Sort by filename
    ckpt_fps.sort()

    n_sample = 5
    n_bar = 64

    mem_rate_results = {}
    pbar = tqdm(ckpt_fps)
    for ckpt_fp in pbar:
        pbar.set_description(f'Processing {os.path.basename(ckpt_fp)}')
        step = os.path.basename(ckpt_fp).split('=')[-1].split('.')[0]
        step_save_dir = os.path.join(save_dir, 'mem_rate_test', f'{n_bar}bar_{n_sample}sample', f'step_{step}')
        t = calculate_mem_rate_for_ckpt(ckpt_fp, n_sample=5, n_bar=64, save_dir=step_save_dir)
        mem_rate_results[step] = t

    # Sort dict by key's integer value
    mem_rate_results = dict(sorted(mem_rate_results.items(), key=lambda item: int(item[0])))

    # Save results to json
    save_fp = os.path.join(save_dir, f'mem_rate_results_{n_bar}bar_{n_sample}sample.json')
    save_json(mem_rate_results, save_fp)

def calculate_mem_rate_for_ckpt(ckpt_fp, n_sample, n_bar, save_dir):
    create_dir_if_not_exist(save_dir)

    metric = Metric()
    model = PhraseLDM(length_control=True, ckpt_fp=ckpt_fp)
    model = model.cuda()

    out, latent = model.generate(n_sample=n_sample, bar_sep=' ', n_bars=n_bar, verbose=False)
    mts = [MultiTrack.from_remiz_str(out[i], remove_repeated_eob=True) for i in range(len(out))]

    for i, mt in enumerate(mts):
        save_fp = os.path.join(save_dir, f'step160k_{n_bar}bar_{i}.mid')
        mt.to_midi(save_fp, tempo=90, verbose=False)
    mem_rate = metric.memorization_rate(mts)
    print(f'Checkpoint: {ckpt_fp}, Memorization Rate: {mem_rate:.4f}')

    return mem_rate

if __name__ == '__main__':
    main()