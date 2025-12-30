''' Latest inference code '''

import os
import sys

# Add project root to sys.path
dirof = os.path.dirname
try:
    dir_of_file = dirof(__file__)
except NameError:
    # .ipynb 文件中没有 __file__，使用当前工作目录
    dir_of_file = os.getcwd()
project_root = dir_of_file
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.ldm_inference import BarLDM
from sonata_utils import create_dir_if_not_exist
from remi_z import MultiTrack
from evaluation.metrics import visualize_ssm_for_song_bar, Metric
from the_utils.latent_utils import truncate_by_self_similarity

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Run this to obtain MIDI from a model

model_fp = '/data1/longshen/Results/AccGenResults/diffusion_prior/bar_latent/full_song/baseline/tb_logs/version_8/checkpoints/step_step=10000.ckpt'
save_root = '/data1/longshen/Results/AccGenResults/test_outputs/ldm/BarLDM'

model_name = 'baseline'
pt = 'scratch'
lencond = False
n_song = 5
n_bar = 64

model = BarLDM(ckpt_fp=model_fp, length_control=lencond)
model = model.cuda()
out, latent = model.generate(n_sample=n_song, bar_sep=' ', n_bars=n_bar)

# Calculate memorization rate
metric = Metric()

sample_lens = []
mem_rates = []
mts = [MultiTrack.from_remiz_str(out[i], remove_repeated_eob=True) for i in range(n_song)]
mem_rates = metric.memorization_rate_mt_batch(mts, return_list=True)
for i in range(n_song):
    mt = MultiTrack.from_remiz_str(out[i], remove_repeated_eob=True)
    sample_lens.append(len(mt))

# # Compute new sample rate
# new_sample_rate, _, top2_ratios = metric.new_sample_rate_batch([MultiTrack.from_remiz_str(s, remove_repeated_eob=True) for s in out])
# print('New sample rate of generated samples:', new_sample_rate)
# memorized_samples = [True if top2_ratios[i] < 0.3333 else False for i in range(n_song)]

# # Compute section recurrence score
# srs = metric.compute_srs(latent, sample_lens)
# print('Section Recurrence Scores of generated samples:', srs)

# # Compute FID
# fid = metric.compute_fid(out_tensor=latent, out_n_bars=sample_lens)
# print('FID of generated samples:', fid)

# Save MIDI
if 'last' in os.path.basename(model_fp):
    step_str = 'last'
else:
    step = os.path.basename(model_fp).split('=')[-1].split('.')[0]
    kstep = int(step) // 1000
    step_str = f'{kstep}k'

save_dir = f'{save_root}/{model_name}/from_{pt}/step_{step_str}'
create_dir_if_not_exist(save_dir)
ssm_dir = os.path.join(save_dir, 'ssm')
create_dir_if_not_exist(ssm_dir)
midi_dir = os.path.join(save_dir, 'midi')
create_dir_if_not_exist(midi_dir)
for i in range(n_song):
    mt = MultiTrack.from_remiz_str(out[i], remove_repeated_eob=True)
    mem_rate = mem_rates[i]
    # save_fn = f'{n_bar}bar_{i}_mem{mem_rate:.2f}_memorized{memorized_samples[i]}.mid'
    save_fn = f'{n_bar}bar_{i}_mem{mem_rate:.2f}.mid'
    save_fp = os.path.join(midi_dir, save_fn)
    mt.to_midi(save_fp, tempo=90, verbose=False)

    # Also save SSM visualization
    n_bars = sample_lens[i]
    z_trunc = latent[i][:n_bars*4]
    ssm_fn = f'{n_bar}bar_{i}.png'
    visualize_ssm_for_song_bar(z_trunc, save_dir=ssm_dir, filename=f'{n_bar}bar_{i}_mem{mem_rate:.2f}.png')

    


