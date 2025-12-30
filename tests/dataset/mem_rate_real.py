'''
Check real-vs-real memorization rate
'''

import os
import sys
sys.path.append('/home/longshen/work/AccGen/AccGen')

from models.ldm_inference import PhraseLDM
from sonata_utils import create_dir_if_not_exist, read_json, save_json
from remi_z import MultiTrack
from evaluation.metrics import Metric, memorization_rate
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def main():
    metric = Metric()

    cnt = 0
    for song_id in metric.mel_dict:
        mel = metric.mel_dict[song_id]
        mem_rate = memorization_rate(mel, {k:v for k,v in metric.mel_dict.items() if k != song_id})
        print(f'Song ID: {song_id}, Memorization Rate: {mem_rate:.4f}')
        
        cnt += 1
        if cnt >= 10:
            break

if __name__ == '__main__':
    main()