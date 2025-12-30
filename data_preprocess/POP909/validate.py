'''
Check phrase annotation integrity for POP909 dataset.
'''

import os
import sys

# Add project root to sys.path
dirof = os.path.dirname
try:
    dir_of_file = dirof(__file__)
except NameError:
    # .ipynb 文件中没有 __file__，使用当前工作目录
    dir_of_file = os.getcwd()
project_root = dirof(dirof(dir_of_file))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sonata_utils import ls, jpath, save_json
from remi_z import MultiTrack
from tqdm import tqdm


def main():
    check_phrase_annotation()


def procedures():
    clean_annotation()
    check_phrase_annotation()


def clean_annotation():
    '''
    i4A4A4B4C12A4A4B4C11X7C11X3
    ->
    i4 A4 A4 B4 C12 A4 A4 B4 C11 X7 C11 X3
    '''
    data_dir = '/data1/longshen/Datasets/Piano/POP909/pop909_longshen/data'
    song_names = ls(data_dir)
    pbar = tqdm(song_names)
    for song_name in pbar:
        pbar.set_description(f'Checking {song_name}')
        
        annot_fp = jpath(data_dir, song_name, 'phrase_annot.txt')
        with open(annot_fp, 'r') as f:
            annot = f.read().strip()
        
        # annotation loos like i4A4B8A4A4b4B8A4A4b4b4A4A4b4A4o3
        new_annot = ''
        i = 0
        while i < len(annot):
            c = annot[i]
            new_annot += c
            if c.isdigit():
                # collect all digits
                j = i + 1
                while j < len(annot) and annot[j].isdigit():
                    new_annot += annot[j]
                    j += 1
                new_annot += ' '
                i = j
            else:
                i += 1
        new_annot = new_annot.strip()

        annot_new_fp = jpath(data_dir, song_name, 'phrase_annot_cleaned.txt')
        with open(annot_new_fp, 'w') as f:
            f.write(new_annot)



def check_phrase_annotation():
    '''
    Check phrase annotation integrity for POP909 dataset.
    To ensure
    1. total number of bars in the phrase annotation equals that in the midi file.
    '''
    data_dir = '/data1/longshen/Datasets/Piano/POP909/pop909_longshen/data_key_normed'
    song_names = ls(data_dir)
    pbar = tqdm(song_names)

    n_errors = 0
    error_ids = []

    for song_name in pbar:
        pbar.set_description(f'Checking {song_name}')
        
        midi_fp = jpath(data_dir, song_name, f'{song_name}.mid')
        mt = MultiTrack.from_midi(midi_fp)
        
        annot_fp = jpath(data_dir, song_name, 'phrase_annot_cleaned.txt')
        with open(annot_fp, 'r') as f:
            annot = f.read().strip()
        
        n_bars_midi = len(mt)

        # annotation loos like i4 A4 B8 A4 A4 b4 B8 A4 A4 b4 b4 A4 A4 b4 A4 o3
        n_bar_annot = 0
        phrase_tokens = annot.split(' ')
        for phrase_token in phrase_tokens:
            assert len(phrase_token) >= 2, f'Invalid phrase token {phrase_token} in {song_name}'
            phrase_len = int(phrase_token[1:])
            # assert phrase_len > 0, f'Invalid phrase length {phrase_len} in {song_name}'
            if phrase_len <= 0:
                print(f'Warning: phrase length <= 0 in {song_name}')
            n_bar_annot += phrase_len

        if n_bar_annot != n_bars_midi:
            print(f'Inconsistency found in {song_name}: '
                  f'n_bars_midi={n_bars_midi}, n_bars_annot={n_bar_annot}')
            # raise ValueError('Phrase annotation inconsistency detected!')
            n_errors += 1
            error_ids.append(song_name)

    print(f'Checking completed. Total {n_errors} errors found.')
    
    # Note down error ids
    dataset_dir = '/data1/longshen/Datasets/Piano/POP909/pop909_longshen'
    error_fp = jpath(dataset_dir, 'phrase_annotation_errors.json')
    save_json(error_ids, error_fp)
        


if __name__ == '__main__':
    main()