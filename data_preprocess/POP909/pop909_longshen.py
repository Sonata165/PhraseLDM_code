'''
Construct Longshen's ver of POP909 dataset
'''

import numpy as np
import os
from remi_z import MultiTrack, Bar
from sonata_utils import jpath, ls, save_jsonl, create_dir_if_not_exist
import shutil
from tqdm import tqdm

def main():
    test_key_norm()


def procedures():
    convert_to_midis()
    test_key_norm()


def test_key_norm():
    # Test key normalization
    song_dir = '/data1/longshen/Datasets/Piano/POP909/pop909_longshen/data'
    out_dir = '/data1/longshen/Datasets/Piano/POP909/pop909_longshen/data_key_normed'
    
    song_fns = ls(song_dir)
    song_fns.sort()
    pbar = tqdm(song_fns)
    for song in pbar:
        pbar.set_description(song)

        song_fp = os.path.join(song_dir, song, f'{song}.mid')
        assert os.path.exists(song_fp)
        mt = MultiTrack.from_midi(song_fp)
        pitch_shifted = mt.normalize_pitch()

        # Create output dir
        song_out_dir = os.path.join(out_dir, song)
        create_dir_if_not_exist(song_out_dir)
        save_fp = os.path.join(song_out_dir, f'{song}.mid')
        mt.to_midi(save_fp, verbose=False)
        
        # Split phrase sections (old ver)
        # There are some blank phrases, means there are failed MIDI files in Ziyu's ver ...
        src_phrase_dir = os.path.join(song_dir, song, 'phrases')
        tgt_phrase_dir = os.path.join(song_out_dir, 'phrases')
        create_dir_if_not_exist(tgt_phrase_dir)
        phrase_fns = ls(src_phrase_dir)
        phrase_fns.sort()
        phrase_mts = []
        for phrase_fn in phrase_fns:

            pbar.set_description(f'{song} {phrase_fn}')

            src_phrase_fp = os.path.join(src_phrase_dir, phrase_fn)
            phrase_mt = MultiTrack.from_midi(src_phrase_fp)
            phrase_mt.shift_pitch(pitch_shifted)
            tgt_phrase_fp = os.path.join(tgt_phrase_dir, phrase_fn)
            phrase_mt.to_midi(tgt_phrase_fp, verbose=False)
            phrase_mts.append(phrase_mt)

        # # Reconstruct the full song
        # full_mt = MultiTrack.concat(phrase_mts)
        # full_save_fp = os.path.join(song_out_dir, f'{song}_full.mid')
        # full_mt.to_midi(full_save_fp, verbose=False)

        # Copy the phrase_sections.txt
        ori_phrase_section_fp = os.path.join(song_dir, song, 'phrase_annot.txt')
        assert os.path.exists(ori_phrase_section_fp)
        save_phrase_section_fp = os.path.join(song_out_dir, 'phrase_annot.txt')
        shutil.copy(ori_phrase_section_fp, save_phrase_section_fp)

        # break



def convert_to_midis():
    data_root = '/data1/longshen/Datasets/Piano/POP909/phrase_splits/POP909 Phrase Split Data/Phrase Split Data'
    save_dir = '/data1/longshen/Datasets/Piano/POP909/pop909_longshen/data'
    tempo_refer_dir = '/data1/longshen/Datasets/Piano/POP909/quantized/midi'
    song_fns = ls(data_root)
    song_fns.sort()
    for song in tqdm(song_fns):

        # Get original tempo
        tempo_refer_fp = os.path.join(tempo_refer_dir, f'{song}.mid')
        assert os.path.exists(tempo_refer_fp)
        mt_tempo = MultiTrack.from_midi(tempo_refer_fp)
        tempo = mt_tempo.tempos[0]

        song_root = os.path.join(data_root, song)
        phrase_fns = ls(song_root)
        phrase_fns.sort()
        pbar = tqdm(phrase_fns)
        phrase_mts = []
        for file in pbar:
            pbar.set_description(file)

            if file.split('.')[-1] == 'npz':
                phrase_lable = file.split('.')[0].split('_')[1][0] #capital letter denotes vocal phrases
                phrase_length = int(file.split('.')[0].split('_')[1][1:])  #phrase length is measured in bars
                phrase_data = np.load(os.path.join(song_root, file))
                """The following four matrix paired with each other"""
                melody_phrase = phrase_data['melody']   # [n_pos, n_pitch]
                bridge_phrase = phrase_data['bridge']   #accompanying melody. Could be a zero matrix, too. Quantized in 16th note
                piano_phrase = phrase_data['piano'] #piano accompaniment, Should NOT be zero at MOST times. Quantized in 16th note
                chord_phrase = phrase_data['chord'] #chord transcription. Could be zero at some times. Quantized in quater notes
                #print(phrase_lable, phrase_length, melody_phrase.shape, bridge_phrase.shape, piano_phrase.shape, chord_phrase.shape)
                pass    #do what you need

                # Process one bar at a time
                melody_bars = []
                bridge_bars = []
                piano_bars = []
                for i in range(phrase_length):
                    mel_bar_proll = melody_phrase[i*16:(i+1)*16]
                    bri_bar_proll = bridge_phrase[i*16:(i+1)*16]
                    pia_bar_proll = piano_phrase[i*16:(i+1)*16]

                    mel_bar = Bar.from_piano_roll(mel_bar_proll, pos_per_bar=16)
                    bri_bar = Bar.from_piano_roll(bri_bar_proll, pos_per_bar=16)
                    pia_bar = Bar.from_piano_roll(pia_bar_proll, pos_per_bar=16)

                    melody_bars.append(mel_bar)
                    bridge_bars.append(bri_bar)
                    piano_bars.append(pia_bar)
                
                mt_piano = MultiTrack.from_bars(piano_bars)
                mt_melody = MultiTrack.from_bars(melody_bars)
                mt_bridge = MultiTrack.from_bars(bridge_bars)

                mt_merge = mt_piano.merge_with(mt_bridge, other_prog_id=25)
                mt_merge = mt_merge.merge_with(mt_melody, other_prog_id=13)

                # Create a dir for that song
                song_save_dir = jpath(save_dir, song)
                create_dir_if_not_exist(song_save_dir)
                phrase_dir = jpath(song_save_dir, 'phrases')
                create_dir_if_not_exist(phrase_dir)
                save_fp = jpath(phrase_dir, f"{file.split('.')[0]}.mid")
                mt_merge.set_tempo(tempo)
                mt_merge.to_midi(save_fp, verbose=False)

                phrase_mts.append(mt_merge)

        # Save the full song
        full_song_mt = MultiTrack.concat(phrase_mts)
        full_song_save_fp = jpath(song_save_dir, f"{song}.mid")
        full_song_mt.set_tempo(tempo)
        full_song_mt.to_midi(full_song_save_fp, verbose=False)

        # Copy the phrase_sections.txt
        ori_phrase_section_fp = jpath(song_root, 'phrase_sections.txt')
        assert os.path.exists(ori_phrase_section_fp)
        save_phrase_section_fp = jpath(song_save_dir, 'phrase_annot.txt')
        shutil.copy(ori_phrase_section_fp, save_phrase_section_fp)

        # break


if __name__ == "__main__":
    main()