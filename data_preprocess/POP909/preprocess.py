import numpy as np
from remi_z import MultiTrack, Bar
from sonata_utils import jpath, ls, save_jsonl, create_dir_if_not_exist, read_jsonl
from tqdm import tqdm


def main():
    construct_phrase_jsonl()


def procedures():
    all_npz_to_midi() # Convert POP909 original data to midi version

    construct_jsonl_bar_flattened()     # Prepare bar-level flattened jsonl file
    construct_jsonl_bar_multitrack()    # Prepare bar-level multitrack jsonl file
    construct_position_level_jsonl()    # Prepare position-level jsonl file
    construct_phrase_jsonl()            # Prepare phrase-level jsonl file
    


def construct_phrase_jsonl():
    """
    Create phrase-level (track within bar) jsonl file for POP909 dataset
    """
    dataset_dir = "/data1/longshen/Datasets/Piano/POP909/pop909_longshen/data_key_normed"
    song_fns = ls(dataset_dir)

    # Randomly select 5% songs as test set
    test_ratio = 0.05
    song_fns = sorted(song_fns)
    test_num = int(len(song_fns) * test_ratio)
    np.random.seed(42)
    test_indices = np.random.choice(len(song_fns), size=test_num, replace=False)
    test_indices = set(test_indices)
    train_fns = [fn for i, fn in enumerate(song_fns) if i not in test_indices]
    val_fns = [fn for i, fn in enumerate(song_fns) if i in test_indices]

    train_lines = []
    for song_fn in tqdm(train_fns):
        song_fp = jpath(dataset_dir, song_fn, f'{song_fn}.mid')
        mt = MultiTrack.from_midi(song_fp)
        for bar in mt.bars:
            for track_id, track in bar.tracks.items():
                track_seq = track.to_remiz_seq(with_velocity=False)
                train_lines.append(" ".join(track_seq))
            train_lines.append('[INST]') # LDM pad token
            train_lines.append("b-1")  # bar separator
        train_lines.append("[SEP]")  # LDM end of song token
        # break

    test_lines = []
    for song_fn in tqdm(val_fns):
        song_fp = jpath(dataset_dir, song_fn, f'{song_fn}.mid')
        mt = MultiTrack.from_midi(song_fp)
        for bar in mt.bars:
            for track_id, track in bar.tracks.items():
                track_seq = track.to_remiz_seq(with_velocity=False)
                test_lines.append(" ".join(track_seq))
            test_lines.append('[INST]')  # LDM pad token
            test_lines.append("b-1")
        test_lines.append("[SEP]")  # LDM end of song token
        # break

    save_dir = "/data1/longshen/Datasets/Piano/POP909/jsonl/phrase_level"
    create_dir_if_not_exist(save_dir)

    save_fn = "train.jsonl"
    save_fp = jpath(save_dir, save_fn)
    save_jsonl(train_lines, save_fp)
    save_fn = "val.jsonl"
    save_fp = jpath(save_dir, save_fn)
    save_jsonl(test_lines, save_fp)


def check_midi_recon_result():
    dataset_fp = "/data1/longshen/Datasets/Piano/POP909/jsonl/bar_level_piano/pop909_piano_track_bars_val.jsonl"
    data = read_jsonl(dataset_fp)
    

    save_dir = '/home/longshen/work/AccGen/AccGen/test_outputs/dataset_preparation/pop909'
    create_dir_if_not_exist(save_dir)
    
    for i in range(10):
        # random select a bar id
        bar_id = np.random.randint(len(data))
        bar = data[bar_id]
        b = MultiTrack.from_remiz_str(bar)
        save_fp = jpath(save_dir, f'test_piano_bar_{i}.mid')
        b.to_midi(save_fp)


def construct_position_level_jsonl():
    '''
    Read the bar-level jsonl file and convert to position-level jsonl file
    '''
    dataset_dir = "/data1/longshen/Datasets/Piano/POP909/jsonl/bar_level"
    out_dir = '/data1/longshen/Datasets/Piano/POP909/jsonl/position_level'
    create_dir_if_not_exist(out_dir)

    bar_level_fns = ["train.jsonl", "val.jsonl"]
    for bar_level_fn in bar_level_fns:
        bar_level_fp = jpath(dataset_dir, bar_level_fn)
        bars = read_jsonl(bar_level_fp)

        pos_level_data = []
        for bar_str in bars:
            assert isinstance(bar_str, str)
            bar_seq = bar_str.strip().split(' ')
            cur_pos = None
            note_of_pos = []
            for token in bar_seq:
                if token.startswith('o-'):
                    # add notes of previous position
                    if cur_pos is not None:
                        pos_level_data.append(" ".join(note_of_pos))
                        note_of_pos = []
                    cur_pos = int(token[2:])
                if not token.startswith('b-'):
                    note_of_pos.append(token)
            if cur_pos is not None and len(note_of_pos) > 0:
                pos_level_data.append(" ".join(note_of_pos))

            pos_level_data.append("b-1")  # bar separator

        save_dir = "/data1/longshen/Datasets/Piano/POP909/jsonl/position_level"
        create_dir_if_not_exist(save_dir)
        save_fp = jpath(save_dir, bar_level_fn)
        save_jsonl(pos_level_data, save_fp)


def construct_jsonl_bar_multitrack():
    """
    Create bar-level jsonl file for POP909 dataset
    All tracks
    """
    dataset_dir = "/data1/longshen/Datasets/Piano/POP909/pop909_longshen/data_key_normed"
    song_fns = ls(dataset_dir)
    lines = []

    # Randomly select 10% songs as test set
    song_fns = sorted(song_fns)
    test_num = int(len(song_fns) * 0.1)
    np.random.seed(42)
    test_indices = np.random.choice(len(song_fns), size=test_num, replace=False)
    test_indices = set(test_indices)
    train_fns = [fn for i, fn in enumerate(song_fns) if i not in test_indices]
    val_fns = [fn for i, fn in enumerate(song_fns) if i in test_indices]

    train_lines = []
    for song_fn in tqdm(train_fns):
        song_fp = jpath(dataset_dir, song_fn, f'{song_fn}.mid')
        mt = MultiTrack.from_midi(song_fp)
        for bar in mt.bars:
            seq = bar.to_remiz_seq(
                with_ts=False,
                with_tempo=False,
                with_velocity=False,
                include_drum=False,
            )
            train_lines.append(" ".join(seq))
        train_lines.append("[SEP]") # end of song token

    test_lines = []
    for song_fn in tqdm(val_fns):
        song_fp = jpath(dataset_dir, song_fn, f'{song_fn}.mid')
        mt = MultiTrack.from_midi(song_fp)
        for bar in mt.bars:
            seq = bar.to_remiz_seq(
                with_ts=False,
                with_tempo=False,
                with_velocity=False,
                include_drum=False,
            )
            test_lines.append(" ".join(seq))
        test_lines.append("[SEP]") # end of song token

    save_dir = "/data1/longshen/Datasets/Piano/POP909/jsonl/bar_level"
    create_dir_if_not_exist(save_dir)
    save_fn = "train.jsonl"
    save_fp = jpath(save_dir, save_fn)
    save_jsonl(train_lines, save_fp)
    save_fn = "val.jsonl"
    save_fp = jpath(save_dir, save_fn)
    save_jsonl(test_lines, save_fp)


def construct_jsonl_bar_flattened():
    """
    Create bar-level jsonl file for POP909 dataset
    Piano track only
    """
    dataset_dir = "/data1/longshen/Datasets/Piano/POP909/pop909_longshen/data_key_normed"
    song_fns = ls(dataset_dir)
    lines = []

    # Randomly select 10% songs as test set
    song_fns = sorted(song_fns)
    test_num = int(len(song_fns) * 0.1)
    np.random.seed(42)
    test_indices = np.random.choice(len(song_fns), size=test_num, replace=False)
    test_indices = set(test_indices)
    train_fns = [fn for i, fn in enumerate(song_fns) if i not in test_indices]
    val_fns = [fn for i, fn in enumerate(song_fns) if i in test_indices]

    train_lines = []
    for song_fn in tqdm(train_fns):
        song_fp = jpath(dataset_dir, song_fn, f'{song_fn}.mid')
        mt = MultiTrack.from_midi(song_fp)
        for bar in mt.bars:
            # bar.filter_tracks(insts=[0])

            flattened = bar.flatten()

            seq = flattened.to_remiz_seq(
                with_ts=True,
                with_tempo=True,
                with_velocity=False,
                include_drum=False,
            )

            # remove time signature, tempo, instrument
            if len(seq) > 3:
                seq = seq[3:]

            train_lines.append(" ".join(seq))
        # break

    test_lines = []
    for song_fn in tqdm(val_fns):
        song_fp = jpath(dataset_dir, song_fn, f'{song_fn}.mid')
        mt = MultiTrack.from_midi(song_fp)
        for bar in mt.bars:
            # bar.filter_tracks(insts=[0])

            flattened = bar.flatten()

            seq = flattened.to_remiz_seq(
                with_ts=True,
                with_tempo=True,
                with_velocity=False,
                include_drum=False,
            )

            # remove time signature, tempo, instrument
            if len(seq) > 3:
                seq = seq[3:]

            test_lines.append(" ".join(seq))
        # break

    save_dir = "/data1/longshen/Datasets/Piano/POP909/jsonl/bar_level"
    create_dir_if_not_exist(save_dir)
    save_fn = "train.jsonl"
    save_fp = jpath(save_dir, save_fn)
    save_jsonl(train_lines, save_fp)
    save_fn = "val.jsonl"
    save_fp = jpath(save_dir, save_fn)
    save_jsonl(test_lines, save_fp)



def all_npz_to_midi_4bin():
    npy_dir = "/data1/longshen/Datasets/Piano/POP909/piano_rolls/POP09-PIANOROLL-4-bin-quantization"
    save_dir = "/data1/longshen/Datasets/Piano/POP909/quantized/midi_4bin"
    ori_dir = "/data1/longshen/Datasets/Piano/POP909/original/POP909"
    song_ids = [t for t in ls(ori_dir) if t.isdigit()]

    create_dir_if_not_exist(save_dir)

    for song_id in tqdm(song_ids):
        npz_fp = jpath(npy_dir, f"{song_id}.npz")
        save_fp = jpath(save_dir, f"{song_id}.mid")
        ori_fp = jpath(ori_dir, f"{song_id}/{song_id}.mid")
        npz_to_midi_one_song(ori_fp, npz_fp, save_fp, bins_per_beat=4, pos_per_bar=16)


def all_npz_to_midi():
    npy_dir = "/data1/longshen/Datasets/Piano/POP909/piano_rolls/POP09-PIANOROLL-12-bin-quantization"
    save_dir = "/data1/longshen/Datasets/Piano/POP909/quantized/midi"
    ori_dir = "/data1/longshen/Datasets/Piano/POP909/original/POP909"
    song_ids = [t for t in ls(ori_dir) if t.isdigit()]

    for song_id in tqdm(song_ids):
        npz_fp = jpath(npy_dir, f"{song_id}.npz")
        save_fp = jpath(save_dir, f"{song_id}.mid")
        ori_fp = jpath(ori_dir, f"{song_id}/{song_id}.mid")
        npz_to_midi_one_song(ori_fp, npz_fp, save_fp)


def npz_to_midi_one_song(ori_fp, npy_fp, save_fp, bins_per_beat=12, pos_per_bar=48):
    """
    Process the npy file to midi file
    """
    t = MultiTrack.from_midi(ori_fp)
    tempo = t.tempos[0]

    d = np.load(npy_fp)
    pos_tot = len(d["beat"]) * bins_per_beat

    N_TRACK = 3  # melody, bridge, piano
    N_PITCH = 128
    pr = np.zeros((N_TRACK, pos_tot, N_PITCH, 2))
    for on_beat, sq, bin_per_beat, off_beat, eq, _, p, v in d["melody"]:
        assert (
            bin_per_beat == _ == bins_per_beat
        )  # should be quantization bins per beat
        s_ind = int(on_beat * bin_per_beat + sq)
        e_ind = int(off_beat * bin_per_beat + eq)
        p = int(p)
        pr[0, s_ind, p, 0] = e_ind - s_ind
        pr[0, s_ind, p, 1] = v

    for on_beat, sq, bin_per_beat, off_beat, eq, _, p, v in d["bridge"]:
        assert (
            bin_per_beat == _ == bins_per_beat
        )  # should be quantization bins per beat
        s_ind = int(on_beat * bin_per_beat + sq)
        e_ind = int(off_beat * bin_per_beat + eq)
        p = int(p)
        pr[1, s_ind, p, 0] = e_ind - s_ind
        pr[1, s_ind, p, 1] = v

    for on_beat, sq, bin_per_beat, off_beat, eq, _, p, v in d["piano"]:
        assert (
            bin_per_beat == _ == bins_per_beat
        )  # should be quantization bins per beat
        s_ind = int(on_beat * bin_per_beat + sq)
        e_ind = int(off_beat * bin_per_beat + eq)
        p = int(p)
        pr[2, s_ind, p, 0] = e_ind - s_ind
        pr[2, s_ind, p, 1] = v

    bars = []

    for pos_start in range(0, pos_tot, pos_per_bar):
        end_pos = pos_start + pos_per_bar
        t = Bar.from_piano_roll(pr[2, pos_start:end_pos, :, 0], pos_per_bar=pos_per_bar)
        bars.append(t)
    mt_piano = MultiTrack.from_bars(bars)
    # mt_piano.to_midi('/home/longshen/work/AccGen/AccGen/temp/test_piano.mid')

    bars = []
    for pos_start in range(0, pos_tot, pos_per_bar):
        end_pos = pos_start + pos_per_bar
        t = Bar.from_piano_roll(pr[1, pos_start:end_pos, :, 0], pos_per_bar=pos_per_bar)
        bars.append(t)
    mt_bridge = MultiTrack.from_bars(bars)
    # mt_bridge.to_midi('/home/longshen/work/AccGen/AccGen/temp/test_bridge.mid')

    bars = []
    for pos_start in range(0, pos_tot, pos_per_bar):
        end_pos = pos_start + pos_per_bar
        t = Bar.from_piano_roll(pr[0, pos_start:end_pos, :, 0], pos_per_bar=pos_per_bar)
        bars.append(t)
    mt_melody = MultiTrack.from_bars(bars)
    # mt_melody.to_midi('/home/longshen/work/AccGen/AccGen/temp/test_melody.mid')

    mt_merge = mt_piano.merge_with(mt_bridge, other_prog_id=25)
    mt_merge = mt_merge.merge_with(mt_melody, other_prog_id=13)

    mt_merge.set_tempo(tempo)

    mt_merge.to_midi(save_fp)


if __name__ == "__main__":
    main()
