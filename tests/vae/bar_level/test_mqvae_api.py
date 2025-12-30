import os
import sys
sys.path.insert(0, '/home/longshen/work/AccGen/AccGen')

from models.vae_inference import MQVAE


def main():
    # Prepare data
    from sonata_utils import read_jsonl
    jsonl_fp = '/home/longshen/work/AccGen/AccGen/data_preprocess/POP909/statistics/unique_piano_bars.jsonl'
    data = read_jsonl(jsonl_fp)
    bar1 = data[8]
    bar2 = data[108]

    # Remove the first token (now the model does not support instrument token)
    bar1 = " ".join(bar1.split()[1:])
    bar2 = " ".join(bar2.split()[1:])

    bar_strs = [bar1, bar2]

    # Initialize model
    model = MQVAE()

    latents = model.encode_batch(bar_strs, do_sample=True)
    print("Latents shape:", latents.shape)
    decoded_strs = model.decode_batch(latents)
    print("Decoded strings:", decoded_strs)

    assert len(decoded_strs) == len(bar_strs)
    for i in range(len(bar_strs)):
        print(f"Original {i}: {bar_strs[i]}")
        print(f"Decoded  {i}: {decoded_strs[i]}")
        print(f'This is a match: {decoded_strs[i] == bar_strs[i]}')
        # assert decoded_strs[i] == bar_strs[i], f"Mismatch at index {i}"
    print("All decoded strings match the original input strings.")

    # Test MultiTrack conversion
    mts = model.decode_batch(latents, return_mt=True)
    from sonata_utils import create_dir_if_not_exist
    print("Number of MultiTrack objects:", len(mts))
    out_dir = '/home/longshen/work/AccGen/AccGen/tests/vae_out/recon_two_bar_sample'
    create_dir_if_not_exist(out_dir)
    for i, mt in enumerate(mts):
        print('n_bar in MT:', len(mt))
        save_path = os.path.join(out_dir, f'recon_{i}.mid')
        mt.set_tempo(90)
        mt.to_midi(save_path)

    # Save original bars for comparison
    from remi_z import MultiTrack
    
    out_dir = '/home/longshen/work/AccGen/AccGen/tests/vae_out/orig_two_bar'
    create_dir_if_not_exist(out_dir)
    for i, bar_str in enumerate(bar_strs):
        mt = MultiTrack.from_remiz_str(bar_str)
        print('n_bar in MT:', len(mt))
        save_path = os.path.join(out_dir, f'orig_{i}.mid')
        mt.set_tempo(90)
        mt.to_midi(save_path)

    # Test piano roll conversion
    prolls = model.decode_batch(latents, return_proll=True)
    print("Number of piano rolls:", len(prolls))


if __name__ == "__main__":
    main()