import os
import sys
# sys.path.append()
sys.path.insert(0, '/home/longshen/work/AccGen/AccGen')

from models.vae_inference import MQAE


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
    model = MQAE()

    latents = model.encode_batch(bar_strs)
    print("Latents shape:", latents.shape)
    decoded_strs = model.decode_batch(latents)
    print("Decoded strings:", decoded_strs)

    assert len(decoded_strs) == len(bar_strs)
    for i in range(len(bar_strs)):
        assert decoded_strs[i] == bar_strs[i], f"Mismatch at index {i}"
    print("All decoded strings match the original input strings.")

    # Test MultiTrack conversion
    mts = model.decode_batch(latents, return_mt=True)
    print("Number of MultiTrack objects:", len(mts))
    for i, mt in enumerate(mts):
        print('n_bar in MT:', len(mt))

    # Test piano roll conversion
    prolls = model.decode_batch(latents, return_proll=True)
    print("Number of piano rolls:", len(prolls))


if __name__ == "__main__":
    main()