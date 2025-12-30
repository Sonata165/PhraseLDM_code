from piano_roll_utils import save_piano_roll
from remi_z import MultiTrack
import numpy as np

def main():
    remiz_str = "s-9 t-30 i-0 o-0 p-47 d-18 o-3 p-54 d-15 o-6 p-59 d-9 o-9 p-66 d-15 o-24 p-49 d-15 o-27 p-56 d-12 o-30 p-61 d-12 o-33 p-65 d-9 b-1"
    bar = MultiTrack.from_remiz_str(remiz_str)[0]
    ret = bar.to_piano_roll(pos_per_bar=48)
    ret = np.clip(ret, 0, 48)  # clip duration to at most 48 (one whole note)
    # ret = ret / 48.0  # 归一化到0~1，最大时值为1
    
    save_piano_roll(ret, "/home/longshen/work/AccGen/AccGen/temp/test_piano_tree/test_vis_roll.png")


if __name__ == "__main__":
    main()