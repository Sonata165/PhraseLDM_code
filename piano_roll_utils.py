import matplotlib.pyplot as plt
import numpy as np


def main():
    remiz_str = "s-9 t-30 i-0 o-0 p-47 d-18 o-3 p-54 d-15 o-6 p-59 d-9 o-9 p-66 d-15 o-24 p-49 d-15 o-27 p-56 d-12 o-30 p-61 d-12 o-33 p-65 d-9 b-1"
    pass


def save_piano_roll(data, save_fp, show=False):
    '''
    data: (time, pitch) with duration values
    '''
    # Visualize the piano roll
    # Accepts 2D (time, pitch) or 3D (batch, time, pitch) arrays
    if hasattr(data, 'numpy'):
        data = data.cpu().numpy()
    if data.ndim == 2:
        piano_roll = np.array(data)
    elif data.ndim == 3:
        piano_roll = np.array(data[0])
    else:
        raise ValueError(f"Unsupported piano roll shape: {data.shape}")

    time, pitch = piano_roll.shape
    expanded = np.zeros((time, pitch), dtype=np.float32)
    for t in range(time):
        for p in range(pitch):
            dur = int(piano_roll[t, p])
            if dur > 0:
                for dt in range(dur):
                    tt = t + dt
                    if tt < time:
                        expanded[tt, p] = 1.0
    piano_roll = expanded.T  # (pitch, time)

    plt.figure(figsize=(6, 3))
    plt.imshow(
        piano_roll,
        aspect='auto',
        origin='lower',
        cmap='gray_r',
        interpolation='nearest'
    )
    plt.xlabel("Time step")
    plt.ylabel("MIDI Pitch")
    plt.title("Piano Roll (duration as stick length)")
    plt.grid(color='lightgray', linestyle='--', linewidth=0.5, alpha=0.6)
    plt.ylim(20, 100)
    plt.yticks(np.arange(20, 101, 12))

    # Draw vertical grid lines at 1/4, 2/4, 3/4 of the bar
    num_steps = piano_roll.shape[1]
    main_beat_positions = [num_steps // 4, num_steps // 2, 3 * num_steps // 4]
    plt.xticks(main_beat_positions)
    plt.grid(axis='x', color='lightgray', linestyle='--', linewidth=0.5, alpha=0.6)

    # Add minor grid lines at every 1/16 of the bar
    minor_beat_positions = [i * num_steps // 16 for i in range(1, 16)]
    for pos in minor_beat_positions:
        plt.axvline(x=pos, color='lightgray', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.savefig(save_fp, dpi=300)
    plt.close()


def save_piano_roll_old(data, save_fp):
    '''
    Duration as color intensity
    '''
    # Visualize the piano roll
    # Accepts 2D (time, pitch) or 3D (batch, time, pitch) arrays
    if data.ndim == 2:
        piano_roll = data.T  # (pitch, time)
    elif data.ndim == 3:
        piano_roll = data[0].T  # take first batch, (pitch, time)
    else:
        raise ValueError(f"Unsupported piano roll shape: {data.shape}")

    plt.figure(figsize=(10, 4))
    plt.imshow(
        piano_roll,
        aspect='auto',
        origin='lower',
        cmap='gray_r',
        interpolation='nearest'
    )
    plt.xlabel("Time step")
    plt.ylabel("MIDI Pitch")
    plt.title("Piano Roll")
    plt.grid(color='lightgray', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.ylim(20, 100)
    plt.yticks(np.arange(20, 101, 12))
    plt.tight_layout()
    plt.savefig(save_fp, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()