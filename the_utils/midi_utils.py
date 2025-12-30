from remi_z import MultiTrack

def save_midi(mt: MultiTrack, save_fp: str):
    '''
    Save a MultiTrack object from pop909 to a MIDI file.

    NOTE: this function changed the melody track to id 87
    '''
    mt.change_instrument(old_inst_id=13, new_inst_id=87)
    mt.shift_pitch(12, track_id=87)
    mt.set_velocity(96, 0)
    mt.set_velocity(70, 87)
    mt.set_velocity(100, 25)
    mt.to_midi(save_fp, verbose=False, tempo=90)
