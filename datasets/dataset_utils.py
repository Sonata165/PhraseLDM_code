
def load_phrase_annot_cleaned(fp: str) -> list[tuple[str, int]]:
    with open(fp, 'r') as f:
        lines = f.read()
    phrase_tokens = lines.strip().split(' ')

    ret = []
    for phrase_token in phrase_tokens:
        phrase = (phrase_token[0], int(phrase_token[1:]))
        ret.append(phrase)

    return ret

def calculate_bar_id_of_phrases(phrase_annot: list[tuple[str, int]]) -> list[tuple[str, int, int]]:
    '''
    Calculate the bar id for each bar based on phrase annotation.
    return:
    (phrase_type, phrase_start_bar_id (start from 0), phrase_end_bar_id (start from 0, exclusive))
    So phrase ret[i] covers bars from ret[i][1] to ret[i][2]-1
    '''
    ret = []
    current_bar_id = 0
    for phrase in phrase_annot:
        phrase_type, phrase_length = phrase
        phrase_start_bar_id = current_bar_id
        phrase_end_bar_id = current_bar_id + phrase_length
        ret.append((phrase_type, phrase_start_bar_id, phrase_end_bar_id))
        current_bar_id += phrase_length
    return ret
