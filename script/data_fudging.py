'''
Hacks and fudges for working with data that doesn't meet all default assumptions (silences trimmed etc). 
Used for working with existing (e.g. Blizzard) data to replicate results.
'''


def pad_speech_to_length(speech, labels):
    '''
    Small mismatches happen a lot with Blizzard STRAIGHT data, so need some hacks to handle them.
    This is rarely/never an issue with world data and labels prepared by Ossian
    '''
    m,dim = speech.shape
    nframe = labels[-1][0][1]

    if math.fabs(nframe - m) > label_length_diff_tolerance:
        print 'Warning: number frames in target cost speech and label do not match (%s vs %s)'%(m, nframe)
        return numpy.array([[0.0]])

    ## Note: small mismatches happen a lot with Blizzard STRAIGHT data, but never with world data prepared by Oliver
    ## Where they occur, use zero padding:

    if nframe > m:
        padding_length = nframe - m
        speech = np.vstack([speech, np.zeros((padding_length, dim))])

    elif nframe < m:
        speech = speech[:nframe,:]

    return speech


def reinsert_terminal_silence(speech, labels, silence_symbols=['#']):
    initial_silence_end = 0
    final_silence_start = -1
    for ((s,e), quinphone) in labels:
        if quinphone[2] in silence_symbols:
            initial_silence_end = e
        else:
            break
    for ((s,e), quinphone) in reversed(labels):
        if quinphone[2] in silence_symbols:
            final_silence_start = s
        else:
            break
    m,n = speech.shape
    label_frames = labels[-1][0][1]
    end_sil_length = label_frames - final_silence_start
    start_sil_length = initial_silence_end

    padded_speech = numpy.vstack([numpy.zeros((start_sil_length, n)) , speech , numpy.zeros((end_sil_length, n))])


    # padded_speech = numpy.zeros((label_frames, n))
    # print speech.shape
    # print padded_speech.shape
    # print initial_silence_end, final_silence_start
    # print padded_speech[initial_silence_end:final_silence_start, :].shape
    # padded_speech[initial_silence_end:final_silence_start, :] = speech
    return padded_speech


def suppress_weird_festival_pauses(label, replace_list=['B_150'], replacement='pau'):
    outlabel = []
    for ((s,e), quinphone) in label:
        new_quinphone = []
        for phone in quinphone:
            if phone in replace_list:
                new_quinphone.append(replacement)
            else:
                new_quinphone.append(phone)
        outlabel.append(((s,e),new_quinphone))
    return outlabel

    