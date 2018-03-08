
import numpy as np
from scipy.interpolate import interp1d

def get_label_frame_centres(nframes, sample_rate, fshift_seconds):
    shift_samples = int(sample_rate * fshift_seconds)
    window_centre = int(shift_samples / 2)
    time_axis = []
    for i in xrange(nframes):
        time_axis.append(window_centre)
        window_centre += shift_samples
        
    # ## and then 1 more frame to catch remainder of wave:
    # time_axis.append(window_centre)
    return np.array( time_axis)


def pitch_synchronous_resample_label(sample_rate, fshift_seconds, pms, labels):

    '''
    input labels like:    [((0, 1), ['xx', 'xx', '#', 'eI', 'm', '2']), ((1, 2), ['xx', 'xx', '#', 'eI', 'm', '3']), ... ]
    '''
    
    frame_labs = []
    for ((s,e), quinphone) in labels:
        mono = quinphone[2]
        dur = e - s
        frame_labs.extend([mono]*dur)

    frame_centres = get_label_frame_centres(len(frame_labs), sample_rate, fshift_seconds)
    
    interpolator = interp1d(frame_centres, np.arange(len(frame_centres)), kind='nearest', axis=0, bounds_error=False, fill_value=(0,len(frame_centres)-1))
    resampled_ixx = np.array(interpolator(pms), dtype='int')
    
    resampled_labels = np.array(frame_labs)[resampled_ixx]

    return resampled_labels