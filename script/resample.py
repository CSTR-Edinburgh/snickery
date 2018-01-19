import numpy as np
import scipy.interpolate

### These functions reverse engineer the mechanisms that various tools use to window
### a waveform, in order to determine the waveform sample which should be considered
### associated with any given frame of features. 

def get_world_frame_centres(wavlength, sample_rate, fshift_seconds):
    ## World makes a time axis like this, then centres anaysis windows on entries in time_index
    ## This was worked out by inspecting World code.
    samples_per_frame = int(sample_rate * fshift_seconds)
    f0_length = (wavlength / float(sample_rate) / fshift_seconds) + 1
    time_axis = []
    for i in range(int(f0_length)):
        time_axis.append(i * samples_per_frame)
    return np.array( time_axis)

def get_snack_frame_centres(wavlength, sample_rate, fshift_seconds):
    ## Frame centres used by Snack formant code, using default window_length (0.049)
    ## This was worked out by inspecting Snack documentation.
    flength_seconds = 0.049
    shift_samples = int(sample_rate * fshift_seconds)
    window_samples = int(sample_rate * flength_seconds)
    window_centre = int(window_samples / 2)
    window_end = window_samples
    time_axis = []
    while window_end <= wavlength:
        time_axis.append(window_centre)
        window_centre += shift_samples
        window_end += shift_samples
    return np.array( time_axis)


def get_mfcc_frame_centres(wavlength, sample_rate, fshift_seconds, flength_seconds):
    ## This was worked out to be consistent with HCopy's output.
    ## MFCCs were extracted from 16k downsampled wave, here comparing with original 48k wave...
    shift_samples = int(sample_rate * fshift_seconds)
    window_samples = int(sample_rate * flength_seconds)
    window_centre = int(window_samples / 2)
    window_end = window_samples
    time_axis = []
    while window_end <= wavlength:
        time_axis.append(window_centre)
        window_centre += shift_samples
        window_end += shift_samples

    ## and then 1 more frame to catch remainder of wave:
    time_axis.append(window_centre)


    return np.array( time_axis)



def upsample(len_wave, sample_rate, fshift_seconds, features, f0_dim=-1, convention='world'):

    if convention == 'world':
        frame_centres = get_world_frame_centres(len_wave, sample_rate, fshift_seconds)
    elif convention == 'snack':
        frame_centres = get_snack_frame_centres(len_wave, sample_rate, fshift_seconds)
    else:
        sys.exit('Unknown windowing convention: %s'%(convention))

    m,n = features.shape
    #print  len(frame_centres), m, n
    
    diff = len(frame_centres) - m

    if diff not in [-0,1]:
        print 'Warning: Inconsistent data sizes -- return empty matrix!'
        return np.zeros(0)

    if diff == 1:
        frame_centres = frame_centres[:-1]      
    
    resampled = np.zeros((len_wave, n))
    for dim in xrange(n):
        if dim == f0_dim:
            y = features[:,dim]

            voiced_ix = np.where( y > 0.0 )  ## equiv to np.nonzero(y)
            unvoiced_ix = np.where( y == 0.0 )
            
            voicing_flag = np.zeros(y.shape)
            voicing_flag[voiced_ix] = 1.0
            
            ## voiced first with linear interp:
            
            interpolator = scipy.interpolate.interp1d(frame_centres[voiced_ix], y[voiced_ix], kind='linear', axis=0, \
                                        bounds_error=False, fill_value='extrapolate')
            v_interpolated = interpolator(np.arange(len_wave))
            
            ## then unvoiced:
            
            interpolator = scipy.interpolate.interp1d(frame_centres, voicing_flag, kind='nearest', axis=0, \
                                        bounds_error=False, fill_value='extrapolate')
            u_interpolated = interpolator(np.arange(len_wave))            
            
            interpolated = v_interpolated # np.minimum(v_interpolated, u_interpolated)
            interpolated[u_interpolated==0] = 0.0
                   
        else:
        
            y = features[:,dim]
        

            interpolator = scipy.interpolate.interp1d(frame_centres, y, kind='linear', axis=0, \
                                        bounds_error=False, fill_value='extrapolate')
            interpolated = interpolator(np.arange(len_wave))
        resampled[:,dim] = interpolated
        

    return resampled
    


