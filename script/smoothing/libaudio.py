# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 02:48:28 2015

My personal library for general audio processing.

@author: Felipe Espic
"""
import numpy as np
import os
from subprocess import call
from binary_io  import BinaryIOCollection
import warnings
import time
import soundfile as sf
import libutils as lu
from scipy import signal
from scipy import interpolate

# Debug:
#from libdevhelpers import *

# tools location: # Hard coded for nom. TODO: Change
_sptk_bin_dir = os.environ['HOME'] + '/Dropbox/Projects/sublimetext_as_python_ide/common/SPTK-3.7'
_reaper_bin   = os.environ['HOME'] + '/Dropbox/Projects/sublimetext_as_python_ide/common/REAPER/build/reaper'



_io = BinaryIOCollection() 
_curr_dir = os.path.dirname(os.path.realpath(__file__))

MAGIC = -1.0E+10 # logarithm floor (the same as SPTK)

#------------------------------------------------------------------------------
def shift_to_pm(v_shift):
    v_pm = np.cumsum(v_shift)
    return v_pm

#------------------------------------------------------------------------------
def pm_to_shift(v_pm):
    v_shift = np.diff(np.hstack((0,v_pm)))
    return v_shift

#------------------------------------------------------------------------------
def gen_non_symmetric_win(left_len, right_len, win_func):
    # Left window:
    v_left_win = win_func(1+2*left_len)
    v_left_win = v_left_win[0:(left_len+1)]
    
    # Right window:
    v_right_win = win_func(1+2*right_len)
    v_right_win = np.flipud(v_right_win[0:(right_len+1)])
    
    # Constructing window:
    return np.hstack((v_left_win, v_right_win[1:]))    
    
#------------------------------------------------------------------------------
# generated centered assymetric window:
# If totlen is even, it is assumed that the center is the first element of the second half of the vector.
# TODO: case win_func == None
def gen_centr_win(winlen_l, winlen_r, totlen, win_func=None):
   
    v_win_shrt = gen_non_symmetric_win(winlen_l, winlen_r, win_func)  
    win_shrt_len = len(v_win_shrt)
    
    nx_cntr  = np.floor(totlen / 2.0).astype(int)
    nzeros_l = nx_cntr - winlen_l    
    
    v_win = np.zeros(totlen)
    v_win[nzeros_l:nzeros_l+win_shrt_len] = v_win_shrt
    return v_win
    
#------------------------------------------------------------------------------
def ola(m_frm, v_pm, win_func=None):
    nfrms, frmlen = m_frm.shape
    v_sig = np.zeros(v_pm[-1] + frmlen)
    
    v_shift = pm_to_shift(v_pm)
    v_shift = np.append(v_shift, v_shift[-1]) # repeating last value   
    strt  = 0
    for i in xrange(nfrms):        

        if win_func is not None:
            #g = m_frm[i,:].copy()
            v_win = gen_centr_win(v_shift[i], v_shift[i+1], frmlen, win_func=win_func)            
            m_frm[i,:] *= v_win

            '''
            holdon()
            plot(g, '-b')
            plot(v_win, '-r')
            plot(m_frm[i,:], '-k')
            holdoff()
            '''
        #print(i)
        #if i==1: import ipdb; ipdb.set_trace()  # breakpoint bcbc872b //

        # Add frames:
        if (strt+frmlen) >= len(v_sig):
            nsmpls = len(v_sig) - strt
            v_sig[strt:] += m_frm[i,:nsmpls]
            break
        else:
            v_sig[strt:(strt+frmlen)] += m_frm[i,:]

        strt += v_shift[i+1]

    # Cut beginning:
    v_sig = v_sig[(frmlen/2 - v_pm[0]):]     
        
    return v_sig    
    
'''
#------------------------------------------------------------------------------
def ola(m_frm, v_pm, win_func=None):
    nfrms, frmlen = m_frm.shape
    v_sig = np.zeros(v_pm[-1] + frmlen)
    
    v_shift    = pm_to_shift(v_pm)
    v_shift[0] = 0
    strt       = 0
    for i in xrange(nfrms):
        # Add frames:
        strt += v_shift[i]

        if win_func is not None:
            g = 1
            
            
        v_sig[strt:(strt+frmlen)] += m_frm[i,:]    
        
    # Cut beginning:
    v_sig = v_sig[(frmlen/2 - v_pm[0]):]     
        
    return v_sig
'''    
#------------------------------------------------------------------------------
def frm_list_to_matrix(l_frames, v_shift, nFFT):
    nFFThalf = nFFT / 2 + 1
    nfrms    = len(v_shift)
    m_frm    = np.zeros((nfrms, nFFT))
    for i in xrange(nfrms):
        rel_shift  = nFFThalf - v_shift[i] - 1
        m_frm[i,:] = frame_shift(l_frames[i], rel_shift, nFFT)  
    
    return m_frm

#------------------------------------------------------------------------------
# Converts magnitude spectrum into minimum phase complex spectrum:
# m_sp: magnitude spectrum
def sp_to_min_phase(m_sp, in_type='sp'):
    m_rceps     = rceps(m_sp, in_type=in_type)
    m_rceps_mph = rceps_to_min_phase(m_rceps)    
    m_sp_mph    = np.exp(np.fft.fft(m_rceps_mph))
    m_sp_mph    = remove_hermitian_half(m_sp_mph)
    return m_sp_mph
    
#------------------------------------------------------------------------------
# frm_cntr: index of the centre of the input frame
# nFFT could be larger than the one used for v_sp
# ph_type: 'minph' (minimum phase) or 'linph' (linear phase)
# TODO: check a better OLA, for example by applying a window at the output.
def stamp_mag_sp(v_frm_shrt, v_sp_targ, frm_cntr, ph_type='minph'):
    
    nFFThalf = len(v_sp_targ)
    nFFT     = 2 * (nFFThalf - 1)    
    shift    = nFFThalf - frm_cntr - 1
    v_frm    = frame_shift(v_frm_shrt, shift, nFFT)
    
    # Stamp Mag spectrum:    
    v_sp_comp_noise = np.fft.fft(v_frm)   
    
    if ph_type == 'linph':
        v_ph      = np.angle(v_sp_comp_noise)
        v_sp_targ = add_hermitian_half(v_sp_targ[None,:])[0]
        v_sp_comp_out = v_sp_targ * np.exp(v_ph * 1j)
        
    elif ph_type == 'minph':        
        v_rceps       = rceps(v_sp_targ[None,:], in_type='sp')
        v_rceps_mph   = rceps_to_min_phase(v_rceps)    
        v_sp_comp_mph = np.exp(np.fft.fft(v_rceps_mph))
        v_sp_comp_out = v_sp_comp_mph * v_sp_comp_noise
                
    v_frm = np.fft.ifft(v_sp_comp_out).real 
    
    '''
    holdon()
    plot(v_sp_targ[:200], '.-b')
    plot(np.absolute(v_sp_mph[0,:200]), '-r')
    holdoff()
    '''
    
    return v_frm

'''
#------------------------------------------------------------------------------
# frm_cntr: index of the centre of the input frame
def stamp_mag_sp(v_frm_shrt, v_sp_targ, frm_cntr):
    nFFThalf   = len(v_sp_targ)
    nFFT       = 2 * (nFFThalf - 1)    
    shift      = nFFThalf - frm_cntr - 1
    v_frm      = frame_shift(v_frm_shrt, shift, nFFT)
    
    # Stamp Mag spectrum:
    v_cmp_sp = np.fft.fft(v_frm)    
    v_ph     = np.angle(v_cmp_sp)
    v_sp_targ     = add_hermitian_half(v_sp_targ[None,:])[0]
    v_cmp_sp = v_sp_targ * np.exp(v_ph * 1j)
    v_frm    = np.fft.ifft(v_cmp_sp).real 
    
    return v_frm
'''
#------------------------------------------------------------------------------
def frame_shift(v_frm, shift, out_len):
    right_len = out_len - (shift + len(v_frm))
    v_frm_out = np.hstack(( np.zeros(shift) , v_frm, np.zeros(right_len)))
    return v_frm_out
    
#------------------------------------------------------------------------------
# "Cosine window": cos_win**2 = hannnig
# power: 1=> coswin, 2=> hanning
def cos_win(N):
    v_x   = np.linspace(0,np.pi,N)
    v_win = np.sin(v_x)
    return v_win
'''
def cos_win(N, power=1):
    v_x   = np.linspace(0,np.pi,N)
    v_win = np.sin(v_x)**power
    return v_win
'''
#------------------------------------------------------------------------------
def hz_to_bin(v_hz, nFFT, fs):    
    return v_hz * nFFT / float(fs)

def bin_to_hz(v_bin, nFFT, fs):         
    return v_bin * fs / float(nFFT)

#------------------------------------------------------------------------------
# m_sp_l: spectrum on the left. m_sp_r: spectrum on the right
# TODO: Processing fo other freq scales, such as Mel.
def spectral_crossfade(m_sp_l, m_sp_r, cut_off, bw, fs, freq_scale='hz'):
    
    '''
    holdon()
    nx = 120
    plot(m_sp_l[nx,:], '-b')
    plot(m_sp_r[nx,:], '-r')
    holdoff()
    '''

    # Hz to bin:
    nFFThalf = m_sp_l.shape[1]
    nFFT     = (nFFThalf - 1) * 2    
    bin_l    = lu.round_to_int(hz_to_bin(cut_off - bw/2, nFFT, fs))     
    bin_r    = lu.round_to_int(hz_to_bin(cut_off + bw/2, nFFT, fs))

    # Gen short windows:
    bw_bin       = bin_r - bin_l
    v_win_shrt   = np.hanning(2*bw_bin + 1)
    v_win_shrt_l = v_win_shrt[bw_bin:]
    v_win_shrt_r = v_win_shrt[:bw_bin+1]
    
    # Gen long windows:
    v_win_l = np.hstack((np.ones(bin_l),  v_win_shrt_l , np.zeros(nFFThalf - bin_r - 1)))
    v_win_r = np.hstack((np.zeros(bin_l), v_win_shrt_r , np.ones(nFFThalf - bin_r - 1)))
    
    # Apply windows:
    m_sp_l_win = m_sp_l * v_win_l[None,:]
    m_sp_r_win = m_sp_r * v_win_r[None,:]
    m_sp       = m_sp_l_win + m_sp_r_win
    
    '''
    holdon()
    nx = 220
    plot(m_sp_l[nx,:], '-.b')
    plot(m_sp_l_win[nx,:], '-.r')
    plot(m_sp_r[nx,:], '-g')
    plot(m_sp_r_win[nx,:], '-k')
    holdoff()
    '''

    '''
    holdon()
    nx = 196
    plot(db(m_sp_l[nx,:]), '.-b')
    #plot(db(m_sp_l_win[nx,:]), '.-r')
    #plot(db(m_sp_r_win[nx,:]), '-g')
    plot(db(m_sp[nx,:]), '-k')
    holdoff()
    '''
    
    return m_sp
    

#------------------------------------------------------------------------------
def rceps_to_min_phase(m_rceps, out_type='rceps'):
    '''
    # out_type: 'rceps', 'td' (time domain)
    # TODO: 'td' implementation!!
    '''
    
    # Debug:
    nFFThalf = m_rceps.shape[1] / 2 + 1
    m_rceps[:,1:nFFThalf-1] *= 2
    m_rceps[:,nFFThalf:]     = 0

    # Debug:
    '''
    holdon()
    plot(m_rceps, '.-b')
    plot(m_rceps2, '-r')
    holdoff()
    '''
    '''
    holdon()
    plot( np.fft.fft(m_rceps).real, '.-b' )
    plot( np.fft.fft(m_rceps2).real, '-r' )
    holdoff()
    '''
        
    return m_rceps
    


#------------------------------------------------------------------------------
# in_type: 'sp', 'splog' (any log base)
# nc: number of coeffs
# fade_to_total: ratio between the length of the fade out over the total ncoeffs
def rceps_spectral_smoothing(m_sp, in_type='splog', nc_total=60, fade_to_total=0.2):
    
    #dp = lu.DimProtect()
    #m_sp = dp.start(m_sp)
   
    nc_fade = lu.round_to_int(fade_to_total * nc_total)
  
    # Getting Cepstrum:
    m_rceps = rceps(m_sp, in_type=in_type)    
    m_minph_rceps   = rceps_to_min_phase(m_rceps)
    v_ener_orig_rms = np.sqrt(np.mean(m_minph_rceps**2,axis=1))
    
    # Create window:
    v_win_shrt = np.hanning(2*nc_fade+3)
    v_win_shrt = v_win_shrt[nc_fade+2:-1]    
        
    # Windowing:    
    m_minph_rceps[:,nc_total:] = 0
    m_minph_rceps[:,nc_total-nc_fade:nc_total] *= v_win_shrt

    # Energy compensation:
    v_ener_after_rms = np.sqrt(np.mean(m_minph_rceps**2,axis=1))     
    v_ener_fact      = v_ener_orig_rms / v_ener_after_rms
    m_minph_rceps    = m_minph_rceps * v_ener_fact[:,None]
    
    # Go back to spectrum:
    nFFT    = m_rceps.shape[1]
    m_sp_sm = np.fft.fft(m_minph_rceps, n=nFFT).real
    m_sp_sm = remove_hermitian_half(m_sp_sm)
    
    # Plots:
#    from libdevhelpers import *
#    holdon()
#    nx = 134
#    plot(np.log(m_sp[nx,:]),    '-b')
#    plot(m_sp_sm[nx,:], '-r')
#    holdoff()
    
    return m_sp_sm

#------------------------------------------------------------------------------
def log(m_x):
    '''
    Protected log: Uses MAGIC number to floor the logarithm.
    '''    
    m_y = np.log(m_x) 
    m_y[np.isinf(m_y)] = MAGIC
    return m_y    
    
#------------------------------------------------------------------------------
# out_type: 'compact' or 'whole'
def rceps(m_data, in_type='log', out_type='compact'):
    """
    in_type: 'abs', 'log' (any log base), 'td' (time domain).
    TODO: 'td' case not implemented yet!!
    """
    ncoeffs = m_data.shape[1]
    if in_type == 'abs':
        m_data = log(m_data)    
        
    m_data  = add_hermitian_half(m_data, data_type='magnitude')
    m_rceps = np.fft.ifft(m_data).real

    # Amplify coeffs in the middle:
    if out_type == 'compact':        
        m_rceps[:,1:(ncoeffs-2)] *= 2
        m_rceps = m_rceps[:,:ncoeffs]
    
    return m_rceps 
    
#------------------------------------------------------------------------------
# in_type: 'splog' or 'sp'
# v2: Uses information from the previous frame as a first guess.
def true_envelope_3(m_sp, in_type='splog', ncoeffs=60, thres=0.1):
    
    # Formatting input:
    if in_type=='sp':
        m_sp = db(m_sp)
        
    # First guesses:  
    #m_first_guess = true_envelope(m_sp, in_type=in_type, ncoeffs=ncoeffs, thres=thres)    

    # Iterates through frames:
    nFrms, nFFThalf = m_sp.shape
    v_sp_env_prev = np.zeros(nFFThalf)
    m_sp_env      = np.zeros(m_sp.shape)    
    n_maxiter     = 100 
    for f in xrange(nFrms):
        
        v_curr_sp    = m_sp[f,:]        
        v_sp_env_raw = true_envelope(v_curr_sp[np.newaxis,:], in_type=in_type, ncoeffs=ncoeffs, thres=thres)[0]
        v_sp_env_raw = v_sp_env_prev - np.max(v_sp_env_prev - v_sp_env_raw) 
        v_curr_sp = np.maximum(v_curr_sp, v_sp_env_raw)

        # Approximations Iterations:
        for i in xrange(n_maxiter): 
            v_curr_sp_sm = rceps_spectral_smoothing(v_curr_sp[np.newaxis,:], nc_total=ncoeffs, fade_to_total=0.7)[0]
            if np.mean(np.abs(v_curr_sp - v_curr_sp_sm)) < thres:
                break
            
            if i==0:
                v_curr_sp_sm  = v_sp_env_prev - np.max(v_sp_env_prev - v_curr_sp_sm) 
                #v_curr_sp_sm  = v_sp_env_prev - np.max(v_sp_env_prev[349:] - v_curr_sp_sm[349:])

            v_curr_sp = np.maximum(v_curr_sp, v_curr_sp_sm)
            
            # Plot:
            '''   
            holdon()
            plot(v_curr_sp_sm, '-b')
            plot(v_sp_env_prev, '-r')
            plot(v_curr_sp, '-g')
            holdoff()
            
            holdon()
            plot(m_sp[f,:], '-b')
            plot(v_curr_sp_sm, '-r')            
            plot(v_curr_sp, '-g')
            holdoff()
            '''
        # Debug:            
        print('max iter: %d' % (i))
        v_sp_env_prev  = v_curr_sp_sm
        m_sp_env[f,:] = v_curr_sp_sm
        
    return m_sp_env

#------------------------------------------------------------------------------
# in_type: 'splog' or 'sp'
# v2: Uses information from the previous frame as a first guess.
def true_envelope_2(m_sp, in_type='splog', ncoeffs=60, thres=0.1):
    
    # Formatting input:
    if in_type=='sp':
        m_sp = db(m_sp)

    # Iterates through frames:
    nFrms, nFFThalf = m_sp.shape
    v_sp_sm_prev = np.ones(nFFThalf)
    m_sp_env     = np.zeros(m_sp.shape)    
    n_maxiter    = 100 
    for f in xrange(nFrms):
        
        v_curr_sp = m_sp[f,:]

        # Approximations Iterations:
        for i in xrange(n_maxiter): 
            v_curr_sp_sm = rceps_spectral_smoothing(v_curr_sp[np.newaxis,:], nc_total=ncoeffs, fade_to_total=0.7)[0]
            if np.mean(np.abs(v_curr_sp - v_curr_sp_sm)) < thres:
                break
            
            if i==0:
                v_curr_sp_sm  = v_sp_sm_prev - np.max(v_sp_sm_prev - v_curr_sp_sm) 
                #v_curr_sp_sm  = v_sp_sm_prev - np.max(v_sp_sm_prev[349:] - v_curr_sp_sm[349:])

            v_curr_sp = np.maximum(v_curr_sp, v_curr_sp_sm)
            
            # Plot:
            '''   
            holdon()
            plot(v_curr_sp_sm, '-b')
            plot(v_sp_sm_prev, '-r')
            plot(v_curr_sp, '-g')
            holdoff()
            
            holdon()
            plot(m_sp[f,:], '-b')
            plot(v_curr_sp_sm, '-r')            
            plot(v_curr_sp, '-g')
            holdoff()
            '''
        # Debug:            
        print('max iter: %d' % (i))
        v_sp_sm_prev  = v_curr_sp_sm
        m_sp_env[f,:] = v_curr_sp_sm
        
    return m_sp_env




#------------------------------------------------------------------------------
# in_type: 'splog' or 'sp'
def true_envelope(m_sp, in_type='splog', ncoeffs=60, thres=0.1):
    
    if in_type=='sp':
        m_sp  = db(m_sp)
    
    m_sp_env = np.zeros(m_sp.shape)
    
    n_maxiter = 100 
    nFrms = m_sp.shape[0]     
    for f in xrange(nFrms):
        
        v_curr_sp = m_sp[f,:]

        for i in xrange(n_maxiter):            
            v_curr_sp_sm = rceps_spectral_smoothing(v_curr_sp[np.newaxis,:], nc_total=ncoeffs, fade_to_total=0.7)[0]
            # Debug:            
            #print('iter: %d' % (i))
            if np.mean(np.abs(v_curr_sp - v_curr_sp_sm)) < thres:
                break

            v_curr_sp = np.maximum(v_curr_sp, v_curr_sp_sm)
            
            # Plot:
            '''
            
            holdon()
            plot(m_sp[f,:], '-b')
            plot(v_curr_sp_sm, '-r')
            plot(v_curr_sp, '-g')
            holdoff()
            '''
        m_sp_env[f,:] = v_curr_sp_sm
        # Debug:            
        #print('last iter: %d' % (i))        
    return m_sp_env

#------------------------------------------------------------------------------
# interp_type: e.g., 'linear', 'slinear', 'zeros'
def interp_unv_regions(m_data, v_voi, voi_cond='>0', interp_type='linear'):

    vb_voiced   = eval('v_voi ' + voi_cond)
    
    if interp_type == 'zeros':
        m_data_intrp = m_data * vb_voiced[:,None]

    else:
        v_voiced_nx = np.nonzero(vb_voiced)[0]
    
        m_strt_and_end_voi_frms = np.vstack((m_data[v_voiced_nx[0],:] , m_data[v_voiced_nx[-1],:]))        
        t_strt_and_end_voi_frms = tuple(map(tuple, m_strt_and_end_voi_frms))
        
        func_intrp  = interpolate.interp1d(v_voiced_nx, m_data[vb_voiced,:], bounds_error=False , axis=0, fill_value=t_strt_and_end_voi_frms, kind=interp_type)
        
        nFrms = np.size(m_data, axis=0)
        m_data_intrp = func_intrp(np.arange(nFrms))
    
    return m_data_intrp

#------------------------------------------------------------------------------
# Generates time-domain non-symmetric "flat top" windows
# Also, it can be used for generating non-symetric windows ("non-flat top")
# func_win: e.g., numpy.hanning
# flat_to_len_ratio: flat_length / total_length. Number [0,1]
def gen_wider_window(func_win,len_l, len_r, flat_to_len_ratio):
    fade_to_len_ratio = 1 - flat_to_len_ratio  
    
    len_l = lu.round_to_int(len_l)
    len_r = lu.round_to_int(len_r)
    
    len_l_fade = lu.round_to_int(fade_to_len_ratio * len_l)     
    len_r_fade = lu.round_to_int(fade_to_len_ratio * len_r) 
        
    v_win_l   = func_win(2 * len_l_fade + 1)
    v_win_l   = v_win_l[:len_l_fade]
    v_win_r   = func_win(2 * len_r_fade + 1)
    v_win_r   = v_win_r[len_r_fade+1:]
    len_total = len_l + len_r
    len_flat  = len_total - (len_l_fade + len_r_fade)
    v_win     = np.hstack(( v_win_l, np.ones(len_flat) , v_win_r ))
        
    return v_win


# Read audio file:-------------------------------------------------------------
def read_audio_file(filepath, **kargs):
    '''
    Wrapper function. For now, just to keep consistency with the library
    '''    
    return sf.read(filepath, **kargs)
    
# Write wav file:--------------------------------------------------------------
# The format is picked automatically from the file extension. ('WAV', 'FLAC', 'OGG', 'AIFF', 'WAVEX', 'RAW', or 'MAT5')
# v_signal be mono (TODO: stereo, comming soon), values [-1,1] are expected if no normalisation is selected.
# option: norm. If False-> No norm, If 'max'-> normalises to the maximum absolute value, If ''
# If norm provided, norm = 'max' by default
def write_audio_file(filepath, v_signal, fs, **kargs):
    
    # Parsing input:
    if 'norm' in kargs:
        if kargs['norm'] == False:
            pass          
            
        elif kargs['norm'] == 'max':
            v_signal = v_signal / np.max(np.abs(v_signal))
            
        del(kargs['norm'])
    else:
        print 'normalizing audio...'
        v_signal = v_signal / np.max(np.abs(v_signal)) # default
        
    # Write:    
    sf.write(filepath, v_signal, fs, **kargs)
    
    return


# 1-D Smoothing by convolution: (from ScyPy Cookbook - not checked yet!)-----------------------------
def smooth_by_conv(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
    output:
        the smoothed signal
        
    example:
    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """ 
     
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."
        
    if window_len<3:
        return x
    
    
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    
    s=numpy.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')
    
    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y 

#------------------------------------------------------------------------------
# data_type: 'magnitude', 'phase' or 'zeros' (for zero padding), 'complex'
def add_hermitian_half(m_data, data_type='magnitude'):  
           
    if data_type == 'magnitude':
        m_data = np.hstack((m_data , np.fliplr(m_data[:,1:-1])))
        
    elif data_type == 'phase':        
        m_data[:,0]  = 0            
        m_data[:,-1] = 0   
        m_data = np.hstack((m_data , -np.fliplr(m_data[:,1:-1])))

    elif data_type == 'zeros':
        nfrms, nFFThalf = m_data.shape
        m_data = np.hstack((m_data , np.zeros((nfrms,nFFThalf-2))))
        
    elif data_type == 'complex':
        m_data_real = add_hermitian_half(m_data.real)
        m_data_imag = add_hermitian_half(m_data.imag, data_type='phase')
        m_data      = m_data_real + m_data_imag * 1j
    
    return m_data

# Remove hermitian half of fft-based data:-------------------------------------
# Works for either even or odd fft lenghts.
def remove_hermitian_half(m_data):
    
    #if np.ndim(m_data) == 1:
    #   m_data = m_data[None,:] 
    
    nFFThalf   = int(np.floor(np.size(m_data,1) / 2)) + 1
    m_data_rem = m_data[:,:nFFThalf]
   
    return m_data_rem


# Remove redundant half of fft-based data:-------------------------------------
def remove_redun_half(m_data):
    
    warnings.warn('Deprecated, use "remove_hemitian_half", instead') 
    
    nFFThalf   = np.size(m_data,1) / 2 + 1
    m_data_rem = m_data[:,:nFFThalf]
    
    return m_data_rem
    
# Add second half to fft-based data:-------------------------------------------    
# b_sym. If True: add second half symmetrically. If False: add second half anti-symmetrically        
def add_redun_half(m_data, b_sym = True):  

    warnings.warn('Deprecated, use "add_hemitian_half", instead')    
    
    fact = 1    
    if not b_sym: # anti-symmetric
        fact = -1            
    if m_data.ndim is 1:
        m_data = m_data[None,:]    
    m_data = np.hstack((m_data , fact * np.fliplr(m_data[:,1:-1])))
    m_data = np.squeeze(m_data)
    
    return m_data
    
# Inserts pid to file name. This is useful when using temp files.--------------
# Example: path/file.wav -> path/file_pid.wav
def ins_pid(filepath):
    filename, ext = os.path.splitext(filepath)
    filename = "%s_%d%s" % (filename, os.getpid(), ext)
    return filename
    
# Inserts date and time to file name. This is useful for output files----------
# Example: path/file.wav -> path/file_prefix_date_time.wav    
def ins_date_time(filepath, prefix=""):
    filename, ext = os.path.splitext(filepath)
    filename = "%s_%s_%s%s" % (filename, prefix, time.strftime("%Y%m%d_%H%M"), ext)
    return filename

#-----------------------------------------------------
def read_est_file(est_file):
    '''
    Generic function to read est files. So far, it reads the first two columns of est files. (TODO: expand)
    '''

    # Get EST_Header_End line number: (TODO: more efficient)
    with open(est_file) as fid:
        header_size = 1 # init
        for line in fid:
            if line == 'EST_Header_End\n':
                break
            header_size += 1

    m_data = np.loadtxt(est_file, skiprows=header_size, usecols=[0,1])
    return m_data

#------------------------------------------------------------------------------
# check_len_smpls= signal length. If provided, it checks and fixes for some pm out of bounds (REAPER bug)
# fs: Must be provided if check_len_smpls is given
def read_reaper_est_file(est_file, check_len_smpls=-1, fs=-1, skiprows=7, usecols=[0,1]):

    # Checking input params:
    if (check_len_smpls > 0) and (fs == -1):
        raise ValueError('If check_len_smpls given, fs must be provided as well.')

    # Read text: TODO: improve skiprows
    m_data = np.loadtxt(est_file, skiprows=skiprows, usecols=usecols)
    v_pm_sec  = m_data[:,0]
    v_voi = m_data[:,1]

    # Protection against REAPER bugs 1:
    vb_correct = np.hstack(( True, np.diff(v_pm_sec) > 0))
    v_pm_sec  = v_pm_sec[vb_correct]
    v_voi = v_voi[vb_correct]

    # Protection against REAPER bugs 2 (maybe it needs a better protection):
    if (check_len_smpls > 0) and ( (v_pm_sec[-1] * fs) >= (check_len_smpls-1) ):
        v_pm_sec  = v_pm_sec[:-1]
        v_voi = v_voi[:-1]
    return v_pm_sec, v_voi

# REAPER wrapper:--------------------------------------------------------------
def reaper(in_wav_file, out_est_file):
    global _reaper_bin
    cmd =  _reaper_bin + " -s -x 400 -m 50 -a -u 0.005 -i %s -p %s" % (in_wav_file, out_est_file)
    call(cmd, shell=True)
    return

#------------------------------------------------------------------------------
# type: 'f0', 'lf0'
def read_f0(filename, kind='lf0'):
    v_data = lu.read_binfile(filename, dim=1)
    if kind=='lf0':
        v_data = np.exp(v_data)
    return v_data
    
#------------------------------------------------------------------------------
def f0_to_lf0(v_f0):
       
    old_settings = np.seterr(divide='ignore') # ignore warning
    v_lf0 = np.log(v_f0)
    np.seterr(**old_settings)  # reset to default
    
    v_lf0[np.isinf(v_lf0)] = MAGIC
    return v_lf0

# Get pitch marks from signal using REAPER:------------------------------------

def get_pitch_marks(v_sig, fs):
    
    temp_wav = ins_pid('temp.wav')
    temp_pm  = ins_pid('temp.pm')
        
    sf.write(temp_wav, v_sig, fs)
    reaper(temp_wav, temp_pm)
    v_pm = np.loadtxt(temp_pm, skiprows=7)
    v_pm = v_pm[:,0]
    
    # Protection against REAPER bugs 1:
    vb_correct = np.hstack(( True, np.diff(v_pm) > 0))
    v_pm = v_pm[vb_correct]
    
    # Protection against REAPER bugs 2 (maybe I need a better protection):
    if (v_pm[-1] * fs) >= (np.size(v_sig)-1):
        v_pm = v_pm[:-1]
    
    # Removing temp files:
    os.remove(temp_wav)
    os.remove(temp_pm)
    
    return v_pm


# Next power of two:-----------------------------------------------------------
def next_pow_of_two(x):
    # Protection:    
    if x < 2: 
        x = 2
    # Safer for older numpy versions:
    x = 2**np.ceil(np.log2(x)).astype(int)
        
    '''
    # Faster, but only works with updated numpy (NO BORRAR):
    x = np.ceil(x).astype(int)
    x = 1<<(x-1).bit_length()    
    '''
    
    return x

# Typical constant frame rate windowing function:
def Windowing(vInSig, vWin, shiftMs, fs):
    
    inSigLen = len(vInSig)
    nFrms = GetNFramesFromSigLen(inSigLen, shiftMs, fs) 
    
    frmLen    = len(vWin)
    nZerosBeg = int(np.floor(frmLen/2))
    nZerosEnd = frmLen
    
    vZerosBeg = np.zeros(nZerosBeg)
    vZerosEnd = np.zeros(nZerosEnd)
    vInSig    = np.concatenate((vZerosBeg, vInSig, vZerosEnd))

    mSig   = np.zeros((nFrms, frmLen))
    shift  = np.round(fs * shiftMs / 1000)
    nxStrt = 0
    for t in xrange(0,nFrms):
        mSig[t,:] = vInSig[nxStrt:(nxStrt+frmLen)] * vWin
        nxStrt = nxStrt + shift   
    
    return mSig

# This function is provided to to avoid confusion about how to compute the exact 
# number of frames from shiftMs and fs    
def GetNFramesFromSigLen(sigLen, shiftMs, fs):
    
    shift = np.round(fs * shiftMs / 1000)
    nFrms = np.ceil(1 + ((sigLen - 1) / shift))
    nFrms = int(nFrms)
    
    return nFrms


# =================================================================================

def read_mgc(in_mgc_file, n_mgc_coeffs):    
    
    #from binary_io import BinaryIOCollection 
    
    m_mgc = _io.load_binary_file(in_mgc_file, n_mgc_coeffs)     
    
    return m_mgc
    
def mgc_to_sp(m_mgc, nFFT, n_coeffs=60, alpha=0.77):
    # Warning: deprecated
    warnings.warn('Deprecated, use "mcep_to_sp", instead')
    
    return

# reads into spectrum
def read_mgc_file( filename, n_coeffs, nFFT, alpha = 0.77):

    # MGC to Spec:
    curr_cmd = _curr_dir + "/SPTK-3.7/bin/mgc2sp -a %1.2f -g 0 -m %d -l %d -o 2 %s > temp.spbin" % (alpha, n_coeffs-1, nFFT, filename)
    call(curr_cmd, shell=True)

    # Binary file to text file:
    #curr_cmd = "./SPTK-3.7/bin/x2x +fa temp.spbin > temp.spfloat"
    #call(curr_cmd, shell=True)

    # Read binary file:
    v_data = np.fromfile("temp.spbin",dtype='float32')

    # vector to mat:
    nFFT_half = 1 + nFFT / 2
    m_data = np.reshape(v_data, ( len(v_data) / nFFT_half , nFFT_half ), order='C')

    return m_data
    


#==============================================================================
# Converts mcep to lin sp, without doing any  Mel warping.
def mcep_to_lin_sp_log(mgc_mat, nFFT):
    
    '''
    ncoefs_in = m_mgc.shape[1]
    
    nFFT = ncoefs_in * 2
    maxnx_body = ncoefs_in-1
    if maxnx_body == nFFT/2:
        maxnx_body = nFFT/2 - 1
    
    m_mgc[:,1:maxnx_body] = m_mgc[:,1:maxnx_body] * 2

    return (np.fft.fft(m_mgc, n=128)).real
    '''
    
    nFrms, n_coeffs = mgc_mat.shape
    nFFTHalf = 1 + nFFT/2
    
    mgc_mat = np.concatenate((mgc_mat, np.zeros((nFrms, (nFFT/2 - n_coeffs + 1)))),1)
    mgc_mat = np.concatenate((mgc_mat, np.fliplr(mgc_mat[:,1:-1])),1)
    sp_log  = (np.fft.fft(mgc_mat, nFFT,1)).real
    sp_log  = sp_log[:,0:nFFTHalf]

    return sp_log 

# Converts mgc to lin sp, without doing any  Mel warping.
def mgc_to_lin_sp_log(mgc_mat, nFFT):
    
    warnings.warn('Deprecated, use "mcep_to_lin_sp_log", instead')
    
    nFrms, n_coeffs = mgc_mat.shape
    nFFTHalf = 1 + nFFT/2
    
    mgc_mat = np.concatenate((mgc_mat, np.zeros((nFrms, (nFFT/2 - n_coeffs + 1)))),1)
    mgc_mat = np.concatenate((mgc_mat, np.fliplr(mgc_mat[:,1:-1])),1)
    sp_log  = (np.fft.fft(mgc_mat, nFFT,1)).real
    sp_log  = sp_log[:,0:nFFTHalf]
    
    return sp_log 

    
#Gets RMS from matrix no matter the number of bins m_data has, 
#it figures out according to the FFT length.
# For example, nFFT = 128 , nBins_data= 60 (instead of 65 or 128)
def get_rms(m_data, nFFT):
    m_data2 = m_data**2
    m_data2[:,1:(nFFT/2)] = 2 * m_data2[:,1:(nFFT/2)]    
    v_rms = np.sqrt(np.sum(m_data2[:,0:(nFFT/2+1)],1) / nFFT)    
    return v_rms   
    
# Converts spectrum to MCEPs using SPTK toolkit--------------------------------  
# if alpha=0, no spectral warping
# m_sp: absolute and non redundant spectrum
# in_type: Type of input spectrum. if 3 => |f(w)|. If 1 => 20*log|f(w)|. If 2 => ln|f(w)|
# fft_len: If 0 => automatic computed from input data, If > 0 , is the value of the fft length
def sp_to_mcep(m_sp, n_coeffs=60, alpha=0.77, in_type=3, fft_len=0):

    #Pre:
    temp_sp  =  ins_pid('temp.sp')
    temp_mgc =  ins_pid('temp.mgc')
    
    # Writing input data:
    _io.array_to_binary_file(m_sp, temp_sp)

    if fft_len is 0: # case fft automatic
        fft_len = 2*(np.size(m_sp,1) - 1)

    # MCEP:      
    curr_cmd = _curr_dir + "/SPTK-3.7/bin/mcep -a %1.2f -m %d -l %d -e 1.0E-8 -j 0 -f 0.0 -q %d %s > %s" % (alpha, n_coeffs-1, fft_len, in_type, temp_sp, temp_mgc)
    call(curr_cmd, shell=True)
    
    # Read MGC File:
    m_mgc = _io.load_binary_file(temp_mgc , n_coeffs)
    
    # Deleting temp files:
    os.remove(temp_sp)
    os.remove(temp_mgc)
    
    #$sptk/mcep -a $alpha -m $mcsize -l $nFFT -e 1.0E-8 -j 0 -f 0.0 -q 3 $sp_dir/$sentence.sp > $mgc_dir/$sentence.mgc
    
    return m_mgc

    
# MCEP to SP (Deprecated).-----------------------------------------------
def mcep_to_sp(m_mgc, nFFT, alpha=0.77, out_type=2): 
    
    warnings.warn('Deprecated, use "mcep_to_sp_sptk" or "mcep_cosmat", instead') 
        
    m_sp = mcep_to_sp_sptk(m_mgc, nFFT, alpha=alpha, out_type=out_type)        
   
    return m_sp    
    
# MCEP to SP using SPTK toolkit.-----------------------------------------------
# m_sp is absolute and non redundant spectrum
# out_type = type of output spectrum. If out_type==0 -> 20*log|H(z)|. If out_type==2 -> |H(z)|
def mcep_to_sp_sptk(m_mgc, nFFT, alpha=0.77, out_type=2): 
  
    n_coeffs = m_mgc.shape[1]    

    temp_mgc =  ins_pid('temp.mgc') 
    temp_sp  =  ins_pid('temp.sp')
    
    _io.array_to_binary_file(m_mgc,temp_mgc)

    # MGC to Spec:
    curr_cmd = _curr_dir + "/SPTK-3.7/bin/mgc2sp -a %1.2f -g 0 -m %d -l %d -o %d %s > %s" % (alpha, n_coeffs-1, nFFT, out_type, temp_mgc, temp_sp)
    call(curr_cmd, shell=True) 

    m_sp = _io.load_binary_file( temp_sp, 1+nFFT/2 )  
    if np.size(m_sp,0) == 1: # protection when it is only one frame
        m_sp = m_sp[0]
    
    os.remove(temp_mgc)
    os.remove(temp_sp)
   
    return m_sp

#============================================================================== 
# out_type: 'db', 'log', 'abs' (absolute)    
def mcep_to_sp_cosmat(m_mcep, n_spbins, alpha=0.77, out_type='abs'):
    '''
    mcep to sp using dot product with cosine matrix.
    '''
    # Warping axis:
    n_cepcoeffs = m_mcep.shape[1]
    v_bins_out  = np.linspace(0, np.pi, num=n_spbins)
    v_bins_warp = np.arctan(  (1-alpha**2) * np.sin(v_bins_out) / ((1+alpha**2)*np.cos(v_bins_out) - 2*alpha) ) 
    v_bins_warp[v_bins_warp < 0] += np.pi
    
    # Building matrix:
    m_trans = np.zeros((n_cepcoeffs, n_spbins))
    for nxin in xrange(n_cepcoeffs):
        for nxout in xrange(n_spbins):
            m_trans[nxin, nxout] = np.cos( v_bins_warp[nxout] * nxin )        
            
    # Apply transformation:
    m_sp = np.dot(m_mcep, m_trans)
    
    if out_type == 'abs':
        m_sp = np.exp(m_sp)
    elif out_type == 'db':
        m_sp = m_sp * (20 / np.log(10))
    elif out_type == 'log':
        pass
    
    return m_sp
            
# Real Cepstrum:---------------------------------------------------------------
#def rceps():
#    y = real(ifft(log(abs(fft(x)))));
#
#    w = [1;2*ones(n/2-1,1);ones(1-rem(n,2),1);zeros(n/2-1,1)];
#    ym = real(ifft(exp(fft(w.*y))));
    
    
# Absolute to Decibels:--------------------------------------------------------
# b_inv: inverse function
def db(m_data, b_inv=False):
    if b_inv==False:
        return 20 * np.log10(m_data) 
    elif b_inv==True:
        return 10 ** (m_data / 20)
         
# Get Spectrum:================================================================
# type: 'db', 'abs' (absolute)
def get_spectrum(v_sig, type='db'):
    
    nFFT = la.next_pow_of_two(np.size(v_sig))        
    v_sp = np.absolute(np.fft.fft(v_sig, n=nFFT))    
    nFFThalf = nFFT / 2 + 1
    v_sp = v_sp[:nFFThalf]    
    if type is 'db':
        v_sp = 20 * np.log10(v_sp)       
    
    return v_sp 

#Prue: ========================================================================

import types
def imports():
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            #yield val.__name__
            yield val


# Warping fBank class:=====================================================
# Not working well yet. TODO: Fix it!
class MelFBank:

    def __init__(self, n_in, n_out, alpha=0.77):

        # For now, hardcoded:
        eval_func=self.eval_hanning
        
        # Bins warping:
        v_bins_in  = np.linspace(0, np.pi, num=n_in)
        v_bins_warp = np.arctan(  (1-alpha**2) * np.sin(v_bins_in) / ((1+alpha**2)*np.cos(v_bins_in) - 2*alpha) ) 
        v_bins_warp[v_bins_warp < 0] += np.pi

        # Bands gen:
        maxval = v_bins_warp[-1]
        tr_width_half = maxval / (n_out - 1)
        v_crit_points = np.linspace(-tr_width_half, maxval+tr_width_half, num=(n_out+2))        
        
        # Eval triangles:
        m_trans = np.zeros((n_in, n_out))
        for nxout in xrange(n_out):
            for nxin in xrange(n_in):
                m_trans[nxin, nxout] = eval_func(v_bins_warp[nxin], v_crit_points[nxout], v_crit_points[nxout+2])
 
        
        # Ener normalisation:
        v_ener_warp   = np.sum(m_trans, axis=0)
        v_ener_unwarp = np.sum(m_trans, axis=1)
        
        self.m_warp   = m_trans / v_ener_warp
        self.m_unwarp = (m_trans / v_ener_unwarp[:,None]).T
        return
                
            
    def eval_triang(self, x, l, r):
        mp = (l + r) / 2 # mid point
        
        if (l <= x) and ( x < mp):
            cb = l
            
        elif (mp <= x) and ( x <= r):
            cb = r
            
        else:
            return 0
            
        a = 1 / (mp - cb)
        b = -cb * a
        return a * x  + b
            

    def eval_hanning(self, x, l, r):
        
        if (x < l) or (r < x):
            return 0        
        
        a  = 2 * np.pi / (r - l)
        b  = np.pi - a * r
        x2 = a * x + b
        return 0.5 * (1 + np.cos(x2)) 
            
#==============================================================================
class McepToSpLog:
    
    def __init__(self, n_in, n_out, alpha=0.77):  
        # Bins warping:
        v_bins_in  = np.linspace(0, np.pi, num=n_in)
        v_bins_warp = np.arctan(  (1-alpha**2) * np.sin(v_bins_in) / ((1+alpha**2)*np.cos(v_bins_in) - 2*alpha) ) 
        v_bins_warp[v_bins_warp < 0] += np.pi

        self.m_trans = np.zeros((n_in, n_out))
        for nxin in xrange(n_in):
            for nxout in xrange(n_out):
                self.m_trans[nxin,nxout] = np.cos( v_bins_warp[nxin] * nxout )
            
        return
             
    def convert(self, m_mcep):
        return np.dot(m_mcep, self.m_trans.T)

# in_type: Type of input spectrum. if 3 => |f(w)|. If 1 => 20*log|f(w)|. If 2 => ln|f(w)|        
def sp_mel_warp(m_sp, nbins_out, alpha=0.77, in_type=3):
    '''
    Info:
    in_type: Type of input spectrum. if 3 => |f(w)|. If 1 => 20*log|f(w)|. If 2 => ln|f(w)|        
    '''
    
    # sp to mcep:
    m_mcep = sp_to_mcep(m_sp, n_coeffs=nbins_out, alpha=alpha, in_type=in_type)
    
    # mcep to sp:
    if in_type == 3:
        out_type = 'abs'
    elif in_type == 1:
        out_type = 'db'
    elif in_type == 2:
        out_type = 'log'
        
    m_sp_wrp = mcep_to_sp_cosmat(m_mcep, nbins_out, alpha=0.0, out_type=out_type)
    return m_sp_wrp
    

#==============================================================================
# in_type: 'abs', 'log'
# TODO: 'db'
def sp_mel_unwarp(m_sp_mel, nbins_out, alpha=0.77, in_type='log'):
    
    ncoeffs = m_sp_mel.shape[1]
    
    if in_type == 'abs':
        m_sp_mel = np.log(m_sp_mel)
    
    #sp to mcep:
    m_sp_mel = add_hermitian_half(m_sp_mel, data_type='magnitude')
    m_mcep   = np.fft.ifft(m_sp_mel).real
    
    # Amplify coeffs in the middle:    
    m_mcep[:,1:(ncoeffs-2)] *= 2
       
    #mcep to sp:    
    m_sp_unwr = mcep_to_sp_cosmat(m_mcep[:,:ncoeffs], nbins_out, alpha=alpha, out_type=in_type)
    
    return m_sp_unwr

#==============================================================================
def label_state_align_to_phone_align(in_file, out_file, n_states_x_ph=5):
    ll_in_lines = lu.read_text_file(in_file, output_mode='split')
       
    n_lines = len(ll_in_lines)
    l_out_lines = []
    
    # Iterating through lines:
    for li in xrange(n_lines):
        if np.remainder(li,n_states_x_ph) == 0:
            l_out_lines.append(ll_in_lines[li][0] + ' ')
            
        if np.remainder(li+1,n_states_x_ph) == 0:
            l_out_lines[-1] = l_out_lines[-1] + ll_in_lines[li][1] + ' ' + ll_in_lines[li][2][:-4]

    # Write file:                
    lu.write_text_file(l_out_lines, out_file, add_newline=True) 
    return
       
#==============================================================================
# MAIN - Used for Dev
#==============================================================================

if __name__ == '__main__':   
    
    nFFThalf = 2049
    nbands   = 60
    alpha    = 0.77
    
    fb = MelFBank(nFFThalf, nbands, alpha)
