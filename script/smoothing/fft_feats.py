"""
Old version of MagPhase vocoder, shipped with the LibWavGen library (Snickery).
@author: Felipe Espic
"""

#==============================================================================
# IMPORTS
#==============================================================================

# Standard:--------------------------------------------------------------------

import numpy as np
import libutils as lu 
import libaudio as la
import soundfile as sf
import os

# Additional:------------------------------------------------------------------
from scipy import interpolate
from scipy import signal


#==============================================================================
# BODY
#==============================================================================

# Todo, add a input param to control the mix:
def voi_noise_window(length):
    return np.bartlett(length)**2.5 # 2.5 optimum # max: 4
    #return np.bartlett(length)**4

#==============================================================================
# If win_func == None, no window is applied (i.e., boxcar)
# win_func: None, window function, or list of window functions.
def windowing(v_sig, v_pm, win_func=np.hanning):
    n_smpls = np.size(v_sig)
    
    # Round to int:
    v_pm = lu.round_to_int(v_pm) 
    
    # Pitch Marks Extension: 
    v_pm_plus = np.hstack((0,v_pm, (n_smpls-1)))    
    n_pm      = np.size(v_pm_plus) - 2     
    v_lens    = np.zeros(n_pm, dtype=int)
    v_shift   = np.zeros(n_pm, dtype=int)
    v_rights  = np.zeros(n_pm, dtype=int)
    l_frames  = []
    
    for f in xrange(0,n_pm):
        left_lim  = v_pm_plus[f]
        pm        = v_pm_plus[f+1]
        right_lim = v_pm_plus[f+2]        
        
        # Curr raw frame:
        v_frm   = v_sig[left_lim:(right_lim+1)]  
        
        # win lengts:
        left_len  = pm - left_lim
        right_len = right_lim - pm
        
        # Apply window:
        if isinstance(win_func, list):
            v_win = la.gen_non_symmetric_win(left_len, right_len, win_func[f])  
            v_frm = v_frm * v_win            
            
        elif callable(open): # if it is a function:
            v_win = la.gen_non_symmetric_win(left_len, right_len, win_func)  
            v_frm = v_frm * v_win
            
        elif None:
            pass               
            
        # Store:
        l_frames.append(v_frm) 
        v_lens[f]   = len(v_frm)
        v_shift[f]  = left_len
        v_rights[f] = right_len
        
    return l_frames, v_lens, v_pm_plus, v_shift, v_rights
        
def analysis(v_in_sig, fs):
    # Pitch Marks:-------------------------------------------------------------
    v_pm = la.get_pitch_marks(v_in_sig, fs)
    v_pm_smpls = v_pm * fs
    
    # Windowing:---------------------------------------------------------------    
    l_frms, v_lens, v_pm_plus, v_shift, v_rights = windowing(v_in_sig, v_pm_smpls)
    
    # FFT:---------------------------------------------------------------------
    len_max = np.max(v_lens) # max frame length in file    
    nFFT = la.next_pow_of_two(len_max)    
    print "current nFFT: %d" % (nFFT)
    
    n_frms = len(l_frms)
    m_frms = np.zeros((n_frms, nFFT))
    for f in xrange(n_frms):        
        m_frms[f,0:v_lens[f]] = l_frms[f]
    
    m_fft = np.fft.fft(m_frms)
    m_sp  = np.absolute(m_fft) 
    m_ph  = np.angle(m_fft) 
    
    # Remove redundant information:--------------------------------------------
    m_sp = la.remove_hermitian_half(m_sp)
    m_ph = la.remove_hermitian_half(m_ph)
    
    return m_sp, m_ph, v_pm_plus

def synthesis(m_sp, m_ph, v_pm_plus):
    
    # Mirorring second half of spectrum:
    m_sp = la.add_hermitian_half(m_sp)
    m_ph = la.add_hermitian_half(m_ph, data_type='phase')    
    
    # To complex:
    m_fft = m_sp * np.exp(m_ph * 1j)

    # To time domain:
    m_frms = np.fft.ifft(m_fft).real 
    
    # OLA:
    n_frms    = len(v_pm_plus) - 2
    v_out_sig = np.zeros(v_pm_plus[-1] + 1)
    for f in xrange(n_frms): 
        strt = v_pm_plus[f]
        curr_len = v_pm_plus[f+2] - strt + 1
        v_out_sig[strt:(strt+curr_len)] += m_frms[f,0:curr_len]
        
    return v_out_sig

#==============================================================================
# From (after) 'analysis_with_del_comp':
# new: returns voi/unv decision.
# new: variable length FFT
def analysis_with_del_comp_from_est_file_2(v_in_sig, est_file, fs):
    # Pitch Marks:-------------------------------------------------------------
    v_pm_sec, v_voi = la.read_reaper_est_file(est_file, check_len_smpls=len(v_in_sig), fs=fs)    
    v_pm_smpls = v_pm_sec * fs
    
    # Windowing:---------------------------------------------------------------    
    l_frms, v_lens, v_pm_plus, v_shift, v_rights = windowing(v_in_sig, v_pm_smpls)  
    
    n_frms = len(l_frms)
    l_sp   = []
    l_ph   = []
    for f in xrange(n_frms):   
                
        v_frm = l_frms[f]        
        # un-delay the signal:
        v_frm = np.hstack((v_frm[v_shift[f]:], v_frm[0:v_shift[f]])) 
        
        v_fft = np.fft.fft(v_frm)
        v_sp  = np.absolute(v_fft)
        v_ph  = np.angle(v_fft)        
            
        # Remove second (hermitian) half:
        v_sp = la.remove_hermitian_half(v_sp)
        v_ph = la.remove_hermitian_half(v_ph)
        
        # Storing:
        l_sp.append(v_sp)
        l_ph.append(v_ph)
    
    return l_sp, l_ph, v_shift, v_voi
#=========================================================================================

def analysis_with_del_comp_from_pm(v_in_sig, v_pm_smpls, nFFT, win_func=np.hanning):

    # Windowing:
    l_frms, v_lens, v_pm_plus, v_shift, v_rights = windowing(v_in_sig, v_pm_smpls, win_func=win_func)
    #import ipdb; ipdb.set_trace()  # breakpoint 2c53771f //

    # FFT:---------------------------------------------------------------------
    len_max = np.max(v_lens) # max frame length in file    
    if nFFT < len_max:
        raise ValueError("nFFT (%d) is shorter than the maximum frame length (%d)" % (nFFT,len_max))
    
    n_frms = len(l_frms)
    m_frms = np.zeros((n_frms, nFFT))   
    
    # For paper:--------------------------------
    #m_frms_orig = np.zeros((n_frms, nFFT))  
    # ------------------------------------------
    
    for f in xrange(n_frms):           
        m_frms[f,0:v_lens[f]] = l_frms[f]
        # un-delay the signal:
        v_curr_frm  = m_frms[f,:]     
        
        # For paper:----------------------------
        #m_frms_orig[f,:] = v_curr_frm        
        # --------------------------------------
                   
        m_frms[f,:] = np.hstack((v_curr_frm[v_shift[f]:], v_curr_frm[0:v_shift[f]]))         
        
                   
    # For paper:----------------------------
    #m_fft_orig = np.fft.fft(m_frms_orig)    
    #m_ph_orig  = np.angle(m_fft_orig)    
    # ---------------------------------------
                   
    m_fft = np.fft.fft(m_frms)
    m_sp  = np.absolute(m_fft) 
    m_ph  = np.angle(m_fft) 
    
    # For paper:----------------------------
    # plotm(np.log(m_sp))
    '''
    i = 88
    nbins = 200
    holdon()
    plot(m_ph_orig[i,:nbins], '-b')
    plot(m_ph_orig[i+1,:nbins], '-r')
    plot(m_ph_orig[i+1,:nbins] - m_ph_orig[i,:nbins], '-k')
    holdoff()
    #import matplotlib.pyplot as plt 
    i = 88
    nbins = 2049
    holdon()
    plot(m_ph[i,:nbins], '-b')
    plot(m_ph[i+1,:nbins], '-r')
    plot(m_ph[i+1,:nbins] - m_ph[i,:nbins], '-k')
    holdoff()
    '''
    # -------------------------------------
    # Remove redundant second half:--------------------------------------------
    m_sp  = la.remove_hermitian_half(m_sp)
    m_ph  = la.remove_hermitian_half(m_ph) 
    m_fft = la.remove_hermitian_half(m_fft)
    
    return m_sp, m_ph, v_shift, m_frms, m_fft, v_lens



#==============================================================================
# From (after) 'analysis_with_del_comp':
# new: returns voi/unv decision.
def analysis_with_del_comp_from_est_file(v_in_sig, est_file, nfft, fs, win_func=np.hanning, b_ph_unv_zero=False):
    # Pitch Marks:-------------------------------------------------------------
    v_pm_sec, v_voi = la.read_reaper_est_file(est_file, check_len_smpls=len(v_in_sig), fs=fs)
    v_pm_smpls = v_pm_sec * fs

    m_sp, m_ph, v_shift, m_frms, m_fft, v_lens = analysis_with_del_comp_from_pm(v_in_sig, v_pm_smpls, nfft, win_func=win_func)

    if b_ph_unv_zero:
        m_ph = m_ph * v_voi[:,None]

    return m_sp, m_ph, v_shift, v_voi, m_frms, m_fft

#==============================================================================

def analysis_with_del_comp(v_in_sig, nFFT, fs):
    # Pitch Marks:-------------------------------------------------------------
    v_pm = la.get_pitch_marks(v_in_sig, fs)
    v_pm_smpls = v_pm * fs
    
    # Windowing:---------------------------------------------------------------    
    l_frms, v_lens, v_pm_plus, v_shift, v_rights = windowing(v_in_sig, v_pm_smpls)
    
    # FFT:---------------------------------------------------------------------
    len_max = np.max(v_lens) # max frame length in file    
    if nFFT < len_max:
        raise ValueError("nFFT (%d) is shorter than the maximum frame length (%d)" % (nFFT,len_max))
    
    n_frms = len(l_frms)
    m_frms = np.zeros((n_frms, nFFT))
    
    for f in xrange(n_frms):           
        m_frms[f,0:v_lens[f]] = l_frms[f]
        # un-delay the signal:
        v_curr_frm  = m_frms[f,:]        
        m_frms[f,:] = np.hstack((v_curr_frm[v_shift[f]:], v_curr_frm[0:v_shift[f]])) 
        
    m_fft = np.fft.fft(m_frms)
    m_sp  = np.absolute(m_fft) 
    m_ph  = np.angle(m_fft) 
    
    # Remove redundant second half:--------------------------------------------
    m_sp = la.remove_hermitian_half(m_sp)
    m_ph = la.remove_hermitian_half(m_ph) 
    
    return m_sp, m_ph, v_shift

'''
#==============================================================================
def synthesis_with_del_comp(m_sp, m_ph, v_shift):
    
    # Enforce int:
    v_shift = lu.round_to_int(v_shift)
    
    # Mirorring second half of spectrum:
    m_sp = la.add_hermitian_half(m_sp)
    m_ph = la.add_hermitian_half(m_ph, data_type='phase')    
    
    # To complex:
    m_fft = m_sp * np.exp(m_ph * 1j)

    # To time domain:
    m_frms = np.fft.ifft(m_fft).real 
    
    # OLA:---------------------------------------------------------------------
    n_frms, nFFT = np.shape(m_sp)
    v_out_sig    = np.zeros(np.sum(v_shift[:-1]) + nFFT + 1) # despues ver como cortar! (debe estar malo este largo!)
       
    # Metodo 2:----------------------------------------------------------------
    # Flip frms:      
    m_frms = np.fft.fftshift(m_frms, axes=1)
    strt   = 0
    for f in xrange(1,n_frms): 
        # wrap frame:
        v_curr_frm  = m_frms[f-1,:]  
        
        # Debug: Simple Window Correction:--------        
#        v_win_shrt = la.gen_wider_window(np.hanning,v_shift[f-1], v_shift[f], 0.5)
#        
#        mid_nx = nFFT / 2
#        v_curr_frm[:(mid_nx-v_shift[f-1])] = 0
#        v_curr_frm[(mid_nx+v_shift[f]):] = 0
        
        # Add frames:
        v_out_sig[strt:(strt+nFFT)] += v_curr_frm        
        strt += v_shift[f] 
        
    # Cut remainders (TODO!!) (only beginning done!):
    v_out_sig = v_out_sig[(nFFT/2 - v_shift[0]):]     
   
    return v_out_sig
'''

#==============================================================================
def synthesis_with_del_comp(m_sp, m_ph, v_shift, win_func=np.hanning, win_flat_to_len=0.3):
    
    # Enforce int:
    v_shift = lu.round_to_int(v_shift)
    
    # Mirorring second half of spectrum:
    m_sp = la.add_hermitian_half(m_sp)
    m_ph = la.add_hermitian_half(m_ph, data_type='phase')    
    
    # To complex:
    m_fft = m_sp * np.exp(m_ph * 1j)

    # To time domain:
    m_frms = np.fft.ifft(m_fft).real 
    
    # OLA:---------------------------------------------------------------------
    n_frms, nFFT = np.shape(m_sp)
    #v_out_sig    = np.zeros(np.sum(v_shift[:-1]) + nFFT + 1) # despues ver como cortar! (debe estar malo este largo!)
    v_out_sig    = np.zeros(la.shift_to_pm(v_shift)[-1] + nFFT)   
    # Metodo 2:----------------------------------------------------------------
    # Flip frms:      
    m_frms = np.fft.fftshift(m_frms, axes=1)
    strt   = 0
    v_win  = np.zeros(nFFT)
    mid_frm_nx = nFFT / 2
    for f in xrange(1,n_frms): 
        # wrap frame:
        v_curr_frm  = m_frms[f-1,:]  
        
        # Window Correction:
        if win_flat_to_len < 1:
            v_win[:] = 0
            v_win_shrt = la.gen_wider_window(win_func,v_shift[f-1], v_shift[f], win_flat_to_len)        
            v_win[(mid_frm_nx-v_shift[f-1]):(mid_frm_nx+v_shift[f])] = v_win_shrt            
            rms_orig   = np.sqrt(np.mean(v_curr_frm**2))
            v_curr_frm = v_curr_frm * v_win
            rms_after_win = np.sqrt(np.mean(v_curr_frm**2))
            # Energy compensation:
            if rms_after_win > 0:
                v_curr_frm = v_curr_frm * rms_orig / rms_after_win        
        
        # Add frames:
        v_out_sig[strt:(strt+nFFT)] += v_curr_frm        
        strt += v_shift[f] 
        
    # Cut remainders (TODO!!) (only beginning done!):
    v_out_sig = v_out_sig[(nFFT/2 - v_shift[0]):]     
   
    return v_out_sig

#==============================================================================

def ph_enc(m_ph):
    m_phs = np.sin(m_ph)    
    m_phc = np.cos(m_ph)
    return m_phs, m_phc  
    

# mode = 'sign': Relies on the cosine value, and uses sine's sign to disambiguate.    
#      = 'angle': Computes the angle between phs (imag) and phc (real)  
def ph_dec(m_phs, m_phc, mode='angle'):  
    
    if mode == 'sign':    
        m_bs = np.arcsin(m_phs)
        m_bc = np.arccos(m_phc)   
        m_ph = np.sign(m_bs) * np.abs(m_bc)   
        
    elif mode == 'angle':
        m_ph = np.angle(m_phc + m_phs * 1j)
        
    return m_ph 

#------------------------------------------------------------------------------
# NOTE: Not finished!!!!
def pm_interp_segment(v_pm, p1nx, p2nx):
    
    # plot(np.diff(v_pm))
    
    p1 = v_pm[p1nx]    
    p2 = v_pm[p2nx]
    
    d1 = p1 - v_pm[p1nx-1]
    d2 = p2 - v_pm[p2nx-1]
    a_ = (d1 - d2) / (p1 - p2)     
    b_ = d1 - a_ * p1
    
    # Gen:
    v_pm_seg = np.zeros(p2nx-p1nx) # maybe not the best?? check!
    p_prev = p1
    for i in xrange(len(v_pm_seg)):
        pc = ( b_ + p_prev ) / (1 - a_)
        v_pm_seg[i] = pc
        if v_pm_seg[i] >= v_pm[p2nx-1]:
            break            
        p_prev = pc
    
    g=1
        
    
        
    
    return
    
'''
# Done after: 'analysis_with_del_comp_and_ph_encoding'=========================
# NOTE: Not finished!!!!
def analysis_with_del_comp_ph_enc_and_pm_interp(est_file, wav_file, nFFT, mvf):

    # Read wav file:
    v_sig, fs = sf.read(wav_file)
    
    # Read est file:
    v_pm_sec, v_voi = la.read_reaper_est_file(est_file, check_len_smpls=len(v_sig), fs=fs) # ??
    v_pm_smpls = v_pm_sec * fs    
    n_frms = len(v_pm_smpls)
    
    p1nx = 101 # ojo: ultimo voiced
    p2nx = 132 # ojo: segundo voiced
    
    pm_interp_segment(v_pm_smpls, p1nx, p2nx)    
    
    # plot(np.diff(v_pm_smpls))
    
    
    # To diff:
    v_pm_diff = np.diff(v_pm_smpls)
    v_pm_diff = np.hstack((v_pm_smpls[0], v_pm_diff))
    
    # Interp in diff domain:
    v_voi_nxs = np.where(v_voi == 1)[0]
    fun_intrp = interpolate.interp1d(v_voi_nxs, v_pm_diff[v_voi_nxs], bounds_error=False,fill_value='extrapolate',  kind='linear')
    v_pm_sec_intrp = fun_intrp(np.arange(n_frms)) 
    
    # Plot:
    pl.figure(3)
    pl.ioff()
    pl.plot(v_voi_nxs, v_pm_diff[v_voi_nxs], '-b')    
    pl.plot(v_pm_diff, '-r')
    pl.plot(v_pm_sec_intrp, '-g')
    pl.show()
    
    
    # To pm:
    v_pm_smpls_rec = np.cumsum(v_pm_diff)
    
 
    

    

    
    plot(np.diff(v_pm_sec_intrp))
    
    
    
    
    return
'''

#==============================================================================
# From 'analysis_with_del_comp_and_ph_encoding_from_files'
# f0_type: 'f0', 'lf0'
def analysis_with_del_comp__ph_enc__f0_norm__from_files(wav_file, est_file, nFFT, mvf, f0_type='f0', b_ph_unv_zero=False, win_func=np.hanning):

    m_spmgc, m_phs_mgc, m_phc_mgc, v_shift, v_voi, fs = analysis_with_del_comp_and_ph_encoding_from_files(wav_file, est_file, nFFT, mvf, b_ph_unv_zero=b_ph_unv_zero, win_func=win_func)
    
    v_f0 = shift_to_f0(v_shift, v_voi, fs, out=f0_type)
    
    return m_spmgc, m_phs_mgc, m_phc_mgc, v_shift, v_voi, v_f0, fs  

    
#==============================================================================    
def get_fft_params_from_complex_data(m_fft):
    m_mag  = np.absolute(m_fft) 
    m_real = m_fft.real / m_mag # = p_phc
    m_imag = m_fft.imag / m_mag # = p_phs
    
    return m_mag, m_real, m_imag


#=======================================================================================

def analysis_with_del_comp__ph_enc__f0_norm__from_files_raw(wav_file, est_file, nFFT, win_func=np.hanning):
    '''
    This function does not perform any Mel warping or data compression
    '''
    # Read wav file:-----------------------------------------------------------
    v_in_sig, fs = sf.read(wav_file)

    # Analysis:----------------------------------------------------------------
    m_sp_dummy, m_ph_dummy, v_shift, v_voi, m_frms, m_fft = analysis_with_del_comp_from_est_file(v_in_sig, est_file, nFFT, fs, win_func=win_func)

    # Get fft-params:----------------------------------------------------------
    m_mag, m_real, m_imag = get_fft_params_from_complex_data(m_fft)

    return m_mag, m_real, m_imag, v_shift, v_voi

    
#==============================================================================   
# v2: New fft feats (mag, real, imag) in Mel-frequency scale.
#     Selection of number of coeffs.
# mvf: Maximum voiced frequency for phase encoding
# After 'analysis_with_del_comp_and_ph_encoding'    
# new: returns voi/unv decision.
# This function performs Mel Warping and vector cutting (for phase)
def analysis_with_del_comp__ph_enc__f0_norm__from_files2(wav_file, est_file, nFFT, mvf, f0_type='f0', win_func=np.hanning, mag_mel_nbins=60, cmplx_ph_mel_nbins=45):

    m_mag, m_real, m_imag, v_shift, v_voi  = analysis_with_del_comp__ph_enc__f0_norm__from_files_raw(wav_file, est_file, nFFT, win_func=win_func)
        
    # Mel warp:----------------------------------------------------------------
    m_mag_mel = la.sp_mel_warp(m_mag, mag_mel_nbins, alpha=0.77, in_type=3)
    m_mag_mel_log = np.log(m_mag_mel)

    # Phase:-------------------------------------------------------------------
    m_imag_mel = la.sp_mel_warp(m_imag, mag_mel_nbins, alpha=0.77, in_type=2)
    m_real_mel = la.sp_mel_warp(m_real, mag_mel_nbins, alpha=0.77, in_type=2)

    # Cutting phase vectors:
    m_imag_mel = m_imag_mel[:,:cmplx_ph_mel_nbins]
    m_real_mel = m_real_mel[:,:cmplx_ph_mel_nbins]
    
    m_real_mel = np.clip(m_real_mel, -1, 1)
    m_imag_mel = np.clip(m_imag_mel, -1, 1)

    # F0:----------------------------------------------------------------------
    v_f0 = shift_to_f0(v_shift, v_voi, fs, out=f0_type)
    
    return m_mag_mel_log, m_real_mel, m_imag_mel, v_shift, v_f0, fs


# mvf: Maximum voiced frequency for phase encoding=============================
# After 'analysis_with_del_comp_and_ph_encoding'    
# new: returns voi/unv decision.
def analysis_with_del_comp_and_ph_encoding_from_files(wav_file, est_file, nFFT, mvf, b_ph_unv_zero=False, win_func=np.hanning):

    # Read wav file:
    v_in_sig, fs = sf.read(wav_file)

    m_sp, m_ph, v_shift, v_voi, m_frms = analysis_with_del_comp_from_est_file(v_in_sig, est_file, nFFT, fs, b_ph_unv_zero=b_ph_unv_zero, win_func=win_func)
    
    '''
    # Debug:
    fb = la.MelFBank(2049, 60, 0.77)
    m_sp_mel = np.dot(m_sp, fb.m_warp)
    m_sp_rec = np.dot(m_sp_mel, fb.m_unwarp)
    
    holdon()
    nx = 90# 90
    plot(la.db(m_sp[nx,:]),'-b')
    plot(la.db(m_sp_rec[nx,:]),'-r')
    holdoff()
    '''
   
    # Phase encoding:
    m_phs, m_phc = ph_enc(m_ph)
    
    # Sp to MGC:
    m_spmgc = la.sp_to_mcep(m_sp)  
    
    '''    
    # Debug:
    fb = la.MelFBank(2049, 60, 0.77)
    m_sp_mel = np.dot(m_spmgc, fb.m_trans.T)

    m_spmgc2 = np.dot(np.log(m_sp), fb.m_trans)
    #m_spmgc2 = np.dot(m_sp, fb.m_trans)

    
    holdon()
    nx = 90# 90
    plot(m_spmgc[nx,:],'-b')
    plot(m_spmgc2[nx,:],'-r')
    holdoff()

    holdon()
    nx = 90# 90
    plot(np.log(m_sp[nx,:]),'-b')
    plot(m_sp_mel[nx,:],'-r')
    holdoff()    
    '''
    '''
    # Debug:
    m_sp_db    = la.db(m_sp)
    m_sp_lq_db = la.db(la.mcep_to_sp(m_spmgc, nFFT))
    
    m_sp_diff_db = m_sp_db - m_sp_lq_db
    m_sp_diff_db[m_sp_diff_db < 0] = 0
    v_sp_diff_db_mean = np.mean(m_sp_diff_db, axis=0)
    
    v_div = np.sum(m_sp_diff_db > 0, axis=0)
    v_sp_diff_db_mean2 = np.sum(m_sp_diff_db, axis=0) / v_div
    #plotm(m_sp_lq_db)
    #plotm(m_sp_log)
    '''
    
    
    '''
    nx = 81
    holdon()
    plot(m_sp_db[nx,:], '-b')
    plot(m_sp_lq_db[nx,:], '-r')
    plot(v_sp_diff_db_mean, '-k')    
    plot(v_sp_diff_db_mean2, '-g') 
    holdoff()
    '''
    
    # Ph to MGC up to MVF:        
    nFFT        = 2*(np.size(m_sp,1) - 1)
    mvf_bin     = lu.round_to_int(mvf * nFFT / np.float(fs))
    nFFThalf_ph = la.next_pow_of_two(mvf_bin) + 1    

    m_phs_shrt       = m_phs[:,:mvf_bin]    
    m_phc_shrt       = m_phc[:,:mvf_bin]
    f_interps        = interpolate.interp1d(np.arange(mvf_bin), m_phs_shrt, kind='cubic')
    f_interpc        = interpolate.interp1d(np.arange(mvf_bin), m_phc_shrt, kind='cubic')
    m_phs_shrt_intrp = f_interps(np.linspace(0,mvf_bin-1,nFFThalf_ph))    
    m_phc_shrt_intrp = f_interpc(np.linspace(0,mvf_bin-1,nFFThalf_ph))    
    m_phs_mgc        = la.sp_to_mcep(m_phs_shrt_intrp, in_type=1)    
    m_phc_mgc        = la.sp_to_mcep(m_phc_shrt_intrp, in_type=1) 
    
    return m_spmgc, m_phs_mgc, m_phc_mgc, v_shift, v_voi, fs

    

# mvf: Maximum voiced frequency for phase encoding    
def analysis_with_del_comp_and_ph_encoding(v_in_sig, nFFT, fs, mvf):

    m_sp, m_ph, v_shift = analysis_with_del_comp(v_in_sig, nFFT, fs)
    
    # Phase encoding:
    m_phs, m_phc = ph_enc(m_ph)
    
    # Sp to MGC:
    m_spmgc    = la.sp_to_mcep(m_sp)     
    
    # Ph to MGC up to MVF:    
    #mvf         = 4500    
    nFFT        = 2*(np.size(m_sp,1) - 1)
    mvf_bin     = lu.round_to_int(mvf * nFFT / np.float(fs))
    nFFThalf_ph = la.next_pow_of_two(mvf_bin) + 1    

    m_phs_shrt       = m_phs[:,:mvf_bin]    
    m_phc_shrt       = m_phc[:,:mvf_bin]
    f_interps        = interpolate.interp1d(np.arange(mvf_bin), m_phs_shrt, kind='cubic')
    f_interpc        = interpolate.interp1d(np.arange(mvf_bin), m_phc_shrt, kind='cubic')
    m_phs_shrt_intrp = f_interps(np.linspace(0,mvf_bin-1,nFFThalf_ph))    
    m_phc_shrt_intrp = f_interpc(np.linspace(0,mvf_bin-1,nFFThalf_ph))    
    m_phs_mgc        = la.sp_to_mcep(m_phs_shrt_intrp, in_type=1)    
    m_phc_mgc        = la.sp_to_mcep(m_phc_shrt_intrp, in_type=1) 
    
    return m_spmgc, m_phs_mgc, m_phc_mgc, v_shift
    
    
#==============================================================================
def synth_only_with_noise(m_sp, v_shift, v_voi, nFFT, fs, mvf, func_win_ana=np.hanning, ph_type='minph'):
    
    # Inputs for now:
    #func_win_ana = la.cos_win
    #func_win_ana = np.hanning
    #func_win_syn = la.cos_win
    
    # TD Noise Gen:    
    v_pm    = la.shift_to_pm(v_shift)
    sig_len = v_pm[-1] + (v_pm[-1] - v_pm[-2]) 
    v_noise = np.random.uniform(-1, 1, sig_len)

    # Extract frames:    
    l_frames, v_lens, v_pm_plus, v_shift_dummy, v_rights = windowing(v_noise, v_pm, win_func=func_win_ana)

    # Frame-based processing:
    nfrms = len(v_shift)
    m_frm = np.zeros((nfrms,nFFT))
    for i in xrange(nfrms): 
        v_sp = m_sp[i,:]

        # Debug:
        #v_sp = m_sp[30,:]            
        
        m_frm[i,:] = la.stamp_mag_sp(l_frames[i], v_sp, v_shift[i], ph_type='minph') 
        # Debug:
        #m_frm[i,:] = la.stamp_mag_sp(l_frames[i], v_sp, v_shift[i], ph_type='linph')
                
    v_sig = la.ola(m_frm, v_pm)
    
    # Debug:
    '''
    holdon()
    max = 129000
    plot(v_noise[:max],'-b')
    plot(v_sig[:max],'-r')
    holdoff()
    '''
    
    '''
    plotm(m_frm[298:313,:100])
    plotm(la.db(m_sp))
    '''
        
    return v_sig
    


#==============================================================================
# Input: f0, instead of shifts (v_shift).
# v_voi: Used to construct the ap mask:
# if v_voi[n] > 0, frame is voiced. If v_voi[n] == 0, frame is unvoiced. 
# If v_voy=='estim', the mask is estimated from phase data.
def synthesis_with_del_comp__ph_enc__from_f0(m_spmgc, m_phs, m_phc, v_f0, nFFT, fs, mvf, ph_hf_gen, v_voi='estim'):
    
    v_shift   = f0_to_shift(v_f0, fs)    
    v_syn_sig = synthesis_with_del_comp_and_ph_encoding(m_spmgc, m_phs, m_phc, v_shift, nFFT, fs, mvf, ph_hf_gen, v_voi=v_voi)
    
    # Debug:
    #v_syn_sig = synthesis_with_del_comp_and_ph_encoding_voi_unv_separated(m_spmgc, m_phs, m_phc, v_shift, v_voi, nFFT, fs, mvf, ph_hf_gen)

    return v_syn_sig

    
#==============================================================================
'''
#==============================================================================
# If ph_hf_gen=='rand', generates random numbers for the phase above mvf
# If ph_hf_gen=='template_mask', uses a phase template to fill the gaps given by the aperiodic mask.
# If ph_hf_gen=='rand_mask' The same as above, but it uses random numbers instead of a template.
# The aperiodic mask is computed (estimated) according to the total phase energy per frame.
# v_voi: Used to construct the ap mask:
# if v_voi[n] > 0, frame is voiced. If v_voi[n] == 0, frame is unvoiced. 
# If v_voy=='estim', the mask is estimated from phase data.
def synthesis_with_del_comp_and_ph_encoding_voi_unv_separated(m_spmgc, m_phs_mgc, m_phc_mgc, v_shift, v_voi, nFFT, fs, mvf, ph_hf_gen="rand_mask"):
    
    # 1.-Magnitude Spectrum:---------------------------------------------------
    # MGC to SP:
    m_sp_syn = la.mcep_to_sp(m_spmgc, nFFT)
    
    # 2.-Deterministic Phase:--------------------------------------------------
    
    # Ph and MVF:
    mvf_bin     = lu.round_to_int(mvf * nFFT / np.float(fs))
    nFFThalf_ph = la.next_pow_of_two(mvf_bin) + 1      
    
    # MGC to Ph up to MVF:
    m_phs_shrt_intrp_syn = la.mcep_to_sp(m_phs_mgc, 2*(nFFThalf_ph-1), out_type=0)
    m_phc_shrt_intrp_syn = la.mcep_to_sp(m_phc_mgc, 2*(nFFThalf_ph-1), out_type=0)
    f_interps_syn        = interpolate.interp1d(np.arange(nFFThalf_ph), m_phs_shrt_intrp_syn, kind='cubic')
    f_interpc_syn        = interpolate.interp1d(np.arange(nFFThalf_ph), m_phc_shrt_intrp_syn, kind='cubic')
    m_phs_shrt_syn       = f_interps_syn(np.linspace(0,nFFThalf_ph-1,mvf_bin))
    m_phc_shrt_syn       = f_interpc_syn(np.linspace(0,nFFThalf_ph-1,mvf_bin))
    
    # Generate phase up to Nyquist:
    nfrms    = np.size(m_phs_shrt_syn,0)
    nFFThalf = nFFT / 2 + 1
    m_phs_shrt_syn = np.clip(m_phs_shrt_syn, -1, 1)  
    m_phc_shrt_syn = np.clip(m_phc_shrt_syn, -1, 1)   
            
    # Phase decoding:
    m_ph_deter = ph_dec(m_phs_shrt_syn, m_phc_shrt_syn, mode='angle') 
    m_ph_deter = np.hstack((m_ph_deter, np.zeros((nfrms,nFFThalf-mvf_bin))))
    
    # 3.-Aperiodic Signal:-----------------------------------------------------
    # Getting aperiodicity mask:
    m_ph_ap_mask = get_ap_mask_from_uv_decision(v_voi, nFFT, fs, mvf)
    
    # Gen aperiodic phase:
    if ph_hf_gen is 'template_mask':   
        m_ap_ph = gen_rand_phase_by_template('../database/ph_template_1.npy',nfrms, nFFThalf)

    elif ph_hf_gen is 'rand_mask':       
        m_ap_ph = np.random.uniform(-np.pi, np.pi, size=(nfrms,nFFThalf))
        
    # Synth of Aperiodic Areas:------------------------------------------------
    v_ap_win   = np.zeros(nFFT)
    mid_frm_nx = nFFT / 2
    v_shift    = lu.round_to_int(v_shift)
    v_ap_sig   = np.zeros(np.sum(v_shift[:-1]) + nFFT + 1)
    strt = 0
    for f in xrange(nfrms-1):
        
        # From Phase to Time-Domain:
        v_ap_ph   = m_ap_ph[f,:]
        v_ap_ph   = la.add_hermitian_half(v_ap_ph[None,:], data_type='phase')[0]
        v_sp_comp = np.exp(v_ap_ph * 1j)
        v_ap_frm  = np.fft.ifft(v_sp_comp).real
        
        # Windowing:
        v_ap_win[:] = 0 # reset        
        v_curr_win_shrt = la.gen_wider_window(np.hanning,v_shift[f], v_shift[f+1], 0.15) # 0.15: value obtained empirically                
        v_ap_win[(mid_frm_nx-v_shift[f]):(mid_frm_nx+v_shift[f+1])] = v_curr_win_shrt        
        v_ap_frm = v_ap_frm * v_ap_win
        
        # To frequency domain - again:
        v_sp_comp = np.fft.fft(v_ap_frm)
        v_curr_ph = np.angle(v_sp_comp)
        
        # Magnitude Spectrum Stamping:
        v_targ_sp = m_sp_syn[f,:] * m_ph_ap_mask[f,:]
        v_sp_comp = la.add_hermitian_half(v_targ_sp[None,:])[0] * np.exp(v_curr_ph * 1j)
        v_ap_frm  = np.fft.ifft(v_sp_comp).real
        
        # Window again:
        rms_prev  = np.sqrt(np.mean(v_ap_frm**2))
        v_ap_frm  = v_ap_frm * v_ap_win
        rms_after = np.sqrt(np.mean(v_ap_frm**2))
        v_ap_frm  = v_ap_frm * rms_prev / rms_after 
        
        # OLA:
        v_ap_sig[strt:(strt+nFFT)] += v_ap_frm        
        strt += v_shift[f]

    v_ap_sig = v_ap_sig[(nFFT/2 - v_shift[0]):]   
         
    # Deterministic Signal:----------------------------------------------------              
    m_ph_det_syn = m_ph_deter * (1 - m_ph_ap_mask) 
    m_sp_det_syn = m_sp_syn   * (1 - m_ph_ap_mask)       

    # Final Synthesis:
    v_det_sig = synthesis_with_del_comp(m_sp_det_syn, m_ph_det_syn, v_shift)       

    return v_det_sig + v_ap_sig
'''
#==============================================================================
# If ph_hf_gen=='rand', generates random numbers for the phase above mvf
# If ph_hf_gen=='template_mask', uses a phase template to fill the gaps given by the aperiodic mask.
# If ph_hf_gen=='rand_mask' The same as above, but it uses random numbers instead of a template.
# The aperiodic mask is computed (estimated) according to the total phase energy per frame.
# v_voi: Used to construct the ap mask:
# if v_voi[n] > 0, frame is voiced. If v_voi[n] == 0, frame is unvoiced. 
# If v_voy=='estim', the mask is estimated from phase data.
def synthesis_with_del_comp_and_ph_encoding_voi_unv_separated(m_spmgc, m_phs_mgc, m_phc_mgc, v_shift, v_voi, nFFT, fs, mvf, ph_hf_gen="rand_mask"):
    
    # 1.-Magnitude Spectrum:---------------------------------------------------
    # MGC to SP:
    m_sp_syn = la.mcep_to_sp(m_spmgc, nFFT)
    
    # 2.-Deterministic Phase:--------------------------------------------------
    
    # Ph and MVF:
    mvf_bin     = lu.round_to_int(mvf * nFFT / np.float(fs))
    nFFThalf_ph = la.next_pow_of_two(mvf_bin) + 1      
    
    # MGC to Ph up to MVF:
    m_phs_shrt_intrp_syn = la.mcep_to_sp(m_phs_mgc, 2*(nFFThalf_ph-1), out_type=0)
    m_phc_shrt_intrp_syn = la.mcep_to_sp(m_phc_mgc, 2*(nFFThalf_ph-1), out_type=0)
    f_interps_syn        = interpolate.interp1d(np.arange(nFFThalf_ph), m_phs_shrt_intrp_syn, kind='cubic')
    f_interpc_syn        = interpolate.interp1d(np.arange(nFFThalf_ph), m_phc_shrt_intrp_syn, kind='cubic')
    m_phs_shrt_syn       = f_interps_syn(np.linspace(0,nFFThalf_ph-1,mvf_bin))
    m_phc_shrt_syn       = f_interpc_syn(np.linspace(0,nFFThalf_ph-1,mvf_bin))
    
    # Generate phase up to Nyquist:
    nfrms    = np.size(m_phs_shrt_syn,0)
    nFFThalf = nFFT / 2 + 1
    m_phs_shrt_syn = np.clip(m_phs_shrt_syn, -1, 1)  
    m_phc_shrt_syn = np.clip(m_phc_shrt_syn, -1, 1)   
            
    # Phase decoding:
    m_ph_deter = ph_dec(m_phs_shrt_syn, m_phc_shrt_syn, mode='angle') 
    m_ph_deter = np.hstack((m_ph_deter, np.zeros((nfrms,nFFThalf-mvf_bin))))
    
    # 3.-Aperiodic Signal:-----------------------------------------------------
    # Getting aperiodicity mask:
    m_ph_ap_mask = get_ap_mask_from_uv_decision(v_voi, nFFT, fs, mvf, fade_len=1)

    # Apply ap mask (PRUE):
    m_sp_ap_syn = m_sp_syn * m_ph_ap_mask     
    #m_sp_ap_syn = m_sp_syn   
    # Synth of Aperiodic Areas:------------------------------------------------
    v_ap_sig = synth_only_with_noise(m_sp_ap_syn, v_shift, v_voi, nFFT, fs, mvf, func_win_ana=la.cos_win)
    
         
    # Deterministic Signal:----------------------------------------------------              
    m_ph_det_syn = m_ph_deter * (1 - m_ph_ap_mask) 
    m_sp_det_syn = m_sp_syn   * (1 - m_ph_ap_mask)       

    # Final Synthesis:
    v_det_sig = synthesis_with_del_comp(m_sp_det_syn, m_ph_det_syn, v_shift)  

    # Debug:
    '''
    play(v_ap_sig, fs)
    play(v_ap_sig + v_det_sig, fs)
    '''     

    return v_det_sig + v_ap_sig

    
#==============================================================================
# v2: Improved phase generation. 
# v3: specific window handling for aperiodic spectrum in voiced segments.
# v4: Splitted window support
# v5: Works with new fft params: mag_mel_log, real_mel, and imag_mel
# If ph_hf_gen=='rand', generates random numbers for the phase above mvf
# If ph_hf_gen=='template_mask', uses a phase template to fill the gaps given by the aperiodic mask.
# If ph_hf_gen=='rand_mask' The same as above, but it uses random numbers instead of a template.
# The aperiodic mask is computed (estimated) according to the total phase energy per frame.
# v_voi: Used to construct the ap mask:
# if v_voi[n] > 0, frame is voiced. If v_voi[n] == 0, frame is unvoiced. 
# If v_voy=='estim', the mask is estimated from phase data.
# hf_slope_coeff: 1=no slope, 2=finishing with twice the energy at highest frequency.
def synthesis_with_del_comp_and_ph_encoding5(m_mag_mel_log, m_real_mel, m_imag_mel, v_f0, nfft, fs, mvf, f0_type='lf0', hf_slope_coeff=1.0, b_use_ap_voi=True, b_voi_ap_win=True):
    
    if f0_type=='lf0':
        v_f0 = np.exp(v_f0)
        
    # Debug:
    '''    
    vb_voi = v_f0 > 1
    v_f02 = np.zeros(len(v_f0))
    #v_f02[vb_voi] = signal.medfilt(v_f0[vb_voi], kernel_size=37)    
    L = 20
    v_win = np.hanning(L)
    v_f02[vb_voi] = np.convolve(v_f0[vb_voi], v_win / np.sum(v_win), mode='same')
    v_f0 = v_f02
    #v_f02 = vb_voi * signal.medfilt(v_f0, kernel_size=11)
    '''
    '''
    holdon()
    nx = 9
    plot(v_f0, '-b')    
    plot(v_f02, '-r')
    holdoff() 
    '''
        
    nfrms, ncoeffs_mag = m_mag_mel_log.shape
    ncoeffs_comp = m_real_mel.shape[1] 
    nfft_half    = nfft / 2 + 1

    # Magnitude mel-unwarp:----------------------------------------------------
    m_mag = np.exp(la.sp_mel_unwarp(m_mag_mel_log, nfft_half, alpha=0.77, in_type='log'))

    # Complex mel-unwarp:------------------------------------------------------
    f_intrp_real = interpolate.interp1d(np.arange(ncoeffs_comp), m_real_mel, kind='nearest', fill_value='extrapolate')
    f_intrp_imag = interpolate.interp1d(np.arange(ncoeffs_comp), m_imag_mel, kind='nearest', fill_value='extrapolate')
    
    m_real_mel = f_intrp_real(np.arange(ncoeffs_mag))
    m_imag_mel = f_intrp_imag(np.arange(ncoeffs_mag)) 
    
    m_real = la.sp_mel_unwarp(m_real_mel, nfft_half, alpha=0.77, in_type='log')   
    m_imag = la.sp_mel_unwarp(m_imag_mel, nfft_half, alpha=0.77, in_type='log') 
    
    # Noise Gen:---------------------------------------------------------------
    v_shift = f0_to_shift(v_f0, fs, unv_frm_rate_ms=5).astype(int)
    v_pm    = la.shift_to_pm(v_shift)
    
    ns_len = v_pm[-1] + (v_pm[-1] - v_pm[-2]) 
    v_ns   = np.random.uniform(-1, 1, ns_len)     
    
    # Noise Windowing:---------------------------------------------------------
    l_ns_win_funcs = [ np.hanning ] * nfrms
    vb_voi = v_f0 > 1 # case voiced  (1 is used for safety)  
    if b_voi_ap_win:        
        for i in xrange(nfrms):
            if vb_voi[i]:         
                l_ns_win_funcs[i] = voi_noise_window

    l_frm_ns, v_lens, v_pm_plus, v_shift_dummy, v_rights = windowing(v_ns, v_pm, win_func=l_ns_win_funcs)   # Checkear!! 
    
    m_frm_ns  = la.frm_list_to_matrix(l_frm_ns, v_shift, nfft)
    m_frm_ns  = np.fft.fftshift(m_frm_ns, axes=1)    
    m_ns_cmplx = la.remove_hermitian_half(np.fft.fft(m_frm_ns))

    # AP-Mask:-----------------------------------------------------------------   
    cf = 5000 #5000
    bw = 2000 #2000 
    
    # Norm gain:
    m_ns_mag  = np.absolute(m_ns_cmplx)
    rms_noise = np.sqrt(np.mean(m_ns_mag**2)) # checkear!!!!
    m_ap_mask = np.ones(m_ns_mag.shape)
    m_ap_mask = m_mag * m_ap_mask / rms_noise

    m_zeros = np.zeros((nfrms, nfft_half))    
    m_ap_mask[vb_voi,:] = la.spectral_crossfade(m_zeros[vb_voi,:], m_ap_mask[vb_voi,:], cf, bw, fs, freq_scale='hz') 
    
    # HF - enhancement:          
    v_slope  = np.linspace(1, hf_slope_coeff, num=nfft_half)
    m_ap_mask[~vb_voi,:] = m_ap_mask[~vb_voi,:] * v_slope 
    
    # Det-Mask:----------------------------------------------------------------    
    m_det_mask = m_mag
    m_det_mask[~vb_voi,:] = 0
    m_det_mask[vb_voi,:]  = la.spectral_crossfade(m_det_mask[vb_voi,:], m_zeros[vb_voi,:], cf, bw, fs, freq_scale='hz')
    
    # Applying masks:----------------------------------------------------------
    m_ap_cmplx  = m_ap_mask  * m_ns_cmplx
    m_det_cmplx = m_real + m_imag * 1j
    m_det_cmplx = m_det_mask * m_det_cmplx / np.absolute(m_det_cmplx)

    # bin width: bw=11.71875 Hz    
    # Final synth:-------------------------------------------------------------
    m_syn_cmplx = la.add_hermitian_half(m_ap_cmplx + m_det_cmplx, data_type='complex')    
    m_syn_td    = np.fft.ifft(m_syn_cmplx).real
    m_syn_td    = np.fft.fftshift(m_syn_td,  axes=1)
    v_syn_sig   = la.ola(m_syn_td,  v_pm, win_func=None) 
       
    # HPF:---------------------------------------------------------------------     
    fc = 60
    order = 4
    fc_norm = fc / (fs / 2.0)
    bc, ac = signal.ellip(order,0.5 , 80, fc_norm, btype='highpass')
    v_syn_sig = signal.lfilter(bc, ac, v_syn_sig)

    # Debug:
    '''
    holdon()
    plot(la.db(m_mag[264,:]), '-b')    
    plot(la.db(m_mag_syn[264,:]), '-r')
    plot(10 * m_ph[264,:], '-k')    
    plot(10 * m_ph_ns[264,:], '.-g')
    holdoff()  
    
    holdon()
    plot(la.db(m_mag[264,:]), '-b')    
    plot(la.db(m_mag_syn[264,:]), '-r')
    plot(10 * m_ph[264,:], '-k')    
    plot(10 * m_ph_syn[264,:], '-g')
    holdoff()    
    
    holdon()
    plot(la.db(m_mag[264,:]), '-b')    
    plot(la.db(m_mag_syn[264,:]), '-r')
    plot(la.db(m_mag[265,:]), '-k')    
    plot(la.db(m_mag_syn[265,:]), '-g')
    holdoff()
    
    holdon()
    plot(la.db(m_mag[264,:]), '-b')
    plot(la.db(m_mag[265,:]), '-r')
    holdoff()
    
    holdon()
    plot(m_ph_syn[264,:], '-b')
    plot(m_ph_syn[265,:], '-r')
    holdoff()    
    
    '''
    # la.write_audio_file(out_dir + '/' + filename + suffix + '.wav', v_sig_syn, fs)
    return v_syn_sig   


'''
def synthesis_with_del_comp_and_ph_encoding5(m_mag_mel_log, m_real_mel, m_imag_mel, v_f0, nfft, fs, mvf, f0_type='lf0', hf_slope_coeff=1.0, b_use_ap_voi=True, b_voi_ap_win=True):
    
    if f0_type=='lf0':
        v_f0 = np.exp(v_f0)

        
    nfrms, ncoeffs_mag = m_mag_mel_log.shape
    ncoeffs_comp = m_real_mel.shape[1] 
    nfft_half    = nfft / 2 + 1

    # Magnitude mel-unwarp:----------------------------------------------------
    m_mag = np.exp(la.sp_mel_unwarp(m_mag_mel_log, nfft_half, alpha=0.77, in_type='log'))

    # Complex mel-unwarp:------------------------------------------------------
    f_intrp_real = interpolate.interp1d(np.arange(ncoeffs_comp), m_real_mel, kind='nearest', fill_value='extrapolate')
    f_intrp_imag = interpolate.interp1d(np.arange(ncoeffs_comp), m_imag_mel, kind='nearest', fill_value='extrapolate')
    
    m_real_mel = f_intrp_real(np.arange(ncoeffs_mag))
    m_imag_mel = f_intrp_imag(np.arange(ncoeffs_mag)) 
    
    m_real = la.sp_mel_unwarp(m_real_mel, nfft_half, alpha=0.77, in_type='log')   
    m_imag = la.sp_mel_unwarp(m_imag_mel, nfft_half, alpha=0.77, in_type='log') 
    
    # Noise Gen:---------------------------------------------------------------
    v_shift = f0_to_shift(v_f0, fs, unv_frm_rate_ms=5).astype(int)
    v_pm    = la.shift_to_pm(v_shift)
    
    ns_len = v_pm[-1] + (v_pm[-1] - v_pm[-2]) 
    v_ns   = np.random.uniform(-1, 1, ns_len)     
    
    # Noise Windowing:---------------------------------------------------------
    l_ns_win_funcs = [ np.hanning ] * nfrms
    vb_voi = v_f0 > 1 # case voiced  (1 is used for safety)  
    if b_voi_ap_win:        
        for i in xrange(nfrms):
            if vb_voi[i]:         
                l_ns_win_funcs[i] = voi_noise_window

    l_frm_ns, v_lens, v_pm_plus, v_shift_dummy, v_rights = windowing(v_ns, v_pm, win_func=l_ns_win_funcs)   # Checkear!! 
    
    m_frm_ns  = la.frm_list_to_matrix(l_frm_ns, v_shift, nfft)
    m_frm_ns  = np.fft.fftshift(m_frm_ns, axes=1)    
    m_ns_cmplx = la.remove_hermitian_half(np.fft.fft(m_frm_ns))
    
    m_ns_mag, m_ns_real, m_ns_imag = get_fft_params_from_complex_data(m_ns_cmplx)
   
    # Norm:
    rms_noise = np.sqrt(np.mean(m_ns_mag**2)) # checkear!!!!
    m_ns_mag  = m_ns_mag / rms_noise
    
    # HF - enhancement:          
    v_slope  = np.linspace(1, hf_slope_coeff, num=nfft_half)
    m_ns_mag[~vb_voi,:] = m_ns_mag[~vb_voi,:] * v_slope    
    
    # Merge data:--------------------------------------------------------------   
    cf_mag = 5000 #5000
    bw_mag = 2000 #2000
    
    cf_cmpx = cf_mag #5000
    bw_cmpx = bw_mag #2000 
    
    # Alloc:
    m_mag_ap  = np.zeros((nfrms, nfft_half))
    m_mag_det = np.zeros((nfrms, nfft_half))
    
  


    # Working:    
    m_ph     = np.angle(m_real    +    m_imag *1j)
    m_ph_ns  = np.angle(m_ns_real + m_ns_imag *1j)
    m_ph_ap  = m_ph_ns
    m_ph_det = m_ph
    m_mag_zeros = np.zeros((nfrms, nfft_half))
    
    if b_use_ap_voi:        
        # Mag - ap: 
        m_mag_ap[vb_voi,:]  = la.spectral_crossfade(m_mag_zeros[vb_voi,:], m_mag[vb_voi,:] * m_ns_mag[vb_voi,:], cf_mag, bw_mag, fs, freq_scale='hz')    
        m_mag_ap[~vb_voi,:] = m_mag[~vb_voi,:] * m_ns_mag[~vb_voi,:]
        

    
        #-------------------------------------------------------------------------------
            
        # Mag - det:
        m_mag_det[vb_voi,:]  = la.spectral_crossfade(m_mag[vb_voi,:], m_mag_zeros[vb_voi,:], cf_mag, bw_mag, fs, freq_scale='hz')

    else: # Check:
        # Mag - ap:                
        m_mag_ap[~vb_voi,:] = m_mag[~vb_voi,:] * m_ns_mag[~vb_voi,:]
        
        # Mag - det:
        m_mag_det[vb_voi,:] = m_mag[vb_voi,:]
        
    # Debug:
    m_syn_cmplx = m_mag_ap  * np.exp(m_ph_ap  * 1j) + m_mag_det * np.exp(m_ph_det * 1j)
    m_syn_cmplx = la.add_hermitian_half(m_syn_cmplx , data_type='complex')    
    

    
    # bin width: bw=11.71875 Hz
    
    # Final synth:-------------------------------------------------------------
    m_syn_td    = np.fft.ifft(m_syn_cmplx).real
    m_syn_td    = np.fft.fftshift(m_syn_td,  axes=1)
    v_syn_sig   = la.ola(m_syn_td,  v_pm, win_func=None) 
    
    
    # HPF:---------------------------------------------------------------------     
    fc = 60
    order = 4
    fc_norm = fc / (fs / 2.0)
    bc, ac = signal.ellip(order,0.5 , 80, fc_norm, btype='highpass')
    v_syn_sig = signal.lfilter(bc, ac, v_syn_sig)



    # la.write_audio_file(out_dir + '/' + filename + suffix + '.wav', v_sig_syn, fs)
    return v_syn_sig   
'''

'''
def synthesis_with_del_comp_and_ph_encoding5(m_mag_mel_log, m_real_mel, m_imag_mel, v_f0, nfft, fs, mvf, f0_type='lf0', hf_slope_coeff=1.0, b_use_ap_voi=True, b_voi_ap_win=True):
    
    if f0_type=='lf0':
        v_f0 = np.exp(v_f0)
     
    nfrms, ncoeffs_mag = m_mag_mel_log.shape
    ncoeffs_comp = m_real_mel.shape[1] 
    nfft_half    = nfft / 2 + 1

    # Magnitude mel-unwarp:----------------------------------------------------
    m_mag = np.exp(la.sp_mel_unwarp(m_mag_mel_log, nfft_half, alpha=0.77, in_type='log'))

    # Complex mel-unwarp:------------------------------------------------------
    f_intrp_real = interpolate.interp1d(np.arange(ncoeffs_comp), m_real_mel, kind='nearest', fill_value='extrapolate')
    f_intrp_imag = interpolate.interp1d(np.arange(ncoeffs_comp), m_imag_mel, kind='nearest', fill_value='extrapolate')
    
    m_real_mel = f_intrp_real(np.arange(ncoeffs_mag))
    m_imag_mel = f_intrp_imag(np.arange(ncoeffs_mag)) 

    # Debug:-------------------------------------------------------------------
    #m_real_mel = np.pad(m_real_mel, ((0,0),(0,ncoeffs_mag-ncoeffs_comp)), 'constant', constant_values=0)
    #m_imag_mel = np.pad(m_imag_mel, ((0,0),(0,ncoeffs_mag-ncoeffs_comp)), 'constant', constant_values=0)   
    
    
    m_real = la.sp_mel_unwarp(m_real_mel, nfft_half, alpha=0.77, in_type='log')   
    m_imag = la.sp_mel_unwarp(m_imag_mel, nfft_half, alpha=0.77, in_type='log') 
    
    
    # Debug:-------------------------------------------------------------------
    #m_cmpx_orig_mag = np.absolute(m_real + m_imag * 1j)
    #m_real = m_real / m_cmpx_orig_mag
    #m_imag = m_imag / m_cmpx_orig_mag
    
    # Noise Gen:---------------------------------------------------------------
    v_shift = f0_to_shift(v_f0, fs, unv_frm_rate_ms=5).astype(int)
    v_pm    = la.shift_to_pm(v_shift)
    
    ns_len = v_pm[-1] + (v_pm[-1] - v_pm[-2]) 
    v_ns   = np.random.uniform(-1, 1, ns_len)     
    
    # Noise Windowing:---------------------------------------------------------
    l_ns_win_funcs = [ np.hanning ] * nfrms
    vb_voi = v_f0 > 1 # case voiced  (1 is used for safety)  
    if b_voi_ap_win:        
        for i in xrange(nfrms):
            if vb_voi[i]:         
                l_ns_win_funcs[i] = voi_noise_window

    l_frm_ns, v_lens, v_pm_plus, v_shift_dummy, v_rights = windowing(v_ns, v_pm, win_func=l_ns_win_funcs)   # Checkear!! 
    
    m_frm_ns  = la.frm_list_to_matrix(l_frm_ns, v_shift, nfft)
    m_frm_ns  = np.fft.fftshift(m_frm_ns, axes=1)    
    m_ns_cmplx = la.remove_hermitian_half(np.fft.fft(m_frm_ns))
    
    m_ns_mag, m_ns_real, m_ns_imag = get_fft_params_from_complex_data(m_ns_cmplx)

   
   
    # Norm:
    rms_noise = np.sqrt(np.mean(m_ns_mag**2)) # checkear!!!!
    m_ns_mag  = m_ns_mag / rms_noise
    
    # HF - enhancement:          
    v_slope  = np.linspace(1, hf_slope_coeff, num=nfft_half)
    m_ns_mag[~vb_voi,:] = m_ns_mag[~vb_voi,:] * v_slope    
    
    # Merge data:--------------------------------------------------------------
    #cf_mag = 5000 #5000
    #bw_mag = 2000 #2000
    
    cf_mag = 6000 #5000
    bw_mag = 4000 #2000
    
    cf_cmpx = cf_mag #5000
    bw_cmpx = bw_mag #2000 
    
    # Alloc:
    m_mag_syn  = np.ones((nfrms, nfft_half))
    m_real_syn = np.zeros((nfrms, nfft_half))
    m_imag_syn = np.zeros((nfrms, nfft_half))
    
    if b_use_ap_voi:
        # Mag:        
        m_mag_syn[vb_voi,:]  = la.spectral_crossfade(m_mag[vb_voi,:], m_mag[vb_voi,:] * m_ns_mag[vb_voi,:], cf_mag, bw_mag, fs, freq_scale='hz')    
        m_mag_syn[~vb_voi,:] = m_mag[~vb_voi,:] * m_ns_mag[~vb_voi,:]

        #Compx - Voi:
        m_real_syn[vb_voi,:] = la.spectral_crossfade(m_real[vb_voi,:], m_ns_real[vb_voi,:], cf_cmpx, bw_cmpx, fs, freq_scale='hz')
        m_imag_syn[vb_voi,:] = la.spectral_crossfade(m_imag[vb_voi,:], m_ns_imag[vb_voi,:], cf_cmpx, bw_cmpx, fs, freq_scale='hz')
        
        #Compx - Unv:
        m_real_syn[~vb_voi,:] = m_ns_real[~vb_voi,:]
        m_imag_syn[~vb_voi,:] = m_ns_imag[~vb_voi,:]
    else:
        # Mag:        
        m_mag_syn[vb_voi,:]  = m_mag[vb_voi,:]
        m_mag_syn[~vb_voi,:] = m_mag[~vb_voi,:] * m_ns_mag[~vb_voi,:]
        
        # Compx - Voi:
        m_real_syn[vb_voi,:] = m_real[vb_voi,:]    
        m_imag_syn[vb_voi,:] = m_imag[vb_voi,:]
        
        # Compx - Unv:
        m_real_syn[~vb_voi,:] = m_ns_real[~vb_voi,:]
        m_imag_syn[~vb_voi,:] = m_ns_imag[~vb_voi,:]        
        
        
        
        
    

    
    
    # Final synth:-------------------------------------------------------------
    
    # Debug:--------------------------------------------------   
    g = (m_mag_syn * m_real_syn + m_mag_syn * m_imag_syn * 1j) / m_cmpx_mag
    m_g_mag = np.absolute(g)
    m_g_ph  = np.angle(g) 
    #m_ph  = np.angle(m_real_syn + m_imag_syn *1j) 
    #m_syn = m_mag_syn * np.exp(m_ph * 1j)
    #m_syn = la.add_hermitian_half(m_syn, data_type='complex')
    
    #m_syn = la.add_hermitian_half(m_mag_syn * m_real_syn + m_mag_syn * m_imag_syn * 1j, data_type='complex')
    
    
    #------------------------------------------------------------------------
    m_cmpx_mag = np.absolute(m_real_syn + m_imag_syn * 1j)
    m_syn = la.add_hermitian_half((m_mag_syn * m_real_syn + m_mag_syn * m_imag_syn * 1j) / m_cmpx_mag, data_type='complex')
    

    
    m_syn = np.fft.ifft(m_syn).real
    m_syn = np.fft.fftshift(m_syn, axes=1)    
    v_sig_syn = la.ola(m_syn, v_pm, win_func=None)    
    
    
    
    
    # HPF:---------------------------------------------------------------------     
    fc = 60
    order = 4
    fc_norm = fc / (fs / 2.0)
    bc, ac = signal.ellip(order,0.5 , 80, fc_norm, btype='highpass')
    v_sig_syn = signal.lfilter(bc, ac, v_sig_syn)
    

    
    return v_sig_syn, m_syn, m_mag_syn, m_real_syn, m_imag_syn    
'''    
    
#==============================================================================
# v2: Improved phase generation. 
# v3: specific window handling for aperiodic spectrum in voiced segments.
# v4: Splitted window support
# If ph_hf_gen=='rand', generates random numbers for the phase above mvf
# If ph_hf_gen=='template_mask', uses a phase template to fill the gaps given by the aperiodic mask.
# If ph_hf_gen=='rand_mask' The same as above, but it uses random numbers instead of a template.
# The aperiodic mask is computed (estimated) according to the total phase energy per frame.
# v_voi: Used to construct the ap mask:
# if v_voi[n] > 0, frame is voiced. If v_voi[n] == 0, frame is unvoiced. 
# If v_voy=='estim', the mask is estimated from phase data.
def synthesis_with_del_comp_and_ph_encoding4(m_spmgc, m_phs_mgc, m_phc_mgc, v_shift, nFFT, fs, mvf, v_voi, b_medfilt=False, win_func=None):
    
    #Protection:
    v_shift = v_shift.astype(int)
    
    # Ph and MVF:
    mvf_bin     = lu.round_to_int(mvf * nFFT / np.float(fs))
    nFFThalf_ph = la.next_pow_of_two(mvf_bin) + 1      

    # MGC to Ph up to MVF:
    m_phs_shrt_intrp_syn = la.mcep_to_sp_sptk(m_phs_mgc, 2*(nFFThalf_ph-1), out_type=0)
    m_phc_shrt_intrp_syn = la.mcep_to_sp_sptk(m_phc_mgc, 2*(nFFThalf_ph-1), out_type=0)
    f_interps_syn        = interpolate.interp1d(np.arange(nFFThalf_ph), m_phs_shrt_intrp_syn, kind='cubic')
    f_interpc_syn        = interpolate.interp1d(np.arange(nFFThalf_ph), m_phc_shrt_intrp_syn, kind='cubic')
    m_phs_shrt_syn       = f_interps_syn(np.linspace(0,nFFThalf_ph-1,mvf_bin))
    m_phc_shrt_syn       = f_interpc_syn(np.linspace(0,nFFThalf_ph-1,mvf_bin))
        
    # Deterministic Phase decoding:----------------------
    nfrms    = np.size(m_phs_shrt_syn,0)
    nFFThalf = nFFT / 2 + 1
    m_phs_shrt_syn = np.clip(m_phs_shrt_syn, -1, 1)  
    m_phc_shrt_syn = np.clip(m_phc_shrt_syn, -1, 1)  
    m_ph_deter     = ph_dec(m_phs_shrt_syn, m_phc_shrt_syn, mode='angle') 
    #m_ph_deter     = np.hstack((m_ph_deter, np.zeros((nfrms,nFFThalf-mvf_bin))))
    
    # Debug:
    f = interpolate.interp1d(np.arange(mvf_bin), m_ph_deter, kind='nearest', fill_value='extrapolate')
    m_ph_deter = f(np.arange(nFFThalf))

    # TD Noise Gen:---------------------------------------    
    v_pm    = la.shift_to_pm(v_shift)
    sig_len = v_pm[-1] + (v_pm[-1] - v_pm[-2]) 
    v_noise = np.random.uniform(-1, 1, sig_len)    
    #v_noise = np.random.normal(size=sig_len)
    
    # Extract noise magnitude and phase for unvoiced segments: (TODO: make it more efficient!)-------------------------------
    win_func_unv = np.hanning    
    if win_func is la.cos_win:
        win_func_unv = la.cos_win    
        
    l_frm_noise, v_lens, v_pm_plus, v_shift_dummy, v_rights = windowing(v_noise, v_pm, win_func=win_func_unv)    
    m_frm_noise = la.frm_list_to_matrix(l_frm_noise, v_shift, nFFT)
    m_frm_noise = np.fft.fftshift(m_frm_noise, axes=1)
    
    '''
    # Debug - randomise sequence of noise frames (NO BORRAR!):
    v_new_nx = np.random.randint(nfrms, size=nfrms)
    m_frm_noise = m_frm_noise[v_new_nx,:]
    #------------------------------------------
    '''
    
    m_noise_sp  = la.remove_hermitian_half(np.fft.fft(m_frm_noise))
    m_noise_ph  = np.angle(m_noise_sp)    
    m_noise_mag = np.absolute(m_noise_sp)
    m_noise_mag_log = np.log(m_noise_mag)
    # Noise amp-normalisation:
    rms_noise = np.sqrt(np.mean(m_noise_mag**2))
    m_noise_mag_log = m_noise_mag_log - np.log(rms_noise)  
    
    # Extract noise magnitude and phase for voiced segments: (TODO: make it more efficient!)-------------------------------------
    l_frm_voi_noise, v_lens, v_pm_plus, v_shift_dummy, v_rights = windowing(v_noise, v_pm, win_func=voi_noise_window)    
    m_frm_voi_noise = la.frm_list_to_matrix(l_frm_voi_noise, v_shift, nFFT)
    m_frm_voi_noise = np.fft.fftshift(m_frm_voi_noise, axes=1)
    m_voi_noise_sp  = la.remove_hermitian_half(np.fft.fft(m_frm_voi_noise))
    m_voi_noise_ph  = np.angle(m_voi_noise_sp)      
    m_voi_noise_mag = np.absolute(m_voi_noise_sp)
    m_voi_noise_mag_log = np.log(m_voi_noise_mag)
    # Noise amp-normalisation:
    rms_voi_noise = np.sqrt(np.mean(m_voi_noise_mag**2))
    m_voi_noise_mag_log = m_voi_noise_mag_log - np.log(rms_voi_noise)      
    
    #------------------------------------------------------------------------------------------------------------------------------
    
    # ap mask:
    v_voi_mask =  np.clip(v_voi, 0, 1)

    # target sp from mgc:
    m_sp_targ = la.mcep_to_sp_sptk(m_spmgc, nFFT)
    
    # medfilt:
    if b_medfilt:
        m_sp_targ = signal.medfilt(m_sp_targ, kernel_size=[3,1])        

    '''
    # Debug - Minimum phase filter for ap signal (NO BORRAR!):
    m_sp_comp_mph  = la.sp_to_min_phase(m_sp_targ, in_type='sp')
    m_sp_ph_mph    = np.angle(m_sp_comp_mph)    
    m_noise_ph     = m_noise_ph + m_sp_ph_mph
    m_voi_noise_ph = m_voi_noise_ph + m_sp_ph_mph
    '''
    
    # Alloc:    
    m_frm_syn = np.zeros((nfrms, nFFT))
    m_mag_syn = np.zeros((nfrms, nFFThalf)) # just for debug
    m_mag     = np.zeros((nfrms, nFFThalf)) # just for debug
   
    # Spectral crossfade constants (TODO: Improve this):
    muf = 3500 # "minimum unvoiced freq."
    bw = (mvf - muf) - 20 # values found empirically. assuming mvf > 4000
    cut_off = (mvf + muf) / 2
    v_zeros = np.zeros((1,nFFThalf))         
    
    # Iterates through frames:
    for i in xrange(nfrms):        
        

        if v_voi_mask[i] == 1: # voiced case            
            # Magnitude:----------------------------------------- 
            v_mag_log = m_voi_noise_mag_log[i,:]                                 
            v_mag_log = la.spectral_crossfade(v_zeros, v_mag_log[None,:], cut_off, bw, fs, freq_scale='hz')[0]    

            # Debug:
            v_mag_log = np.squeeze(v_zeros)
    

            # Phase:--------------------------------------------                      
            v_ph = la.spectral_crossfade(m_ph_deter[None, i,:], m_voi_noise_ph[None,i,:], cut_off, bw, fs, freq_scale='hz')[0]

            # Debug:
            
            v_ph_deters, v_ph_deterc = ph_enc(m_ph_deter[i,:])
            v_voi_noise_phs, v_voi_noise_phc = ph_enc(m_voi_noise_ph[i,:])
            
            v_phsA = la.spectral_crossfade(v_ph_deters[None,:], v_voi_noise_phs[None,:], 5000, 2000, fs, freq_scale='hz')[0]
            v_phcA = la.spectral_crossfade(v_ph_deterc[None,:], v_voi_noise_phc[None,:], 5000, 2000, fs, freq_scale='hz')[0]
            
            v_ph = ph_dec(v_phsA, v_phcA)
            
            #v_ph = m_ph_deter[i,:]
            
            
            '''
            holdon()
            plot(v_ph_deters, '.-b')
            plot(v_voi_noise_phs, '.-r')
            plot(v_phsA, '.-k')
            holdoff()
            '''

            
            '''
            holdon()
            plot(m_ph_deter[None, i,:], '.-b')
            plot(m_voi_noise_ph[None,i,:], '.-r')
            plot(v_ph, '.-k')
            holdoff()
            '''
            
        elif v_voi_mask[i] == 0: # unvoiced case
            # Magnitude:---------------------------------------
            v_mag_log = m_noise_mag_log[i,:]       
            # Debug:
            v_mag_log = np.squeeze(v_zeros)
            
            # Phase:--------------------------------------------
            v_ph = m_noise_ph[i,:]      
            
        # To complex:
        m_mag[i,:] = np.exp(v_mag_log) # just for debug
        v_mag = np.exp(v_mag_log) * m_sp_targ[i,:]
        v_sp  = v_mag * np.exp(v_ph * 1j) 
        v_sp  = la.add_hermitian_half(v_sp[None,:], data_type='complex')        
        
        '''                
        # Debug:
        holdon()
        plot(np.log(m_sp_targ[i,:]), '.-b')
        plot(v_mag_log, '.-r')
        plot(np.log(v_mag), '.-k')
        plot(m_voi_noise_mag_log[i,:], '-b')
        holdoff()
        '''
        
        # Save:
        #print(i)
        m_mag_syn[i,:] = v_mag # for inspection    
        m_frm_syn[i,:] = np.fft.fftshift(np.fft.ifft(v_sp).real)     
        
    v_sig_syn = la.ola(m_frm_syn, v_pm, win_func=win_func)
     
    return v_sig_syn, m_frm_syn, m_mag_syn, m_sp_targ, m_frm_noise, m_frm_voi_noise, m_mag    

    
#==============================================================================
# v3: specific window handling for aperiodic spectrum in voiced segments.
# v2: Improved phase generation. 
# If ph_hf_gen=='rand', generates random numbers for the phase above mvf
# If ph_hf_gen=='template_mask', uses a phase template to fill the gaps given by the aperiodic mask.
# If ph_hf_gen=='rand_mask' The same as above, but it uses random numbers instead of a template.
# The aperiodic mask is computed (estimated) according to the total phase energy per frame.
# v_voi: Used to construct the ap mask:
# if v_voi[n] > 0, frame is voiced. If v_voi[n] == 0, frame is unvoiced. 
# If v_voy=='estim', the mask is estimated from phase data.
def synthesis_with_del_comp_and_ph_encoding3(m_spmgc, m_phs_mgc, m_phc_mgc, v_shift, nFFT, fs, mvf, v_voi, b_medfilt=False):
    
    #Protection:
    v_shift = v_shift.astype(int)
    
    # Ph and MVF:
    mvf_bin     = lu.round_to_int(mvf * nFFT / np.float(fs))
    nFFThalf_ph = la.next_pow_of_two(mvf_bin) + 1      

    # MGC to Ph up to MVF:
    m_phs_shrt_intrp_syn = la.mcep_to_sp(m_phs_mgc, 2*(nFFThalf_ph-1), out_type=0)
    m_phc_shrt_intrp_syn = la.mcep_to_sp(m_phc_mgc, 2*(nFFThalf_ph-1), out_type=0)
    f_interps_syn        = interpolate.interp1d(np.arange(nFFThalf_ph), m_phs_shrt_intrp_syn, kind='cubic')
    f_interpc_syn        = interpolate.interp1d(np.arange(nFFThalf_ph), m_phc_shrt_intrp_syn, kind='cubic')
    m_phs_shrt_syn       = f_interps_syn(np.linspace(0,nFFThalf_ph-1,mvf_bin))
    m_phc_shrt_syn       = f_interpc_syn(np.linspace(0,nFFThalf_ph-1,mvf_bin))
        
    # Deterministic Phase decoding:----------------------
    nfrms    = np.size(m_phs_shrt_syn,0)
    nFFThalf = nFFT / 2 + 1
    m_phs_shrt_syn = np.clip(m_phs_shrt_syn, -1, 1)  
    m_phc_shrt_syn = np.clip(m_phc_shrt_syn, -1, 1)  
    m_ph_deter     = ph_dec(m_phs_shrt_syn, m_phc_shrt_syn, mode='angle') 
    m_ph_deter     = np.hstack((m_ph_deter, np.zeros((nfrms,nFFThalf-mvf_bin))))

    # TD Noise Gen:---------------------------------------    
    v_pm    = la.shift_to_pm(v_shift)
    sig_len = v_pm[-1] + (v_pm[-1] - v_pm[-2]) 
    v_noise = np.random.uniform(-1, 1, sig_len)    
    #v_noise = np.random.normal(size=sig_len)
    
    # Extract noise magnitude and phase for unvoiced segments: (TODO: make it more efficient!)-------------------------------
    l_frm_noise, v_lens, v_pm_plus, v_shift_dummy, v_rights = windowing(v_noise, v_pm, win_func=np.hanning)    
    m_frm_noise = la.frm_list_to_matrix(l_frm_noise, v_shift, nFFT)
    m_frm_noise = np.fft.fftshift(m_frm_noise, axes=1)
    
    '''
    # Debug - randomise sequence of noise frames (NO BORRAR!):
    v_new_nx = np.random.randint(nfrms, size=nfrms)
    m_frm_noise = m_frm_noise[v_new_nx,:]
    #------------------------------------------
    '''
    
    m_noise_sp  = la.remove_hermitian_half(np.fft.fft(m_frm_noise))
    m_noise_ph  = np.angle(m_noise_sp)    
    m_noise_mag = np.absolute(m_noise_sp)
    m_noise_mag_log = np.log(m_noise_mag)
    # Noise amp-normalisation:
    rms_noise = np.sqrt(np.mean(m_noise_mag**2))
    m_noise_mag_log = m_noise_mag_log - np.log(rms_noise)  
    
    # Extract noise magnitude and phase for voiced segments: (TODO: make it more efficient!)-------------------------------------
    l_frm_voi_noise, v_lens, v_pm_plus, v_shift_dummy, v_rights = windowing(v_noise, v_pm, win_func=voi_noise_window)    
    m_frm_voi_noise = la.frm_list_to_matrix(l_frm_voi_noise, v_shift, nFFT)
    m_frm_voi_noise = np.fft.fftshift(m_frm_voi_noise, axes=1)
    m_voi_noise_sp  = la.remove_hermitian_half(np.fft.fft(m_frm_voi_noise))
    m_voi_noise_ph  = np.angle(m_voi_noise_sp)      
    m_voi_noise_mag = np.absolute(m_voi_noise_sp)
    m_voi_noise_mag_log = np.log(m_voi_noise_mag)
    # Noise amp-normalisation:
    rms_voi_noise = np.sqrt(np.mean(m_voi_noise_mag**2))
    m_voi_noise_mag_log = m_voi_noise_mag_log - np.log(rms_voi_noise)      
    
    #------------------------------------------------------------------------------------------------------------------------------
    
    # ap mask:
    v_voi_mask =  np.clip(v_voi, 0, 1)

    # target sp from mgc:
    m_sp_targ = la.mcep_to_sp(m_spmgc, nFFT)
    
    # medfilt:
    if b_medfilt:
        m_sp_targ = signal.medfilt(m_sp_targ, kernel_size=[3,1])        

    '''
    # Debug - Minimum phase filter for ap signal (NO BORRAR!):
    m_sp_comp_mph  = la.sp_to_min_phase(m_sp_targ, in_type='sp')
    m_sp_ph_mph    = np.angle(m_sp_comp_mph)    
    m_noise_ph     = m_noise_ph + m_sp_ph_mph
    m_voi_noise_ph = m_voi_noise_ph + m_sp_ph_mph
    '''
    
    # Alloc:    
    m_frm_syn = np.zeros((nfrms, nFFT))
    m_mag_syn = np.zeros((nfrms, nFFThalf)) # just for debug
    m_mag     = np.zeros((nfrms, nFFThalf)) # just for debug
   
    # Spectral crossfade constants (TODO: Improve this):
    muf = 3500 # "minimum unvoiced freq."
    bw = (mvf - muf) - 20 # values found empirically. assuming mvf > 4000
    cut_off = (mvf + muf) / 2
    v_zeros = np.zeros((1,nFFThalf))         
    
    # Iterates through frames:
    for i in xrange(nfrms):        
        

        if v_voi_mask[i] == 1: # voiced case            
            # Magnitude: 
            v_mag_log = m_voi_noise_mag_log[i,:]                                 
            v_mag_log = la.spectral_crossfade(v_zeros, v_mag_log[None,:], cut_off, bw, fs, freq_scale='hz')[0]        

            # Phase:   
            #v_ph = la.spectral_crossfade(m_ph_deter[None, i,:], m_noise_ph[None,i,:], cut_off, bw, fs, freq_scale='hz')[0]        
            v_ph = la.spectral_crossfade(m_ph_deter[None, i,:], m_voi_noise_ph[None,i,:], cut_off, bw, fs, freq_scale='hz')[0]
            
        elif v_voi_mask[i] == 0: # unvoiced case
            # Magnitude:
            v_mag_log = m_noise_mag_log[i,:]       

            # Phase:
            v_ph = m_noise_ph[i,:]      
            
        # To complex:
        m_mag[i,:] = np.exp(v_mag_log) # just for debug
        v_mag = np.exp(v_mag_log) * m_sp_targ[i,:]
        v_sp  = v_mag * np.exp(v_ph * 1j) 
        v_sp  = la.add_hermitian_half(v_sp[None,:], data_type='complex')        
        
        # Save:
        #print(i)
        m_mag_syn[i,:] = v_mag # for inspection    
        m_frm_syn[i,:] = np.fft.fftshift(np.fft.ifft(v_sp).real)     
        
    v_sig_syn = la.ola(m_frm_syn, v_pm)
     
    return v_sig_syn, m_frm_syn, m_mag_syn, m_sp_targ, m_frm_noise, m_frm_voi_noise, m_mag
 
    
    
#==============================================================================
# v2: Improved phase generation. 
# If ph_hf_gen=='rand', generates random numbers for the phase above mvf
# If ph_hf_gen=='template_mask', uses a phase template to fill the gaps given by the aperiodic mask.
# If ph_hf_gen=='rand_mask' The same as above, but it uses random numbers instead of a template.
# The aperiodic mask is computed (estimated) according to the total phase energy per frame.
# v_voi: Used to construct the ap mask:
# if v_voi[n] > 0, frame is voiced. If v_voi[n] == 0, frame is unvoiced. 
# If v_voy=='estim', the mask is estimated from phase data.
def synthesis_with_del_comp_and_ph_encoding2(m_spmgc, m_phs_mgc, m_phc_mgc, v_shift, nFFT, fs, mvf, v_voi, win_func=np.hanning):
    
    # Ph and MVF:
    mvf_bin     = lu.round_to_int(mvf * nFFT / np.float(fs))
    nFFThalf_ph = la.next_pow_of_two(mvf_bin) + 1      

    # MGC to Ph up to MVF:
    m_phs_shrt_intrp_syn = la.mcep_to_sp(m_phs_mgc, 2*(nFFThalf_ph-1), out_type=0)
    m_phc_shrt_intrp_syn = la.mcep_to_sp(m_phc_mgc, 2*(nFFThalf_ph-1), out_type=0)
    f_interps_syn        = interpolate.interp1d(np.arange(nFFThalf_ph), m_phs_shrt_intrp_syn, kind='cubic')
    f_interpc_syn        = interpolate.interp1d(np.arange(nFFThalf_ph), m_phc_shrt_intrp_syn, kind='cubic')
    m_phs_shrt_syn       = f_interps_syn(np.linspace(0,nFFThalf_ph-1,mvf_bin))
    m_phc_shrt_syn       = f_interpc_syn(np.linspace(0,nFFThalf_ph-1,mvf_bin))
        
    # Deterministic Phase decoding:----------------------
    nfrms    = np.size(m_phs_shrt_syn,0)
    nFFThalf = nFFT / 2 + 1
    m_phs_shrt_syn = np.clip(m_phs_shrt_syn, -1, 1)  
    m_phc_shrt_syn = np.clip(m_phc_shrt_syn, -1, 1)  
    m_ph_deter     = ph_dec(m_phs_shrt_syn, m_phc_shrt_syn, mode='angle') 
    m_ph_deter     = np.hstack((m_ph_deter, np.zeros((nfrms,nFFThalf-mvf_bin))))
    
    # Estimating aperiodicity mask:-----------------------
    #m_ph_ap_mask = get_ap_mask_from_uv_decision(v_voi, nFFT, fs, mvf)

    # TD Noise Gen:---------------------------------------    
    v_pm    = la.shift_to_pm(v_shift)
    sig_len = v_pm[-1] + (v_pm[-1] - v_pm[-2]) 
    v_noise = np.random.uniform(-1, 1, sig_len)    
    #v_noise = np.random.normal(size=sig_len)
    
    # Extract noise magnitude and phase:    
    l_frm_noise, v_lens, v_pm_plus, v_shift_dummy, v_rights = windowing(v_noise, v_pm, win_func=win_func)    
    m_frm_noise = la.frm_list_to_matrix(l_frm_noise, v_shift, nFFT)
    m_frm_noise = np.fft.fftshift(m_frm_noise, axes=1)
    m_noise_sp  = la.remove_hermitian_half(np.fft.fft(m_frm_noise))
    m_noise_ph  = np.angle(m_noise_sp)
    
    m_noise_mag     = np.absolute(m_noise_sp)
    m_noise_mag_log = np.log(m_noise_mag)
    
    # Debug:
    '''
    ga2 = np.fft.fftshift(m_frm_noise, axes=1)
    nx = 114
    holdon()
    plot(m_frm_noise[nx,:], '-b')
    plot(ga2[nx,:], '-r')    
    holdoff()
    '''
    
    # ap mask:
    v_voi_mask =  np.clip(v_voi, 0, 1)

    # target sp from mgc:
    m_sp_targ = la.mcep_to_sp(m_spmgc, nFFT)
    
    # Debug:
    #v_voi_mask[:] = 0
    # m_noise_ph = gen_rand_phase_by_template('../database/ph_template_1.npy',nfrms, nFFThalf)

    # Minimum phase filter for ap signal:
    #m_sp_targ = np.tile(m_sp_targ[30,:], (nfrms,1))
    m_sp_comp_mph = la.sp_to_min_phase(m_sp_targ, in_type='sp')
    m_sp_ph_mph   = np.angle(m_sp_comp_mph)
    m_noise_ph    = m_noise_ph + m_sp_ph_mph
    
    
    # Alloc:    
    m_frm_syn = np.zeros((nfrms, nFFT))
    m_mag_syn = np.zeros((nfrms, nFFThalf)) # just for debug
    
    # Noise amp-normalisation:
    '''    
    mag_ave = np.mean(m_noise_mag_log)
    m_noise_mag_log -= mag_ave 
    '''
    rms_noise = np.sqrt(np.mean(m_noise_mag**2))
    m_noise_mag_log = m_noise_mag_log - np.log(rms_noise)  
   
    # Spectral crossfade constants (TODO: Improve this):
    muf = 3500 # "minimum unvoiced freq."
    bw = (mvf - muf) - 20 # values found empirically. assuming mvf > 4000
    cut_off = (mvf + muf) / 2
    v_zeros = np.zeros((1,nFFThalf))         
    
    # Iterates through frames:
    for i in xrange(nfrms):        
        v_mag_log = m_noise_mag_log[i,:]

        if v_voi_mask[i] == 1: # voiced case            
            # Magnitude:                                    
            #v_mag_log[:mvf_bin] = 0
            v_mag_log = la.spectral_crossfade(v_zeros, v_mag_log[None,:], cut_off, bw, fs, freq_scale='hz')[0]        

            # Phase:
            #v_ph = np.hstack((m_ph_deter[i,:], m_noise_ph[i,mvf_bin:]))       
            v_ph = la.spectral_crossfade(m_ph_deter[None, i,:], m_noise_ph[None,i,:], cut_off, bw, fs, freq_scale='hz')[0]        
            
        elif v_voi_mask[i] == 0: # unvoiced case
            # Phase:
            v_ph = m_noise_ph[i,:]
      
            
        # To complex:          
        v_mag = np.exp(v_mag_log) * m_sp_targ[i,:]
        #Debug:
        #v_mag = np.exp(v_mag_log) 
        #v_mag = m_sp_targ[114,:]

        v_sp  = v_mag * np.exp(v_ph * 1j) 
        v_sp  = la.add_hermitian_half(v_sp[None,:], data_type='complex')        
        
        # Save:
        print(i)
        m_mag_syn[i,:] = v_mag # for inspection    
        m_frm_syn[i,:] = np.fft.fftshift(np.fft.ifft(v_sp).real)       
        

    v_sig_syn = la.ola(m_frm_syn, v_pm)
    # la.write_audio_file('hola.wav', v_sig, fs)
    
    return v_sig_syn, m_frm_syn, m_mag_syn, m_sp_targ, m_frm_noise

#==============================================================================
    
    
#==============================================================================
# If ph_hf_gen=='rand', generates random numbers for the phase above mvf
# If ph_hf_gen=='template_mask', uses a phase template to fill the gaps given by the aperiodic mask.
# If ph_hf_gen=='rand_mask' The same as above, but it uses random numbers instead of a template.
# The aperiodic mask is computed (estimated) according to the total phase energy per frame.
# v_voi: Used to construct the ap mask:
# if v_voi[n] > 0, frame is voiced. If v_voi[n] == 0, frame is unvoiced. 
# If v_voy=='estim', the mask is estimated from phase data.
def synthesis_with_del_comp_and_ph_encoding(m_spmgc, m_phs_mgc, m_phc_mgc, v_shift, nFFT, fs, mvf, ph_hf_gen="rand", v_voi='estim', win_func=np.hanning, win_flat_to_len=0.3):
    
    # MGC to SP:
    m_sp_syn = la.mcep_to_sp(m_spmgc, nFFT)
    
    # Ph and MVF:
    mvf_bin     = lu.round_to_int(mvf * nFFT / np.float(fs))
    nFFThalf_ph = la.next_pow_of_two(mvf_bin) + 1      
    
    # MGC to Ph up to MVF:
    m_phs_shrt_intrp_syn = la.mcep_to_sp(m_phs_mgc, 2*(nFFThalf_ph-1), out_type=0)
    m_phc_shrt_intrp_syn = la.mcep_to_sp(m_phc_mgc, 2*(nFFThalf_ph-1), out_type=0)
    f_interps_syn        = interpolate.interp1d(np.arange(nFFThalf_ph), m_phs_shrt_intrp_syn, kind='cubic')
    f_interpc_syn        = interpolate.interp1d(np.arange(nFFThalf_ph), m_phc_shrt_intrp_syn, kind='cubic')
    m_phs_shrt_syn       = f_interps_syn(np.linspace(0,nFFThalf_ph-1,mvf_bin))
    m_phc_shrt_syn       = f_interpc_syn(np.linspace(0,nFFThalf_ph-1,mvf_bin))
    
    # Generate phase up to Nyquist:
    nfrms    = np.size(m_phs_shrt_syn,0)
    nFFThalf = nFFT / 2 + 1
    m_phs_shrt_syn = np.clip(m_phs_shrt_syn, -1, 1)  
    m_phc_shrt_syn = np.clip(m_phc_shrt_syn, -1, 1)   
    
    if ph_hf_gen is 'rand':   
        m_phs_syn  = np.hstack((m_phs_shrt_syn, np.random.uniform(-1, 1, size=(nfrms,nFFThalf-mvf_bin))))
        m_phc_syn  = np.hstack((m_phc_shrt_syn, np.random.uniform(-1, 1, size=(nfrms,nFFThalf-mvf_bin))))
        
        # Phase decoding:
        m_ph_syn = ph_dec(m_phs_syn, m_phc_syn) 

    elif ph_hf_gen is 'template_mask' or 'rand_mask':
        
        # Deterministic Phase decoding:----------------------
        m_ph_deter     = ph_dec(m_phs_shrt_syn, m_phc_shrt_syn, mode='angle') 
        m_ph_deter     = np.hstack((m_ph_deter, np.zeros((nfrms,nFFThalf-mvf_bin))))
        
        # Estimating aperiodicity mask:-----------------------
        if v_voi is 'estim':
            m_ph_ap_mask = estim_ap_mask_from_ph_data(m_phs_shrt_syn, nFFT, fs, mvf)
            
        elif type(v_voi) is np.ndarray: 
            
            # Debug:
            #v_voi[:] = 0
            
            m_ph_ap_mask = get_ap_mask_from_uv_decision(v_voi, nFFT, fs, mvf)
        
        # Gen aperiodic phase:--------------------------------
        if ph_hf_gen is 'template_mask':   
            m_ap_ph = gen_rand_phase_by_template('../database/ph_template_1.npy',nfrms, nFFThalf)
    
        elif ph_hf_gen is 'rand_mask':       
            m_ap_ph = np.random.uniform(-np.pi, np.pi, size=(nfrms,nFFThalf))

            
        # Mix:
        m_ph_syn = m_ap_ph * m_ph_ap_mask + m_ph_deter * (1 - m_ph_ap_mask)       

    # Final Synthesis:
    v_syn_sig = synthesis_with_del_comp(m_sp_syn, m_ph_syn, v_shift, win_func=win_func, win_flat_to_len=win_flat_to_len)       
    
    # Debug:
    #v_syn_sig = synthesis_with_del_comp_2(m_sp_syn, m_ph_syn, m_ph_ap_mask, v_shift) 
    
    return v_syn_sig

#==============================================================================

def get_ap_mask_from_uv_decision(v_voi, nFFT, fs, mvf, fade_len=40):

    # Body:-------------------------------------    
    v_ph_ap_mask =  1 - np.clip(v_voi, 0, 1)
    mvf_bin      = lu.round_to_int(mvf * nFFT / np.float(fs)) 
    m_ph_ap_mask = np.tile(v_ph_ap_mask[:,None],[1,mvf_bin])
    
    # Smoothing of the mask arounf mvf:    
    v_ramp = np.linspace(1,0,fade_len)
    m_ph_ap_mask = 1 - m_ph_ap_mask
    m_ph_ap_mask[:,-fade_len:] = m_ph_ap_mask[:,-fade_len:] * v_ramp
    m_ph_ap_mask = 1 - m_ph_ap_mask
    
    nfrms    = len(v_voi)
    nFFThalf = nFFT / 2 + 1    
    m_ph_ap_mask = np.hstack((m_ph_ap_mask, np.ones((nfrms,nFFThalf-mvf_bin))))
    
    return m_ph_ap_mask


#==============================================================================
def estim_ap_mask_from_ph_data(m_mask_ref, nFFT, fs, mvf):        
    # Constants:    
    clip_range = [-28.1 , -10.3]
    fade_len   = 40
    
    # Body:-------------------------------------------------
    v_mask_ref = la.db(np.sqrt(np.mean(m_mask_ref**2,1)))
    
    v_ph_ap_mask = -np.clip(v_mask_ref, clip_range[0], clip_range[1])        
    v_ph_ap_mask = (v_ph_ap_mask + clip_range[1]) / float(clip_range[1] - clip_range[0])  
    
    # Phase mask in 3D:
    
    mvf_bin = lu.round_to_int(mvf * nFFT / np.float(fs))     
    m_ph_ap_mask = np.tile(v_ph_ap_mask[:,None],[1,mvf_bin])
    
    # Smoothing of the mask arounf mvf:
    
    v_ramp = np.linspace(1,0,fade_len)
    m_ph_ap_mask = 1 - m_ph_ap_mask
    m_ph_ap_mask[:,-fade_len:] = m_ph_ap_mask[:,-fade_len:] * v_ramp
    m_ph_ap_mask = 1 - m_ph_ap_mask
    
    nFFThalf = nFFT / 2 + 1
    nfrms = np.size(m_mask_ref,0)
    m_ph_ap_mask = np.hstack((m_ph_ap_mask, np.ones((nfrms,nFFThalf-mvf_bin))))

    return m_ph_ap_mask


#==============================================================================
# Transform data from picth sync to constant rate in provided in ms.
def to_constant_rate(m_data, targ_shift_ms, v_shift, fs, interp_kind='linear'):
    
    v_in_cntr_nxs    = np.cumsum(v_shift)
    in_est_sig_len   = v_in_cntr_nxs[-1] + v_shift[-1] # Instead of using sig_len, it could be estimated like this    
    targ_shift_smpls = targ_shift_ms / 1000.0 * fs   
    v_targ_cntr_nxs  = np.arange(targ_shift_smpls, in_est_sig_len, targ_shift_smpls) # checkear que el codigo DNN indexe los frames asi tamnbien!
    v_targ_cntr_nxs  = v_targ_cntr_nxs.astype(int)   
        
    # Interpolation:     
    f_interp = interpolate.interp1d(v_in_cntr_nxs, m_data, axis=0, fill_value='extrapolate', kind=interp_kind)            
    m_data   = f_interp(v_targ_cntr_nxs)  
    
    return m_data
'''
def to_pitch_sync(m_data, shift_ms, v_shift_cons, fs, interp_kind='linear'):
    
    nInfrms     = np.size(m_data,0)
    shift_smpls = shift_ms / 1000.0 * fs 
    est_sig_len = nInfrms * shift_smpls

    v_in_cntr_nxs   = np.arange(shift_smpls, est_sig_len, shift_smpls)
    v_targ_cntr_nxs = v_targ_cntr_nxs.astype(int)     

    
    v_in_cntr_nxs    = np.cumsum(v_shift_5ms)
    in_est_sig_len   = v_in_cntr_nxs[-1] + v_shift[-1] # Instead of using sig_len, it could be estimated like this    
        
    # Interpolation:     
    f_interp = interpolate.interp1d(v_in_cntr_nxs, m_data, axis=0, fill_value='extrapolate', kind=interp_kind)            
    m_data   = f_interp(v_targ_cntr_nxs)  
    
    return
'''   

# NOT FINISHED!!!! 
def const_shifts_to_pitch_sync(v_const_lefts, shift_ms, fs, interp_kind='linear'):   
    
    nConstFrms = len(v_const_lefts)
    
    shift_smpls = shift_ms / 1000.0 * fs 
    v_const_cntr_nxs = np.arange(1, nConstFrms+1) * shift_smpls  
    
    f_interp = interpolate.interp1d(v_const_cntr_nxs, v_const_lefts, axis=0, fill_value='extrapolate', kind=interp_kind)            
    #m_data   = f_interp(v_targ_cntr_nxs)     
    
    v_shift = np.zeros(nConstFrms * 2) # Twice should be enough, although maybe not, check!!!        
    for con_left in v_const_lefts:
        g=1
    
    return
   
#==============================================================================
# v2: allows fine frame state position (adds relative position within the state as decimal number).
# shift file in samples
def frame_to_state_mapping2(shift_file, state_lab_file, fs, states_per_phone=5, b_refine=True):
    #Read files:
    v_shift = lu.read_binfile(shift_file, dim=1)
    v_pm = la.shift_to_pm(v_shift)
    m_state_times = np.loadtxt(state_lab_file, usecols=(0,1))    
    
    # to miliseconds:
    v_pm_ms = 1000 * v_pm / fs
    m_state_times_ms = m_state_times / 10000.0    
    
    # Compare:
    nfrms = len(v_pm_ms)
    v_st = np.zeros(nfrms) - 1 # init
    for f in xrange(nfrms):
        vb_greater = (v_pm_ms[f] >= m_state_times_ms[:,0])  # * (v_pm_ms[f] <  m_state_times_ms[:,1])
        state_nx   = np.where(vb_greater)[0][-1]
        v_st[f]    = np.remainder(state_nx, states_per_phone)

        # Refining:
        if b_refine:
            state_len_ms = m_state_times_ms[state_nx,1] - m_state_times_ms[state_nx,0]
            fine_pos = ( v_pm_ms[f] - m_state_times_ms[state_nx,0] ) / state_len_ms
            v_st[f] += fine_pos 
            
    # Protection against wrong ended label files:
    np.clip(v_st, 0, states_per_phone, out=v_st)      
            
    return v_st
    
#==============================================================================

def frame_to_state_mapping(shift_file, lab_file, fs, states_per_phone=5):
    #Read files:
    v_shift = lu.read_binfile(shift_file, dim=1)
    v_pm = la.shift_to_pm(v_shift)
    m_state_times = np.loadtxt(lab_file, usecols=(0,1))    
    
    # to miliseconds:
    v_pm_ms = 1000 * v_pm / fs
    m_state_times_ms = m_state_times / 10000.0    
    
    # Compare:
    nfrms = len(v_pm_ms)
    v_st = np.zeros(nfrms) - 1 # init
    for f in xrange(nfrms):
        vb_greater = (v_pm_ms[f] >= m_state_times_ms[:,0])  # * (v_pm_ms[f] <  m_state_times_ms[:,1])
        state_nx   = np.where(vb_greater)[0][-1]
        v_st[f]    = np.remainder(state_nx, states_per_phone)
    return v_st
    
#==============================================================================
def get_n_frms_per_unit(v_shifts, in_lab_state_al_file, fs, unit_type='phone', n_sts_x_ph=5):
    raise ValueError('Deprecated. Use "get_num_of_frms_per_phon_unit", instead')
    return
    
#==============================================================================
# in_lab_aligned_file: in HTS format
# n_lines_x_unit: e.g., number of states per phoneme. (each state in one line)
# TODO: Change name of variables. e.g, states -> lines
# v_shift in samples.
# nfrms_tolerance: Maximum number of frames of difference between shifts and lab file allowed (Some times, the end of lab files is not acurately defined).
def get_num_of_frms_per_phon_unit(v_shift, in_lab_aligned_file, fs, n_lines_x_unit=5, nfrms_tolerance=1):   

    # Read lab file:
    m_labs_state = np.loadtxt(in_lab_aligned_file, usecols=(0,1))
    m_labs_state_ms = m_labs_state / 10000.0
    
    # Epoch Indexes:
    v_ep_nxs = np.cumsum(v_shift)
    v_ep_nxs_ms = v_ep_nxs * 1000.0 / fs
    
    # Get number of frames per state:
    n_states        = np.size(m_labs_state_ms,axis=0)
    v_nfrms_x_state = np.zeros(n_states)    
    
    for st in xrange(n_states):
        vb_to_right = (m_labs_state_ms[st,0] <= v_ep_nxs_ms)
        vb_to_left  = (v_ep_nxs_ms < m_labs_state_ms[st,1])        
        vb_inter    = vb_to_right * vb_to_left        
        v_nfrms_x_state[st] = sum(vb_inter) 
        
    # Correct if there is only one frame of difference:  
    nfrms_diff = np.size(v_shift) - np.sum(v_nfrms_x_state)     
    if (nfrms_diff > 0) and (nfrms_diff <= nfrms_tolerance):
        v_nfrms_x_state[-1] += nfrms_diff 
        
    # Checking number of frames:  
    if np.sum(v_nfrms_x_state) != np.size(v_shift):
        raise ValueError('Total number of frames is different to the number of frames of the shifts.')

    m_nfrms_x_ph  = np.reshape(v_nfrms_x_state, (n_states/n_lines_x_unit, n_lines_x_unit) )
    v_nfrms_x_ph  = np.sum(m_nfrms_x_ph, axis=1)
            
    # Checking that the number of frames per phoneme should be greater than 0:
    if any(v_nfrms_x_ph == 0.0):
        raise ValueError('There is some phoneme(s) that do(es) not contain any frame.') 
        
    return v_nfrms_x_ph

'''
def get_num_of_frms_per_phon_unit(v_shift, in_lab_aligned_file, fs, n_lines_x_unit=5):   

    # Read lab file:
    m_labs_state = np.loadtxt(in_lab_aligned_file, usecols=(0,1))
    m_labs_state_ms = m_labs_state / 10000.0
    
    # Epoch Indexes:
    v_ep_nxs = np.cumsum(v_shift)
    v_ep_nxs_ms = v_ep_nxs * 1000.0 / fs
    
    # Get number of frames per state:
    n_states        = np.size(m_labs_state_ms,axis=0)
    v_nfrms_x_state = np.zeros(n_states)    
    
    for st in xrange(n_states):
        vb_to_right = (m_labs_state_ms[st,0] <= v_ep_nxs_ms)
        vb_to_left  = (v_ep_nxs_ms < m_labs_state_ms[st,1])        
        vb_inter    = vb_to_right * vb_to_left        
        v_nfrms_x_state[st] = sum(vb_inter) 
        
    # Correct if there is only one frame of difference:  
    if (np.sum(v_nfrms_x_state) + 1) == np.size(v_shift):
        v_nfrms_x_state[-1] += 1        
        
    # Checking number of frames:  
    if np.sum(v_nfrms_x_state) != np.size(v_shift):
        raise ValueError('Total number of frames is different to the number of frames of the shifts.')

    m_nfrms_x_ph  = np.reshape(v_nfrms_x_state, (n_states/n_lines_x_unit, n_lines_x_unit) )
    v_nfrms_x_ph  = np.sum(m_nfrms_x_ph, axis=1)
            
    # Checking that the number of frames per phoneme should be greater than 0:
    if any(v_nfrms_x_ph == 0.0):
        raise ValueError('There is some phoneme(s) that do(es) not contain any frame.') 
        
    return v_nfrms_x_ph
'''
#==============================================================================    
def gen_rand_phase_by_template(tmplt_file, nfrms, nBins):
    
    # Read template:
    m_ph_tmplt = np.load(tmplt_file)   
    
    # Mirror phase (check!):   
    m_ph_tile = np.vstack((m_ph_tmplt, m_ph_tmplt))    
    m_ph_tile = np.hstack((m_ph_tile, m_ph_tile))
    #m_ph_mirror = np.hstack((m_ph_mirror, np.fliplr(m_ph_mirror))) # keep as a comment

    # Size of mirrored phase:
    nfrmsT, nBinsT = m_ph_tile.shape
    
    # Tile phase:
    times_frms = int(np.ceil(nfrms / float(nfrmsT)))
    times_bins = int(np.ceil(nBins / float(nBinsT)))
    
    m_ph_tile = np.tile(m_ph_tile,[times_frms, times_bins])
    m_ph_tile = m_ph_tile[:nfrms, :nBins]
    
    return m_ph_tile
    

def main_phase_template():    
    #==============================================================================
    # INPUT
    #==============================================================================

    in_file = os.getenv("HOME") + "/Dropbox/Education/UoE/Projects/DirectFFTWaveformModelling/data/prue_nat/herald_182.wav"
    mvf = 4500   
    nFFT = 2048 #128 #2048
    shift_ms = 5
    #==============================================================================
    # BODY
    #==============================================================================
         
    v_in_sig, fs = sf.read(in_file) 
    m_sp, m_ph, v_shift = analysis_with_del_comp(v_in_sig, nFFT, fs)
    m_ph_tmplate = m_ph[:50,50:820] # values found empirically
    
    # Write template:
    tmplt_file = '/home/s1373426/Dropbox/Education/UoE/Projects/DirectFFTWaveformModelling/database/ph_template_prue.npy'
    np.save(tmplt_file, m_ph_tmplate)    
    
    return

def main_pitch_marks_analysis():
    fs = 48000
    v_in_sig = sf.read('/afs/inf.ed.ac.uk/group/cstr/projects/Felipe_Espic/Databases/Nick-Zhizheng_dnn_baseline_practice/data/wav/hvd_649.wav')[0]
    v_pm = la.get_pitch_marks(v_in_sig, fs)
    #v_pm_ms = v_pm * 1000
    v_pm_smpls = v_pm * fs
    
    holdon()
    plot(v_in_sig, '-b')
    stem(v_pm_smpls, np.ones(len(v_pm_smpls)), '-r')
    holdoff()
    
    plot(v_pm_smpls[1:] , np.diff(v_pm_smpls), '.-r')
    return
'''
def pm_to_shift(v_pm):
    v_shift = np.diff(np.hstack((0,v_pm)))
    return v_shift
    
def shift_to_pm(v_shift):
    v_pm = np.cumsum(v_shift)
    return v_pm
'''    
#==============================================================================
# out: 'f0' or 'lf0'
def shift_to_f0(v_shift, v_voi, fs, out='f0', b_filt=True):
    v_f0 = v_voi * fs / v_shift.astype('float64')
    if b_filt:
        v_f0 = v_voi * signal.medfilt(v_f0)

    if out == 'lf0':
        v_f0 = la.f0_to_lf0(v_f0)
     
    return v_f0
    
#==============================================================================
    
def f0_to_shift(v_f0_in, fs, unv_frm_rate_ms=5.0):
    v_f0 = v_f0_in.copy()
    v_f0[v_f0 == 0] = 1000.0 / unv_frm_rate_ms    
    v_shift = fs / v_f0
    
    return v_shift     
    
#============================================================================== 
def interp_from_variable_to_const_frm_rate(m_data, v_pm_smpls, const_rate_ms, fs, interp_type='linear') :
    
    dur_total_smpls  = v_pm_smpls[-1]
    const_rate_smpls = fs * const_rate_ms / 1000
    #cons_frm_rate_frm_len = 2 * frm_rate_smpls # This assummed according to the Merlin code. E.g., frame_number = int((end_time - start_time)/50000)
    v_c_rate_centrs_smpls = np.arange(const_rate_smpls, dur_total_smpls, const_rate_smpls) 
  
    # Interpolation m_spmgc:         
    f_interp = interpolate.interp1d(v_pm_smpls, m_data, axis=0, kind=interp_type)  
    m_data_const_rate = f_interp(v_c_rate_centrs_smpls) 

    return m_data_const_rate

#==============================================================================
def interp_from_const_to_variable_rate(m_data, v_frm_locs_smpls, frm_rate_ms, fs, interp_type='linear'):

    n_c_rate_frms  = np.size(m_data,0)                
    frm_rate_smpls = fs * frm_rate_ms / 1000
    
    v_c_rate_centrs_smpls = frm_rate_smpls * np.arange(1,n_c_rate_frms+1)

    f_interp     = interpolate.interp1d(v_c_rate_centrs_smpls, m_data, axis=0, kind=interp_type)            
    m_data_intrp = f_interp(v_frm_locs_smpls) 
    
    return m_data_intrp 

#==============================================================================
# NOTE: "v_frm_locs_smpls" are the locations of the target frames (centres) in the constant rate data to sample from.
# This function should be used along with the function "interp_from_const_to_variable_rate"
def get_shifts_and_frm_locs_from_const_shifts(v_shift_c_rate, frm_rate_ms, fs, interp_type='linear'):
        
    # Interpolation in reverse:
    n_c_rate_frms      = np.size(v_shift_c_rate,0)                
    frm_rate_smpls     = fs * frm_rate_ms / 1000
    
    v_c_rate_centrs_smpls = frm_rate_smpls * np.arange(1,n_c_rate_frms+1)
    f_interp   = interpolate.interp1d(v_c_rate_centrs_smpls, v_shift_c_rate, axis=0, kind=interp_type)            
    v_shift_vr = np.zeros(n_c_rate_frms * 2) # * 2 just in case, Improve these!
    v_frm_locs_smpls = np.zeros(n_c_rate_frms * 2)
    curr_pos_smpl = v_c_rate_centrs_smpls[-1]
    for i_vr in xrange(len(v_shift_vr)-1,0, -1):        
        #print(i_vr)
        v_frm_locs_smpls[i_vr] = curr_pos_smpl            
        try:                
            v_shift_vr[i_vr] = f_interp(curr_pos_smpl)
        except ValueError:
            v_frm_locs_smpls = v_frm_locs_smpls[i_vr+1:]                
            v_shift_vr  = v_shift_vr[i_vr+1:]
            break
        
        curr_pos_smpl = curr_pos_smpl - v_shift_vr[i_vr]
        
    return v_shift_vr, v_frm_locs_smpls
   
    
#==============================================================================
# MAIN (for Dev)
#==============================================================================

if __name__ == '__main__':      
    import libaudio as la
#==============================================================================
# INPUT
#==============================================================================
    mvf  = 4500   
    nFFT = 4096 #128 #2048    
    fs = 48000
    
    
    
    data_dir = '/afs/inf.ed.ac.uk/group/cstr/projects/Felipe_Espic/Projects/DirectFFTWaveModelling/dnn/expers_nick/nick_fft_feats_new_expr_2_SLSTM_5_layers_state_feat_ref_60_45_unv_zero_learn_rate_0_002/gen/DNN_TANH_TANH_TANH_TANH_SLSTM_LINEAR__mag_lf0_real_imag_vuv_0_2400_597_454_5_1024_512'
    filename = 'hvd_622'
    
    #data_dir = '/afs/inf.ed.ac.uk/group/cstr/projects/Felipe_Espic/Projects/DirectFFTWaveModelling/dnn/expers_laura/laura_fft_feats_new_expr_2_SLSTM_5_layers_state_feat_ref_60_45_unv_zero_learn_rate_0_002/gen/DNN_TANH_TANH_TANH_TANH_SLSTM_LINEAR__mag_lf0_real_imag_vuv_0_4500_486_454_5_1024_512'
    #filename = '1106_1' #'3589_1'#'3087_1'#'3589_1'
    
    out_dir = '/home/s1373426/Dropbox/Education/UoE/Projects/DirectFFTWaveformModelling/data/out_dir_prue3'
    
    #filename = 'hvd_618'
    suffix = '_prue_new2'
    
    hf_slope_coeff = 1.8 # def: 1.0
    b_use_ap_voi = True # def: True
    b_voi_ap_win = True # def: True

#==============================================================================
# BODY
#==============================================================================
    
    m_mag_mel_log = la.read_mgc(data_dir + '/' + filename + '.mag_pf', 60)
    m_real_mel    = la.read_mgc(data_dir + '/' + filename + '.real', 45)
    m_imag_mel    = la.read_mgc(data_dir + '/' + filename + '.imag', 45)
    v_f0          = la.read_f0(data_dir  + '/' + filename + '.lf0', kind='lf0')
    
    
    # Synth:
    v_syn_sig = synthesis_with_del_comp_and_ph_encoding5(m_mag_mel_log, m_real_mel, m_imag_mel, v_f0, nFFT, fs, mvf, f0_type='f0', hf_slope_coeff=hf_slope_coeff, b_use_ap_voi=b_use_ap_voi, b_voi_ap_win=b_voi_ap_win)
    
    
    play(v_syn_sig, fs)
    
    
    
    la.write_audio_file(out_dir + '/' + filename + suffix + '.wav', v_syn_sig, fs)
    
   
    dummy=1
  
