import os
from scipy.stats import mode
from scipy import signal
import os.path
import libaudio as la
import libutils as lu
import numpy as np
import fft_feats as ff
#from plot_funcs import pl
#pl.close('all')
# v4: Simple synth  just concatenation in time domain. (NOTE!!: frame "duplicated" in joins!!)
# v5: Simple synth with fft_feats. Simple concatenation of fft_feats. (NOTE!!: frame "duplicated" in joins!!)
# v6: Concatenation using slope adjustment of fft_feats. Also, new style of passing data: everything in just one dictionary. (NOTE!!: frame "duplicated" in joins!!)
# v7: Concatenation by mixing of fft_feats. (ongoing)
#-------------------------------------------------------------------------
def parse_utt_file(utt_file, pm_dir):
    l_data = []
    diphnx = 0
    with open(utt_file) as file_id:
        for line in file_id:
            l_line = line.split()
            if (len(l_line) >= 8) and (l_line[7] == 'ph1'):
                diphone    = l_line[5]
                audio_file = l_line[23]
                nframes    = int(l_line[42])
                middle_frm = int(l_line[20])
                source_end_sec_orig = float(l_line[30])

                # Read pm file:
                pm_file  = pm_dir  + '/' + audio_file + '.pm'
                v_pm_sec = np.loadtxt(pm_file, skiprows=9, usecols=[0])

                # Get position and length in source:---------------------
                mid_point_sec = source_end_sec_orig
                nx_mid_pm = np.argmin(np.abs((mid_point_sec - v_pm_sec)))
                source_strt_sec = v_pm_sec[nx_mid_pm - middle_frm]
                source_end_sec  = v_pm_sec[nx_mid_pm - middle_frm + nframes]
                source_ph_bound_sec = v_pm_sec[nx_mid_pm]

                l_data.append([ diphone, audio_file, source_strt_sec, source_end_sec, source_ph_bound_sec, diphnx])
                diphnx += 1

    # Packing:
    return l_data


#-------------------------------------------------------------------------
def parse_utt_src_file(utt_src_file):

    d_data = {}
    #d_data['diphone'] = []
    #d_data['srcfile'] = []
    #d_data['src_strt_sec'] = np.array([])
    #d_data['src_end_sec']  = np.array([])
    #d_data['src_bnd_sec']  = np.array([])


    m_file_data = np.loadtxt(utt_src_file, comments=None, usecols=(0,1,2,3), dtype='string')
    '''
    l_src_file_list = m_file_data[:,0].tolist()
    v_src_strt_sec  = m_file_data[:,1]
    v_src_end_sec   = m_file_data[:,2]
    l_ph_list       = m_file_data[:,3].tolist()
    '''

    d_data['srcfile']      = m_file_data[:,0].tolist()
    d_data['src_strt_sec'] = m_file_data[:,1].astype('float64')
    d_data['src_end_sec']  = m_file_data[:,2].astype('float64')
    d_data['diphone']      = m_file_data[:,3].tolist()

    # Packing:
    return d_data


#-------------------------------------------------------------------------
def parse_utt_file2(utt_file, pm_dir):
    #l_data = []
    #diphnx = 0

    d_data = {}
    d_data['diphone'] = []
    d_data['srcfile'] = []
    d_data['src_strt_sec'] = np.array([])
    d_data['src_end_sec']  = np.array([])
    d_data['src_bnd_sec']  = np.array([])
    d_data['src_end_sec']  = np.array([])


    with open(utt_file) as file_id:
        for line in file_id:
            l_line = line.split()
            if (len(l_line) >= 8) and (l_line[7] == 'ph1'):
                diphone    = l_line[5]
                wavfile = l_line[23]
                nframes    = int(l_line[42])
                middle_frm = int(l_line[20])
                source_end_sec_orig = float(l_line[30])

                # Read pm file:
                pm_file  = pm_dir  + '/' + wavfile + '.pm'
                v_pm_sec = np.loadtxt(pm_file, skiprows=9, usecols=[0])

                # Get position and length in source:---------------------
                mid_point_sec = source_end_sec_orig
                nx_mid_pm = np.argmin(np.abs((mid_point_sec - v_pm_sec)))
                source_strt_sec = v_pm_sec[nx_mid_pm - middle_frm]
                #source_end_sec  = v_pm_sec[nx_mid_pm - middle_frm + nframes]
                if (nx_mid_pm - middle_frm + nframes)>=len(v_pm_sec):
                    source_end_sec  = v_pm_sec[-1]
                else:
                    source_end_sec  = v_pm_sec[nx_mid_pm - middle_frm + nframes]
                source_ph_bound_sec = v_pm_sec[nx_mid_pm]

                # Storing:
                d_data['diphone'].append(diphone)
                d_data['srcfile'].append(wavfile)
                d_data['src_strt_sec'] = np.append(d_data['src_strt_sec'], source_strt_sec)
                d_data['src_bnd_sec']  = np.append(d_data['src_bnd_sec'] , source_ph_bound_sec)
                d_data['src_end_sec']  = np.append(d_data['src_end_sec'] , source_end_sec)


                #l_data.append([ diphone, wavfile, source_strt_sec, source_end_sec, source_ph_bound_sec, diphnx])
                #diphnx += 1

    # Packing:
    return d_data

#------------------------------------------------------------------------------
def extract_segments(l_utt_data, wav_dir, pm_dir):
    '''
    pm_dir: It could be the original pm or another extracted by e.g., Reaper
    '''

    # Constants:-------------------------
    npm_margin = 3

    # Body:------------------------------
    l_segs_data = []
    ndiph = len(l_utt_data)
    for i  in xrange(ndiph):

        # Read source file:
        wav_file = wav_dir + '/' + l_utt_data[i][1] + '.wav'
        pm_file  = pm_dir  + '/' + l_utt_data[i][1] + '.pm'

        v_src_sig, fs = la.read_audio_file(wav_file)
        v_pm_sec      = la.read_est_file(pm_file)[:,0]

        # Get original position and length in source:---------------------
        src_strt_orig_sec = l_utt_data[i][2]
        src_end_orig_sec  = l_utt_data[i][3]

        # Get start and end pm indexes:
        src_pm_strt_nx = np.argmin(np.abs(v_pm_sec - src_strt_orig_sec))
        src_pm_end_nx  = np.argmin(np.abs(v_pm_sec - src_end_orig_sec))

        # Expanding boundaries:
        src_pm_strt_nx_w_marg = src_pm_strt_nx - npm_margin
        src_pm_end_nx_w_marg  = src_pm_end_nx  + npm_margin

        # Extract signal:
        v_pm_smps     = lu.round_to_int(v_pm_sec * fs)
        #v_sig_seg2 = v_src_sig[v_pm_smps[src_pm_strt_nx_w_marg-1]:(v_pm_smps[src_pm_end_nx_w_marg+1]+1)]
        #v_pm_smps_seg2 = v_pm_smps[]
        v_pm_smps_seg = v_pm_smps[(src_pm_strt_nx_w_marg-1):(src_pm_end_nx_w_marg+2)] # the added ones are because it is needed one extra pm as boundaries.
        v_sig_seg     = v_src_sig[v_pm_smps_seg[0]:(v_pm_smps_seg[-1]+1)]
        v_pm_smps_seg = v_pm_smps_seg - v_pm_smps_seg[0]
        v_pm_smps_seg = v_pm_smps_seg[1:-1] # cutting first and last value, since the windowing function will add them.

        l_segs_data.append([v_sig_seg, v_pm_smps_seg])

    return l_segs_data


#------------------------------------------------------------------------------
def extract_segments2(l_utt_data, wav_dir, pm_reaper_dir, npm_margin):
    '''
    pm_reaper_dir: They must be the pm's extracted by REAPER.
    '''
    # Body:------------------------------
    l_segs_data = []
    ndiph = len(l_utt_data)
    nfrms = 0
    for i  in xrange(ndiph):

        # Read source file:
        wav_file = wav_dir + '/' + l_utt_data[i][1] + '.wav'
        pm_file  = pm_reaper_dir  + '/' + l_utt_data[i][1] + '.pm'

        v_src_sig, fs   = la.read_audio_file(wav_file)
        v_pm_sec, v_voi = la.read_reaper_est_file(pm_file)

        # Get original position and length in source:---------------------
        src_strt_orig_sec   = l_utt_data[i][2]
        src_end_orig_sec    = l_utt_data[i][3]
        src_ph_bnd_orig_sec = l_utt_data[i][4]

        # Get start and end pm indexes:
        src_pm_strt_nx   = np.argmin(np.abs(v_pm_sec - src_strt_orig_sec))
        src_pm_end_nx    = np.argmin(np.abs(v_pm_sec - src_end_orig_sec))
        src_pm_ph_bnd_nx = np.argmin(np.abs(v_pm_sec - src_ph_bnd_orig_sec))

        # Expanding boundaries:
        src_pm_strt_nx_w_marg = src_pm_strt_nx - npm_margin
        src_pm_end_nx_w_marg  = src_pm_end_nx  + npm_margin

        # Extract signal:
        v_pm_smps = lu.round_to_int(v_pm_sec * fs)

        v_pm_smps_seg = v_pm_smps[(src_pm_strt_nx_w_marg-1):(src_pm_end_nx_w_marg+2)] # the added ones are because it is needed one extra pm as boundaries.
        v_voi_seg     = v_voi[(src_pm_strt_nx_w_marg):(src_pm_end_nx_w_marg+1)]
        v_sig_seg     = v_src_sig[v_pm_smps_seg[0]:(v_pm_smps_seg[-1]+1)]
        v_pm_smps_seg = v_pm_smps_seg - v_pm_smps_seg[0]
        v_pm_smps_seg = v_pm_smps_seg[1:-1] # cutting first and last value, since the windowing function will add them.

        # Protection:
        len_diff = len(v_voi_seg) - len(v_pm_smps_seg)
        #print('len_diff: %d' % (len_diff))
        if len_diff > 0:
            v_voi_seg = v_voi_seg[:-len_diff]

        src_pm_ph_bnd_nx_seg = src_pm_ph_bnd_nx - src_pm_strt_nx_w_marg

        nfrms += len(v_pm_smps_seg)
        l_segs_data.append([v_sig_seg, v_pm_smps_seg, v_voi_seg, src_pm_ph_bnd_nx_seg])

    d_data = { 'nfrms': nfrms, 'segs': l_segs_data}
    return d_data


#----------------------------------------------------------------------------------

def extract_segments3(d_utt_data, wav_dir, pm_reaper_dir, npm_margin):
    '''
    pm_reaper_dir: They must be the pm's extracted by REAPER.
    '''
    d_utt_data['v_sig']      = []
    d_utt_data['v_pm_smpls'] = []
    d_utt_data['v_voi']      = []
    #d_utt_data['pm_bnd_nx']  = np.array([]).astype('int64')

    # Body:------------------------------
    #l_segs_data = []
    #ndiph = len(d_utt_data['diphone']) # commented this CVB
    ndiph = len(d_utt_data['srcfile'])
    nfrms = 0
    for i  in xrange(ndiph):

        # Read source file:
        wav_file = wav_dir       + '/' + d_utt_data['srcfile'][i] + '.wav'
        pm_file  = pm_reaper_dir + '/' + d_utt_data['srcfile'][i] + '.pm'

        v_src_sig, fs   = la.read_audio_file(wav_file)
        v_pm_sec, v_voi = la.read_reaper_est_file(pm_file)

        # Get original position and length in source:---------------------
        src_strt_orig_sec   = d_utt_data['src_strt_sec'][i]
        #src_ph_bnd_orig_sec = d_utt_data['src_bnd_sec' ][i]
        src_end_orig_sec    = d_utt_data['src_end_sec' ][i]

        # Get start and end pm indexes:
        src_pm_strt_nx   = np.argmin(np.abs(v_pm_sec - src_strt_orig_sec))
        src_pm_end_nx    = np.argmin(np.abs(v_pm_sec - src_end_orig_sec))
        #src_pm_ph_bnd_nx = np.argmin(np.abs(v_pm_sec - src_ph_bnd_orig_sec))

        # Expanding boundaries:
        src_pm_strt_nx_w_marg = src_pm_strt_nx - npm_margin
        src_pm_end_nx_w_marg  = src_pm_end_nx  + npm_margin

        # Expanding beginning:======================================================
        #src_pm_strt_nx_w_marg = src_pm_strt_nx - npm_margin
        if i==0:
            src_pm_strt_nx_w_marg = src_pm_strt_nx
        # Protection: (the ones are for protection for waveform extraction)
        if (src_pm_strt_nx_w_marg - 1)<0:
            src_pm_strt_nx_w_marg = 1

        # Expanding end:--------------------------------------------------
        #src_pm_end_nx_w_marg = src_pm_end_nx  + npm_margin
        if i==(ndiph-1):
            src_pm_end_nx_w_marg = src_pm_end_nx
        # Protection: (the ones are for protection for waveform extraction)
        nfrms_src = len(v_pm_sec)
        if (src_pm_end_nx_w_marg+1)>(nfrms_src-1):
            src_pm_end_nx_w_marg = nfrms_src - 2
        # END ==============================================================
        # Extract signal:
        v_pm_smps = lu.round_to_int(v_pm_sec * fs)

        v_pm_smps_seg = v_pm_smps[(src_pm_strt_nx_w_marg-1):(src_pm_end_nx_w_marg+2)] # the added ones are because it is needed one extra pm as boundaries.
        v_voi_seg     = v_voi[(src_pm_strt_nx_w_marg):(src_pm_end_nx_w_marg+1)]
        #Protection:
        if v_voi_seg.size==0:
            v_voi_seg = np.array([0.0])

        v_sig_seg     = v_src_sig[v_pm_smps_seg[0]:(v_pm_smps_seg[-1]+1)]
        v_pm_smps_seg = v_pm_smps_seg - v_pm_smps_seg[0]

        # Protection:
        if len(v_pm_smps_seg)<=2:
            v_pm_smps_seg = v_pm_smps_seg[-1]
        else:
            v_pm_smps_seg = v_pm_smps_seg[1:-1] # cutting first and last value, since the windowing function will add them.

        # Protection:
        len_diff = v_voi_seg.size - v_pm_smps_seg.size
        if len_diff > 0:
            v_voi_seg = v_voi_seg[:-len_diff]

        #src_pm_ph_bnd_nx_seg = src_pm_ph_bnd_nx - src_pm_strt_nx_w_marg

        nfrms += v_pm_smps_seg.size

        #if v_pm_smps_seg.size>0:
        d_utt_data['v_sig'     ].append(v_sig_seg)
        d_utt_data['v_pm_smpls'].append(v_pm_smps_seg)
        d_utt_data['v_voi'     ].append(v_voi_seg)
        #d_utt_data['pm_bnd_nx' ] = np.append(d_utt_data['pm_bnd_nx'], src_pm_ph_bnd_nx_seg)

    d_utt_data['nfrms_tot'] = nfrms

    return d_utt_data



#-----------------------------------------------------------------------------------------

def analysis_fft_feats(d_segs_data, nfft):
    # Input:
    l_segs_data = d_segs_data['segs']
    nfrms_tot   = d_segs_data['nfrms']

    # Alloc:
    nffthalf    = 1 + nfft/2
    m_mag_tot   = np.zeros((nfrms_tot, nffthalf))
    m_real_tot  = np.zeros((nfrms_tot, nffthalf))
    m_imag_tot  = np.zeros((nfrms_tot, nffthalf))
    v_shift_tot = np.zeros(nfrms_tot, dtype='int64')
    v_voi_tot   = np.zeros(nfrms_tot, dtype='int64')
    #v_bnd_segs  = np.zeros(nfrms_tot, dtype='int64') # marker for boundaries between segments

    nsegs = len(l_segs_data)
    v_start_segs = np.zeros(nsegs, dtype='int64')
    curr_nx = 0
    for i in xrange(nsegs):
        # Debug:
        #print('i: %d / %d' % (i, nsegs))

        v_sig     = l_segs_data[i][0]
        v_pm_smps = l_segs_data[i][1]
        v_voi     = l_segs_data[i][2]

        # Analyis:
        m_sp, m_ph, v_shift, m_frms, m_fft, v_lens = ff.analysis_with_del_comp_from_pm(v_sig, v_pm_smps, nfft)
        m_mag, m_real, m_imag = ff.get_fft_params_from_complex_data(m_fft)

        # Save:
        curr_nfrms = len(v_shift)
        m_mag_tot[curr_nx:(curr_nx+curr_nfrms),:]  = m_mag
        m_real_tot[curr_nx:(curr_nx+curr_nfrms),:] = m_real
        m_imag_tot[curr_nx:(curr_nx+curr_nfrms),:] = m_imag
        v_shift_tot[curr_nx:(curr_nx+curr_nfrms)]  = v_shift
        v_voi_tot[curr_nx:(curr_nx+curr_nfrms)]    = v_voi
        #v_bnd_segs[curr_nx] = 1
        #v_bnd_segs[curr_nx+curr_nfrms-1] = 1
        v_start_segs[i] = curr_nx
        curr_nx += curr_nfrms

    # Fixing first and last sample:
    #v_bnd_segs[0]  = 0
    #v_bnd_segs[-1] = 0
    return m_mag_tot, m_real_tot, m_imag_tot, v_shift_tot, v_voi_tot, v_start_segs


def analysis_fft_feats2(d_utt_data, nfft):
    # Input:
    nfrms_tot   = d_utt_data['nfrms_tot']

    # Alloc:
    nffthalf    = 1 + nfft/2
    m_mag_tot   = np.zeros((nfrms_tot, nffthalf))
    m_real_tot  = np.zeros((nfrms_tot, nffthalf))
    m_imag_tot  = np.zeros((nfrms_tot, nffthalf))
    v_shift_tot = np.zeros(nfrms_tot, dtype='int64')
    v_voi_tot   = np.zeros(nfrms_tot, dtype='int64')
    #v_bnd_segs  = np.zeros(nfrms_tot, dtype='int64') # marker for boundaries between segments

    #nsegs = len(d_utt_data['diphone'])# commented this CVB
    nsegs = len(d_utt_data['srcfile'])
    v_start_segs = np.zeros(nsegs, dtype='int64')
    curr_nx = 0
    for i in xrange(nsegs):
        # Debug:
        #print('i: %d / %d' % (i, nsegs))
        v_sig     = d_utt_data['v_sig'     ][i]
        v_pm_smps = d_utt_data['v_pm_smpls'][i]
        v_voi     = d_utt_data['v_voi'     ][i]
        #print(i)
        #if i==23: import ipdb; ipdb.set_trace()  # breakpoint b47d1713 //

        # Analyis:
        m_sp, m_ph, v_shift, m_frms, m_fft, v_lens = ff.analysis_with_del_comp_from_pm(v_sig, v_pm_smps, nfft)
        m_mag, m_real, m_imag = ff.get_fft_params_from_complex_data(m_fft)

        # Save:
        curr_nfrms = len(v_shift)
        m_mag_tot[curr_nx:(curr_nx+curr_nfrms),:]  = m_mag
        m_real_tot[curr_nx:(curr_nx+curr_nfrms),:] = m_real
        m_imag_tot[curr_nx:(curr_nx+curr_nfrms),:] = m_imag
        v_shift_tot[curr_nx:(curr_nx+curr_nfrms)]  = v_shift
        v_voi_tot[curr_nx:(curr_nx+curr_nfrms)]    = v_voi
        v_start_segs[i] = curr_nx
        curr_nx += curr_nfrms

    return m_mag_tot, m_real_tot, m_imag_tot, v_shift_tot, v_voi_tot, v_start_segs

def synthesis_fft_feats(m_mag, m_real, m_imag, v_shift):

    m_ph_cmpx = (m_real + m_imag * 1j)
    m_fft     = m_mag * m_ph_cmpx / np.absolute(m_ph_cmpx)
    m_fft     = la.add_hermitian_half(m_fft, data_type='complex')
    m_frm     = np.fft.ifft(m_fft).real
    m_frm     = np.fft.fftshift(m_frm,  axes=1)
    v_pm      = la.shift_to_pm(v_shift)

    v_syn_sig = la.ola(m_frm,v_pm)
    return v_syn_sig

def synthesis_fft_feats2(m_mag, m_real, m_imag, v_shift, v_start_segs=None):
    '''
    If v_start_segs provided, the last frame of each segment is removed
    '''
    if any(np.atleast_1d(v_start_segs)):
        v_end_segs = v_start_segs[1:] - 1
        m_mag  = np.delete(m_mag,  v_end_segs, axis=0)
        m_real = np.delete(m_real, v_end_segs, axis=0)
        m_imag = np.delete(m_imag, v_end_segs, axis=0)
        v_shift = np.delete(v_shift, v_end_segs)

    m_ph_cmpx = (m_real + m_imag * 1j)
    m_fft     = m_mag * m_ph_cmpx / np.absolute(m_ph_cmpx)
    m_fft     = la.add_hermitian_half(m_fft, data_type='complex')
    m_frm     = np.fft.ifft(m_fft).real
    m_frm     = np.fft.fftshift(m_frm,  axes=1)
    v_pm      = la.shift_to_pm(v_shift)

    v_syn_sig = la.ola(m_frm,v_pm)
    return v_syn_sig


def synthesis_fft_feats_from_fft_matrix(m_fft, v_shift):
    '''
    '''
    m_fft     = la.add_hermitian_half(m_fft, data_type='complex')
    m_frm     = np.fft.ifft(m_fft).real
    m_frm     = np.fft.fftshift(m_frm,  axes=1)
    v_pm      = la.shift_to_pm(v_shift)

    v_syn_sig = la.ola(m_frm,v_pm)
    return v_syn_sig

def concat_fft_feats_simple(l_fft_feats, npm_margin):
    nsegs = len(l_fft_feats)
    nfft  = l_fft_feats[0]['m_mag'].shape[1]

    # Concat frames from different segments:
    m_mag   = np.array([]).reshape((0,nfft))
    m_real  = np.array([]).reshape((0,nfft))
    m_imag  = np.array([]).reshape((0,nfft))
    v_shift = np.array([], dtype='int64')
    for i in xrange(nsegs):
        d_data = l_fft_feats[i]
        m_mag   = np.append(m_mag,   d_data['m_mag'][npm_margin:-(npm_margin+1),:] , axis=0)
        m_real  = np.append(m_real,  d_data['m_real'][npm_margin:-(npm_margin+1),:], axis=0)
        m_imag  = np.append(m_imag,  d_data['m_imag'][npm_margin:-(npm_margin+1),:], axis=0)
        v_shift = np.append(v_shift, d_data['v_shift'][npm_margin:-(npm_margin+1)] , axis=0)

    return m_mag, m_real, m_imag, v_shift

def smooth_by_slope_to_average_mat(m_data, v_start_segs, in_type=None):
    m_data = m_data.copy()
    #v_join_locs = v_start_segs[1:]
    #njoins = len(v_join_locs)
    #for jnx in xrange(njoins):
    nbins = m_data.shape[1]
    for nxb in xrange(nbins):
        m_data[:,nxb] = smooth_by_slope_to_average(m_data[:,nxb], v_start_segs, in_type=in_type)
    return m_data

def smooth_by_slope_to_average(v_in_data, v_start_segs, in_type=None):
    '''
    in_type: if 'f0' : it protects for unvoiced segments.
    '''

    max_val = 450
    min_val = 20


    v_data = v_in_data.copy()
    if in_type is 'f0':
        v_voi  = (v_data > 0).astype(int)
    nsegs  = len(v_start_segs)

    v_start_segs = np.append(v_start_segs, len(v_data)) # For protection

    for nxj in xrange(1,nsegs):

        strt = v_start_segs[nxj]
        # If values are equal, do nothing
        if v_data[strt] == v_data[strt-1]:
            continue

        if (in_type is 'f0') and ( (v_voi[strt]==0) or (v_voi[strt-1]==0) ):
            continue

        # Adjustment:
        data_mean = (v_data[strt] + v_data[strt-1]) / 2.0

        # TODO: Make more elegant and efficient

        # From the left:
        diff_w_mean_l = data_mean - v_data[strt-1]
        v_line_l  = np.linspace(0, diff_w_mean_l, strt - v_start_segs[nxj-1])
        v_data[v_start_segs[nxj-1]:strt] += v_line_l

        # From the right:
        diff_w_mean_r = data_mean - v_data[strt]
        v_line_r  = np.linspace(diff_w_mean_r, 0, v_start_segs[nxj+1] - strt)
        v_data[strt:v_start_segs[nxj+1]] += v_line_r

        # Protection:
        if any(v_data[strt:v_start_segs[nxj+1]]<min_val) or any(v_data[strt:v_start_segs[nxj+1]]>max_val):
            v_data[strt:v_start_segs[nxj+1]] = v_in_data[strt:v_start_segs[nxj+1]]


    if in_type is 'f0':
        v_data *= v_voi
    return v_data


#----------------------------------------------------------------------------------------
def get_rms_energy_norm(m_mag, v_shift):
    m_mag  = la.add_hermitian_half(m_mag)
    v_ener = np.sqrt(np.sum(m_mag**2, axis=1))
    v_lens = v_shift[:-1] + v_shift[1:] + 1
    v_lens = np.append(v_lens, 2 * v_shift[-1] + 1)
    v_ener_norm = v_ener / v_lens
    return v_ener_norm

def mod_rms_energy(m_mag, v_new_energy):
    m_mag  = la.add_hermitian_half(m_mag)
    m_mag  = np.multiply(m_mag, v_new_energy[:,None])
    m_mag  = la.remove_hermitian_half(m_mag)
    return m_mag

def print_dict(d_data, width=22):

    def format_column(col_data, width):
        data_type = type(col_data)
        l_column  = []

        # Case numpy array:----------------------
        if data_type is np.ndarray:
            v_data = np.atleast_1d(col_data)
            nlines = len(v_data)

            # Protection for long vectors:
            if v_data.ndim > 1:
                return None

            for nxl in xrange(nlines):
                l_column.append(("{:<%d}" % width).format(v_data[nxl]))

        # Case list of strings:-------------------
        if data_type is list:
            l_data = col_data

            # Protection agains no strings:
            if type(l_data[0]) is not str:
                return None

            nlines = len(l_data)
            for nxl in xrange(nlines):
                l_column.append(("{:%d.%d}" % (width, width-2)).format(l_data[nxl]))
        return l_column

    #BODY:------------------------------------------------------
    l_keys   = d_data.keys()
    ncols    = len(l_keys)
    ll_table = []
    for nxc in xrange(ncols):
        l_curr_col = format_column(d_data[l_keys[nxc]], width)
        if (l_curr_col is None) or (len(l_curr_col) == 0):
            continue

        s_header = ("{:%d.%d}" % (width, width-2)).format(l_keys[nxc])
        l_curr_col.insert(0,s_header)
        l_curr_col.insert(1,'=' * width)
        ll_table.append(l_curr_col)

    # Concatenate strings:
    nlines     = len(ll_table[0])
    ncols_defi = len(ll_table)
    for nxl in xrange(nlines):
        curr_line = ''
        for nxc in xrange(ncols_defi):
            curr_line += ll_table[nxc][nxl]
        print(curr_line)
    return

def analysis_fft_feats3(d_utt_data, nfft, fs):
    '''
    v3: output appended to d_utt_data
    '''

    d_utt_data['m_mag']    = []
    d_utt_data['m_real']   = []
    d_utt_data['m_imag']   = []
    d_utt_data['v_shift']  = []
    d_utt_data['v_f0']     = []
    d_utt_data['m_mag_db'] = []
    d_utt_data['m_fft']    = []

    # Input:
    #nfrms_tot   = d_utt_data['nfrms_tot']

    # Alloc:
    #nffthalf    = 1 + nfft/2
    #nsegs = len(d_utt_data['diphone'])# commented this CVB
    nsegs = len(d_utt_data['srcfile'])
    v_strt_segs = np.zeros(nsegs, dtype='int64')
    curr_nx = 0
    for i in xrange(nsegs):
        # Debug:
        #print('i: %d / %d' % (i, nsegs))
        v_sig     = d_utt_data['v_sig'     ][i]
        v_pm_smps = d_utt_data['v_pm_smpls'][i]
        v_voi     = d_utt_data['v_voi'     ][i]

        # Analyis:
        m_sp, m_ph, v_shift, m_frms, m_fft, v_lens = ff.analysis_with_del_comp_from_pm(v_sig, v_pm_smps, nfft)
        m_mag, m_real, m_imag = ff.get_fft_params_from_complex_data(m_fft)
        v_f0 = ff.shift_to_f0(v_shift, v_voi, fs, b_filt=False)

        # Save:
        curr_nfrms = len(v_shift)
        d_utt_data['m_mag'   ].append(m_mag)
        d_utt_data['m_real'  ].append(m_mag)
        d_utt_data['m_imag'  ].append(m_imag)
        d_utt_data['v_shift' ].append(v_shift)
        d_utt_data['v_f0'    ].append(v_f0)
        d_utt_data['m_mag_db'].append(la.db(m_mag))
        d_utt_data['m_fft'   ].append(m_fft) # It is delay compensated

        v_strt_segs[i] = curr_nx
        curr_nx += curr_nfrms

    d_utt_data['v_strt_segs'] = v_strt_segs

    return d_utt_data

#----------------------------------------------------------------------------------

def extract_segments4(d_utt_data, wav_dir, pm_reaper_dir, npm_margin):
    '''
    pm_reaper_dir: They must be the pm's extracted by REAPER.
    Tries to add npm_margin for the joins.
    '''
    d_utt_data['v_sig']      = []
    d_utt_data['v_pm_smpls'] = []
    d_utt_data['v_voi']      = []
    #d_utt_data['pm_bnd_nx']  = np.array([]).astype('int64')

    # Debug:
    d_utt_data['nfrms_add_strt'] = np.array([]).astype('int64')
    d_utt_data['nfrms_add_end']  = np.array([]).astype('int64')

    # Body:------------------------------
    #ndiph = len(d_utt_data['diphone']) # commented this CVB
    ndiph = len(d_utt_data['srcfile'])
    nfrms_tot = 0
    for i  in xrange(ndiph):
        #print(i)

        # Read source file:
        wav_file = wav_dir       + '/' + d_utt_data['srcfile'][i] + '.wav'
        pm_file  = pm_reaper_dir + '/' + d_utt_data['srcfile'][i] + '.pm'

        v_src_sig, fs   = la.read_audio_file(wav_file)
        v_pm_sec, v_voi = la.read_reaper_est_file(pm_file)

        # Get original position and length in source:---------------------
        src_strt_orig_sec   = d_utt_data['src_strt_sec'][i]
        #src_ph_bnd_orig_sec = d_utt_data['src_bnd_sec' ][i]
        src_end_orig_sec    = d_utt_data['src_end_sec' ][i]

        # Get start and end pm indexes:
        src_pm_strt_nx   = np.argmin(np.abs(v_pm_sec - src_strt_orig_sec))
        src_pm_end_nx    = np.argmin(np.abs(v_pm_sec - src_end_orig_sec))
        #src_pm_ph_bnd_nx = np.argmin(np.abs(v_pm_sec - src_ph_bnd_orig_sec))

        # Expanding beginning:--------------------------------------------
        src_pm_strt_nx_w_marg = src_pm_strt_nx - npm_margin
        if i==0:
            src_pm_strt_nx_w_marg = src_pm_strt_nx
        # Protection: (the ones are for protection for waveform extraction)
        if (src_pm_strt_nx_w_marg - 1)<0:
            src_pm_strt_nx_w_marg = 1

        # Expanding end:--------------------------------------------------
        src_pm_end_nx_w_marg = src_pm_end_nx  + npm_margin
        if i==(ndiph-1):
            src_pm_end_nx_w_marg = src_pm_end_nx
        # Protection: (the ones are for protection for waveform extraction)
        nfrms_src = len(v_pm_sec)
        if (src_pm_end_nx_w_marg+1)>(nfrms_src-1):
            src_pm_end_nx_w_marg = nfrms_src - 2

        # Getting v_pm_smpls and v_voi:
        v_voi_seg       = v_voi[src_pm_strt_nx_w_marg:(src_pm_end_nx_w_marg+1)]

        v_pm_smpls      = lu.round_to_int(v_pm_sec * fs)
        v_pm_smpls_seg  = v_pm_smpls[(src_pm_strt_nx_w_marg-1):(src_pm_end_nx_w_marg+1)]
        v_pm_smpls_seg  = v_pm_smpls_seg - v_pm_smpls_seg[0]
        v_pm_smpls_seg  = v_pm_smpls_seg[1:]

        # Extracting waveform:
        nx_strt_wav = v_pm_smpls[src_pm_strt_nx_w_marg-1]
        nx_end_wav  = v_pm_smpls[src_pm_end_nx_w_marg +1]
        v_sig_seg   = v_src_sig[nx_strt_wav:(nx_end_wav+1)]

        # Phone boundary (middle point):
        #src_pm_ph_bnd_nx_seg = src_pm_ph_bnd_nx - src_pm_strt_nx_w_marg

        # Frames added defi:
        # nfrms_add_strt = src_pm_strt_nx - src_pm_strt_nx_w_marg # TODO: Maybe use maximum as well.
        nfrms_add_strt = np.maximum(0,src_pm_strt_nx - src_pm_strt_nx_w_marg) # CVB
        nfrms_add_end  = np.maximum(src_pm_end_nx_w_marg - src_pm_end_nx,0) # protection against negative values


        # Storing:
        d_utt_data['v_sig'     ].append(v_sig_seg)
        d_utt_data['v_pm_smpls'].append(v_pm_smpls_seg)
        d_utt_data['v_voi'     ].append(v_voi_seg)
        #d_utt_data['pm_bnd_nx' ] = np.append(d_utt_data['pm_bnd_nx'], src_pm_ph_bnd_nx_seg)

        d_utt_data['nfrms_add_strt'] = np.append(d_utt_data['nfrms_add_strt'], nfrms_add_strt)
        d_utt_data['nfrms_add_end']  = np.append(d_utt_data['nfrms_add_end'] , nfrms_add_end)

        # Update:
        nfrms_tot += len(v_pm_smpls_seg)

    d_utt_data['nfrms_tot'] = nfrms_tot

    return d_utt_data

#----------------------------------------------------------------------------------------
def crossfade_gen(fade_len):
    len_tot = (fade_len + 1) * 2 + 1
    v_win   = np.hanning(len_tot)

    v_fade_in  = v_win[1:(fade_len+1)]
    v_fade_out = np.flipud(v_fade_in)

    return v_fade_in, v_fade_out


def fade_gen(fade_len, ftype):
    '''
    ftype: 'fade_in' or 'fade_out'
    '''
    if fade_len < 1:
        return None

    len_tot = (fade_len + 1) * 2 + 1
    v_win   = np.hanning(len_tot)

    v_fade  = v_win[1:(fade_len+1)]
    if ftype is 'fade_out':
        v_fade = np.flipud(v_fade)

    return v_fade

def smooth_by_mixing_params(str_data, d_utt_data):
    '''
    #in_type: if 'f0' : it protects for unvoiced segments.
    '''
    # Init:
    lm_data = d_utt_data[str_data]
    v_strt_segs = d_utt_data['v_strt_segs']

    # Dim protection:
    b_dim_protect = False
    if lm_data[0].ndim == 1:
        b_dim_protect = True

    # Alloc output data:
    ncols = 1
    if not b_dim_protect:
        ncols = lm_data[0].shape[1]

    in_dtype   = lm_data[0].dtype
    m_data_out = np.zeros((d_utt_data['nfrms_tot'], ncols)).astype(in_dtype)

    # First Segment:
    fade_len_strt   = 0 # init
    nx_frm_strt_out = 0 # init
    nsegs = len(v_strt_segs)
    for nxs in xrange(nsegs):

        # Starting and ending points of original seg (w/o added frames):
        nfrms   = len(d_utt_data['v_shift'][nxs])
        nx_strt = d_utt_data['nfrms_add_strt'][nxs]
        nx_end  = nfrms - d_utt_data['nfrms_add_end'][nxs] - 1
        if nxs==(nsegs-1):
            fade_len_end  = 0
        else:
            fade_len_end  = 1 + 2 * np.minimum(d_utt_data['nfrms_add_end'][nxs], d_utt_data['nfrms_add_strt'][nxs+1]) # TODO: protections!!!!!

        # Data:
        m_data = lm_data[nxs].copy()
        if b_dim_protect:
            m_data = m_data[:,None]

        # Debug:
        #m_data[:] = 1
        #print(nxs)

        # Protection:
        if fade_len_strt>=nfrms:
            return None

        # Applying fade-in:------------------------------------
        fade_len_strt_half = np.floor(fade_len_strt/2.0).astype('int64')
        nx_strt_fade = nx_strt - fade_len_strt_half
        if fade_len_strt > 0:
            v_fade_in = fade_gen(fade_len_strt, 'fade_in')
            #if nxs==73: import ipdb; ipdb.set_trace()  # breakpoint 1600b182 //
            m_data[nx_strt_fade:(nx_strt_fade+fade_len_strt),:] = np.multiply(m_data[nx_strt_fade:(nx_strt_fade+fade_len_strt),:], v_fade_in[:,None])


        # Applying fade-out:-----------------------------------
        fade_len_end_half = np.floor(fade_len_end/2.0).astype('int64')
        nx_end_fade = nx_end - fade_len_end_half
        # Protection:
        if nx_end_fade<0:
            return None # backup


        if fade_len_end > 0:
            v_fade_out = fade_gen(fade_len_end,  'fade_out')
            # Protection:
            #if nx_end_fade>=0:
            m_data[nx_end_fade:(nx_end_fade+fade_len_end),:] = np.multiply(m_data[nx_end_fade:(nx_end_fade+fade_len_end),:], v_fade_out[:,None])
            #else:
            #    m_data[0:(nx_end_fade+fade_len_end),:] = np.multiply(m_data[0:(nx_end_fade+fade_len_end),:], v_fade_out[(-nx_end_fade):,None])

        # Adding to out matrix:---------------------------------
        nfrms_defi =  (nx_end + fade_len_end_half) - nx_strt_fade + 1
        m_data_out[nx_frm_strt_out:(nx_frm_strt_out+nfrms_defi),:] = m_data_out[nx_frm_strt_out:(nx_frm_strt_out+nfrms_defi),:] + m_data[nx_strt_fade:(nx_end + fade_len_end_half+1),:]

        # Updating values:
        nx_frm_strt_out = nx_frm_strt_out + nx_end_fade
        fade_len_strt   = fade_len_end

    m_data_out = m_data_out[:(nx_frm_strt_out+1),:]

    # Dim protect:
    if b_dim_protect:
        m_data_out = np.squeeze(m_data_out)

    return m_data_out


def transpose_f0(v_f0, v_voi, v_strt_segs, nref=4, nfrms_btwn_voi=8):

    '''
    Number of frames for reference
    '''
    # Constants:
    max_f0 = 420 # 400 + some margin
    min_f0 = 48  # 50 minus some margin


    # Body:
    nsegs = len(v_strt_segs)
    v_f0_transp = v_f0.copy()
    curr_strt = 0
    for nxs in xrange(1,nsegs):

        prev_strt = curr_strt
        curr_strt = v_strt_segs[nxs]
        next_strt = None
        if nxs<(nsegs-1):
            next_strt = v_strt_segs[nxs+1]

        v_f0_curr = v_f0_transp[curr_strt:next_strt]
        v_f0_prev = v_f0_transp[prev_strt:curr_strt]


        # Protection if no voiced section exists:
        if all(v_f0_prev==0) or all(v_f0_curr==0):
            continue

        #print(nxs)
        # Protection against voiced sections further apart:
        curr_voi_strt = np.nonzero(v_f0_curr > 0)[0][0]  + curr_strt
        prev_voi_end = np.nonzero(v_f0_prev > 0)[0][-1] + prev_strt
        if (curr_voi_strt-prev_voi_end) >= nfrms_btwn_voi:
            continue

        #v_f0_prev_ref = np.mean(v_f0_prev[v_f0_prev > 0][-nref:])  # using the last n voiced values as reference (if possible)
        #v_f0_curr_ref = np.mean(v_f0_curr[v_f0_curr > 0][:nref])


        v_f0_prev_ref = v_f0_prev[v_f0_prev > 0][-(nref+1):-1]
        v_f0_curr_ref = v_f0_curr[v_f0_curr > 0][1:(nref+1)]

        if (len(v_f0_prev_ref) < nref) or (len(v_f0_curr_ref) < nref):
            continue

        #f0_prev_ref = np.mean(v_f0_prev_ref)  # using the last n voiced values as reference (if possible)
        #f0_curr_ref = np.mean(v_f0_curr_ref)   # It does not take the first and the last because they are misleading sometimes.

        f0_prev_ref = np.median(v_f0_prev_ref)  # using the last n voiced values as reference (if possible)
        f0_curr_ref = np.median(v_f0_curr_ref)   # It does not take the first and the last because they are misleading sometimes.

        # Protection:
        #if np.isnan(f0_prev_ref) or np.isnan(f0_curr_ref): # parece que no es necesaria esta proteccion
        #    continue

        # Apply transpose:
        if f0_curr_ref > f0_prev_ref:
            v_transp = v_f0_transp[curr_strt:next_strt] / np.round(f0_curr_ref / f0_prev_ref)
            if all(v_transp[1:]>min_f0):
                v_f0_transp[curr_strt:next_strt] = v_transp
        else:
            v_transp = v_f0_transp[curr_strt:next_strt] * np.round(f0_prev_ref / f0_curr_ref)
            if all(v_transp[1:]<max_f0):
                v_f0_transp[curr_strt:next_strt] = v_transp

    # Return to around original F0 (in case most of the f0 was scaled):
    #v_facts = v_f0[v_f0 > 0]  / v_f0_transp[v_f0_transp > 0]
    v_facts = v_f0_transp[v_f0_transp > 0] / v_f0[v_f0 > 0]

    fact_mode   = mode(v_facts).mode[0]
    v_f0_transp = v_f0_transp / fact_mode

    # DEBUG:
    '''
    from scipy import signal
    v_voi = (v_f0_transp > 0).astype('int64')
    v_f0_transp_mf = v_voi * signal.medfilt(v_f0_transp)
    pl.figure(2)
    v_bnd_segs = np.zeros(len(v_f0))
    v_bnd_segs[v_strt_segs] = 1
    pl.stem(300 * v_bnd_segs)
    pl.plot(v_f0, '.-g', label='v_f0')
    pl.plot(v_f0_transp, '.-r', label='v_f0_transp')
    pl.plot(v_f0_transp_mf, '.-k', label='v_f0_transp_mf')

    pl.grid()
    pl.legend()
    #'''

    # debug:
    #v_f0_transp[72:181] = v_f0_transp[72:181] * 2

    # Register applied transpose:
    v_facts_defi = np.ones(len(v_f0))
    v_facts_defi[v_f0 > 0] = v_f0_transp[v_f0_transp > 0] / v_f0[v_f0 > 0]



    return v_f0_transp, v_facts_defi



def wavgen_simple_concat_f0_slope(uttfile, db_dir, pm_reaper_dir, nfft, fs):

    # DIRS:===========================
    #wav_dir = db_dir + '/wav' # changed CVB
    wav_dir = db_dir # changed CVB
    pm_dir  = db_dir  + '/pm'
    #pm_dir  = pm_reaper_dir
    #pm_reaper_dir = db_dir  + '/pm_reaper'

    ## BODY:============================
    if type(uttfile) is str : # added CVB - to pass data directly
        d_utt_data_slope = parse_utt_file2(uttfile, pm_dir)
    else : # added CVB - to pass the data directly
        d_utt_data_slope = uttfile

    d_utt_data_slope = extract_segments3(d_utt_data_slope, wav_dir, pm_reaper_dir, 0)
    m_mag, m_real, m_imag, v_shift, v_voi, v_start_segs = analysis_fft_feats2(d_utt_data_slope, nfft)

    v_f0 = ff.shift_to_f0(v_shift, v_voi, fs, b_filt=False)
    v_f0_fix_slope = smooth_by_slope_to_average(v_f0, v_start_segs, in_type='f0')
    v_shift_fix_slope_syn = ff.f0_to_shift(v_f0_fix_slope, fs).astype('int64')


    v_syn_sig = synthesis_fft_feats2(m_mag, m_real, m_imag, v_shift_fix_slope_syn, v_start_segs)

    return v_syn_sig



def wavgen_simple_concat(uttfile, db_dir, pm_reaper_dir, nfft, fs):

    # DIRS:===========================
    wav_dir = db_dir + '/wav'
    pm_dir  = db_dir  + '/pm'
    #pm_dir  = pm_reaper_dir
    #pm_reaper_dir = db_dir  + '/pm_reaper'

    ## BODY:============================
    ext = os.path.splitext(uttfile)[1]
    if ext=='.utt':
        d_utt_data_slope = parse_utt_file2(uttfile, pm_dir)
    elif ext=='.source':
        d_utt_data_slope = parse_utt_src_file(uttfile)

    d_utt_data_slope = extract_segments3(d_utt_data_slope, wav_dir, pm_reaper_dir, 0)
    m_mag, m_real, m_imag, v_shift, v_voi, v_start_segs = analysis_fft_feats2(d_utt_data_slope, nfft)

    v_syn_sig = synthesis_fft_feats2(m_mag, m_real, m_imag, v_shift, v_start_segs)

    return v_syn_sig

#----------------------------------------------------------------------------------------------
def wavgen_improved(uttfile, db_dir, pm_reaper_dir, nfft, fs, npm_margin=3, diff_mf_tres=25, f0_trans_nfrms_btwn_voi=8):

    # BODY:===========================
    #wav_dir = db_dir + '/wav' # commented CVB
    wav_dir = db_dir # changed CVB
    pm_dir  = db_dir  + '/pm'
    #pm_dir  = pm_reaper_dir

    ##====================================================================
    ## Common Parsing:
    ##====================================================================

    d_utt_data_common = parse_utt_file2(uttfile, pm_dir)
    d_utt_data_mix    = d_utt_data_common.copy()
    d_utt_data_slope  = d_utt_data_common.copy()

    ##====================================================================
    ## Analysis MIXING:
    ##====================================================================

    d_utt_data_mix = extract_segments4(d_utt_data_mix, wav_dir, pm_reaper_dir, npm_margin)
    d_utt_data_mix = analysis_fft_feats3(d_utt_data_mix, nfft, fs)

    # Smoothing F0:========================================================

    # Extract F0:
    d_utt_data_slope = extract_segments3(d_utt_data_slope, wav_dir, pm_reaper_dir, 0)
    m_mag_dummy, m_real_dummy, m_imag_dummy, v_shift, v_voi, v_start_segs = analysis_fft_feats2(d_utt_data_slope, nfft)
    v_f0 = ff.shift_to_f0(v_shift, v_voi, fs, b_filt=False)

    # Transpose and slope adjustment:------------------------
    v_f0_transp, v_transp_facts = transpose_f0(v_f0, v_voi, v_start_segs, nfrms_btwn_voi=f0_trans_nfrms_btwn_voi)

    # Med filter:---------------------------------------------
    medfilt_size = 15
    v_f0_transp_mf = v_voi * signal.medfilt(v_f0_transp, kernel_size=medfilt_size)
    v_diff = np.abs(v_f0_transp_mf - v_f0_transp)
    v_f0_transp[v_diff > diff_mf_tres] = v_f0_transp_mf[v_diff > diff_mf_tres]

    # Smooth by slope:
    v_f0_fix_slope = smooth_by_slope_to_average(v_f0_transp, v_start_segs, in_type='f0')

    # DEBUG:----------------------------------------------------
    '''
    pl.figure(4)
    v_bnd_segs = np.zeros(len(v_f0))
    v_bnd_segs[v_start_segs] = 1
    pl.stem(300 * v_bnd_segs)
    pl.plot(v_f0, '.-g', label='v_f0')
    pl.plot(v_f0_transp, '*-c', label='v_f0_transp')
    pl.plot(v_f0_fix_slope, '.-r', label='v_f0_fix_slope')
    #pl.plot(v_f0_fix_slope_mf, '.-k', label='v_f0_fix_slope_mf')
    pl.grid()
    pl.legend()
    #'''

    # Smoothing Complex Spectrum:===============================================
    m_fft_fix    = smooth_by_mixing_params('m_fft', d_utt_data_mix)

    # Remove boundary frames in F0 and transpose factors:=================
    v_end_segs = v_start_segs[1:] - 1
    v_f0_fix_slope_syn = np.delete(v_f0_fix_slope, v_end_segs)
    v_transp_facts     = np.delete(v_transp_facts, v_end_segs)

    # Protection:
    nfrms_diff = m_fft_fix.shape[0] - len(v_f0_fix_slope_syn)
    if nfrms_diff>0:
        m_fft_fix = m_fft_fix[:-nfrms_diff,:]
    elif nfrms_diff<0:
        v_f0_fix_slope_syn = v_f0_fix_slope_syn[:nfrms_diff]
        v_transp_facts     = v_transp_facts[:nfrms_diff]




    # Removing frames if downsampled:========================================
    # Debug:
    #v_transp_facts[200:220] = 0.5
    v_nx_to_remove = np.nonzero(v_transp_facts < 1.0)[0]
    if v_nx_to_remove.size>0:
        v_diff = np.diff(v_nx_to_remove)
        v_nx_to_remove = np.delete(v_nx_to_remove, np.nonzero(v_diff > 1)[0]+1)
        v_nx_to_remove = v_nx_to_remove[1:] # removing the first frame
        nnx_to_remove = len(v_nx_to_remove)
        nx = 0
        v_nx_to_rem_defi = np.array([]).astype('int64')
        while nx<nnx_to_remove:
            curr_in_nx = v_nx_to_remove[nx]
            v_nx_to_rem_defi = np.append(v_nx_to_rem_defi, curr_in_nx)
            curr_jump = int(1 / v_transp_facts[curr_in_nx])
            nx += curr_jump

        # Delete frames:
        v_f0_fix_slope_syn = np.delete(v_f0_fix_slope_syn, v_nx_to_rem_defi)
        m_fft_fix          = np.delete(m_fft_fix, v_nx_to_rem_defi, axis=0)
        v_transp_facts     = np.delete(v_transp_facts, v_nx_to_rem_defi)
    # Add frames if upsampled:=================================================
    v_facts_up         = np.maximum(v_transp_facts, 1).astype('int64')
    v_f0_fix_slope_syn = np.repeat(v_f0_fix_slope_syn, v_facts_up)
    m_fft_fix          = np.repeat(m_fft_fix, v_facts_up, axis=0)

    # Convert to shifts:
    v_shift_fix_slope_syn = ff.f0_to_shift(v_f0_fix_slope_syn, fs).astype('int64')

    # Final Synthesis:
    v_syn_sig = synthesis_fft_feats_from_fft_matrix(m_fft_fix, v_shift_fix_slope_syn)

    return v_syn_sig

#----------------------------------------------------------------------------------------------
def wavgen_improved_just_slope(uttfile, db_dir, pm_reaper_dir, nfft, fs, npm_margin=3, diff_mf_tres=25, f0_trans_nfrms_btwn_voi=8):

    # BODY:===========================
    #wav_dir = db_dir + '/wav' # commented CVB
    wav_dir = db_dir # changed CVB
    pm_dir  = db_dir  + '/pm'
    #pm_dir  = pm_reaper_dir

    ##====================================================================
    ## Common Parsing:
    ##====================================================================
    if type(uttfile) is str : # added CVB - to pass data directly
        ext = os.path.splitext(uttfile)[1]
    else :
        ext = ''
        
    if ext=='.utt':
        d_utt_data_common = parse_utt_file2(uttfile, pm_dir)
    elif ext=='.source':
        d_utt_data_common = parse_utt_src_file(uttfile)
    else : # added CVB - to pass the data directly
        d_utt_data_common = uttfile

    d_utt_data_mix    = d_utt_data_common.copy()
    d_utt_data_slope  = d_utt_data_common.copy()

    ##====================================================================
    ## Analysis MIXING:
    ##====================================================================

    d_utt_data_mix = extract_segments4(d_utt_data_mix, wav_dir, pm_reaper_dir, npm_margin)
    d_utt_data_mix = analysis_fft_feats3(d_utt_data_mix, nfft, fs)

    # Smoothing F0:========================================================

    # Extract F0:
    d_utt_data_slope = extract_segments3(d_utt_data_slope, wav_dir, pm_reaper_dir, 0)
    m_mag_dummy, m_real_dummy, m_imag_dummy, v_shift, v_voi, v_start_segs = analysis_fft_feats2(d_utt_data_slope, nfft)
    v_f0 = ff.shift_to_f0(v_shift, v_voi, fs, b_filt=False)

    # Transpose and slope adjustment:------------------------
    #v_f0_transp, v_transp_facts = transpose_f0(v_f0, v_voi, v_start_segs, nfrms_btwn_voi=f0_trans_nfrms_btwn_voi)

    v_f0_fix_slope = smooth_by_slope_to_average(v_f0, v_start_segs, in_type='f0')
    #v_f0_transp = v_f0_fix_slope
    # Med filter:---------------------------------------------
    '''
    medfilt_size = 15
    #medfilt_size = 25
    v_f0_transp_mf = v_voi * signal.medfilt(v_f0_fix_slope, kernel_size=medfilt_size)
    v_diff = np.abs(v_f0_transp_mf - v_f0_fix_slope)
    v_f0_fix_slope[v_diff > diff_mf_tres] = v_f0_transp_mf[v_diff > diff_mf_tres]
    #'''
    # Smooth by slope:


    # DEBUG:----------------------------------------------------
    '''
    pl.figure(4)
    v_bnd_segs = np.zeros(len(v_f0))
    v_bnd_segs[v_start_segs] = 1
    pl.stem(300 * v_bnd_segs)
    pl.plot(v_f0, '.-g', label='v_f0')
    pl.plot(v_f0_transp, '*-c', label='v_f0_transp')
    pl.plot(v_f0_fix_slope, '.-r', label='v_f0_fix_slope')
    #pl.plot(v_f0_fix_slope_mf, '.-k', label='v_f0_fix_slope_mf')
    pl.grid()
    pl.legend()
    #'''

    # DEBUG:
    '''
    m_mag, m_real, m_imag, v_shift, v_voi, v_start_segs = analysis_fft_feats2(d_utt_data_mix, nfft)
    v_shift_fix_slope_syn = ff.f0_to_shift(v_f0_fix_slope, fs).astype('int64')
    v_syn_sig = synthesis_fft_feats2(m_mag, m_real, m_imag, v_shift_fix_slope_syn, v_start_segs)
    # END:
    '''

    # Smoothing Complex Spectrum:===============================================
    m_fft_fix    = smooth_by_mixing_params('m_fft', d_utt_data_mix)
    if m_fft_fix is None: # Backup plan
        v_syn_sig = wavgen_simple_concat_f0_slope(uttfile, db_dir, pm_reaper_dir, nfft, fs)
        return v_syn_sig

    # Remove boundary frames in F0 and transpose factors:=================
    v_end_segs = v_start_segs[1:] - 1
    v_f0_fix_slope_syn = np.delete(v_f0_fix_slope, v_end_segs)
    #v_transp_facts     = np.delete(v_transp_facts, v_end_segs)

    # Protection:
    nfrms_diff = m_fft_fix.shape[0] - len(v_f0_fix_slope_syn)
    if nfrms_diff>0:
        m_fft_fix = m_fft_fix[:-nfrms_diff,:]
    elif nfrms_diff<0:
        v_f0_fix_slope_syn = v_f0_fix_slope_syn[:nfrms_diff]


    # Convert to shifts:
    v_shift_fix_slope_syn = ff.f0_to_shift(v_f0_fix_slope_syn, fs).astype('int64')

    # Final Synthesis:
    v_syn_sig = synthesis_fft_feats_from_fft_matrix(m_fft_fix, v_shift_fix_slope_syn)

    return v_syn_sig

if __name__ == '__main__':




    #import matplotlib
    #matplotlib.rcParams['lines.antialiased'] = False
    #matplotlib.rcParams['lines.linewidth']   = 1.0

    #m_est_data = read_est_file('/home/felipe/Dropbox/Education/UoE/SideProjects/BlizzardChallenge2017_CSTR_entry/database/pm/WindInTheWillows_youngreading_000_001.pm')

    #'''
    # INPUT:==========================
    # NOTE (no delete): original wav files: /afs/inf.ed.ac.uk/group/cstr/projects/blizzard_entries/blizzard2016/multisyn_voice/cstr_edi_fls_multisyn/fls
    #syn_utt_file = '/home/felipe/Dropbox/Education/UoE/SideProjects/BlizzardChallenge2017_CSTR_entry/data/AMidsummerNightsDream_000_002.utt'
    #syn_utt_file = os.environ['HOME'] + '/Dropbox/Education/UoE/SideProjects/BlizzardChallenge2017_CSTR_entry/data/samples_110417/selected/AMidsummerNightsDream_006_000.utt'
    #syn_utt_file = os.environ['HOME'] + '/Dropbox/Education/UoE/SideProjects/BlizzardChallenge2017_CSTR_entry/data/samples_110417/selected/AMidsummerNightsDream_007_004.utt'
    syn_utt_file = os.environ['HOME'] + '/Dropbox/Education/UoE/SideProjects/BlizzardChallenge2017_CSTR_entry/data/samples_110417/selected/AMidsummerNightsDream_006_006.utt'
    db_dir       = os.environ['HOME'] + '/Dropbox/Education/UoE/SideProjects/BlizzardChallenge2017_CSTR_entry/database'
    out_dir      = os.environ['HOME'] + '/Dropbox/Education/UoE/SideProjects/BlizzardChallenge2017_CSTR_entry/data/out'
    nfft         = 4096
    npm_margin_mix   = 3 #3
    fs = 48000


    v_syn_sig = wavgen(syn_utt_file, db_dir, nfft, fs)


    # Synthesis:=================================
    if b_synth:
        v_syn_sig_mix_f0_transp = synthesis_fft_feats_from_fft_matrix(m_fft_fix, v_shift_transp_syn)
        v_syn_sig_mix_f0_transp_slope = synthesis_fft_feats_from_fft_matrix(m_fft_fix, v_shift_fix_slope_syn)
        v_syn_sig_mix_f0_transp_slope_debug = synthesis_fft_feats_from_fft_matrix(m_fft_fix_debug, v_shift_fix_slope_syn_debug)

    # Save:======================================
    if b_save:
        la.write_audio_file(out_dir + '/prue_fix_mix_f0_transp.wav', v_syn_sig_mix_f0_transp, fs)
        la.write_audio_file(out_dir + '/prue_fix_mix_f0_transp_slope.wav', v_syn_sig_mix_f0_transp_slope, fs)
        la.write_audio_file(out_dir + '/prue_fix_mix_f0_transp_slope_debug.wav', v_syn_sig_mix_f0_transp_slope_debug, fs)


