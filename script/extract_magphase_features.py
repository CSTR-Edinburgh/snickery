
import os, sys, glob
from util import safe_makedir
from argparse import ArgumentParser
   
# modify import path to obtain modules from the tools/magphase/src directory:
snickery_dir = os.path.split(os.path.realpath(os.path.abspath(os.path.dirname(__file__))))[0]+'/'
sys.path.append(os.path.join(snickery_dir, 'tool', 'magphase', 'src'))

import numpy as np

# sys.path.append('/Users/owatts/repos/magphase/magphase/src/')
import magphase as mp
import functools

# Turn:

#    out_dir_uncompressed=None  -->   out_dir_uncompressed=hidir

# ... to also dump full FFTs (v. big size)
# extract_hi = True


import libutils as lu
import libaudio as la
import soundfile as sf



def make_magphase_directory_structure(outdir, cepstra=False):
    outdir_hi = os.path.join(outdir, 'high')
    outdir_lo = os.path.join(outdir, 'low')
    for direc in [outdir, outdir_hi, outdir_lo]:
        safe_makedir(direc)
    for subdir in ['mag', 'real', 'imag']:
        for direc in [outdir_hi, outdir_lo]:
            new_direc = os.path.join(direc, subdir)
            safe_makedir(new_direc)
    for subdir in ['shift', 'pm']:
        new_direc = os.path.join(outdir, subdir)
        safe_makedir(new_direc)
    safe_makedir(os.path.join(outdir_hi, 'f0'))
    safe_makedir(os.path.join(outdir_lo, 'lf0'))

    if cepstra:
        for subdir in ['mag_cc', 'imag_cc', 'real_cc']:
            safe_makedir(os.path.join(outdir_lo, subdir))        





def magphase_analysis(wav_file, outdir='', fft_len=None, nbins_mel=60, nbins_phase=45, pm_dir='', skip_low=False, cepstra=False):
    '''
    Function to combine Felipe's analysis_lossless and analysis_compressed with 
    little redundancy, and storing pitchmark files.
    '''

    try:
        outdir_hi = os.path.join(outdir, 'high')
        outdir_lo = os.path.join(outdir, 'low')

        file_id = os.path.basename(wav_file).split(".")[0]

        # Read file:
        v_sig, fs = sf.read(wav_file)

        if not pm_dir:
            # Epoch detection:
            est_file = os.path.join(outdir, 'pm', file_id + '.pm') 
            la.reaper(wav_file, est_file)
        else:
            est_file = os.path.join(pm_dir, file_id + '.pm') 
        v_pm_sec, v_voi = la.read_reaper_est_file(est_file, check_len_smpls=len(v_sig), fs=fs)
        v_pm_smpls = v_pm_sec * fs



        # Spectral analysis:
        m_fft, v_shift = mp.analysis_with_del_comp_from_pm(v_sig, fs, v_pm_smpls, fft_len=fft_len)

        # Getting high-ress magphase feats:
        m_mag, m_real, m_imag, v_f0 = mp.compute_lossless_feats(m_fft, v_shift, v_voi, fs)

        ### write high-dimensional data:
        lu.write_binfile(m_mag, os.path.join(outdir_hi, 'mag', file_id + '.mag'))
        lu.write_binfile(m_real, os.path.join(outdir_hi, 'real', file_id + '.real'))
        lu.write_binfile(m_imag, os.path.join(outdir_hi, 'imag', file_id + '.imag'))
        lu.write_binfile(v_f0, os.path.join(outdir_hi, 'f0', file_id + '.f0'))
        lu.write_binfile(v_shift, os.path.join(outdir, 'shift', file_id + '.shift'))

        if not skip_low:
            # Low dimension (Formatting for Acoustic Modelling):
            m_mag_mel_log, m_real_mel, m_imag_mel, v_lf0_smth = mp.format_for_modelling(m_mag, m_real, m_imag, v_f0, fs, mag_dim=nbins_mel, phase_dim=nbins_phase)
            # fft_len = 2*(np.size(m_mag,1) - 1)

            ### write low-dim data:
            lu.write_binfile(m_mag_mel_log, os.path.join(outdir_lo, 'mag', file_id + '.mag'))
            lu.write_binfile(m_real_mel, os.path.join(outdir_lo, 'real', file_id + '.real'))
            lu.write_binfile(m_imag_mel, os.path.join(outdir_lo, 'imag', file_id + '.imag'))
            lu.write_binfile(v_lf0_smth, os.path.join(outdir_lo, 'lf0', file_id + '.lf0'))

        if cepstra:
            alpha = {48000: 0.77, 16000: 58}[fs]
            m_mag_mcep = la.sp_to_mcep(m_mag, n_coeffs=nbins_mel, alpha=alpha, in_type=3)
            m_real_mcep = la.sp_to_mcep(m_real, n_coeffs=nbins_phase, alpha=alpha, in_type=2)
            m_imag_mcep = la.sp_to_mcep(m_imag, n_coeffs=nbins_phase, alpha=alpha, in_type=2)

            lu.write_binfile(m_mag_mcep, os.path.join(outdir_lo, 'mag_cc', file_id + '.mag_cc'))
            lu.write_binfile(m_real_mcep, os.path.join(outdir_lo, 'real_cc', file_id + '.real_cc'))
            lu.write_binfile(m_imag_mcep, os.path.join(outdir_lo, 'imag_cc', file_id + '.imag_cc'))
    except KeyboardInterrupt, e:
        pass

def main_work():

    #################################################
      
    # ======== process command line ==========

    a = ArgumentParser()
    a.add_argument('-w', dest='wave_dir', required=True)
    a.add_argument('-o', dest='output_dir', required=True)    
    a.add_argument('-N', dest='nfiles', type=int, default=0)  
    a.add_argument('-m', type=int, default=60, help='low dim feature size (compressed mel magnitude spectrum & cepstrum)')  
    a.add_argument('-p', type=int, default=45, help='low dim feature size (compressed mel phase spectra & cepstra)')          
    a.add_argument('-fftlen', type=int, default=1024)          
    a.add_argument('-ncores', type=int, default=0)   

    a.add_argument('-pm_dir', type=str, default='', help='Specify a directory of existing pitchmark files to use, instead of starting from scratch')
    a.add_argument('-cepstra', default=False, action='store_true', help='Extract cepstral coefficients as well as magphase representations.')
    opts = a.parse_args()
    

    wav_datadir = opts.wave_dir
    make_magphase_directory_structure(opts.output_dir, cepstra=opts.cepstra)

    wavlist = sorted(glob.glob(wav_datadir + '/*.wav'))
    if opts.nfiles > 0:
        wavlist = wavlist[:opts.nfiles]

    print wavlist

    print len(wavlist)

    if opts.ncores > 0:
        import multiprocessing

        ## Use partial to pass fixed arguments to the func (https://stackoverflow.com/questions/5442910/python-multiprocessing-pool-map-for-multiple-arguments):

        pool = multiprocessing.Pool(processes=opts.ncores) 
        results = pool.map(functools.partial(magphase_analysis, outdir=opts.output_dir, fft_len=opts.fftlen, nbins_mel=opts.m, nbins_phase=opts.p, pm_dir=opts.pm_dir, cepstra=opts.cepstra), wavlist)         

        try:
            print 'done!'
        except KeyboardInterrupt:
            print 'parent received control-c'
            return        

        pool.close() 


    else:

        for wav_file in wavlist:
            magphase_analysis(wav_file, outdir=opts.output_dir, fft_len=opts.fftlen, nbins_mel=opts.m, nbins_phase=opts.p, pm_dir=opts.pm_dir, cepstra=opts.cepstra) 




if __name__ == '__main__':

    main_work()
