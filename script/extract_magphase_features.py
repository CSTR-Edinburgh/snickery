
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



def make_magphase_directory_structure(outdir):
    outdir_hi = os.path.join(outdir, 'high')
    outdir_lo = os.path.join(outdir, 'low')
    for direc in [outdir, outdir_hi, outdir_lo]:
        safe_makedir(direc)
    for subdir in ['mag', 'real', 'imag']:
        for direc in [outdir_hi, outdir_lo]:
            new_direc = os.path.join(direc, subdir)
            safe_makedir(new_direc)
    for subdir in ['shift', 'f0', 'lf0', 'pm']:
        new_direc = os.path.join(outdir, subdir)
        safe_makedir(new_direc)


def magphase_analysis(wav_file, outdir='', fft_len=None, nbins_mel=60, nbins_phase=45):
    '''
    Function to combine Felipe's analysis_lossless and analysis_compressed with 
    little redundancy, and storing pitchmark files.
    '''
    outdir_hi = os.path.join(outdir, 'high')
    outdir_lo = os.path.join(outdir, 'low')

    file_id = os.path.basename(wav_file).split(".")[0]

    # Read file:
    v_sig, fs = sf.read(wav_file)

    # Epoch detection:
    est_file = os.path.join(outdir, 'pm', file_id + '.pm') 
    la.reaper(wav_file, est_file)
    v_pm_sec, v_voi = la.read_reaper_est_file(est_file, check_len_smpls=len(v_sig), fs=fs)
    v_pm_smpls = v_pm_sec * fs

    # Spectral analysis:
    m_fft, v_shift = mp.analysis_with_del_comp_from_pm(v_sig, fs, v_pm_smpls, fft_len=fft_len)

    # Getting high-ress magphase feats:
    m_mag, m_real, m_imag, v_f0 = mp.compute_lossless_feats(m_fft, v_shift, v_voi, fs)

    # Low dimension (Formatting for Acoustic Modelling):
    m_mag_mel_log, m_real_mel, m_imag_mel, v_lf0_smth = mp.format_for_modelling(m_mag, m_real, m_imag, v_f0, fs, nbins_mel=nbins_mel, nbins_phase=nbins_phase)
    fft_len = 2*(np.size(m_mag,1) - 1)

    ### write high-dimensional data:
    lu.write_binfile(m_mag, os.path.join(outdir_hi, 'mag', file_id + '.mag'))
    lu.write_binfile(m_real, os.path.join(outdir_hi, 'real', file_id + '.real'))
    lu.write_binfile(m_imag, os.path.join(outdir_hi, 'imag', file_id + '.imag'))
    lu.write_binfile(v_f0, os.path.join(outdir, 'f0', file_id + '.f0'))
    lu.write_binfile(v_shift, os.path.join(outdir, 'shift', file_id + '.shift'))

    ### write low-dim data:
    lu.write_binfile(m_mag_mel_log, os.path.join(outdir_lo, 'mag', file_id + '.mag'))
    lu.write_binfile(m_real_mel, os.path.join(outdir_lo, 'real', file_id + '.real'))
    lu.write_binfile(m_imag_mel, os.path.join(outdir_lo, 'imag', file_id + '.imag'))
    lu.write_binfile(v_lf0_smth, os.path.join(outdir, 'lf0', file_id + '.lf0'))








if __name__ == '__main__':



    #################################################
      
    # ======== process command line ==========

    a = ArgumentParser()
    a.add_argument('-w', dest='wave_dir', required=True)
    a.add_argument('-o', dest='output_dir', required=True)    
    a.add_argument('-N', dest='nfiles', type=int, default=0)  
    a.add_argument('-ncores', type=int, default=0)            
    opts = a.parse_args()
    

    wav_datadir = opts.wave_dir
    make_magphase_directory_structure(opts.output_dir)

    # hidir = os.path.join(opts.output_dir, 'magphse_hi')
    # lodir = os.path.join(opts.output_dir, 'magphse_lo')
    # pm_dir = os.path.join(opts.output_dir, 'mag_pm')

    # safe_makedir(hidir)     
    # safe_makedir(lodir)     
    # safe_makedir(pm_dir)     

    # if extract_hi:
    #     use_hidir = hidir
    # else:
    #     use_hidir = None

    # for extn in ['mag', 'imag', 'real', 'lf0']:
    #     stream_dir = os.path.join(opts.output_dir, extn)
    #     safe_makedir(stream_dir)

    # if use_hidir != None:
    #     for extn in ['mag', 'imag', 'real', 'f0']:
    #         stream_dir = os.path.join(opts.output_dir, extn + '_full')
    #         safe_makedir(stream_dir)


    
    wavlist = sorted(glob.glob(wav_datadir + '/*.wav'))
    if opts.nfiles > 0:
        wavlist = wavlist[:opts.nfiles]

    print wavlist


    # lofz = glob.glob(lodir + '/*.lf0')
    # lofz_base = [os.path.split(name)[-1].replace('.lf0','') for name in lofz]
    # lofz_base = dict(zip(lofz_base, lofz_base))
    # wavlist = [name for name in wavlist if os.path.split(name)[-1].replace('.wav','') not in lofz_base]

    print len(wavlist)


    ## wrap the call so we can set all args except wavefile to be constant:
    # def wrapped_magphase_call(wav_file):
    #     base = os.path.split(wav_file)[-1].replace('.wav','')
    #     print 'processing %s'%(base)

    #     if not os.path.isfile(os.path.join(lodir, base + '.lf0')):  
    #         magphase.analysis_compressed(wav_file, out_dir=lodir, out_dir_uncompressed=use_hidir, fft_len=1024, nbins_mel=60, nbins_phase=45, pm_dir=pm_dir) ## explicitly specify defaults (at 48k)
    #     else:
    #         print 'skip extraction!'

    #     ### finally, sort out dir structure for lo features:
    #     for extn in ['mag', 'imag', 'real', 'lf0']:
    #         print 'Move low dimensional represenrations for stream %s'%(extn)
    #         stream_dir = os.path.join(opts.output_dir, extn)
    #         os.system('mv %s/%s.%s %s'%(lodir, base, extn, stream_dir))

    #     if use_hidir != None:
    #         for extn in ['mag', 'imag', 'real', 'f0']:
    #             print 'Move high dimensional represenrations for stream %s'%(extn)
    #             stream_dir = os.path.join(opts.output_dir, extn + '_full')
    #             os.system('mv %s/%s.%s %s'%(hidir, base, extn, stream_dir))




    if opts.ncores > 0:
        import multiprocessing

        ## Use partial to pass fixed arguments to the func (https://stackoverflow.com/questions/5442910/python-multiprocessing-pool-map-for-multiple-arguments):

        pool = multiprocessing.Pool(processes=opts.ncores) 
        results = pool.map(functools.partial(magphase_analysis, outdir=opts.output_dir, fft_len=1024, nbins_mel=60, nbins_phase=45), wavlist)         
        pool.close() #we are not adding any more processes
        #pool.join() #tell it to wait until all threads are done before going on


    else:

        for wav_file in wavlist:
            magphase_analysis(wav_file, outdir=opts.output_dir, fft_len=1024, nbins_mel=60, nbins_phase=45) 


