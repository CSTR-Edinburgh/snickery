
'''
e.g.:
 python ~/proj/slm-local/snickery/script/resynthesise_magphase.py -f /group/project/cstr2/cvbotinh/SCRIPT/Nick/feats/magphase/low/ -o ~/sim2/oliver/temp/nickout -fftlen 2048 -ncores 25 -pattern hvd


'''
import os, sys, glob
from util import safe_makedir, basename
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


def synthesis(base, feature_dir='', output_dir='', fft_len=2048, nbins_mel=60, nbins_phase=45, fs=48000):

    m_mag_mel_log = lu.read_binfile(os.path.join(feature_dir, 'mag', base+'.mag'), dim=nbins_mel)
    m_real_mel = lu.read_binfile(os.path.join(feature_dir, 'real', base+'.real'), dim=nbins_phase)
    m_imag_mel = lu.read_binfile(os.path.join(feature_dir, 'imag', base+'.imag'), dim=nbins_phase)
    v_lf0 = lu.read_binfile(os.path.join(feature_dir, 'lf0', base+'.lf0'), dim=1)

    try:
        v_syn_sig = mp.synthesis_from_compressed(m_mag_mel_log, m_real_mel, m_imag_mel, v_lf0, fs, fft_len)
        wav_file_syn = os.path.join(output_dir, base+'.wav')
        la.write_audio_file(wav_file_syn, v_syn_sig, fs)
        print 'written %s'%(wav_file_syn)
    except:
        print 'synth failed for %s'%(base)

if __name__ == '__main__':



    #################################################
      
    # ======== process command line ==========

    a = ArgumentParser()
    a.add_argument('-f', dest='feature_dir', required=True)
    a.add_argument('-o', dest='output_dir', required=True)    
    a.add_argument('-N', dest='nfiles', type=int, default=0)  
    a.add_argument('-m', type=int, default=60, help='low dim feature size (compressed mel magnitude spectrum & cepstrum)')  
    a.add_argument('-p', type=int, default=45, help='low dim feature size (compressed mel phase spectra & cepstra)')          
    a.add_argument('-fftlen', type=int, default=1024)          
    a.add_argument('-ncores', type=int, default=0)   
    a.add_argument('-fs', type=int, default=48000)  
    a.add_argument('-pattern', type=str, default='', help='only synthesise files with this substring in their basename')  
    opts = a.parse_args()
    
    safe_makedir(opts.output_dir)
    
    baselist = [basename(fname) for fname in sorted(glob.glob(opts.feature_dir + '/lf0/*.lf0'))]

    #### temp
    # baselist2 = []
    # for base in baselist:
    #     if int(base.replace('hvd_', '')) > 600:
    #         baselist2.append(base)
    # baselist = baselist2


    if opts.pattern:
        baselist = [b for b in baselist if opts.pattern in b]

    if opts.nfiles > 0:
        baselist = baselist[:opts.nfiles]

    print baselist

    if opts.ncores > 0:
        import multiprocessing

        ## Use partial to pass fixed arguments to the func (https://stackoverflow.com/questions/5442910/python-multiprocessing-pool-map-for-multiple-arguments):

        pool = multiprocessing.Pool(processes=opts.ncores) 
        results = pool.map(functools.partial(synthesis, feature_dir=opts.feature_dir, output_dir=opts.output_dir, fft_len=opts.fftlen, nbins_mel=opts.m, nbins_phase=opts.p, fs=opts.fs), baselist)         
        pool.close() 

    else:

        for base in baselist:
            synthesis(base, feature_dir=opts.feature_dir, output_dir=opts.output_dir, fft_len=opts.fftlen, nbins_mel=opts.m, nbins_phase=opts.p)


