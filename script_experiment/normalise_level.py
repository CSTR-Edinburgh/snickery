#!/usr/bin/env python
# -*- coding: utf-8 -*-
## Project: SCRIPT - March 2018 
## Contact: Oliver Watts - owatts@staffmail.ed.ac.uk
  
import sys
import os
import glob
import os
from argparse import ArgumentParser

import soundfile as sf

## Check required executables are available:

from distutils.spawn import find_executable

required_executables = ['sv56demo']

for executable in required_executables:
    if not find_executable(executable):
        sys.exit('%s command line tool must be on system path '%(executable))
    



def main_work():

    #################################################
      
    # ======== Get stuff from command line ==========

    a = ArgumentParser()
    a.add_argument('-i', dest='indir', required=True)    
    a.add_argument('-o', dest='outdir', required=True, \
                    help= "Put output here: make it if it doesn't exist")
    a.add_argument('-pattern', default='', \
                    help= "If given, only normalise files whose base contains this substring")
    opts = a.parse_args()
    
    # ===============================================
    
    for direc in [opts.outdir]:
        if not os.path.isdir(direc):
            os.makedirs(direc)

    flist = sorted(glob.glob(opts.indir + '/*.wav'))
    
    for wavefile in flist:
        _, base = os.path.split(wavefile)
        
        if opts.pattern:
            if opts.pattern not in base:
                continue

        print base

        raw_in = os.path.join(opts.outdir, base.replace('.wav','.raw'))
        raw_out = os.path.join(opts.outdir, base.replace('.wav','_norm.raw'))
        logfile = os.path.join(opts.outdir, base.replace('.wav','.log'))
        wav_out = os.path.join(opts.outdir, base)
        
        data, samplerate = sf.read(wavefile, dtype='int16')
        sf.write(raw_in, data, samplerate, subtype='PCM_16')
        os.system('sv56demo -log %s -q -lev -26.0 -sf %s %s %s'%(logfile, samplerate, raw_in, raw_out))
        norm_data, samplerate = sf.read(raw_out, dtype='int16', samplerate=samplerate, channels=1, subtype='PCM_16')
        sf.write(wav_out, norm_data, samplerate)

        os.system('rm %s %s'%(raw_in, raw_out))


# data, samplerate = sf.read('myfile.raw', channels=1, samplerate=44100,
#                            subtype='FLOAT')        



if __name__=="__main__":

    main_work()

