#!/usr/bin/env python
# -*- coding: utf-8 -*-
## Project: Natural Speech Technology - February 2015 - www.natural-speech-technology.org
## Contact: Oliver Watts - owatts@staffmail.ed.ac.uk
  
import sys
import os
import glob
import os
import fileinput
from argparse import ArgumentParser

import numpy as np

## Check required executables are available:

from distutils.spawn import find_executable

required_executables = ['tclsh8.4']

for executable in required_executables:
    if not find_executable(executable):
        sys.exit('%s command line tool must be on system path '%(executable))
    

# modify import path to obtain modules from the script/ directory:
snickery_dir = os.path.split(os.path.realpath(os.path.abspath(os.path.dirname(__file__))))[0]+'/'
sys.path.append(os.path.join(snickery_dir, 'script'))

from speech_manip import  get_speech, put_speech, read_wave

here = os.path.realpath(os.path.abspath(os.path.dirname(__file__)))



def main_work():

    #################################################
      
    # ======== Get stuff from command line ==========

    a = ArgumentParser()
    a.add_argument('-i', dest='indir', required=True)
    a.add_argument('-f', dest='foutdir', required=True, \
                    help= "Put formant freq output here: make it if it doesn't exist")
    a.add_argument('-b', dest='boutdir', required=True, \
                    help= "Put formant bandwidth output here: make it if it doesn't exist")    
    # a.add_argument('-c', dest='clear', action='store_true', \
    #                 help= "clear any previous training data first")
    # a.add_argument('-p', dest='max_cores', required=False, type=int, help="maximum number of CPU cores to use in parallel")
    opts = a.parse_args()
    
    # ===============================================
    
    for direc in [opts.foutdir, opts.boutdir]:
        if not os.path.isdir(direc):
            os.makedirs(direc)


    for wavefile in glob.glob(opts.indir + '/*.wav'):
        _, base = os.path.split(wavefile)
        base = base.replace('.wav', '')
        print base
        os.system('tclsh8.4 %s/get_formant.tcl %s > %s/%s.tmp'%(here, wavefile, opts.foutdir, base))
        mat = np.loadtxt('%s/%s.tmp'%(opts.foutdir, base))
        formfreq = mat[:,:4]
        formband = mat[:,4:]
        put_speech(formfreq, '%s/%s.formfreq'%(opts.foutdir, base))
        put_speech(formband, '%s/%s.formband'%(opts.boutdir, base))        

    for tempfile in glob.glob(opts.foutdir + '/*.tmp'):
        os.system('rm %s'%(tempfile))

if __name__=="__main__":

    main_work()

