#!/usr/bin/env python
# -*- coding: utf-8 -*-
## Project: SCRIPT - February 2018 
## Contact: Oliver Watts - owatts@staffmail.ed.ac.uk
  
import sys
import os
import glob
import os
import fileinput
from argparse import ArgumentParser

import speech_manip
from util import basename

from speech_manip import get_speech
## Check required executables are available:
# 
# from distutils.spawn import find_executable
# 
# required_executables = ['sox', 'ch_wave']
# 
# for executable in required_executables:
#     if not find_executable(executable):
#         sys.exit('%s command line tool must be on system path '%(executable))



# class HDFMagphaseData(object):
#     def __init__(fname):
#         self.fname = fname
#         if os.path.isfile(self.fname):


import h5py

def make_hdf_magphase(datadir, database_fname, fftlength):

    HALFFFTLEN = (fftlength / 2) + 1
    for stream in ['mag', 'real', 'imag', 'f0']:
        assert os.path.isdir(os.path.join(datadir, stream))

    f = h5py.File(database_fname, "w")

    for magfile in sorted(glob.glob(os.path.join(datadir, 'mag/*.mag'))):
        base = basename(magfile)
        print base
        skip_file = False
        for stream in ['mag', 'real', 'imag', 'f0']:
            if not os.path.isfile(os.path.join(datadir, stream, base+'.'+stream)):
                skip_file = True
        if skip_file:
            print '  ---> skip!'
            continue


        utt_group = f.create_group(base)
        for stream in ['mag', 'real', 'imag']:
            speech = get_speech(os.path.join(datadir, stream, base + '.' + stream), HALFFFTLEN)
            utt_group.create_dataset(stream, data=speech)
        f0 = get_speech(os.path.join(datadir, 'f0', base + '.f0'), 1)            
        f0_interp, vuv = speech_manip.lin_interp_f0(f0)
        utt_group.create_dataset('f0_interp', data=f0_interp)
        utt_group.create_dataset('vuv', data=vuv)

    f.close()






def main_work():

    #################################################
      
    # ======== Get stuff from command line ==========

    a = ArgumentParser()
    a.add_argument('-d', dest='datadir', required=True)
    a.add_argument('-o', dest='outfile', required=True)
    a.add_argument('-f', dest='fftlength', default=2048, type=int)    
    opts = a.parse_args()
    
    # ===============================================
    make_hdf_magphase(opts.datadir, opts.outfile, opts.fftlength)
    



if __name__=="__main__":

    main_work()

