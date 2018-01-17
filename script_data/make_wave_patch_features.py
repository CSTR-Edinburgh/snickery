#!/usr/bin/env python
# -*- coding: utf-8 -*-
## Project: ... - February 2017 - ...
## Contact: Oliver Watts - owatts@staffmail.ed.ac.uk
  

#  python make_wave_patch_features.py -w /afs/inf.ed.ac.uk/group/cstr/projects/blizzard_entries/blizzard2017/data/segmented/wav/ -p /afs/inf.ed.ac.uk/group/cstr/projects/nst/oliver/hybrid_work/data/fls_data/world_reaper/pm/ -o /tmp/testfeats

import sys
import os
import glob
import os
import fileinput
import re
import math
from argparse import ArgumentParser
   
import numpy as np

# modify import path to obtain modules from the script/ directory:
snickery_dir = os.path.split(os.path.realpath(os.path.abspath(os.path.dirname(__file__))))[0]+'/'
sys.path.append(os.path.join(snickery_dir, 'script'))

from speech_manip import  get_speech, put_speech, read_wave
from train_halfphone import read_pm
from const import vuv_stream_names

from data_provider import DataProvider
from segmentaxis import segment_axis
import mulaw2

import pylab

EXTN = 'wpf'

# from keras_train import train_network_from_generators

def read_pm(fname):

    f = open(fname, 'r')
    lines = f.readlines()
    f.close()

    for (i,line) in enumerate(lines):
        if line.startswith('EST_Header_End'):
            start = i+1
            break
    lines = lines[start:]
    lines = [float(re.split('\s+',line)[0]) for line in lines]
    lines = np.array(lines)
    
    ## debug: make sure monotonic increase
    start_end = segment_axis(lines, 2, overlap=1)
    diffs = start_end[:,1] - start_end[:,0]
    neg_diffs = (diffs < 0.0)
    if sum(neg_diffs) > 0:
        print ('WARNING: pitch marks not monotonically increasing in %s'%(fname))
        return np.ones((1,1))

    return lines
  


def get_one_sentence_data(wave_file, pitch_mark_file, outfile, rate, width, warp='lin', debug=False  ):

    print wave_file
    if debug:
        print 'get_one_sentence_data'
        

    if not (os.path.isfile(wave_file) and os.path.isfile(pitch_mark_file)):
        return np.zeros((0,0))

    pitchmarks = read_pm(pitch_mark_file) * rate

    if pitchmarks.size==1:
        return np.zeros((0,0))
    pitchmarks = np.array(pitchmarks, dtype='int')

    _, basename = os.path.split(wave_file)
    os.system('sox %s -r %s /tmp/%s'%(wave_file, rate, basename))
    wave, sample_rate = read_wave('/tmp/%s'%(basename))

    if warp == 'mu':
        wave = mulaw2.lin2mu(wave)

    assert width % 2 == 1, 'please choose odd number for width'

    halfwidth = (width - 1) / 2
    
    starts = np.clip(pitchmarks - halfwidth, 0, len(wave))
    ends = np.clip(pitchmarks + halfwidth + 1, 0, len(wave))

    # starts = pitchmarks - halfwidth#, 0, len(wave)
    # ends = pitchmarks + halfwidth + 1 # , 0, len(wave)

    frags = [wave[s:e] for (s,e) in zip(starts, ends)]
    # print [len(f) for f in frags]

    fragmat = np.zeros((len(frags), width))
    #frags = np.vstack(frags)
    for (i,f) in enumerate(frags):
        fragmat[i,:len(f)] = f


    if debug:
        pylab.plot(fragmat.transpose())
        pylab.show()
        sys.exit('advadv')
    
    # print fragmat
    put_speech(fragmat, outfile)





if __name__ == '__main__':



    #################################################
      
    # ======== process command line ==========

    a = ArgumentParser()
    a.add_argument('-w', dest='wave_dir', required=True)
    a.add_argument('-p', dest='pitchmark_dir', required=True)
    a.add_argument('-o', dest='output_dir', required=True)    
    a.add_argument('-r', dest='outrate', type=int, default=48000)  
    a.add_argument('-d', dest='outdim', type=int, default=9)      
    a.add_argument('-N', dest='nfiles', type=int, default=0)   
    a.add_argument('-warp', dest='linear_or_mu', type=str, default='lin')   
    opts = a.parse_args()

    assert opts.linear_or_mu in ['lin', 'mu']
    
    full_extension = '%s_%s_%s_%s'%(EXTN, opts.linear_or_mu, opts.outrate, opts.outdim)


    output_dir = opts.output_dir
    wav_datadir = opts.wave_dir
    pm_datadir = opts.pitchmark_dir

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if not os.path.isdir(output_dir + '/'+full_extension+'/'):
        os.makedirs(output_dir + '/'+full_extension+'/')        

    wavelist = sorted(glob.glob(wav_datadir + '/*.wav'))
    print wavelist
    
    if opts.nfiles == 0:
        nfiles = len(wavelist)
    else:
        nfiles = min(opts.nfiles, len(wavelist))

    wavelist = wavelist[:nfiles]
    
    for wave_file in wavelist:
        _, base = os.path.split(wave_file)
        base = base.replace('.wav', '')
        pitch_mark_file = os.path.join(pm_datadir, base + '.pm')
        outfile = os.path.join(output_dir+'/'+full_extension+'/', base + '.' + full_extension)        
        get_one_sentence_data(wave_file, pitch_mark_file, outfile, opts.outrate, opts.outdim, warp=opts.linear_or_mu)





