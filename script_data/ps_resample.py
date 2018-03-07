#!/usr/bin/env python
# -*- coding: utf-8 -*-
## Project: ... - February 2017 - ...
## Contact: Oliver Watts - owatts@staffmail.ed.ac.uk
  
import sys
import os
import glob
import os
import fileinput
from argparse import ArgumentParser
   
import numpy as np

# modify import path to obtain modules from the script/ directory:
snickery_dir = os.path.split(os.path.realpath(os.path.abspath(os.path.dirname(__file__))))[0]+'/'
sys.path.append(os.path.join(snickery_dir, 'script'))

from speech_manip import  get_speech, put_speech, read_wave
from train_halfphone import read_pm
from const import vuv_stream_names

import scipy # , pylab
from scipy.interpolate import interp1d

from resample import get_snack_frame_centres



MGCDIM = 60
#fshift_seconds = 0.005 # (0.001 * config['frameshift_ms'])

def main_work():

    #################################################
      
    # ======== Get stuff from command line ==========

    a = ArgumentParser()
    a.add_argument('-w', dest='wavfile', required=True)
    a.add_argument('-f', dest='feature_dir', required=True)
    a.add_argument('-p', dest='pm_dir', required=True)
    a.add_argument('-o', dest='outdir', required=True)
    a.add_argument('-x', dest='feature_extension', required=True)
    a.add_argument('-d', dest='feature_dim', type=int, required=True)
    a.add_argument('-s', dest='fshift_seconds', type=float, default=0.005, required=False)
    a.add_argument('-l', dest='labdir', default=None, help='not currently used')    
    opts = a.parse_args()
    
    # ===============================================
    
    # for filetype in ['f0','mgc','ap']:
    #     direc = os.path.join(opts.outdir, filetype)
    #     if not os.path.isdir(direc):
    #         os.makedirs(direc)

    ## temporary check not to use labels:
    assert opts.labdir == None

    if not os.path.isdir(opts.outdir):
        os.makedirs(opts.outdir)

    junk, base = os.path.split(opts.wavfile)
    base = base.replace('.wav', '')

    pm_fname = os.path.join(opts.pm_dir, base + '.pm')

    # mgc_fname = os.path.join(opts.feature_dir, 'mgc', base + '.mgc')
    # ap_fname = os.path.join(opts.feature_dir, 'ap', base + '.ap')
    # f0_fname = os.path.join(opts.feature_dir, 'f0', base + '.f0')
    
    feature_fname = os.path.join(opts.feature_dir, base + '.' + opts.feature_extension)

    for fname in [opts.wavfile, pm_fname, feature_fname]:
        if not os.path.isfile(fname):
            sys.exit('File does not exist: %s'%(fname))

    ## read data from files
    wave, sample_rate = read_wave(opts.wavfile)

    # mgc = get_speech(mgc_fname, MGCDIM)
    # ap = get_speech(ap_fname, get_world_bap_dim(sample_rate))
    # fz = get_speech(f0_fname, 1)

    if opts.feature_extension=='mfcc':
        features = get_speech(feature_fname, opts.feature_dim, remove_htk_header=True)
    else:
        features = get_speech(feature_fname, opts.feature_dim, remove_htk_header=False)

    pms_seconds = read_pm(pm_fname)
    
    #print get_world_bap_dim(sample_rate)
    
    # if not (mgc.shape[0] == ap.shape[0] == fz.shape[0]):
    #     print mgc.shape[0] , ap.shape[0] , fz.shape[0]
    #     sys.exit('dims do not match')
    
    ## Convert seconds -> waveform sample numbers:-
    pms = np.asarray(np.round(pms_seconds * sample_rate), dtype=int)
    len_wave = len(wave)
            
    if opts.feature_extension == 'mfcc':
        windowing_convention='HTK'
    elif opts.feature_extension in ['formfreq', 'formband']:
        windowing_convention='snack'
    else:
        windowing_convention='world'

    if opts.feature_extension in vuv_stream_names: 
        ## then we need to handle voicing decision specially:
        features, vuv = interp_fzero(features)
        ps_features = pitch_synchronous_resample(len_wave, sample_rate, opts.fshift_seconds, pms, features, windowing_convention=windowing_convention)
        ps_vuv = pitch_synchronous_resample(len_wave, sample_rate, opts.fshift_seconds, pms, vuv, int_type='nearest', windowing_convention=windowing_convention)
        assert ps_features.shape == ps_vuv.shape
        ## reimpose voicing decision on resampled F0:
        ps_features[ps_vuv==0] = 0
    else:
        ps_features = pitch_synchronous_resample(len_wave, sample_rate, opts.fshift_seconds, pms, features, windowing_convention=windowing_convention)

    # ps_mgc = pitch_synchronous_resample(len_wave, sample_rate, fshift_seconds, pms, mgc)
    # ps_ap = pitch_synchronous_resample(len_wave, sample_rate, fshift_seconds, pms, ap)
    
    # put_speech(ps_fz, os.path.join(opts.outdir, 'f0', base + '.f0'))
    # put_speech(ps_mgc, os.path.join(opts.outdir, 'mgc', base + '.mgc'))
    # put_speech(ps_ap, os.path.join(opts.outdir, 'ap', base + '.ap'))

    put_speech(ps_features, os.path.join(opts.outdir, base + '.' + opts.feature_extension))

    
    if opts.labdir != None:
        labfile = os.path.join(opts.labdir, base + '.lab')
        print 'TODO -- labels!'
        pms_htkunit = np.asarray(np.round(pms_seconds * 10000000), dtype=int)
        label = read_label(labfile)
        assign_pm_to_labels(pms_htkunit, label)


def assign_pm_to_labels(pms, label):
    print pms
    print label

    bins = [end for start,end,lab in label]
    labs = [lab for start,end,lab in label]
    ixx = np.digitize(pms, bins, right=True)
    np.clip(ixx, 0, len(labs)-1, out=ixx) ## make sure items in upper bin are not too large
    for state in range(len(labs)):
        print '===='
        print labs[state]
        print state
        print np.where(ixx==state)


def read_label(labfile):
    f = open(labfile, 'r')
    lines = f.readlines()
    f.close()
    outlabel = []
    for line in lines:
        start,end,lab = line.strip(' \n').split(' ')[:3]
        quinphone = lab.split(':')[0]
        state = lab.strip(']').split('[')[-1]
        newlab = quinphone + '_' + state
        outlabel.append((int(start), int(end), newlab))
    return outlabel
 
def get_world_bap_dim(rate):
    return int(min(15000.0, (( rate / 2.0) - 3000.0)) / 3000.0) 

 
def interp_fzero(fz):
    vuv = np.ones(fz.shape)
    vuv[fz<=0.0] = 0
        
    flat_fz = fz.flatten()
    x = np.arange(len(flat_fz))[flat_fz>0.0]
    y = flat_fz[flat_fz>0.0]
    
    interpolator = interp1d(x, y, \
                kind='linear', axis=0, bounds_error=False, fill_value='extrapolate')
    interpolated_fz = interpolator(np.arange(len(fz)))
    fz = interpolated_fz.reshape(vuv.shape)
    
    return fz, vuv      
        





### These functions reverse engineer the mechanisms that various tools use to window
### a waveform, in order to determine the waveform sample which should be considered
### associated with any given frame of features. 

def get_world_frame_centres(wavlength, sample_rate, fshift_seconds):
    ## World makes a time axis like this, then centres anaysis windows on entries in time_index
    ## This was worked out by inspecting World code.
    samples_per_frame = int(sample_rate * fshift_seconds)
    f0_length = (wavlength / float(sample_rate) / fshift_seconds) + 1
    time_axis = []
    for i in range(int(f0_length)):
        time_axis.append(i * samples_per_frame)

    return np.array( time_axis)


def get_mfcc_frame_centres(wavlength, sample_rate, fshift_seconds, flength_seconds):
    ## This was worked out to be consistent with HCopy's output.
    ## MFCCs were extracted from 16k downsampled wave, here comparing with original 48k wave...
    shift_samples = int(sample_rate * fshift_seconds)
    window_samples = int(sample_rate * flength_seconds)
    window_centre = int(window_samples / 2)
    window_end = window_samples
    time_axis = []
    while window_end <= wavlength:
        time_axis.append(window_centre)
        window_centre += shift_samples
        window_end += shift_samples

    ## and then 1 more frame to catch remainder of wave:
    time_axis.append(window_centre)


    return np.array( time_axis)


def get_straight_frame_centres(wavlength, sample_rate, fshift_seconds):
    ## !!!! This is hacked  from the world version -- not checked !!!!!!
    samples_per_frame = int(sample_rate * fshift_seconds)
    f0_length = (wavlength / float(sample_rate) / fshift_seconds) + 1
    time_axis = []
    for i in range(int(f0_length)):
        time_axis.append(i * samples_per_frame)

    time_axis = time_axis[:-1] ## omit last value for STRAIGHT...
    return np.array(time_axis)


    
    
#     offset = int(2.5 * fshift)  ## 2 compensates for missing frames, 0.5 shifts to centre
#     len_mono_frames = nframe - 2 - 2  ## 2 incomplete frames and beginning and end cut off
#     frame_centres = np.arange(0,len_mono_frames) * fshift
#     frame_centres += offset

def pitch_synchronous_resample_one_coef_at_a_time(len_wave, sample_rate, fshift_seconds, pms, features, int_type='linear', windowing_convention='world', analysis_window_length_seconds=0.01):

    if windowing_convention == 'world':
        frame_centres = get_world_frame_centres(len_wave, sample_rate, fshift_seconds)
    elif windowing_convention == 'snack':
        frame_centres = get_snack_frame_centres(len_wave, sample_rate, fshift_seconds)          
    elif windowing_convention == 'HTK':
        frame_centres = get_mfcc_frame_centres(len_wave, sample_rate, fshift_seconds, analysis_window_length_seconds)
    else:
        sys.exit('Unknown value for windowing_convention: %s'%(windowing_convention))

    m,n = features.shape

    assert len(frame_centres) == m, 'Length of features (%s) provided not consistent with length of wave and windowing_convention (which gives %s)'%(m, len(frame_centres))
    
    resampled = np.zeros((len(pms), n))
    for dim in xrange(n):
        y = features[:,dim]
        

        interpolator = interp1d(frame_centres, y, kind=int_type, axis=0, \
                                    bounds_error=False, fill_value='extrapolate')
        interpolated = interpolator(pms)
        resampled[:,dim] = interpolated
        
#     print interpolated
#     pylab.scatter(pms, interpolated)
#     pylab.scatter(frame_centres, y, color='r')
#     pylab.xlim(40000,50000)
#     pylab.show()

    return resampled
    
def pitch_synchronous_resample(len_wave, sample_rate, fshift_seconds, pms, features, int_type='linear', windowing_convention='world', analysis_window_length_seconds=0.01):

    if windowing_convention == 'world':
        frame_centres = get_world_frame_centres(len_wave, sample_rate, fshift_seconds)
    elif windowing_convention == 'straight':
        frame_centres = get_straight_frame_centres(len_wave, sample_rate, fshift_seconds)        
    elif windowing_convention == 'snack':
        frame_centres = get_snack_frame_centres(len_wave, sample_rate, fshift_seconds)        
    elif windowing_convention == 'HTK':
        frame_centres = get_mfcc_frame_centres(len_wave, sample_rate, fshift_seconds, analysis_window_length_seconds)
    else:
        sys.exit('Unknown value for windowing_convention: %s'%(windowing_convention))

    m,n = features.shape

    if windowing_convention == 'straight':
        newlength = min(len(frame_centres), m)
        features = features[:newlength, :]
        frame_centres = frame_centres[:newlength]
        m,n = features.shape


    if not len(frame_centres) == m:
        print  'Length of features (%s) provided not consistent with length of wave and windowing_convention (which gives %s)'%(m, len(frame_centres))
        sys.exit() 

    interpolator = interp1d(frame_centres, features, kind=int_type, axis=0, \
                                bounds_error=False, fill_value='extrapolate')
    resampled = interpolator(pms)
        
#     print interpolated
#     pylab.scatter(pms, interpolated)
#     pylab.scatter(frame_centres, y, color='r')
#     pylab.xlim(40000,50000)
#     pylab.show()

    return resampled


def test():
    mfccs = glob.glob('/afs/inf.ed.ac.uk/user/o/owatts/sim2/oliver/slm_data_work/fls_hybrid/feat_29/world_reaper/mfcc/*.mfcc')
    for mfcc in mfccs:
        wavfile = mfcc.replace('.mfcc', '.wav').replace('/mfcc/','/tmp/') # '/afs/inf.ed.ac.uk/user/o/owatts/sim2/oliver/slm_data_work/fls_hybrid/feat_29/world_reaper/tmp/AMidsummerNightsDream_011_000.wav'
        wave, sample_rate = read_wave(wavfile)
        mf = get_speech(mfcc, 13, remove_htk_header=True)
        c = get_mfcc_frame_centres(len(wave), 48000, 0.002, 0.010)
        print c
        print len(c)
        print mf.shape       
            
if __name__=="__main__":
    #test()
    main_work()

