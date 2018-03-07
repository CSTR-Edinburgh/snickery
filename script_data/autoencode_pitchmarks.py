#!/usr/bin/env python
# -*- coding: utf-8 -*-
## Project: ... - February 2017 - ...
## Contact: Oliver Watts - owatts@staffmail.ed.ac.uk
  
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



from keras_train import train_network_from_generators

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
  

def get_pm_wavefrag(pm_trip, wave, pad_length):
    '''
    New in PS2: don't do windowing during training
    '''
    first, middle, last = pm_trip
    length = (last - first)
    left_length = middle - first
    right_length = last - middle 
    
    ## new -- trim wave to fit in pad length
    halfwidth = pad_length / 2
    if left_length > halfwidth:
        # print '      (adjust left)'
        diff = left_length - halfwidth
        first = first + diff
        left_length = halfwidth
    if right_length > halfwidth:
        # print '      (adjust right)'
        diff = right_length - halfwidth
        last = last - diff
        right_length = halfwidth

    nlength = last - first
    #print nlength, pad_length
    assert nlength <= pad_length

    #print [first, last]
    frag = wave[first:last]   
    #print frag
    pad_frag = np.zeros(pad_length)
    extra = int((pad_length/2) - left_length) 
    if extra+nlength > pad_length:
        sys.exit('wave fragment plus padding (%s) longer than pad length (%s)'%(extra+nlength, pad_length))
    if False: ## debug
        print pm_trip
        print nlength
        print frag.shape
        print pad_frag[extra:extra+nlength].shape
        print pad_frag.shape
        print extra
        print extra+nlength
        print '--'
    pad_frag[extra:extra+nlength] = frag

    ### scale with symmetric hanning:
    win = np.concatenate([np.hanning(left_length*2)[:left_length], np.hanning(right_length*2)[right_length:]])
    win_frag = np.zeros(pad_length)
    win_frag[extra:extra+nlength] = win
    pad_frag *= win_frag

    return pad_frag
          


class PitchPeriodProvider(DataProvider):

    def get_number_points_in_one_file(self):
        '''
        This will work OK, but can be overridden with something more efficient
        which just needs to work out data size without returning it.
        '''
        (wave_file, pitch_mark_file) = self.filelist[self.file_index]
        pitchmarks = read_pm(pitch_mark_file) 
        if pitchmarks.size==1:
            return 0
        else:
            print len(pitchmarks) - 2   
            return len(pitchmarks) - 2     


    def get_file_data_from_one_file(self):
        '''
        Here is most of the database specific stuff.
        This one should be provided by subclasses to manipulate data appropriately.
        '''
        #print '----> get_file_data_from_one_file (%s)'%(self.operation)
        (wave_file, pitch_mark_file) = self.filelist[self.file_index]
        pitchmarks = read_pm(pitch_mark_file) * 48000
        if pitchmarks.size==1:
            return (np.zeros((0,0)), np.zeros((0,0)))
        pitchmarks = np.array(pitchmarks, dtype='int')
        pitchmark_triples = segment_axis(pitchmarks, 3, 2, axis=0)
        wave, sample_rate = read_wave(wave_file)
        
        pad_length = 1000
        frags = [get_pm_wavefrag(pm_trip, wave, pad_length) for pm_trip in pitchmark_triples]
        frags = np.vstack(frags)
        
        ### do 0-1 range normalisation, asssume 16bit signed audio:
        data_range = math.pow(2,16)
        frags -=  (data_range / 2) * -1.0
        frags /= data_range

        return (frags, frags)





if __name__ == '__main__':



    #################################################
      
    # ======== process command line ==========

    a = ArgumentParser()
    a.add_argument('-w', dest='wave_dir', required=True)
    a.add_argument('-p', dest='pitchmark_dir', required=True)
    a.add_argument('-o', dest='output_dir', required=True)    
    a.add_argument('-d', dest='outdim', type=int, default=12)  
    a.add_argument('-N', dest='nfiles', type=int, default=0)        
    opts = a.parse_args()
    
    
    output_dir = opts.output_dir
    wav_datadir = opts.wave_dir
    pm_datadir = opts.pitchmark_dir

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if not os.path.isdir(output_dir + '/aef/'):
        os.makedirs(output_dir + '/aef/')        

    #  ./submit.sh ./autoencode_pitchmarks.py -w /afs/inf.ed.ac.uk/group/cstr/projects/blizzard_entries/blizzard2017/data/segmented/wav/ -p /afs/inf.ed.ac.uk/group/cstr/projects/nst/oliver/hybrid_work/data/fls_data/world_reaper/pm/ -o /afs/inf.ed.ac.uk/group/cstr/projects/nst/oliver/hybrid_work/data/fls_data/autoencoder_feats -d 50


    train_provider = PitchPeriodProvider([wav_datadir, pm_datadir], ['wav', 'pm'], batch_size=1024, partition_size=100000, \
                shuffle=True, limit_files=opts.nfiles)
    val_provider = PitchPeriodProvider([wav_datadir, pm_datadir], ['wav', 'pm'],  batch_size=1024, partition_size=100000, \
                shuffle=True, limit_files=opts.nfiles, validation=True)
    (X, Y) = train_provider.get_file_data_from_one_file()
    _, insize = X.shape
    outsize = insize



    encoder = train_network_from_generators(train_provider, val_provider, insize, outsize, \
           opts.output_dir + '/model.krs', architecture=[2048, 2048, opts.outdim, 2048, 2048], activation='relu', max_epoch=3, \
                    patience=5, classification=False, bottleneck=2, truncate_at_bottleneck=True)

    # encoder = truncate_model(autoencoder, 1)


    train_provider.reset()
    while train_provider.file_index < len(train_provider.filelist):
        X, Y = train_provider.get_file_data_from_one_file()
        base = train_provider.get_filename()
        if X.size == 0:
            print 'skip %s'%(base)
            continue
        encoded = encoder.predict(X)
        put_speech(encoded, opts.output_dir + '/aef/' + base + '.aef')
        train_provider.file_index += 1
    val_provider.reset()
    while val_provider.file_index < len(val_provider.filelist):
        X, Y = val_provider.get_file_data_from_one_file()
        base = val_provider.get_filename()
        if X.size == 0:
            print 'skip %s'%(base)
            continue        
        encoded = encoder.predict(X)
        put_speech(encoded, opts.output_dir + '/aef/' + base + '.aef')
        val_provider.file_index += 1

    #import pylab
    # pylab.subplot(211)
    # pylab.plot(X.transpose())
    # pylab.subplot(212)
    # pylab.plot(encoded.transpose())

    #pylab.plot(encoded)

    #pylab.show()

    #numpy.save('/tmp/tmp.data', encoded)





