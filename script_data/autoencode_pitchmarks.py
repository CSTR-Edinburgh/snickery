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
        data = self.get_file_data_from_one_file()
        return data[0].shape[0]

    def get_file_data_from_one_file(self):
        '''
        Here is most of the database specific stuff.
        This one should be provided by subclasses to manipulate data appropriately.
        '''
        print '----> get_file_data_from_one_file (%s)'%(self.operation)
        (wave_file, pitch_mark_file) = self.filelist[self.file_index]
        pitchmarks = read_pm(pitch_mark_file) * 48000
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


TOPDIR='/afs/inf.ed.ac.uk/group/cstr/projects/nst/oliver/hybrid_work/data/fls_data/'

### 48k, mono, 16-bit wav-headered audio:
wav_datadir = '/afs/inf.ed.ac.uk/group/cstr/projects/blizzard_entries/blizzard2017/data/segmented/wav/'

### Reaper pitchmarks:
pm_datadir = TOPDIR + '/world_reaper/pm/'





train_provider = PitchPeriodProvider([wav_datadir, pm_datadir], ['wav', 'pm'], batch_size=32, partition_size=1000, \
            shuffle=True, limit_files=5)
val_provider = PitchPeriodProvider([wav_datadir, pm_datadir], ['wav', 'pm'],  batch_size=32, partition_size=1000, \
            shuffle=True, limit_files=5, validation=True)
(X, Y) = train_provider.get_next_batch()

# import pylab
# pylab.plot(X.transpose())
# pylab.show()

train_provider.reset()
_, insize = X.shape
outsize = insize



encoder = train_network_from_generators(train_provider, val_provider, insize, outsize, \
       '/tmp/tmp.model', architecture=[512, 512, 10, 512, 512], activation='tanh', max_epoch=5, \
                patience=5, classification=False, bottleneck=2, truncate_at_bottleneck=True)

# encoder = truncate_model(autoencoder, 1)


train_provider.reset()
X, Y = train_provider.get_file_data_from_one_file()
print train_provider.get_filename()
encoded = encoder.predict(X)
print encoded

import pylab
# pylab.subplot(211)
# pylab.plot(X.transpose())
# pylab.subplot(212)
# pylab.plot(encoded.transpose())

pylab.plot(encoded)

pylab.show()

numpy.save('/tmp/tmp.data', encoded)