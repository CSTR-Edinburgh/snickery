#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import struct
from numpy import array, reshape, shape
import numpy as np
import numpy
import math
import scipy.interpolate
import wave



def write_wave(v, fname, rate, quiet=True):

    ## to avoid this when packing data:
       # struct.error: short format requires SHRT_MIN <= number <= SHRT_MAX
    ### .. enforce ceiling:
    SHRT_MIN = -32768
    SHRT_MAX = 32767
    if v.max() > SHRT_MAX or v.min() < SHRT_MIN:
        print 'Warning: write_wave squashed data to short range'
        v = np.maximum(np.minimum(v, SHRT_MAX), SHRT_MIN)

    v = list(v)
    packed = ''
    for i in range(len(v)):
        packed += wave.struct.pack('h',v[i]) # transform to binary
    f = wave.open(fname, "w")
    f.setparams((1, 2, rate, rate*4, 'NONE', 'noncompressed'))
    f.writeframesraw(packed)
    f.close()
    if not quiet:
        print 'Wrote wave file: %s'%(fname)

def read_wave(wavefile, enforce_mono=False):
    f = wave.open(wavefile)
    frames = f.readframes(f._nframes)
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = f.getparams()
    if enforce_mono:
        assert nchannels == 1
    v = numpy.array(struct.unpack("%sh"% f._nframes*f._nchannels,
              frames), dtype='float' ).flatten()
    f.close()
    return v, framerate
    
## http://stackoverflow.com/questions/6811183/rolling-window-for-1d-arrays-in-numpy
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return numpy.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    
    
    
    
def rate2fftlength(sample_rate):

    ## replicate GetFFTSizeForCheapTrick in src/cheaptrick.cpp:
    kLog2 = 0.69314718055994529  # set in src/world/constantnumbers.h 
    f0_floor = 71.0  ## set in analysis.cpp
    fftl = math.pow(2.0, (1.0 + int(math.log(3.0 * sample_rate / f0_floor + 1) / kLog2)))

    return fftl

def rate2worldapsize(rate):
    ## these are computed by World internally on the basis of sample rate
    ## and 2 constants set in src/world/constantnumbers.h 
    kFrequencyInterval = 3000.0
    kUpperLimit = 15000.0  
    apsize = int(min(kUpperLimit, (( rate / 2.0) - kFrequencyInterval)) / kFrequencyInterval)
    return apsize

def extract_portion_and_write(infile, outfile, old_dim, new_dim_start, new_dim_width, remove_htk_header):

    new_data = extract_portion(infile, old_dim, new_dim_start, new_dim_width, remove_htk_header)
    put_speech(new_data, outfile)

def extract_portion(infile, old_dim, new_dim_start, new_dim_width, remove_htk_header):

    assert new_dim_start >= 1
    new_dim_start -= 1
    new_dim_stop = new_dim_start + new_dim_width

    data = get_speech(infile, old_dim, remove_htk_header=remove_htk_header)
    new_data = data[:, new_dim_start : new_dim_stop]
    
    return new_data


def get_speech_OLD(infile, dim, remove_htk_header=False):

    data = read_floats(infile)
    if remove_htk_header:
        data = data[3:]  ## 3 floats correspond to 12 byte htk header

    assert len(data) % float(dim) == 0,"%s -- Bad dimension: %s!"%(infile, dim)
    m = len(data) / dim
    data = array(data).reshape((m,dim))
    return data

def get_speech(infile, dim, remove_htk_header=False):

    f = open(infile, 'rb')
    data = np.fromfile(f, dtype=np.float32)
    f.close()
    if remove_htk_header:
        data = data[3:]  ## 3 floats correspond to 12 byte htk header
    assert data.size % float(dim) == 0.0,'specified dimension %s not compatible with data'%(dim)
    data = data.reshape((-1, dim))
    return data

def put_speech(data, outfile):
    m,n = shape(data)
    size = m*n
    flat_data = list(data.reshape((size, 1)))
    write_floats(flat_data, outfile)

def write_floats(data, outfile):
    m = len(data)             
    format = str(m)+"f"

    packed = struct.pack(format, *data)
    f = open(outfile, "w")
    f.write(packed)
    f.close()

def read_floats(infile):
    f = open(infile, "r")
    l = os.stat(infile)[6]  # length in bytes
    data = f.read(l)        # = read until bytes run out (l)
    f.close()

    m = l / 4               
    format = str(m)+"f"

    unpacked = struct.unpack(format, data)
    unpacked = list(unpacked)
    return unpacked

def read_shorts(infile):
    f = open(infile, "r")
    l = os.stat(infile)[6]  # length in bytes
    data = f.read(l)        # = read until bytes run out (l)
    f.close()

    m = l / 2               
    format = str(m)+"h"

    unpacked = struct.unpack(format, data)
    unpacked = list(unpacked)
    return unpacked
    
    
    
## adapted from zhizheng:-
### interpolate F0, if F0 has already been interpolated, nothing will be changed after passing this function
def interpolate_f0(data):
    
    #data = numpy.reshape(data, (datasize, 1))
    datasize,n = np.shape(data)
    
    vuv_vector = numpy.zeros((datasize, 1))
    vuv_vector[data > 0.0] = 1.0
    vuv_vector[data <= 0.0] = 0.0        

    ip_data = data        

    frame_number = datasize
    last_value = 0.0
    for i in xrange(frame_number):
        if data[i] <= 0.0:
            j = i+1
            for j in range(i+1, frame_number):
                if data[j] > 0.0:
                    break
            if j < frame_number-1:
                if last_value > 0.0:
                    step = (data[j] - data[i-1]) / float(j - i)
                    for k in range(i, j):
                        ip_data[k] = data[i-1] + step * (k - i + 1)
                else:
                    for k in range(i, j):
                        ip_data[k] = data[j]
            else:
                for k in range(i, frame_number):
                    ip_data[k] = last_value
        else:
            ip_data[i] = data[i]
            last_value = data[i]

    return  ip_data, vuv_vector   
    
    
def split_cmp(cmp, stream_dims):
    split_data = []
    start = 0
    m,n = np.shape(cmp)
    for width in stream_dims:
        end = start + width
        stream_data = cmp[:,start:end]
        if width == 1:
            stream_data=stream_data.reshape((m,1))
        split_data.append(stream_data)        
        start = end
    return split_data        


def weight(speech, weight_vec):
    m,n = np.shape(speech)        
    weight_vec = np.array(weight_vec).reshape((1,-1)) 
    speech = speech * weight_vec
    return speech

def deweight(speech, weight_vec):
    m,n = np.shape(speech)
    weight_vec = np.array(weight_vec).reshape((1,-1)) 
    speech = speech / weight_vec
    return speech
            
        
def lin_interp_f0(fz):

    y = fz.flatten()

    voiced_ix = np.where( y > 0.0 )[0]  ## equiv to np.nonzero(y)    
    voicing_flag = np.zeros(y.shape)
    voicing_flag[voiced_ix] = 1.0
    
    ## linear interp voiced:
    if voiced_ix.shape[0] == 0:
        v_interpolated = fz
    else:    
        interpolator = scipy.interpolate.interp1d(voiced_ix, y[voiced_ix], kind='linear', axis=0, \
                                bounds_error=False, fill_value='extrapolate')
        v_interpolated = interpolator(np.arange(y.shape[0]))

    if 0:
        import pylab
        pylab.plot(y)
        pylab.plot(v_interpolated)
        pylab.show()
        sys.exit('wvw9e8h98whvw')
    return (v_interpolated.reshape((-1,1)), voicing_flag.reshape((-1,1)))

