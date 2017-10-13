#!/usr/bin/env python
# -*- coding: utf-8 -*-
## Project:  
## Author: Oliver Watts - owatts@staffmail.ed.ac.uk

import sys
import os
import glob
import math
import tempfile
import struct   
import re
import copy
import h5py
from argparse import ArgumentParser

import numpy 
import numpy as np
# import pylab
import numpy as np
#import scipy.signal

from segmentaxis import segment_axis
from speech_manip import get_speech
from label_manip import extract_quinphone
from util import splice_data, unsplice, safe_makedir
from const import label_delimiter, vuv_stream_names, label_length_diff_tolerance


def main_work(config, overwrite_existing_data=False):
    
    ## (temporary) assertions:-
    assert config['standardise_target_data'] == True
    
    database_fname = get_data_dump_name(config)

    if os.path.isfile(database_fname):
        if not overwrite_existing_data:
            sys.exit('Data already exists at %s -- run with -X to overwrite it'%(database_fname))
        else:
            os.system('rm '+database_fname)
            
    target_feat_dir = config['target_datadir']
    join_feat_dir = config['join_datadir']

    datadims_join = config['datadims_join']
    datadims_target = config['datadims_target']
    n_train_utts = config['n_train_utts']
    

    stream_list_join = config['stream_list_join']    
    stream_list_target = config['stream_list_target'] 
    
    ## check feature directories exist
    for stream in stream_list_target:
        stream_dir = os.path.join(target_feat_dir, stream)
        assert os.path.isdir(stream_dir), 'Directory %s not accessible'%(stream_dir)
    for stream in stream_list_join:        
        stream_dir = os.path.join(join_feat_dir, stream)
        assert os.path.isdir(stream_dir), 'Directory %s not accessible'%(stream_dir)
    

    ## work out initial list of training utterances based on files present in first stream subdir: 
    first_stream = stream_list_target[0] ## <-- typically, mgc
    utt_list = sorted(glob.glob(os.path.join(target_feat_dir, first_stream)+'/*.' + first_stream))
    flist = [os.path.split(fname)[-1].replace('.'+first_stream,'') for fname in utt_list]
    
    if type(n_train_utts) == int:
        if (n_train_utts == 0 or n_train_utts > len(flist)):
            n_train_utts = len(flist)
        flist = flist[:n_train_utts]
    elif type(n_train_utts) == str:
        match_expression = n_train_utts
        flist = [name for name in flist if match_expression in name]
        print 'Selected %s utts with pattern %s'%(len(flist), match_expression)
        
    ## also filter for test material, in case they are in same directory:
    test_flist = []
    for fname in test_flist:
        for pattern in self.config['test_patterns']:
            if pattern in fname:
                test_flist.append(fname)
    flist = [name for name in flist if name not in test_flist]

    assert len(flist) > 0    


    ## 1A) First pass: get mean and std per stream for each of {target,join}
    (mean_vec_target, std_vec_target) = get_mean_std(target_feat_dir, stream_list_target, datadims_target, flist)
    (mean_vec_join, std_vec_join) = get_mean_std(join_feat_dir, stream_list_join, datadims_join, flist)




    ## 1B) Initialise HDF5; store mean and std in HDF5: 

    f = h5py.File(database_fname, "w")

    mean_target_dset = f.create_dataset("mean_target", np.shape(mean_vec_target), dtype='f')
    std_target_dset = f.create_dataset("std_target", np.shape(std_vec_target), dtype='f')

    mean_join_dset = f.create_dataset("mean_join", np.shape(mean_vec_join), dtype='f')
    std_join_dset = f.create_dataset("std_join", np.shape(std_vec_join), dtype='f')

    mean_target_dset[:] = mean_vec_target[:]
    std_target_dset[:] = std_vec_target[:]
    mean_join_dset[:] = mean_vec_join[:]
    std_join_dset[:] = std_vec_join[:]            
    
    
    ## Set some values....
    
    target_dim = mean_vec_target.shape[0]
    join_dim = mean_vec_join.shape[0]

    fshift_seconds = (0.001 * config['frameshift_ms'])
    fshift = int(config['sample_rate'] * fshift_seconds)    
    samples_per_frame = fshift
    
    
    ## go through data to find number of units:-
    n_units = 0
    for base in flist:
        labfile = os.path.join(config['label_datadir'], base + '.' + config['lab_extension'])
        n_states = len(read_label(labfile, config['quinphone_regex']))
        assert n_states % 5 == 0
        n_halfphones = (n_states / 5) * 2
        n_units += n_halfphones

    print '%s halfphones'%(n_units)
    
    ## 2) get ready to store data in HDF5:
    ## maxshape makes a dataset resizable
    train_dset = f.create_dataset("train_unit_features", (n_units, 2*target_dim), maxshape=(n_units, 2*target_dim), dtype='f') 
    phones_dset = f.create_dataset("train_unit_names", (n_units,), maxshape=(n_units,), dtype='|S50') 
    filenames_dset = f.create_dataset("filenames", (n_units,), maxshape=(n_units,), dtype='|S50') 
    cutpoints_dset = f.create_dataset("cutpoints", (n_units,2), maxshape=(n_units,2), dtype='i') 

    # TODO: hardcoded for natural2
    join_contexts_dset = f.create_dataset("join_contexts", (n_units + 1, 2*join_dim), maxshape=(n_units + 1, 2*join_dim), dtype='f') 

    ## Optionally dump some extra data which can be used for training a better join cost:-
    if config['dump_join_data']:
        join_database_fname = get_data_dump_name(config, joindata=True)
        fjoin = h5py.File(join_database_fname, "w")
        halfwin = config['join_cost_halfwidth']
        start_join_feats_dset = fjoin.create_dataset("start_join_feats", (n_units, halfwin*join_dim), maxshape=(n_units, halfwin*join_dim), dtype='f') 
        end_join_feats_dset = fjoin.create_dataset("end_join_feats", (n_units, halfwin*join_dim), maxshape=(n_units, halfwin*join_dim), dtype='f') 


    ## Standardise data (within streams), compose, add VUV, fill F0 gaps with utterance mean voiced value: 
    start = 0

    print 'Composing ....'
    print flist
    new_flist = []
    for base in flist:

        print base    
        wname = os.path.join(config['wav_datadir'], base + '.wav')
        pm_file = os.path.join(config['pm_datadir'], base + '.pm')
        labfile = os.path.join(config['label_datadir'], base + '.' + config['lab_extension'])
        if not (os.path.isfile(wname) or os.path.isfile(pm_file)):
            print 'Warning: no wave or pm -- skip!'
            continue
                   

        ### Get speech params for target cost (i.e. probably re-generated speech for consistency):
        t_speech = compose_speech(target_feat_dir, base, stream_list_target, datadims_target) 
        if t_speech.shape == [1,1]:  ## bad return value  
            continue                    
        if config['standardise_target_data']:
            t_speech = standardise(t_speech, mean_vec_target, std_vec_target)


        ### Get speech params for join cost (i.e. probably natural speech):
        j_speech = compose_speech(join_feat_dir, base, stream_list_join, datadims_join)
        if j_speech.shape == [1,1]:  ## bad return value  
            continue                    
        if config['standardise_join_data']:
            j_speech = standardise(j_speech, mean_vec_join, std_vec_join)

        ### get labels:
        labs = read_label(labfile, config['quinphone_regex'])
        label_frames = labs[-1][0][1]

        ## Has silence been trimmed from either t_speech or j_speech?
        if config.get('untrim_silence_join_speech', False):
            print 'Add trimmed silence back to join cost speech features'
            j_speech = reinsert_terminal_silence(j_speech, labs)

        if config.get('untrim_silence_target_speech', False):
            print 'Add trimmed silence back to target cost speech features'
            t_speech = reinsert_terminal_silence(t_speech, labs)


        # ### TODO: Length of T and J does not quite match here :-(  need to debug.
        # print 'T'
        # print t_speech.shape
        # print 'J'
        # print j_speech.shape
        # print 'L'
        # print label_frames
     
        ## Pad or trim speech to match the length of the labels (within a certain tolerance):-
        t_speech = pad_speech_to_length(t_speech, labs)
        j_speech = pad_speech_to_length(j_speech, labs)

        ## Discard sentences where length of speech and labels differs too much:- 
        if t_speech.size==1:
            print 'Skip utterance'
            continue
        if j_speech.size==1:
            print 'Skip utterance'            
            continue

        ## Get representations of half phones to use in target cost:-
        unit_names, unit_features, timings = get_halfphone_stats(t_speech, labs)

        if config['dump_join_data']:
            start_join_feats, end_join_feats = get_join_data_AL(j_speech, timings, config['join_cost_halfwidth'])

        ## Get pitchmarks (to join halfphones on detected GCIs):-
        pms_seconds = read_pm(pm_file)
        if pms_seconds.shape == (1,1):
            print 'Warning: trouble reading pm file -- skip!'
            continue                
        cutpoints = get_cutpoints(timings, pms_seconds)


        context_data = get_contexts_for_natural_joincost(j_speech, timings, width=2)

        filenames = [base] * len(cutpoints)

        m,n = unit_features.shape
        o,p = context_data.shape
        assert o == m+1

        ## Add everything to database:
        train_dset[start:start+m, :] = unit_features
        phones_dset[start:start+m] = unit_names
        filenames_dset[start:start+m] = filenames
        cutpoints_dset[start:start+m,:] = cutpoints

        ## cut off last join context... (kind of messy)
        join_contexts_dset[start:start+m, :] = context_data[:-1,:]

        if config['dump_join_data']:
            start_join_feats_dset[start:start+m, :] = start_join_feats
            end_join_feats_dset[start:start+m, :] = end_join_feats            


        start += m        
        new_flist.append(base)

    ## add database final join context back on (kind of messy)
    join_contexts_dset[m, :] = context_data[-1,:]

    ## Number of units was computed before without considering dropped utterances, actual number
    ## will be smaller. Resize the data:
    actual_nframes = start
    print '\n\n\nNumber of units actually written:'
    print actual_nframes
    print 

    train_dset.resize(actual_nframes, axis=0)
    phones_dset.resize(actual_nframes, axis=0)
    filenames_dset.resize(actual_nframes, axis=0)
    cutpoints_dset.resize(actual_nframes, axis=0)

    join_contexts_dset.resize(actual_nframes+1, axis=0)

    print 
    print 'Storing hybrid voice data:'
    for thing in f.values():
        print thing
    f.close()
    
    print 'Stored training data for %s sentences to %s'%(n_train_utts, database_fname)
       
    if config['dump_join_data']:       
        start_join_feats_dset.resize(actual_nframes, axis=0)
        end_join_feats_dset.resize(actual_nframes, axis=0)
        print 
        print 'Storing data for learning join cost:'
        for thing in fjoin.values():
            print thing
        fjoin.close()


def reinsert_terminal_silence(speech, labels, silence_symbols=['#']):
    initial_silence_end = 0
    final_silence_start = -1
    for ((s,e), quinphone) in labels:
        if quinphone[2] in silence_symbols:
            initial_silence_end = e
        else:
            break
    for ((s,e), quinphone) in reversed(labels):
        if quinphone[2] in silence_symbols:
            final_silence_start = s
        else:
            break
    m,n = speech.shape
    label_frames = labels[-1][0][1]
    end_sil_length = label_frames - final_silence_start
    start_sil_length = initial_silence_end

    padded_speech = numpy.vstack([numpy.zeros((start_sil_length, n)) , speech , numpy.zeros((end_sil_length, n))])


    # padded_speech = numpy.zeros((label_frames, n))
    # print speech.shape
    # print padded_speech.shape
    # print initial_silence_end, final_silence_start
    # print padded_speech[initial_silence_end:final_silence_start, :].shape
    # padded_speech[initial_silence_end:final_silence_start, :] = speech
    return padded_speech


def get_mean(flist, dim, exclude_uv=False):
    '''
    '''
    frame_sum = np.zeros(dim)
    frame_count = 0
    for fname in flist:
        if not os.path.isfile(fname):
            continue    
        print 'mean: ' + fname
        
        speech = get_speech(fname, dim)
        if np.sum(np.isnan(speech)) + np.sum(np.isinf(speech)) > 0:
            print 'EXCLUDE ' + fname
            continue
        
        if exclude_uv:
            ## remove speech where first column is <= 0.0
            speech = speech[speech[:,0]>0.0, :]
        
        frame_sum += speech.sum(axis=0)
        m,n = np.shape(speech)
        frame_count += m
        
    mean_vec = frame_sum / float(frame_count)
    return mean_vec, frame_count
    
def get_std(flist, dim, mean_vec, exclude_uv=False):
    diff_sum = np.zeros(dim)
    frame_count = 0    
    for fname in flist:
        if not os.path.isfile(fname):
            continue
        print 'std: ' + fname
        
        speech = get_speech(fname, dim)
        if np.sum(np.isnan(speech)) + np.sum(np.isinf(speech)) > 0:
            print 'EXCLUDE ' + fname
            continue

        if exclude_uv:
            ## remove speech where first column is <= 0.0
            speech = speech[speech[:,0]>0.0, :]
                                
        m,n = np.shape(speech)
        mean_mat = np.tile(mean_vec,(m,1))
        sq_diffs = (speech - mean_mat) ** 2
        diff_sum += sq_diffs.sum(axis=0)
        frame_count += m
    std_vec = (diff_sum / float(frame_count)) ** 0.5
    return std_vec
    
def standardise(speech, mean_vec, std_vec):

    m,n = np.shape(speech)
        
    ## TODO: switch to broadcasting here
    mean_mat = np.tile(mean_vec,(m,1))
    std_mat = np.tile(std_vec,(m,1))
    
    ## standardise:-
    speech = (speech - mean_mat) / std_mat
    
    ## leave weighting till later!
    return speech

def destandardise(speech, mean_vec, std_vec):

    m,n = np.shape(speech)
        
    mean_mat = np.tile(mean_vec,(m,1))
    std_mat = np.tile(std_vec,(m,1))
    #weight_mat = np.tile(weight_vec,(m,1))
    
    ## standardise:-
    speech = (speech * std_mat) + mean_mat
    
    ## leave weighting till later!
    # speech = speech * weight_mat
    return speech
    


DODEBUG = False
def debug(msg):
    if DODEBUG:
        print msg
    
def compose_speech(indir, base, stream_list, datadims):
    '''
    where there is trouble, signal this by returning a 1 x 1 matrix
    '''

    mgc_fn = os.path.join(indir, 'mgc', base+'.mgc' ) 
    f0_fn = os.path.join(indir, 'f0', base+'.f0' ) 
    ap_fn = os.path.join(indir, 'ap', base+'.ap' ) 

    stream_data_list = []
    for stream in stream_list:
        stream_fname = os.path.join(indir, stream, base+'.'+stream ) 
        if not os.path.isfile(stream_fname):
            return np.zeros((1,1))
        stream_data = get_speech(stream_fname, datadims[stream])
        
        if stream in vuv_stream_names:
            uv_ix = np.arange(stream_data.shape[0])[stream_data[:,0]<=0.0]
            vuv = np.ones(stream_data.shape)
            vuv[uv_ix, :] = 0.0
            ## set F0 to utterance's voiced frame mean in unvoiced frames:   
            voiced_mean = stream_data[stream_data>0.0].mean()
            stream_data[stream_data<=0.0] = voiced_mean 
            stream_data_list.append(stream_data)
            stream_data_list.append(vuv)
        else:
            stream_data_list.append(stream_data)

    ## where data has different number of frames per stream, chop off the extra frames:
    frames = [np.shape(data)[0] for data in stream_data_list]
    nframe = min(frames)
    stream_data_list = [data[:nframe,:] for data in stream_data_list]
    
    speech = np.hstack(stream_data_list)

    return speech




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
  
          
def get_data_dump_name(config, joindata=False, joinsql=False):
    safe_makedir(os.path.join(config['workdir'], 'data_dumps'))
    condition = make_train_condition_name(config)
    assert not (joindata and joinsql)
    if joindata:
        last_part = '.joindata.hdf5'
    elif joinsql:
        last_part = '.joindata.sql'
    else:
        last_part = '.hdf5'
    database_fname = os.path.join(config['workdir'], "data_dumps", condition + last_part)
    return database_fname

def make_train_condition_name(config):
    '''
    condition name including any important hyperparams
    '''
    return '%s_utts'%(config['n_train_utts'])


def read_label(labfile, quinphone_regex):
    f = open(labfile, 'r')
    lines = f.readlines()
    f.close()
    outlabel = []
    for line in lines:
        start,end,lab = re.split('\s+', line.strip(' \n'))[:3]
        quinphone = extract_quinphone(lab, quinphone_regex) # lab.split('/5:')[0] # (':')[0]
        state = lab.strip(']').split('[')[-1]
        newlab = list(quinphone) + [state]
        #for thing in newlab:
        #    assert label_delimiter not in thing, 'quinphone plus state (%s) contains label_delimiter (%s)'%(newlab, label_delimiter)
        #newlab = label_delimiter.join(newlab)
        start_frame = int(start) / 50000
        end_frame = (int(end) / 50000)     ## TODO: de-hardcode frameshift

        #length_in_frames = (int(end) - int(start)) / 50000
        #print length_in_frames
        outlabel.append(((start_frame, end_frame), newlab))
    return outlabel


def get_cutpoints(timings, pms):
    '''
    Find GCIs which are nearest to the start and end of each unit
    '''
    cutpoints = []
    for (start, end) in timings:
        start_sec = start * 0.005  # TODO: unhardcode frameshift and rate
        end_sec = (end) * 0.005
        start_closest_ix = numpy.argmin(numpy.abs(pms - start_sec))
        end_closest_ix = numpy.argmin(numpy.abs(pms - end_sec))
        cutpoints.append((pms[start_closest_ix], pms[end_closest_ix]))
    cutpoints = np.array(cutpoints)
    cutpoints *= 48000
    cutpoints = np.array(cutpoints, dtype=int)
    return cutpoints


def get_halfphone_stats(speech, labels):
    '''
    Arbitrarily assign states 1 & 2 to halfphone 1, and states 3, 4 and 5 to halfphone 2.
    Characterise a halfphone by the first and last frames appearing in it.

    Where there are N hafphones in an utt, return (names, features, timings) where
        -- names is N-element array like (array(['xx~xx-#_L+p=l', 'xx~xx-#_R+p=l', 'xx~#-p_L+l=i', ...
        -- timings is N-element list like [(0, 40), (41, 60), (61, 62), ...
        -- features is N x D array, where D is size of feature vector
    '''
    
    m,dim = speech.shape

    assert len(labels) % 5 == 0, 'There must be 5 states for each phone in label'
    nphones = len(labels) / 5 
    features = numpy.zeros((nphones*2, dim*2))
    names = []
    starts = []
    ends = []
    halfphone_counter = 0
    for ((s,e),lab) in labels:
        #print ((s,e),lab)
        if e > m-1:
            e = m-1
        assert len(lab) == 6

        #quinphone_plus_state = lab.split(label_delimiter)  # lab.split('_')[0]
        quinphone = lab[:5]
        state = lab[-1]

        debug( '%s/%s halfphones' %(halfphone_counter, nphones*2) )
        debug( 's,e: %s %s'%(s,e) )
        if state == '2':
            halfphone_name = copy.copy(quinphone)
            halfphone_name[2] += '_L'
            assert label_delimiter not in ''.join(halfphone_name), 'delimiter %s occurs in one or more name element (%s)'%(label_delimiter, halfphone_name)
            halfphone_name = label_delimiter.join(halfphone_name)
            names.append(halfphone_name)
            #names.append(quinphone.replace('+', '_L+'))
            features[halfphone_counter, :dim] = speech[s,:]
            starts.append(s)
        elif state == '3':
            features[halfphone_counter, dim:] = speech[e,:]  
            ends.append(e)
            halfphone_counter += 1                     
        elif state == '4':
            halfphone_name = copy.copy(quinphone)
            halfphone_name[2] += '_R'
            assert label_delimiter not in ''.join(halfphone_name), 'delimiter %s occurs in one or more name element (%s)'%(label_delimiter, halfphone_name)
            halfphone_name = label_delimiter.join(halfphone_name)
            names.append(halfphone_name)
            features[halfphone_counter, :dim] = speech[s,:]    
            starts.append(s)  
        elif state == '5':
            pass                      
        elif state == '6':
            features[halfphone_counter, dim:] = speech[e,:]   
            ends.append(e)        
            halfphone_counter += 1  
        else:
            sys.exit('bad state number')
                    
    assert len(names) == nphones*2 == len(starts) == len(ends)
    names = np.array(names)
    timings = zip(starts,ends)
    return (names, features, timings)



def get_contexts_for_natural_joincost(speech, timings, width=2):
    '''
    TODO: defaults to natural2
    timings: start and end frame indices: [(0, 2), (2, 282), (282, 292), (292, 297), (297, 302)]
    Where timings gives times for n units, return (n+1 x dim) matrix, each row of which 
    gives a 'join context' for the end of a unit. Row p gives the start join context for 
    unit p, and the end join context for unit p-1. Explain 'natural join' ...

    ''' 
    assert width % 2 == 0, 'context width for natural joincost should be even valued'

    label_frames = timings[-1][1]
    speech_frames, dim = speech.shape

    ## Note: small mismatches happen a lot with Blizzard STRAIGHT data, but never with world data prepared by Oliver
    ## Where they occur, use zero padding:
    ### ===== This should no longer be necessary (done in a previous step) =====
    # if label_frames > speech_frames:
    #     padding_length = label_frames - speech_frames
    #     speech = np.vstack([speech, np.zeros((padding_length, dim))])

    ## starting parts
    last_end = timings[-1][1]
    starts = np.array([s for (s,e) in timings] + [last_end])

    ### reduplicate start and end frames -- assuming silence at end of utts, this gives a reasonable context
    halfwidth = width / 2
    prepadding = numpy.tile(speech[0,:], (halfwidth, 1))
    postpadding = numpy.tile(speech[-1,:], (halfwidth, 1))    
    speech = numpy.vstack([prepadding, speech, postpadding])

    frames = segment_axis(speech, width, overlap=width-1, axis=0)
    context_frames = frames[starts,:,:]

    ## flatten the last 2 dimensions of the data:--
    context_frames = context_frames.reshape((-1, width*dim))

    return context_frames



def get_join_data_AL(speech, timings, halfwidth):
    '''
    Output of this operation is not yet used -- it will be used for actively learned join cost
    '''

    #print speech
    #print timings
    # pylab.plot(speech)
    # pylab.show()

    ## starting parts
    starts = [s for (s,e) in timings]
    ## do we need to pad the end of the speech?
    m,n = speech.shape
    ##print 'N'
    ##print n
    if max(starts) + halfwidth > m:
        diff = (max(starts) + halfwidth) - m
        start_speech = np.vstack([speech, np.zeros((diff, n))])
        debug('correct start')
    else:
        start_speech = speech
    #print start_speech.shape
    frames = segment_axis(start_speech, halfwidth, overlap=halfwidth-1, axis=0)
    #print frames.shape
    start_frames = frames[starts,:,:]
    #print start_frames.shape

    ends = np.array([e for (s,e) in timings])
    ## do we need to pad the start of the speech?
    if min(ends) - halfwidth < 0:
        diff = 0 - (min(ends) - halfwidth)
        end_speech = np.vstack([np.zeros((diff, n)), speech])
        ends += diff
        debug('correct end')
    else:
        end_speech = speech
    ends -=  halfwidth ###  to get starting point of end segments
    frames = segment_axis(end_speech, halfwidth, overlap=halfwidth-1, axis=0)
    #print frames.shape
    end_frames = frames[ends,:,:]
    
    ## flatten the last 2 dimensions of the data:--
    #print start_frames.shape
    #print halfwidth, n
    start_frames = start_frames.reshape((-1, halfwidth*n))
    end_frames = end_frames.reshape((-1, halfwidth*n))

    return (start_frames, end_frames)



def get_mean_std(feat_dir, stream_list, datadims, flist):

    means = {}
    stds = {}
    for stream in stream_list:
        stream_files = [os.path.join(feat_dir, stream, base+'.'+stream) for base in flist]
        if stream in vuv_stream_names:
            means[stream], _ = get_mean(stream_files, datadims[stream], exclude_uv=True)
            stds[stream] = get_std(stream_files, datadims[stream], means[stream], exclude_uv=True)
        else:
            means[stream], nframe = get_mean(stream_files, datadims[stream])
            stds[stream] = get_std(stream_files, datadims[stream], means[stream])


    mean_vec = []
    for stream in stream_list:
        mean_vec.append(means[stream])
        if stream in vuv_stream_names: ## add fake stats for VUV which will leave values unaffected
            mean_vec.append(numpy.zeros(means[stream].shape))

    std_vec = []
    for stream in stream_list:
        std_vec.append(stds[stream])
        if stream in vuv_stream_names: ## add fake stats for VUV which will leave values unaffected
            std_vec.append(numpy.ones(stds[stream].shape))

    mean_vec = np.hstack(mean_vec)
    std_vec = np.hstack(std_vec)

    return mean_vec, std_vec    


def pad_speech_to_length(speech, labels):
    '''
    Small mismatches happen a lot with Blizzard STRAIGHT data, so need some hacks to handle them.
    This is rarely/never an issue with world data and labels prepared by Ossian
    '''
    m,dim = speech.shape
    nframe = labels[-1][0][1]

    if math.fabs(nframe - m) > label_length_diff_tolerance:
        print 'Warning: number frames in target cost speech and label do not match (%s vs %s)'%(m, nframe)
        return numpy.array([[0.0]])

    ## Note: small mismatches happen a lot with Blizzard STRAIGHT data, but never with world data prepared by Oliver
    ## Where they occur, use zero padding:

    if nframe > m:
        padding_length = nframe - m
        speech = np.vstack([speech, np.zeros((padding_length, dim))])

    elif nframe < m:
        speech = speech[:nframe,:]

    return speech


if __name__ == '__main__':



    #################################################
      
    # ======== process command line ==========

    a = ArgumentParser()
    a.add_argument('-c', dest='config_fname', required=True)
    a.add_argument('-X', dest='overwrite_existing_data', action='store_true', \
                    help= "clear any previous training data first")
    opts = a.parse_args()
    
    config = {}
    execfile(opts.config_fname, config)
    del config['__builtins__']
    print config
    main_work(config, overwrite_existing_data=opts.overwrite_existing_data)
    
    
    
    