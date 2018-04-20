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
from speech_manip import get_speech, read_wave
from mulaw2 import lin2mu
from label_manip import extract_quinphone
from util import splice_data, unsplice, safe_makedir, readlist
import const
from const import label_delimiter, vuv_stream_names, label_length_diff_tolerance, target_rep_widths


import resample
import resample_labels
import varying_filter

NORMWAVE=False # False

def locate_stream_directories(directories, streams): 
    '''
    For each stream in streams, find a subdirectory for some directory in 
    directories, directory/stream. Make sure that there is only 1 such subdirectory
    named after the stream. Return dict mapping from stream names to directory locations. 
    '''
   
    stream_directories = {}
    for stream in streams:
        for directory in directories:
            candidate_dir = os.path.join(directory, stream)
            if os.path.isdir(candidate_dir):
                ## check unique:
                if stream in stream_directories:
                    sys.exit('Found at least 2 directories for stream %s: %s and %s'%(stream, stream_directories[stream], candidate_dir))
                stream_directories[stream] = candidate_dir
    ## check we found a location for each stream:
    for stream in streams:
        if stream not in stream_directories:
            sys.exit('No subdirectory found under %s for stream %s'%(','.join(directories), stream))
    return stream_directories




def main_work(config, overwrite_existing_data=False):
    
    ## (temporary) assertions:-
    config['standardise_target_data'] = True
    assert config['standardise_target_data'] == True
    
    config['joincost_features'] = True    ## want to use self. here, but no class defined...
    if config['target_representation'] == 'sample':
        config['joincost_features'] = False


    database_fname = get_data_dump_name(config)

    if os.path.isfile(database_fname):
        if not overwrite_existing_data:
            sys.exit('Data already exists at %s -- run with -X to overwrite it'%(database_fname))
        else:
            os.system('rm '+database_fname)
            

    n_train_utts = config['n_train_utts']

    target_feat_dirs = config['target_datadirs']
    datadims_target = config['datadims_target']
    stream_list_target = config['stream_list_target'] 
    ## get dicts mapping e.g. 'mgc': '/path/to/mgc/' : -
    target_stream_dirs = locate_stream_directories(target_feat_dirs, stream_list_target)
    
    if config['joincost_features']:
        join_feat_dirs = config['join_datadirs']
        datadims_join = config['datadims_join']
        stream_list_join = config['stream_list_join']    
        ## get dicts mapping e.g. 'mgc': '/path/to/mgc/' : -
        join_stream_dirs   = locate_stream_directories(join_feat_dirs, stream_list_join)
        
    
    
    

    # for stream in stream_list_target:
    #     stream_dir = os.path.join(target_feat_dir, stream)
    #     assert os.path.isdir(stream_dir), 'Directory %s not accessible'%(stream_dir)
    # for stream in stream_list_join:        
    #     stream_dir = os.path.join(join_feat_dir, stream)
    #     assert os.path.isdir(stream_dir), 'Directory %s not accessible'%(stream_dir)
    
    ## First, work out initial list of training utterances based on files present in first stream subdir: 
    first_stream = stream_list_target[0] ## <-- typically, mgc
    utt_list = sorted(glob.glob(target_stream_dirs[first_stream] +'/*.' + first_stream))
    flist = [os.path.split(fname)[-1].replace('.'+first_stream,'') for fname in utt_list]
    
    ## Next, limit training utterances by number or by pattern:
    if type(n_train_utts) == int:
        if (n_train_utts == 0 or n_train_utts > len(flist)):
            n_train_utts = len(flist)
        flist = flist[:n_train_utts]
    elif type(n_train_utts) == str:
        match_expression = n_train_utts
        flist = [name for name in flist if match_expression in name]
        print 'Selected %s utts with pattern %s'%(len(flist), match_expression)
        
    ## Also filter for test material, in case they are in same directory:
    test_flist = []
    for fname in flist:
        for pattern in config['test_patterns']:
            if pattern in fname:
                test_flist.append(fname)
    flist = [name for name in flist if name not in test_flist]


    ## Finally, only take utterances which occur in train_list, if it is given in config:
    if 'train_list' in config:
        assert os.path.isfile(config['train_list']), 'File %s does not exist'%(config['train_list'])
        train_list = readlist(config['train_list'])
        train_list = dict(zip(train_list, train_list))
        flist = [name for name in flist if name in train_list]


    assert len(flist) > 0    


    ## 1A) First pass: get mean and std per stream for each of {target,join}
    (mean_vec_target, std_vec_target) = get_mean_std(target_stream_dirs, stream_list_target, datadims_target, flist)
    if config['joincost_features']:  
        (mean_vec_join, std_vec_join) = get_mean_std(join_stream_dirs, stream_list_join, datadims_join, flist)


    ## Get std of (transformed) waveform if doing sample synthesis
    if config['target_representation'] == 'sample':
        wave_mu_sigma = get_wave_mean_std(config['wav_datadir'], flist, config['sample_rate'], nonlin_wave=config['nonlin_wave'])




    ## 1B) Initialise HDF5; store mean and std in HDF5: 

    f = h5py.File(database_fname, "w")

    mean_target_dset = f.create_dataset("mean_target", np.shape(mean_vec_target), dtype='f', track_times=False)
    std_target_dset = f.create_dataset("std_target", np.shape(std_vec_target), dtype='f', track_times=False)

    if config['joincost_features']:
        mean_join_dset = f.create_dataset("mean_join", np.shape(mean_vec_join), dtype='f', track_times=False)
        std_join_dset = f.create_dataset("std_join", np.shape(std_vec_join), dtype='f', track_times=False)

    mean_target_dset[:] = mean_vec_target[:]
    std_target_dset[:] = std_vec_target[:]

    if config['joincost_features']:    
        mean_join_dset[:] = mean_vec_join[:]
        std_join_dset[:] = std_vec_join[:]            
    
   

    ## Set some values....
    
    target_dim = mean_vec_target.shape[0]
    if config['joincost_features']:
        join_dim = mean_vec_join.shape[0]

    target_rep_size = target_dim * target_rep_widths[config.get('target_representation', 'twopoint')]

    fshift_seconds = (0.001 * config['frameshift_ms'])
    fshift = int(config['sample_rate'] * fshift_seconds)    
    samples_per_frame = fshift
 
    print 'go through data to find number of units:- '  
    
    n_units = 0


    if config['target_representation'] in ['epoch', 'sample']:
        new_flist = []
        print target_stream_dirs
        first_stream, first_streamdir = sorted(target_stream_dirs.items())[0]
        for base in flist:
            featfile = os.path.join(first_streamdir, base + '.' + first_stream)
            if not os.path.exists(featfile):
                print 'skipping %s'%(featfile)
                continue
            speech = get_speech(featfile, datadims_target[first_stream])
            npoint, _ = speech.shape
            n_units += npoint
            new_flist.append(base)
        flist = new_flist
    else:
        for base in flist:
            labfile = os.path.join(config['label_datadir'], base + '.' + config['lab_extension'])
            n_states = len(read_label(labfile, config['quinphone_regex']))
            assert n_states % 5 == 0
            n_halfphones = (n_states / 5) * 2
            n_units += n_halfphones

    if config['target_representation']  == 'sample': 
        n_units *= (config['sample_rate']*fshift_seconds)

    print '%s units (%s)'%(n_units,  config['target_representation'])
    
    ## 2) get ready to store data in HDF5:
    total_target_dim = target_rep_size
    if config['target_representation']  == 'sample': 
        total_target_dim = config['wave_context_length'] + target_rep_size

    ## maxshape makes a dataset resizable
    train_dset = f.create_dataset("train_unit_features", (n_units, total_target_dim), maxshape=(n_units, total_target_dim), dtype='f', track_times=False) 

    if config['target_representation']  == 'sample': 
        #wavecontext_dset = f.create_dataset("wavecontext", (n_units, config['wave_context_length']), maxshape=(n_units,config['wave_context_length']), dtype='i') 
        nextsample_dset = f.create_dataset("nextsample", (n_units, 1), maxshape=(n_units,1), dtype='f', track_times=False) 

    else:
        phones_dset = f.create_dataset("train_unit_names", (n_units,), maxshape=(n_units,), dtype='|S50', track_times=False) 
        filenames_dset = f.create_dataset("filenames", (n_units,), maxshape=(n_units,), dtype='|S50', track_times=False) 
        unit_index_within_sentence_dset = f.create_dataset("unit_index_within_sentence_dset", (n_units,), maxshape=(n_units,), dtype='i', track_times=False) 

        if config['target_representation'] == 'epoch':
            cutpoints_dset = f.create_dataset("cutpoints", (n_units,3), maxshape=(n_units,3), dtype='i', track_times=False) 
        else:
            cutpoints_dset = f.create_dataset("cutpoints", (n_units,2), maxshape=(n_units,2), dtype='i', track_times=False) 

        # hardcoded for pitch sync cost, unless epoch selection, in whcih case natural 2:
        if config['target_representation'] == 'epoch':
            join_dim *= 2

        join_contexts_dset = f.create_dataset("join_contexts", (n_units + 1, join_dim), maxshape=(n_units + 1, join_dim), dtype='f', track_times=False) 


    if config.get('store_full_magphase', False):
        mp_mag_dset = f.create_dataset("mp_mag", (n_units, 513), maxshape=(n_units, 513), dtype='f', track_times=False) 
        mp_imag_dset = f.create_dataset("mp_imag", (n_units, 513), maxshape=(n_units, 513), dtype='f', track_times=False) 
        mp_real_dset = f.create_dataset("mp_real", (n_units, 513), maxshape=(n_units, 513), dtype='f', track_times=False)   
        mp_fz_dset = f.create_dataset("mp_fz", (n_units, 1), maxshape=(n_units, 1), dtype='f', track_times=False)   


    ## Optionally dump some extra data which can be used for training a better join cost:-
    if config.get('dump_join_data', False):
        join_database_fname = get_data_dump_name(config, joindata=True)
        fjoin = h5py.File(join_database_fname, "w")
        halfwin = config['join_cost_halfwidth']
        start_join_feats_dset = fjoin.create_dataset("start_join_feats", (n_units, halfwin*join_dim), maxshape=(n_units, halfwin*join_dim), dtype='f', track_times=False) 
        end_join_feats_dset = fjoin.create_dataset("end_join_feats", (n_units, halfwin*join_dim), maxshape=(n_units, halfwin*join_dim), dtype='f', track_times=False) 


    ## Standardise data (within streams), compose, add VUV, fill F0 gaps with utterance mean voiced value: 
    start = 0



    print 'Composing ....'
    print flist
    new_flist = []
    for base in flist:

        print base    
        
        pm_file = os.path.join(config['pm_datadir'], base + '.pm')

        ## only actually need wave in sample case:-
        if config['target_representation'] == 'sample':
            wname = os.path.join(config['wav_datadir'], base + '.wav')
            if not os.path.isfile(wname):
                print 'Warning: no wave -- skip!'
                continue
                
        if not(os.path.isfile(pm_file)):
            print 'Warning: no pm -- skip!'
            continue


        ## Get pitchmarks (to join halfphones on detected GCIs):-
        pms_seconds = read_pm(pm_file)
        if pms_seconds.shape == (1,1):
            print 'Warning: trouble reading pm file -- skip!'
            continue                    

        ### Get speech params for target cost (i.e. probably re-generated speech for consistency):
        t_speech = compose_speech(target_stream_dirs, base, stream_list_target, datadims_target) 

        # print t_speech
        # print t_speech.shape
        # sys.exit('sedvsbvsfrb')
        if t_speech.shape == [1,1]:  ## bad return value  
            continue                    

        ### upsample before standardisation (inefficient, but standardisation rewrites uv values?? TODO: check this)
        if config['target_representation'] == 'sample':
            nframes, _ = t_speech.shape
            ### orignally:
            #len_wave = int(config['sample_rate'] * fshift_seconds * nframes)
            wavecontext, nextsample = get_waveform_fragments(wname, config['sample_rate'], config['wave_context_length'], nonlin_wave=config['nonlin_wave'], norm=wave_mu_sigma, wave_context_type=config.get('wave_context_type', 0))
            len_wave, _ = wavecontext.shape

            t_speech = resample.upsample(len_wave, config['sample_rate'], fshift_seconds, t_speech, f0_dim=-1, convention='world')
            if t_speech.size == 0:
                print 'Warning: trouble upsampling -- skip!'
                continue                    



        if config['standardise_target_data']:
            t_speech = standardise(t_speech, mean_vec_target, std_vec_target)


        if config['target_representation'] == 'sample':
            t_speech = np.hstack([wavecontext, t_speech])

            



        if config['joincost_features']:
            ### Get speech params for join cost (i.e. probably natural speech).
            ### These are now expected to have already been resampled so that they are pitch-synchronous. 
            j_speech = compose_speech(join_stream_dirs, base, stream_list_join, datadims_join)
            print 'j shape'
            print j_speech.shape
            if j_speech.size == 1:  ## bad return value  
                continue 
            if config.get('standardise_join_data', True):
                j_speech = standardise(j_speech, mean_vec_join, std_vec_join) 
                  

            j_frames, j_dim = j_speech.shape
            if j_frames != len(pms_seconds):      
                print (j_frames, len(pms_seconds))
                print 'Warning: number of rows in join cost features not same as number of pitchmarks:'
                print 'these features should be pitch synchronous. Skipping utterance!'
                continue  


        if  config['target_representation'] == 'epoch':
            t_frames, t_dim = t_speech.shape
            print 't shape'
            print t_speech.shape            
            if j_frames != len(pms_seconds):      
                print (t_frames, len(pms_seconds))
                print 'Warning: number of rows in target cost features not same as number of pitchmarks:'
                print 'these features should be pitch synchronous (when target_representation == epoch). Skipping utterance!'
                continue  


        # j_speech = j_speech[1:-1,:]  ## remove first and last frames corresponding to terminal pms
        # j_frames -= 2

        if not config['target_representation'] in ['epoch', 'sample']:  ### TODO: pitch synchronise labels...
            ### get labels:
            labfile = os.path.join(config['label_datadir'], base + '.' + config['lab_extension'])
            labs = read_label(labfile, config['quinphone_regex'])   ### __pp:  pitch sync label?
            label_frames = labs[-1][0][1] ## = How many (5msec) frames does label correspond to?

            ## Has silence been trimmed from either t_speech or j_speech?

            ## Assume pitch synch join features are not silence trimmed
            # if config.get('untrim_silence_join_speech', False):
            #     print 'Add trimmed silence back to join cost speech features'
            #     j_speech = reinsert_terminal_silence(j_speech, labs)

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

            if DODEBUG:
                check_pitch_sync_speech(j_speech, labs, pms_seconds)
            #j_speech = pad_speech_to_length(j_speech, labs) ## Assume pitch synch join features are all OK

            ## Discard sentences where length of speech and labels differs too much:- 
            if t_speech.size==1:
                print 'Skip utterance'
                continue
            # if j_speech.size==1:
            #     print 'Skip utterance'            
            #     continue

        if config['target_representation'] == 'sample':
            
            unit_features = t_speech

        elif config['target_representation'] == 'epoch':
            ## Get representations of half phones to use in target cost:-
            unit_features = t_speech[1:-1, :]

            ## Find 'cutpoints': pitchmarks which are considered to be the boudnaries of units, and where those
            ## units will be concatenated:
            #cutpoints, cutpoint_indices = get_cutpoints(timings, pms_seconds)
            pms_samples = np.array(pms_seconds * 48000, dtype=int)

            cutpoints = segment_axis(pms_samples, 3, overlap=2, axis=0)

            #context_data = j_speech[1:-1, :]
            m,n = j_speech.shape
            context_data = segment_axis(j_speech, 2, overlap=1, axis=0).reshape((m-1, n*2))

            ADD_PHONETIC_EPOCH = False
            if ADD_PHONETIC_EPOCH:
                labfile = os.path.join(config['label_datadir'], base + '.' + config['lab_extension'])
                labs = read_label(labfile, config['quinphone_regex'])
                unit_names = resample_labels.pitch_synchronous_resample_label(48000, 0.005, pms_samples, labs)
                unit_names = unit_names[1:-1]
            else:                
                unit_names = np.array(['_']*(t_speech.shape[0]-2))

        else:
            ## Get representations of half phones to use in target cost:-
            unit_names, unit_features, timings = get_halfphone_stats(t_speech, labs, config.get('target_representation', 'twopoint'))

            ## Find 'cutpoints': pitchmarks which are considered to be the boudnaries of units, and where those
            ## units will be concatenated:
            cutpoints, cutpoint_indices = get_cutpoints(timings, pms_seconds)

            #context_data = get_contexts_for_natural_joincost(j_speech, timings, width=2)
            context_data = get_contexts_for_pitch_synchronous_joincost(j_speech, cutpoint_indices)            

        m,n = unit_features.shape
        if config['joincost_features']:  ## i.e. don't store this in sample-based case
            filenames = [base] * len(cutpoints)
            o,p = context_data.shape
            # if config['target_representation'] == 'epoch':
            #     assert o == m, (o, m)
            # else:
            assert o == m+1, (o, m)

            unit_index_within_sentence = np.arange(m)


            if config.get('dump_join_data', False):
                start_join_feats, end_join_feats = get_join_data_AL(j_speech, cutpoint_indices, config['join_cost_halfwidth'])

        CHECK_MAGPHASE_SIZES = False
        if CHECK_MAGPHASE_SIZES: # config.get('store_full_magphase', False):
            print 'CHECK_MAGPHASE_SIZES'
            for extn in  ['mag','imag','real','f0']:
                direc = extn + '_full'
                if extn == 'f0':
                    sdim = 1
                else:
                    sdim = 513
                fname = os.path.join(config['full_magphase_dir'], direc, base+'.'+extn)
                full_stream = get_speech(fname, sdim)
                #full_stream = full_stream[1:-1,:]
                print direc
                print full_stream.shape
                


        if config.get('store_full_magphase', False):
            mp_data = []
            for extn in  ['mag','imag','real','f0']:
                direc = extn + '_full'
                if extn == 'f0':
                    sdim = 1
                else:
                    sdim = 513
                fname = os.path.join(config['full_magphase_dir'], direc, base+'.'+extn)
                full_stream = get_speech(fname, sdim)
                full_stream = full_stream[1:-1,:]
                print direc
                print full_stream.shape
                mp_data.append(full_stream)


        ## Add everything to database:
        train_dset[start:start+m, :] = unit_features
        if config['joincost_features']:
            phones_dset[start:start+m] = unit_names
            filenames_dset[start:start+m] = filenames
            unit_index_within_sentence_dset[start:start+m] = unit_index_within_sentence
            cutpoints_dset[start:start+m,:] = cutpoints
            join_contexts_dset[start:start+m, :] = context_data[:-1,:]

            if config.get('dump_join_data', False):
                start_join_feats_dset[start:start+m, :] = start_join_feats
                end_join_feats_dset[start:start+m, :] = end_join_feats            

        if config['target_representation'] == 'sample':
            #wavecontext_dset[start:start+m, :] = wavecontext
            nextsample_dset[start:start+m, :] = nextsample

        if config.get('store_full_magphase', False):
            (mp_mag, mp_imag, mp_real, mp_fz) = mp_data

            mp_mag_dset[start:start+m, :] = mp_mag
            mp_imag_dset[start:start+m, :] = mp_imag
            mp_real_dset[start:start+m, :] = mp_real
            mp_fz_dset[start:start+m, :] = mp_fz


        start += m        
        new_flist.append(base)

    
    if config['target_representation'] not in ['epoch', 'sample']:      
        ## add database final join context back on (kind of messy)
        join_contexts_dset[m, :] = context_data[-1,:]

    ## Number of units was computed before without considering dropped utterances, actual number
    ## will be smaller. Resize the data:
    actual_nframes = start
    print '\n\n\nNumber of units actually written:'
    print actual_nframes
    print 

    train_dset.resize(actual_nframes, axis=0)

    if config['joincost_features']:
        phones_dset.resize(actual_nframes, axis=0)
        filenames_dset.resize(actual_nframes, axis=0)
        unit_index_within_sentence_dset.resize(actual_nframes, axis=0)
        cutpoints_dset.resize(actual_nframes, axis=0)

        join_contexts_dset.resize(actual_nframes+1, axis=0)

    if config['target_representation'] == 'sample':
        # wavecontext_dset.resize(actual_nframes, axis=0)
        nextsample_dset.resize(actual_nframes, axis=0)


        ## Store waveform standardisation info:
        wave_mu_sigma_dset = f.create_dataset("wave_mu_sigma", np.shape(wave_mu_sigma), dtype='f', track_times=False)
        wave_mu_sigma_dset[:] = wave_mu_sigma 

    if config.get('store_full_magphase', False):
        mp_mag_dset.resize(actual_nframes, axis=0)
        mp_imag_dset.resize(actual_nframes, axis=0)
        mp_real_dset.resize(actual_nframes, axis=0)
        mp_fz_dset.resize(actual_nframes, axis=0)


    print 
    print 'Storing hybrid voice data:'
    for thing in f.values():
        print thing



    # print '-------a' 
    # t = f["train_unit_features"][:,:]
    # print np.mean(t, axis=0).tolist()
    # print np.std(t, axis=0).tolist() 
    # print np.min(t, axis=0).tolist()
    # print np.max(t, axis=0).tolist()      
    # sys.exit('uuuuuuuu')



    f.close()
    


    print 'Stored training data for %s sentences to %s'%(n_train_utts, database_fname)
       
    if config.get('dump_join_data', False):    
        start_join_feats_dset.resize(actual_nframes, axis=0)
        end_join_feats_dset.resize(actual_nframes, axis=0)
        print 
        print 'Storing data for learning join cost:'
        for thing in fjoin.values():
            print thing
        fjoin.close()



def check_pitch_sync_speech(j_speech, labs, pms_seconds):
    print '-----------------------'
    print 'check_pitch_sync_speech'
    print j_speech.shape
    print labs[-1]
    print len(pms_seconds)
    print




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
    Take mean over each coeff, to centre their trajectories around zero.
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
    '''
    Unlike mean, use single std value over all coeffs in stream, to preserve relative differences in range of coeffs within a stream
    The value we use is the largest std across the coeffs, which means that this stream when normalised
    will have std of 1.0, and other streams decreasing. 
    Reduplicate this single value to vector the width of the stream.
    '''
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
        #mean_mat = np.tile(mean_vec,(m,1))
        mean_vec = mean_vec.reshape((1,-1))
        sq_diffs = (speech - mean_vec) ** 2
        diff_sum += sq_diffs.sum(axis=0)
        frame_count += m

    max_diff_sum = diff_sum.max()
    print mean_vec.tolist()
    print max_diff_sum.tolist()
    std_val = (max_diff_sum / float(frame_count)) ** 0.5
    std_vec = np.ones((1,dim)) * std_val
    return std_vec
    
def standardise(speech, mean_vec, std_vec):

    m,n = np.shape(speech)
        
    ### record where unvoiced values are with Boolean array, so we can revert them later:
    uv_positions = (speech==const.special_uv_value)

    mean_vec = mean_vec.reshape((1,-1))
    
    ## standardise:-
    speech = (speech - mean_vec) / std_vec
    
    uv_values = std_vec * -1.0 * const.uv_scaling_factor

    for column in range(n):
        # print speech[:,column].shape
        # print uv_positions[:,column].shape
        # print speech[:,column]
        # print uv_positions[:,column]
        # print column
        #if True in uv_positions[:,column]:
        speech[:,column][uv_positions[:,column]] = uv_values[0, column]

    ## leave weighting till later!
    return speech

def destandardise(speech, mean_vec, std_vec):

    m,n = np.shape(speech)
        
    mean_vec = mean_vec.reshape((1,-1))
    #std_mat = np.tile(std_vec,(m,1))
    #weight_mat = np.tile(weight_vec,(m,1))
    
    ## standardise:-
    speech = (speech * std_vec) + mean_vec
    
    ## leave weighting till later!
    # speech = speech * weight_mat
    return speech
    


DODEBUG = False
def debug(msg):
    if DODEBUG:
        print msg
    


def compose_speech(feat_dir_dict, base, stream_list, datadims, ignore_streams=['triphone']): 
    '''
    where there is trouble, signal this by returning a 1 x 1 matrix
    '''

    stream_list = [stream for stream in stream_list if stream not in ignore_streams]
    # mgc_fn = os.path.join(indir, 'mgc', base+'.mgc' ) 
    # f0_fn = os.path.join(indir, 'f0', base+'.f0' ) 
    # ap_fn = os.path.join(indir, 'ap', base+'.ap' ) 

    stream_data_list = []
    for stream in stream_list:
        stream_fname = os.path.join(feat_dir_dict[stream], base+'.'+stream ) 
        if not os.path.isfile(stream_fname):
            print stream_fname + ' does not exist'
            return np.zeros((1,1))
        stream_data = get_speech(stream_fname, datadims[stream])
        if stream == 'aef':
            stream_data = np.vstack([np.zeros((1,datadims[stream])), stream_data, np.zeros((1,datadims[stream]))])
        ### previously:        
        # if stream in vuv_stream_names:
        #     uv_ix = np.arange(stream_data.shape[0])[stream_data[:,0]<=0.0]
        #     vuv = np.ones(stream_data.shape)
        #     vuv[uv_ix, :] = 0.0
        #     ## set F0 to utterance's voiced frame mean in unvoiced frames:   
        #     voiced = stream_data[stream_data>0.0]
        #     if voiced.size==0:
        #         voiced_mean = 100.0 ### TODO: fix artibrary nnumber!
        #     else:
        #         voiced_mean = voiced.mean()
        #     stream_data[stream_data<=0.0] = voiced_mean 
        #     stream_data_list.append(stream_data)
        #     stream_data_list.append(vuv)

        ### Now, just set unvoiced frames to -1.0 (they will be specially weighted later):
        if stream in vuv_stream_names:
            # uv_ix = np.arange(stream_data.shape[0])[stream_data[:,0]<=0.0]
            # vuv = np.ones(stream_data.shape)
            # vuv[uv_ix, :] = 0.0
            ## set F0 to utterance's voiced frame mean in unvoiced frames:   
            # voiced = stream_data[stream_data>0.0]
            # if voiced.size==0:
            #     voiced_mean = 100.0 ### TODO: fix artibrary nnumber!
            # else:
            #     voiced_mean = voiced.mean()
            stream_data[stream_data<=0.0] = const.special_uv_value
            stream_data_list.append(stream_data)
            # stream_data_list.append(vuv)
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
  
          
def get_data_dump_name(config, joindata=False, joinsql=False, searchtree=False):
    safe_makedir(os.path.join(config['workdir'], 'data_dumps'))
    condition = make_train_condition_name(config)
    assert not (joindata and joinsql)
    if joindata:
        last_part = '.joindata.hdf5'
    elif joinsql:
        last_part = '.joindata.sql'
    elif searchtree:
        last_part = '.searchtree.hdf5'
    else:
        last_part = '.hdf5'
    database_fname = os.path.join(config['workdir'], "data_dumps", condition + last_part)
    return database_fname

def make_train_condition_name(config):
    '''
    condition name including any important hyperparams
    '''
    ### N-train_utts doesn't account for exclusions due to train_list, bad data etc. TODO - fix?
    if not config['target_representation'] == 'sample':
        jstreams = '-'.join(config['stream_list_join'])
        tstreams = '-'.join(config['stream_list_target'])
        return '%s_utts_jstreams-%s_tstreams-%s_rep-%s'%(config['n_train_utts'], jstreams, tstreams, config.get('target_representation', 'twopoint'))
    else:        
        streams = '-'.join(config['stream_list_target'])
        return '%s_utts_streams-%s_rep-%s'%(config['n_train_utts'], streams, config.get('target_representation', 'twopoint'))
        

def read_label(labfile, quinphone_regex):
    '''
    Return list with entries like:  ((start_frame, end_frame), [ll,l,c,r,,rr,state_number]).
    The typical labels input mean that end frame of item at t-1 is same as start frame at t. 
    '''
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
    Find GCIs which are nearest to the start and end of each unit.
    Also return indices of GCIs so we can map easily to pitch-synchronous features.
    '''

    cutpoints = []
    indices = []
    for (start, end) in timings:
        start_sec = start * 0.005  # TODO: unhardcode frameshift and rate
        end_sec = (end) * 0.005
        start_closest_ix = numpy.argmin(numpy.abs(pms - start_sec))
        end_closest_ix = numpy.argmin(numpy.abs(pms - end_sec))
        indices.append((start_closest_ix, end_closest_ix))
        cutpoints.append((pms[start_closest_ix], pms[end_closest_ix]))

    indices = np.array(indices, dtype=int)
    cutpoints = np.array(cutpoints)
    cutpoints *= 48000             # TODO: unhardcode frameshift and rate
    cutpoints = np.array(cutpoints, dtype=int)
    return (cutpoints, indices)





def get_halfphone_stats(speech, labels, representation_type='twopoint'):
    '''
    Where there are N hafphones in an utt, return (names, features, timings) where
        -- names is N-element array like (array(['xx~xx-#_L+p=l', 'xx~xx-#_R+p=l', 'xx~#-p_L+l=i', ...
        -- timings is N-element list like [(0, 40), (41, 60), (61, 62), ...
        -- features is N x D array, where D is size of feature vector

    To get from s 5-state alignment for a phone to 2 halfphones, we arbitrarily 
    assign states 1 & 2 to halfphone 1, and states 3, 4 and 5 to halfphone 2.

    Given this division, various types of representation are possible. A unit can
    be represented by:
        -- onepoint: middle frame appearing in it
        -- twopoint: first and last frames appearing in it
        -- threepoint: first, middle, and last frames appearing in it
    We use state alignment in effect to do downsmpling which is non-linear in time.
    Hence, the 'middle' point is not necessarily equidistant from the start and end
    of a unit, but rather the last frame in state 2 (for first halfphone) or in state 
    5 (for second halfphone). Other choices for middle frame are possible. 
    '''

    if 0:
        print speech
        print labels
        print speech.shape
        print len(labels)
        sys.exit('stop here 8293438472938')

    if representation_type not in ['onepoint', 'twopoint', 'threepoint']:
        sys.exit('Unknown halfphone representation type: %s '%(representation_type))

    
    m,dim = speech.shape

    assert len(labels) % 5 == 0, 'There must be 5 states for each phone in label'
    nphones = len(labels) / 5 
    features = numpy.zeros((nphones*2, dim*2))
    names = []
    starts = []
    middles = []
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
            #features[halfphone_counter, :dim] = speech[s,:]
            #if representation_type in ['twopoint', 'threepoint']:
            starts.append(s)
            #if representation_type in ['onepoint', 'threepoint']:
            middles.append(e)
        elif state == '3':
            #features[halfphone_counter, dim:] = speech[e,:]  
            # if representation_type in ['twopoint', 'threepoint']:
            ends.append(e)
            #halfphone_counter += 1                     
        elif state == '4':
            halfphone_name = copy.copy(quinphone)
            halfphone_name[2] += '_R'
            assert label_delimiter not in ''.join(halfphone_name), 'delimiter %s occurs in one or more name element (%s)'%(label_delimiter, halfphone_name)
            halfphone_name = label_delimiter.join(halfphone_name)
            names.append(halfphone_name)
            #features[halfphone_counter, :dim] = speech[s,:]   
            # if representation_type in ['twopoint', 'threepoint']: 
            starts.append(s)  
        elif state == '5':
            # if representation_type in ['onepoint', 'threepoint']:
            middles.append(e)                      
        elif state == '6':
            #features[halfphone_counter, dim:] = speech[e,:]  
            # if representation_type in ['twopoint', 'threepoint']:  
            ends.append(e)        
            #halfphone_counter += 1  
        else:
            sys.exit('bad state number')
                    
    assert len(names) == nphones*2 == len(starts) == len(ends) == len(middles)

    # if representation_type in ['twopoint', 'threepoint']: 
    #     assert len(names) == len(starts) == len(ends)
    # if representation_type in ['onepoint', 'threepoint']: 
    #     assert len(names) == len(middles)    

    names = np.array(names)
    timings = zip(starts,ends)

    ### construct features with advanced indexing:--
    if representation_type == 'onepoint':
        features = speech[middles, :]    
    elif representation_type == 'twopoint':
        features = np.hstack([speech[starts,:], speech[ends,:]])    
    elif representation_type == 'threepoint':
        features = np.hstack([speech[starts,:], speech[middles,:], speech[ends,:]])
    else:
        sys.exit('eifq2o38rf293f')
        
    return (names, features, timings)



def get_prosody_targets(speech, timings, ene_dim=0, lf0_dim=-1, fshift_sec=0.005):
    '''
    Return list of triplets, containing (dur,ene,lf0) where these are averages over
    the halfphone. If all speech in the halfphone is unvoiced, return negative lf0
    value, else the mean of the voiced frames.
    '''
    prosody_targets = []
    for (start, end) in timings:
        energy = speech[start:end, ene_dim].mean()

        pitch = speech[start:end, lf0_dim]
        voiced = pitch[pitch>0.0]
        if voiced.size==0:
            pitch = -1000.0
        else:
            pitch = voiced.mean()

        duration = (end - start) * fshift_sec
        duration = duration * 1000.0  ## to msec
        
        prosody_targets.append((duration, energy, pitch))
    return prosody_targets


def get_contexts_for_pitch_synchronous_joincost(speech, pm_indices):
    '''
    pm_indices: start and end indices of pitchmarks considered to be unit cutpoints: 
        [[  0   1]
         [  1   5]
         [  5  12]
         [ 12  17] ...
    Because speech features used for join cost are expected to be already pitch synchronous or 
    synchronised, we can index rows of speech directly with these.

    Where pm_indices gives indices for n units (n rows), return (n+1 x dim) matrix, each row of which 
    gives a 'join context' for the end of a unit. Row p gives the start join context for 
    unit p, and the end join context for unit p-1. 
    ''' 

    # enforce that t end is same as t+1 start -- TODO: do this check sooner, on the labels?  
    assert pm_indices[1:, 0].all() == pm_indices[:-1, 1].all()

    ## convert n -> n+1 with shared indices:
    last_end = pm_indices[-1][1]
    starts = np.array([s for (s,e) in pm_indices] + [last_end])

    # print '=========='
    # print speech.shape
    # print starts
    context_frames = speech[starts, :]

    return context_frames



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



def get_join_data_AL(speech, pm_indices, halfwidth):
    '''
    Newer version: pitch synchronous features.
    Output of this operation can be used by later scripts for actively learning a join cost.

    pm_indices: start and end indices of pitchmarks considered to be unit cutpoints: 
        [[  0   1]
         [  1   5]
         [  5  12]
         [ 12  17] ...
    Because speech features used for join cost are expected to be already pitch synchronous or 
    synchronised, we can index rows of speech directly with these.
    '''
    # enforce that t end is same as t+1 start -- TODO: do this check sooner, on the labels?  
    assert pm_indices[1:, 0].all() == pm_indices[:-1, 1].all()    

    starts = pm_indices[:,0]
    ends = pm_indices[:,1]
    start_speech = copy.copy(speech)
    if starts[-1] + halfwidth > ends[-1]:
        difference = starts[-1] + halfwidth - ends[-1]
        padding = speech[-1,:].reshape((1,-1))
        start_speech = np.vstack([start_speech] +    difference * [padding])
    start_speech = segment_axis(start_speech, halfwidth, overlap=halfwidth-1, axis=0)
    start_contexts = start_speech[starts,:,:].reshape((len(starts), -1))

    end_speech = copy.copy(speech)
    if ends[0] - (halfwidth+1) < 0:
        difference = (ends[0] - (halfwidth+1)) * -1
        padding = speech[0,:].reshape((1,-1))
        end_speech = np.vstack(difference * [padding] +   [end_speech])
    ends -= (halfwidth+1)
    end_speech = segment_axis(end_speech, halfwidth, overlap=halfwidth-1, axis=0)
    end_contexts = end_speech[ends,:,:].reshape((len(ends), -1))

    return (start_contexts, end_contexts)




##### fixed framerate version:
# def get_join_data_AL(speech, timings, halfwidth):
#     '''
#     Output of this operation is not yet used -- it will be used for actively learned join cost
#     '''

#     print speech
#     print timings
#     sys.exit('wefswrb545')
#     # pylab.plot(speech)
#     # pylab.show()

#     ## starting parts
#     starts = [s for (s,e) in timings]
#     ## do we need to pad the end of the speech?
#     m,n = speech.shape
#     ##print 'N'
#     ##print n
#     if max(starts) + halfwidth > m:
#         diff = (max(starts) + halfwidth) - m
#         start_speech = np.vstack([speech, np.zeros((diff, n))])
#         debug('correct start')
#     else:
#         start_speech = speech
#     #print start_speech.shape
#     frames = segment_axis(start_speech, halfwidth, overlap=halfwidth-1, axis=0)
#     #print frames.shape
#     start_frames = frames[starts,:,:]
#     #print start_frames.shape

#     ends = np.array([e for (s,e) in timings])
#     ## do we need to pad the start of the speech?
#     if min(ends) - halfwidth < 0:
#         diff = 0 - (min(ends) - halfwidth)
#         end_speech = np.vstack([np.zeros((diff, n)), speech])
#         ends += diff
#         debug('correct end')
#     else:
#         end_speech = speech
#     ends -=  halfwidth ###  to get starting point of end segments
#     frames = segment_axis(end_speech, halfwidth, overlap=halfwidth-1, axis=0)
#     #print frames.shape
#     end_frames = frames[ends,:,:]
    
#     ## flatten the last 2 dimensions of the data:--
#     #print start_frames.shape
#     #print halfwidth, n
#     start_frames = start_frames.reshape((-1, halfwidth*n))
#     end_frames = end_frames.reshape((-1, halfwidth*n))

#     return (start_frames, end_frames)



def get_mean_std(feat_dir_dict, stream_list, datadims, flist):

    means = {}
    stds = {}
    for stream in stream_list:
        stream_files = [os.path.join(feat_dir_dict[stream], base+'.'+stream) for base in flist]
        if stream in vuv_stream_names:
            means[stream], _ = get_mean(stream_files, datadims[stream], exclude_uv=True)
            stds[stream] = get_std(stream_files, datadims[stream], means[stream], exclude_uv=True)
        else:
            means[stream], nframe = get_mean(stream_files, datadims[stream])
            stds[stream] = get_std(stream_files, datadims[stream], means[stream])


    mean_vec = []
    for stream in stream_list:
        mean_vec.append(means[stream])
        # if stream in vuv_stream_names: ## add fake stats for VUV which will leave values unaffected
        #     mean_vec.append(numpy.zeros(means[stream].shape))

    std_vec = []
    for stream in stream_list:
        std_vec.append(stds[stream])
        # if stream in vuv_stream_names: ## add fake stats for VUV which will leave values unaffected
        #     std_vec.append(numpy.ones(stds[stream].shape))

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



def get_waveform_fragments(wave_fname, rate, context_length, nonlin_wave=True, norm=np.zeros((0)), wave_context_type=0):
    '''
    wave_context_type: 0 = 

    if wave_context_type == 1: leftmost output values will correspond to rightmost (most recent) waeform samples  
    '''
    wave, fs = read_wave(wave_fname)
    assert fs == rate

    if wave_context_type == 0:
        wavefrag_length = context_length
    elif wave_context_type == 1:
        DILATION_FACTOR = 1.2
        filter_matrix = varying_filter.make_filter_01(DILATION_FACTOR, context_length)
        wavefrag_length, nfeats = filter_matrix.shape
        assert nfeats == context_length
    else:
        sys.exit('unknown wave_context_type: %s'%(wave_context_type))

    wave = np.concatenate([np.zeros(wavefrag_length), wave])


    # print 'Linear wave stats:'
    # print wave.mean()
    # print wave.std()

    if nonlin_wave:
        wave = lin2mu(wave)

        # print 'Prenormed omed mulaw wave stats:'
        # print wave.mean()
        # print wave.std()
        

    if NORMWAVE:
        if norm.size > 0:
            assert norm.size == 2
            (mu, sigma) = norm
            # import pylab
            # pylab.subplot(211)
            # pylab.plot(wave)
            #print type(wave[0])
            wave = (wave - mu) / sigma
            #print wave[:10]

            # pylab.subplot(212)
            # pylab.plot(wave)
            # pylab.show()
            #sys.exit('esdvsvsdfbv0000')
            # print 'Nomed mulaw wave stats:'

            # print wave.mean()
            # print wave.std()
            # print 'Normed with:'
            # print (mu, sigma)
            



    frags = segment_axis(wave, wavefrag_length+1, overlap=wavefrag_length, axis=0)
    context = frags[:,:-1]
    next_sample = frags[:,-1].reshape((-1,1))

    if wave_context_type > 0:
        context = np.dot(context, filter_matrix)


    return (context, next_sample)


def get_wave_mean_std(wav_datadir, flist, rate, nonlin_wave=True, nutts=100):
    '''
    By default, find mean and std of 1st 100 sentences only
    '''
    waves = []
    for fname in flist[:min(nutts,len(flist))]:
        wave_fname = os.path.join(wav_datadir, fname + '.wav')
        wave, fs = read_wave(wave_fname)
        assert fs == rate
        if nonlin_wave:
            wave = lin2mu(wave)
        waves.append(wave)

    waves = np.concatenate(waves)
    mu = waves.mean()
    sigma = waves.std()

    return np.array([mu, sigma])
    


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
    
    
    
    