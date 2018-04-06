#!/usr/bin/env python
# -*- coding: utf-8 -*-
## Project:  
## Author: Oliver Watts - owatts@staffmail.ed.ac.uk

import sys
import os
import glob

from argparse import ArgumentParser

import h5py
import numpy as np

from const import target_rep_widths 
from speech_manip import get_speech 
from util import safe_makedir, readlist
from file_naming import get_data_dump_name
from data_manipulation import locate_stream_directories, get_mean_std, compose_speech, standardise

DODEBUG = False
def debug(msg):
    if DODEBUG:
        print msg
    

def main_work(config, overwrite_existing_data=False):
    
    assert config['target_representation'] == 'epoch'

    database_fname = get_data_dump_name(config)

    if os.path.isfile(database_fname):
        if not overwrite_existing_data:
            sys.exit('Data already exists at %s -- run with -X to overwrite it'%(database_fname))
        else:
            os.system('rm '+database_fname)
            
    n_train_utts = config.get('n_train_utts', 0) ## default (0): use all sentences

    target_feat_dirs = config['target_datadirs']
    datadims_target = config['datadims_target']
    stream_list_target = config['stream_list_target'] 
    ## get dicts mapping e.g. 'mgc': '/path/to/mgc/' : -
    target_stream_dirs = locate_stream_directories(target_feat_dirs, stream_list_target)
    
    join_feat_dirs = config['join_datadirs']
    datadims_join = config['datadims_join']
    stream_list_join = config['stream_list_join']    
    ## get dicts mapping e.g. 'mgc': '/path/to/mgc/' : -
    join_stream_dirs   = locate_stream_directories(join_feat_dirs, stream_list_join)
    
    ## First, work out initial list of training utterances based on files present in first stream subdir: 
    first_stream = stream_list_target[0] ## <-- typically, mgc, but not really important
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
    (mean_vec_join, std_vec_join) = get_mean_std(join_stream_dirs, stream_list_join, datadims_join, flist)

 
    ## 1B) Initialise HDF5; store mean and std in HDF5: 

    f = h5py.File(database_fname, "w")

    mean_target_dset = f.create_dataset("mean_target", np.shape(mean_vec_target), dtype='f', track_times=False)
    std_target_dset = f.create_dataset("std_target", np.shape(std_vec_target), dtype='f', track_times=False)
    mean_join_dset = f.create_dataset("mean_join", np.shape(mean_vec_join), dtype='f', track_times=False)
    std_join_dset = f.create_dataset("std_join", np.shape(std_vec_join), dtype='f', track_times=False)

    mean_target_dset[:] = mean_vec_target[:]
    std_target_dset[:] = std_vec_target[:]
    mean_join_dset[:] = mean_vec_join[:]
    std_join_dset[:] = std_vec_join[:]            
    
    ## Set some values....
    target_dim = mean_vec_target.shape[0]
    join_dim = mean_vec_join.shape[0]

    target_rep_size = target_dim * target_rep_widths[config.get('target_representation', 'epoch')]

    fshift_seconds = (0.001 * config['frameshift_ms'])
    fshift = int(config['sample_rate'] * fshift_seconds)    
    samples_per_frame = fshift
 
    print 'Go through data to find number of units:- '  
    
    n_units = 0
    new_flist = []
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

    print '%s units (%s)'%(n_units,  config.get('target_representation', 'epoch'))
    
    ## 2) Get ready to store data in HDF5:
    total_target_dim = target_rep_size
 
    ## maxshape makes a dataset resizable
    train_dset = f.create_dataset("train_unit_features", (n_units, total_target_dim), maxshape=(n_units, total_target_dim), dtype='f', track_times=False) 

    phones_dset = f.create_dataset("train_unit_names", (n_units,), maxshape=(n_units,), dtype='|S50', track_times=False) 
    filenames_dset = f.create_dataset("filenames", (n_units,), maxshape=(n_units,), dtype='|S50', track_times=False) 
    unit_index_within_sentence_dset = f.create_dataset("unit_index_within_sentence_dset", (n_units,), maxshape=(n_units,), dtype='i', track_times=False) 
    join_contexts_dset = f.create_dataset("join_contexts", (n_units+1, join_dim), maxshape=(n_units+1, join_dim), dtype='f', track_times=False) 

    ### TODO: use? 
    if config.get('store_full_magphase', False):
        mp_mag_dset = f.create_dataset("mp_mag", (n_units, 513), maxshape=(n_units, 513), dtype='f', track_times=False) 
        mp_imag_dset = f.create_dataset("mp_imag", (n_units, 513), maxshape=(n_units, 513), dtype='f', track_times=False) 
        mp_real_dset = f.create_dataset("mp_real", (n_units, 513), maxshape=(n_units, 513), dtype='f', track_times=False)   
        mp_fz_dset = f.create_dataset("mp_fz", (n_units, 1), maxshape=(n_units, 1), dtype='f', track_times=False)   


    ## Standardise data (within streams), compose, add VUV, fill F0 gaps with utterance mean voiced value: 
    start = 0

    print 'Composing ....'
    print flist
    new_flist = []
    for base in flist:

        print base    
        
        #! pm_file = os.path.join(config['pm_datadir'], base + '.pm')              
        # if not(os.path.isfile(pm_file)):
        #     print 'Warning: no pm -- skip!'
        #     continue

        #! ## Get pitchmarks (to join halfphones on detected GCIs):-
        # pms_seconds = read_pm(pm_file)
        # if pms_seconds.shape == (1,1):
        #     print 'Warning: trouble reading pm file -- skip!'
        #     continue                    

        ### Get speech params for target cost (i.e. probably re-generated speech for consistency):
        t_speech = compose_speech(target_stream_dirs, base, stream_list_target, datadims_target) 
        if t_speech.shape == [1,1]:  ## bad return value  
            continue                    

        t_speech = standardise(t_speech, mean_vec_target, std_vec_target)
            
        ### Get speech params for join cost (i.e. probably natural speech).
        ### These are expected to have already been resampled so that they are pitch-synchronous. 
        j_speech = compose_speech(join_stream_dirs, base, stream_list_join, datadims_join)
        if j_speech.size == 1:  ## bad return value  
            continue 
        j_speech = standardise(j_speech, mean_vec_join, std_vec_join) 
              

        j_frames, j_dim = j_speech.shape
        # if j_frames != len(pms_seconds):      
        #     print (j_frames, len(pms_seconds))
        #     print 'Warning: number of rows in join cost features not same as number of pitchmarks:'
        #     print 'these features should be pitch synchronous. Skipping utterance!'
        #     continue  


        t_frames, t_dim = t_speech.shape
        if j_frames != t_frames:      
            print (j_frames, t_frames)
            print 'Warning: number of rows in target cost features not same as number in join cost features:'
            print ' Skipping utterance!'
            continue  


        first_sentence_in_corpus = base==flist[0]
        if config.get('REPLICATE_IS2018_EXP', False):
            unit_features = t_speech[1:-1, :]  ## Representations for target cost

            if first_sentence_in_corpus:
                context_data = j_speech[:-1, :]
            else:
                context_data = j_speech[1:-1, :]
        else: ## this should be consistent with how hi-dim frames are selected and remove a bug
            unit_features = t_speech  ## do not trim frames

            if first_sentence_in_corpus:
                initial_history = j_speech[0,:].reshape((1,-1)) ### assume that first frame is silence
                context_data = np.vstack([initial_history, j_speech])
            else:
                context_data = j_speech           


        ## TODO: reinstate this?:--
        ADD_PHONETIC_EPOCH = False
        if ADD_PHONETIC_EPOCH:
            labfile = os.path.join(config['label_datadir'], base + '.' + config['lab_extension'])
            labs = read_label(labfile, config['quinphone_regex'])
            unit_names = resample_labels.pitch_synchronous_resample_label(48000, 0.005, pms_samples, labs)
            unit_names = unit_names[1:-1]
        else:                
            unit_names = np.array(['_']*(t_speech.shape[0]-2))


        m,n = unit_features.shape   
        filenames = [base] * m

        unit_index_within_sentence = np.arange(m)

        ## TODO: reinstate this as hi-dim writer?:--
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
                

        ## TODO: reinstate this as hi-dim writer?:--
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

        phones_dset[start:start+m] = unit_names
        filenames_dset[start:start+m] = filenames
        unit_index_within_sentence_dset[start:start+m] = unit_index_within_sentence
        #! cutpoints_dset[start:start+m,:] = cutpoints

        ### join_contexts has extra initial frame of history -- deal with it:
        if first_sentence_in_corpus:
            join_contexts_dset[start:start+m+1, :] = context_data
        else:
            join_contexts_dset[start+1:start+m+1, :] = context_data

        ### TODO: use?
        if config.get('store_full_magphase', False):
            (mp_mag, mp_imag, mp_real, mp_fz) = mp_data

            mp_mag_dset[start:start+m, :] = mp_mag
            mp_imag_dset[start:start+m, :] = mp_imag
            mp_real_dset[start:start+m, :] = mp_real
            mp_fz_dset[start:start+m, :] = mp_fz


        start += m        
        new_flist.append(base)

    
    ## Number of units was computed before without considering dropped utterances, actual number
    ## will be smaller. Resize the data:
    actual_nframes = start
    print '\n\n\nNumber of units actually written:'
    print actual_nframes
    print 

    train_dset.resize(actual_nframes, axis=0)

    phones_dset.resize(actual_nframes, axis=0)
    filenames_dset.resize(actual_nframes, axis=0)
    unit_index_within_sentence_dset.resize(actual_nframes, axis=0)

    join_contexts_dset.resize(actual_nframes+1, axis=0)

    ### TODO
    if config.get('store_full_magphase', False):
        mp_mag_dset.resize(actual_nframes, axis=0)
        mp_imag_dset.resize(actual_nframes, axis=0)
        mp_real_dset.resize(actual_nframes, axis=0)
        mp_fz_dset.resize(actual_nframes, axis=0)


    print 
    print 'Storing hybrid voice data:'
    for thing in f.values():
        print thing

    f.close()
    
    print 'Stored training data for %s sentences to %s'%(n_train_utts, database_fname)
       




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
    
    
    
    