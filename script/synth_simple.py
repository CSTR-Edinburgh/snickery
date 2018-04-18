#!/usr/bin/env python
# -*- coding: utf-8 -*-
## Project:  
## Author: Oliver Watts - owatts@staffmail.ed.ac.uk

import sys
import os
import glob
import re
import timeit
import random
from argparse import ArgumentParser

# modify import path to obtain modules from the tools/magphase/src directory:
snickery_dir = os.path.split(os.path.realpath(os.path.abspath(os.path.dirname(__file__))))[0]+'/'
sys.path.append(os.path.join(snickery_dir, 'tool', 'magphase', 'src'))
import magphase
import libaudio as la

import numpy as np
import scipy

import h5py

from sklearn.cluster import KMeans

import speech_manip

from util import safe_makedir, basename, writelist   
from speech_manip import read_wave, write_wave, weight, get_speech
from file_naming import get_data_dump_name, make_train_condition_name, make_synthesis_condition_name
from data_manipulation import locate_stream_directories, compose_speech, standardise, random_subset_data
from matrix_operations import zero_pad_matrix, taper_matrix
from resample import pitch_synchronise

from segmentaxis import segment_axis

DODEBUG=False ## print debug information?
from train_simple import debug

import const
from const import FFTHALFLEN 

APPLY_JCW_ON_TOP = True     ## for IS2018 -- scale weights by jcw -- this should prob always be true henceforth?

        

class Synthesiser(object):

    def __init__(self, config_file, holdout_percent=0.0):

        self.mode_of_operation = 'normal'  ### !TODO: remove?
        self.verbose = True

        print 'Load config...'
        self.config = {}
        execfile(config_file, self.config)
        del self.config['__builtins__']
        
        self.config_file = config_file   ## in case we need to refresh...

        self.stream_list_target = self.config['stream_list_target']
        self.stream_list_join = self.config['stream_list_join']

        print 'Prepare weights from config'
        self.datadims_target = self.config['datadims_target']
        self.datadims_join = self.config['datadims_join']

        self.target_representation = 'epoch'

        print 'load database...'        
        datafile = get_data_dump_name(self.config)
        if not os.path.isfile(datafile):
            sys.exit('data: \n   %s   \ndoes not exist -- try other?'%(datafile))
            
        f = h5py.File(datafile, "r")

        print 
        print 'Database file: %s'%(datafile)
        print 
        print 'Loading hybrid voice data:'
        for thing in f.values():
            print thing
        print


        self.train_unit_features_unweighted = f["train_unit_features"][:,:]
        self.train_unit_names = f["train_unit_names"][:] 
        #! self.train_cutpoints = f["cutpoints"][:] 
        self.train_filenames = f["filenames"][:]                 
        self.mean_vec_target = f["mean_target"][:] 
        self.std_vec_target = f["std_target"][:] 
        self.mean_vec_join = f["mean_join"][:] 
        self.std_vec_join = f["std_join"][:] 

        self.join_contexts_unweighted = f["join_contexts"][:,:]
        self.unit_index_within_sentence = f['unit_index_within_sentence_dset'][:]

        ## !TODO! use?
        if self.config.get('store_full_magphase', False):
            self.mp_mag = f["mp_mag"][:] 
            self.mp_imag = f["mp_imag"][:] 
            self.mp_real = f["mp_real"][:] 
            self.mp_fz = f["mp_fz"][:]                                     

        f.close()

        self.number_of_units, _ = self.train_unit_features_unweighted.shape

        ### !TODO: remove this from basic version? refactor?
        self.holdout_percent = holdout_percent
        self.holdout_samples = 0
        if holdout_percent > 0.0:
            holdout_samples = int(self.number_of_units * (holdout_percent/100.0))
            print 'holdout_samples:'
            print holdout_samples

            self.train_unit_features_unweighted_dev = self.train_unit_features_unweighted[-holdout_samples:,:]
            self.train_unit_features_unweighted = self.train_unit_features_unweighted[:-holdout_samples,:]
        
            self.train_unit_names_dev = self.train_unit_names[-holdout_samples:]
            self.train_unit_names = self.train_unit_names[:-holdout_samples]

            self.number_of_units -= holdout_samples
            #sys.exit('evwservgwsrbv')
            self.holdout_samples = holdout_samples

        ## !TODO: always this way? Enforce sum-to-1 of stream weights?
        if APPLY_JCW_ON_TOP:
            self.set_target_weights(np.array(self.config['target_stream_weights']) * (1.0 - self.config['join_cost_weight'])) 
            self.set_join_weights(np.array(self.config['join_stream_weights']) * self.config['join_cost_weight'])
        else:
            self.set_target_weights(self.config['target_stream_weights'])
            self.set_join_weights(self.config['join_stream_weights'])

        ## !TODO: reinstate?
        # if 'truncate_target_streams' in self.config:
        #     self.truncate_target_streams(self.config['truncate_target_streams'])
        # if 'truncate_join_streams' in self.config:
        #     self.truncate_join_streams(self.config['truncate_join_streams'])

        self.first_silent_unit = 0 ## assume first unit is a silence, for v naive backoff

        self.config['preselection_method'] = 'acoustic'

        self.waveforms = {} ## !TODO: rename

        self.use_hdf_magphase = self.config.get('use_hdf_magphase', '')
        if self.use_hdf_magphase:
            self.hdf_magphase_pointer = h5py.File(self.use_hdf_magphase, 'r')


        elif self.config['hold_waves_in_memory']:
            print 'load waves into memory'
            for base in np.unique(self.train_filenames):
                print '.',
                wavefile = os.path.join(self.config['wav_datadir'], base + '.wav')
                wave, sample_rate = read_wave(wavefile)
                self.waveforms[base] = wave
        elif self.config['preload_all_magphase_utts']:
            print 'load magphase into memory'
            self.preload_all_magphase_utts()


        print

        assert self.config['greedy_search'] == True

        
        if self.config.get('multiple_search_trees', 1) > 1:
            sys.exit('multiple_search_trees not implemented yet -- try adjusting search_epsilon instead to speed up search')
            self.get_multiple_trees_for_greedy_search()  ### !TODO: implement this properly?
        else:
            self.get_tree_for_greedy_search()

        print 'Database loaded'
        print '\n\n----------\n\n'

        self.test_data_target_dirs = locate_stream_directories(self.config['test_data_dirs'], self.stream_list_target)

        print 'Found target directories: %s'%(self.test_data_target_dirs)
        print 
        print 
        if self.config.get('tune_data_dirs', ''):
            self.tune_data_target_dirs = locate_stream_directories(self.config['tune_data_dirs'], self.stream_list_target)


    ##### ========== set-up methods ==========


    def get_tree_for_greedy_search(self):

        #! m,n = self.unit_start_data.shape

        self.prev_join_rep = self.unit_start_data    ## !osw
        self.current_join_rep = self.unit_end_data   ## !osw

        multiepoch = self.config.get('multiepoch', 1)
        if multiepoch > 1:
            t = self.start_clock('reshape data for multiepoch...')
            overlap = multiepoch-1
            ### reshape target rep:
            m,n = self.train_unit_features.shape
            self.train_unit_features = segment_axis(self.train_unit_features, multiepoch, overlap=overlap, axis=0)
            self.train_unit_features = self.train_unit_features.reshape(m-overlap,n*multiepoch)

            if self.config.get('last_frame_as_target', False):   ### !TODO: keep this option?
                print 'test -- take last frame only as target...'  
                # self.train_unit_features = self.train_unit_features[:,-n:]
                self.train_unit_features = np.hstack([self.train_unit_features[:,:n], self.train_unit_features[:,-n:]])

            ### alter join reps: -- first tried taking first vs. last
            m,n = self.current_join_rep.shape
            self.current_join_rep = self.current_join_rep[overlap:,:]
            self.prev_join_rep = self.prev_join_rep[:-overlap, :]

            ### !TODO: revisit this?
            ### then, whole comparison for join:
            # m,n = self.current_join_rep.shape
            # self.current_join_rep = segment_axis(self.current_join_rep, multiepoch, overlap=overlap, axis=0).reshape(m-overlap,n*multiepoch)
            # self.prev_join_rep = segment_axis(self.prev_join_rep, multiepoch, overlap=overlap, axis=0).reshape(m-overlap,n*multiepoch)
            self.stop_clock(t)

        t = self.start_clock('stack data to train joint tree...')
        combined_rep = np.hstack([self.prev_join_rep, self.train_unit_features])
        self.stop_clock(t)
        
        t = self.start_clock('make joint join + target tree...')
        ### For now, build each time from scratch -- compare resurrection time with rebuild time.
        self.joint_tree = scipy.spatial.cKDTree(combined_rep, leafsize=100, balanced_tree=False) # , compact_nodes=False)
        self.stop_clock(t)
        


    def set_join_weights(self, weights):
        assert len(weights) == len(self.stream_list_join)
        
        ## get from per-stream to per-coeff weights:
        join_weight_vector = []
        for (i,stream) in enumerate(self.stream_list_join):
            # if stream in const.vuv_stream_names:
            #     join_weight_vector.extend([weights[i]]*2)
            # else:
                join_weight_vector.extend([weights[i]]*self.datadims_join[stream])

        join_weight_vector = np.array(join_weight_vector)
        ## TODO: be more explicit about how this copies and does NOT weight original self.join_contexts_unweighted
        join_contexts_weighted = weight(self.join_contexts_unweighted, join_weight_vector)   

        ## This should not copy:
        self.unit_end_data = join_contexts_weighted[1:,:]
        self.unit_start_data = join_contexts_weighted[:-1,:]   ## <-- only this one is used!

        if self.holdout_samples > 0:
            self.unit_end_data = self.unit_end_data[:-self.holdout_samples,:]
            self.unit_start_data = self.unit_start_data[:-self.holdout_samples,:]

    def set_target_weights(self, weights):
        assert len(weights) == len(self.stream_list_target), (weights, self.stream_list_target)
        ## get from per-stream to per-coeff weights:
        target_weight_vector = []
        for (i,stream) in enumerate(self.stream_list_target):
            # if stream in const.vuv_stream_names:
            #     target_weight_vector.extend([weights[i]]*2)
            # else:
                target_weight_vector.extend([weights[i]]*self.datadims_target[stream])
        nrepetitions = const.target_rep_widths[self.target_representation]
        target_weight_vector = np.array(target_weight_vector * nrepetitions)   
        self.train_unit_features = weight(self.train_unit_features_unweighted, target_weight_vector)   

        if self.holdout_samples > 0:
            self.train_unit_features_dev = weight(self.train_unit_features_unweighted_dev, target_weight_vector)   

        ## save this so we can weight incoming predicted acoustics: 
        self.target_weight_vector = target_weight_vector


    ##### ========== generation methods ==========

    def synth_from_config(self, inspect_join_weights_only=False, synth_type='test', outdir=''):

        self.report('synth_from_config')
        test_flist = self.get_sentence_set(synth_type)
        for fname in test_flist:
            self.synth_utt(fname, synth_type=synth_type, outdir=outdir)    

   
    def get_sentence_set(self, set_name): 
        assert set_name in ['test', 'tune']

        first_stream = self.stream_list_target[0]

        if set_name == 'test':
            data_dirs = self.test_data_target_dirs[first_stream]
            name_patterns = self.config.get('test_patterns', [])
            limit = self.config['n_test_utts']
        elif set_name == 'tune':
            data_dirs = self.tune_data_target_dirs[first_stream]
            name_patterns = self.config.get('tune_patterns', [])
            limit = self.config['n_tune_utts']            
        else:
            sys.exit('Set name unknown: "%s"'%(set_name))

        flist = sorted(glob.glob(data_dirs + '/*.' + first_stream))
        flist = [basename(fname) for fname in flist]

        ## find all files containing one of the patterns in test_patterns
        L = len(flist)
        if name_patterns:
            selected_flist = []
            for (i,fname) in enumerate(flist):
                for pattern in name_patterns:
                    if pattern in fname:
                        if fname not in selected_flist:
                            selected_flist.append(fname)
                        
            flist = selected_flist 

        ## check sentences not in training:
        if 1:
            train_names = dict(zip(self.train_filenames, self.train_filenames))
            selected_flist = []
            for name in flist:
                if name in train_names:
                    print 'Warning: %s in train utternances!'%(name)
                else:
                    selected_flist.append(name)
            flist = selected_flist 


        ### Only synthesise n sentences:
        if limit > 0:
            flist = flist[:limit]

        nfiles = len(flist)
        if nfiles == 0:
            print 'No files found for set "%s" based on configured test_data_dir and test_pattern'%(set_name)
        else:
            self.report('get_sentence_set: synthesising %s utternances based on config'%(nfiles))
        return flist


    def synth_utt(self, base, synth_type='tune', outstem='', outdir=''): 

        if synth_type == 'test':
            data_dirs = self.test_data_target_dirs
            lab_dir = self.config.get('test_lab_dir', '') ## default added for pure acoustic epoch case
        elif synth_type == 'tune':
            data_dirs = self.tune_data_target_dirs
            lab_dir = self.config.get('tune_lab_dir', '') ## default added for pure acoustic epoch case
        else:
            sys.exit('Unknown synth_type  9489384')

        if outdir:
            assert not outstem

        if not outstem:
            train_condition = make_train_condition_name(self.config)
            synth_condition = make_synthesis_condition_name(self.config)
            if outdir:
                synth_dir = outdir
            else:
                synth_dir = os.path.join(self.config['workdir'], 'synthesis_%s'%(synth_type), train_condition, synth_condition)
            safe_makedir(synth_dir)
                
            self.report('               ==== SYNTHESISE %s ===='%(base))
            outstem = os.path.join(synth_dir, base)       
        else:
            self.report('               ==== SYNTHESISE %s ===='%(outstem))

        start_time = self.start_clock('Get speech ')
        unnorm_speech = compose_speech(data_dirs, base, self.stream_list_target, \
                                self.config['datadims_target']) 

        if self.config.get('pitch_synchronise_test_data', False):
            unnorm_speech = pitch_synchronise(unnorm_speech, self.stream_list_target, \
                                self.config['datadims_target'])
            #unnorm_speech = unnorm_speech_b

        m,dim = unnorm_speech.shape

        speech = standardise(unnorm_speech, self.mean_vec_target, self.std_vec_target)         
            

        if self.config.get('REPLICATE_IS2018_EXP', False):
            unit_features = speech[1:-1, :]  
        else:
            unit_features = speech

        unit_features = weight(unit_features, self.target_weight_vector)       
        n_units, _ = unit_features.shape
        self.stop_clock(start_time)

        if self.config.get('debug_with_adjacent_frames', False):
            print 'Concatenate naturally contiguous units to debug concatenation!'
            assert not self.config.get('magphase_use_target_f0', True), 'set magphase_use_target_f0 to False for using debug_with_adjacent_frames'
            multiepoch = self.config.get('multiepoch', 1)
            if multiepoch > 1:
                best_path = np.arange(0,500, multiepoch)
            else:
                best_path = np.arange(500)

        else:
            assert self.config['greedy_search']
            assert self.config.get('target_representation') == 'epoch'
            best_path = self.greedy_joint_search(unit_features)
 

        if self.mode_of_operation == 'stream_weight_balancing':
            self.report( '\n\n balancing stream weights -- skip making waveform \n\n')
        else:
            PRELOAD_UTTS = False  ### !TODO?
            if PRELOAD_UTTS:
                start_time = self.start_clock('Preload magphase utts for sentence')
                self.preload_magphase_utts(best_path)
                self.stop_clock(start_time) 

            start_time = self.start_clock('Extract and join units')
            
            if self.config.get('store_full_magphase_sep_files', False):
                assert self.config['target_representation'] == 'epoch'
                target_fz = unnorm_speech[:,-1]  ## TODO: unhardcode position and lf0!
                target_fz = np.exp(target_fz)
                magphase_overlap = self.config.get('magphase_overlap', 0)


                if self.config.get('magphase_use_target_f0', True):
                    self.concatenateMagPhaseEpoch_sep_files(best_path, outstem + '.wav', fzero=target_fz, overlap=magphase_overlap)                
                else:
                    self.concatenateMagPhaseEpoch_sep_files(best_path, outstem + '.wav', overlap=magphase_overlap)                

            elif self.config.get('store_full_magphase', False):
                target_fz = unnorm_speech[:,-1]
                target_fz = np.exp(target_fz)
                self.concatenateMagPhaseEpoch(best_path, outstem + '.wav', fzero=target_fz)
            else:
                sys.exit('only support store_full_magphase_sep_files / store_full_magphase')
            self.stop_clock(start_time)          
            self.report( 'Output wave: %s.wav\n\n'%(outstem ))

        if self.mode_of_operation == 'stream_weight_balancing':
            tscores = self.get_target_scores_per_stream(unit_features, best_path)
            jscores = self.get_join_scores_per_stream(best_path)
            return (tscores, jscores)

        if self.config['get_selection_info']:
            trace_lines = self.get_path_information_epoch(unit_features, best_path)
            writelist(trace_lines, outstem + '.trace.txt')
            print 'Wrote trace file %s'%(outstem + '.trace.txt')
        
    def greedy_joint_search(self, unit_features, start_state=-1, holdout=[]):

        assert self.config['target_representation'] == 'epoch'

        start_time = self.start_clock('Greedy search')
        path = []
        m,n = self.current_join_rep.shape
        #m,n = self.join_contexts_unweighted.shape

        if start_state < 0:
            prev_join_vector = np.zeros((n,))
        else:
            prev_join_vector = self.prev_join_rep[start_state, :]


        multiepoch = self.config.get('multiepoch', 1)
        if multiepoch > 1:
            ### reshape target rep:
            m,n = unit_features.shape
            unit_features = segment_axis(unit_features, multiepoch, overlap=0, axis=0)
            unit_features = unit_features.reshape(m/multiepoch,n*multiepoch)

            if self.config.get('last_frame_as_target', False):
                print 'test -- take last frame only as target...'  ## TODO99
                # unit_features = unit_features[:,-n:]
                unit_features = np.hstack([unit_features[:,:n], unit_features[:,-n:]])

        ix = -1 
        final_dists = []   ### for debugging
        for target_vector in unit_features:
            both = np.concatenate([prev_join_vector, target_vector]).reshape((1,-1))
            # dists, indexes = self.joint_tree.query(both, k=7 + len(holdout)) # , n_jobs=4)
            dists, indexes = self.joint_tree.query(both, k=1+len(holdout), eps=self.config.get('search_epsilon', 0.0)) # , n_jobs=4)
            
            dindexes = zip(dists.flatten(), indexes.flatten())
            # if ix > -1:
            #     ## TODO: forbid regression -- configurable
            #     dindexes = [(d,i) for (d,i) in dindexes if i not in range(ix-5, ix+1)]
            #     dindexes = [(d,i) for (d,i) in dindexes if i not in holdout]
            
            (d, ix) = dindexes[0]
            path.append(ix)
            final_dists.append(d)
            prev_join_vector = self.current_join_rep[ix,:]
        self.stop_clock(start_time)
        return path



    ##### ========== concatenation methods ==========

    def preload_magphase_utts(self, path):
        '''
        preload utts used for a given path
        '''
        for index in path:
            if self.train_filenames[index] in self.waveforms: # self.config['hold_waves_in_memory']:  ### i.e. waves or magphase FFT spectra
                (mag_full, real_full, imag_full, f0_interp, vuv) = self.waveforms[self.train_filenames[index]]  
            else:     
                mag_full = get_speech(os.path.join(self.config['full_magphase_dir'], 'mag', self.train_filenames[index] + '.mag'), FFTHALFLEN)
                real_full = get_speech(os.path.join(self.config['full_magphase_dir'], 'real',  self.train_filenames[index] + '.real'), FFTHALFLEN)
                imag_full = get_speech(os.path.join(self.config['full_magphase_dir'], 'imag',  self.train_filenames[index] + '.imag'), FFTHALFLEN)
                f0_full = get_speech(os.path.join(self.config['full_magphase_dir'], 'f0',  self.train_filenames[index] + '.f0'), 1)            
                f0_interp, vuv = speech_manip.lin_interp_f0(f0_full)
                self.waveforms[self.train_filenames[index]] = (mag_full, real_full, imag_full, f0_interp, vuv)


    def preload_all_magphase_utts(self):
        start_time = self.start_clock('Preload magphase utts for corpus')
        for base in np.unique(self.train_filenames):
            print base
            mag_full = get_speech(os.path.join(self.config['full_magphase_dir'], 'mag', base + '.mag'), FFTHALFLEN)
            real_full = get_speech(os.path.join(self.config['full_magphase_dir'], 'real',  base + '.real'), FFTHALFLEN)
            imag_full = get_speech(os.path.join(self.config['full_magphase_dir'], 'imag',  base + '.imag'), FFTHALFLEN)
            f0_full = get_speech(os.path.join(self.config['full_magphase_dir'], 'f0', base + '.f0'), 1)            
            f0_interp, vuv = speech_manip.lin_interp_f0(f0_full)
            self.waveforms[base] = (mag_full, real_full, imag_full, f0_interp, vuv)
        self.stop_clock(start_time) 


    def retrieve_magphase_frag(self, index, extra_frames=0):

        if 0:
            print 'retrieving fragment'
            print self.train_filenames[index]
            print self.unit_index_within_sentence[index]

        if self.use_hdf_magphase:
            base = self.train_filenames[index]
            mag_full = self.hdf_magphase_pointer[base]['mag'][:]
            real_full = self.hdf_magphase_pointer[base]['real'][:]
            imag_full = self.hdf_magphase_pointer[base]['imag'][:]
            f0_interp = self.hdf_magphase_pointer[base]['f0_interp'][:]
            vuv = self.hdf_magphase_pointer[base]['vuv'][:]

        else:
            ## side effect -- data persists in self.waveforms. TODO: Protect against mem errors
            if False: # self.train_filenames[index] in self.waveforms: # self.config['hold_waves_in_memory']:  ### i.e. waves or magphase FFT spectra
                (mag_full, real_full, imag_full, f0_interp, vuv) = self.waveforms[self.train_filenames[index]]  
            else:     
                mag_full = get_speech(os.path.join(self.config['full_magphase_dir'], 'mag', self.train_filenames[index] + '.mag'), FFTHALFLEN)
                real_full = get_speech(os.path.join(self.config['full_magphase_dir'], 'real',  self.train_filenames[index] + '.real'), FFTHALFLEN)
                imag_full = get_speech(os.path.join(self.config['full_magphase_dir'], 'imag',  self.train_filenames[index] + '.imag'), FFTHALFLEN)
                f0_full = get_speech(os.path.join(self.config['full_magphase_dir'], 'f0',  self.train_filenames[index] + '.f0'), 1)            
                f0_interp, vuv = speech_manip.lin_interp_f0(f0_full)
                self.waveforms[self.train_filenames[index]] = (mag_full, real_full, imag_full, f0_interp, vuv)

        start_index = self.unit_index_within_sentence[index]
        #start_index -= 1  ### because magphase have extra pms beginning and end
        multiepoch = self.config.get('multiepoch', 1)
        end_index = start_index + multiepoch

        ## 
        start_pad = 0
        end_pad = 0        
        if extra_frames > 0:
            new_start_index = start_index - extra_frames
            new_end_index = end_index + extra_frames

            ## check out of bounds and record to zero pad later if necessary:
            nframes, _ = mag_full.shape

            if new_start_index < 0:
                start_pad = new_start_index * -1
            if new_end_index > nframes:
                end_pad = new_end_index - nframes 

            if start_pad > 0:
                start_index = 0
            else:
                start_index = new_start_index

            if end_pad > 0:
                end_index = nframes
            else:
                end_index = new_end_index

        if 0:
            print 'se'
            print (start_pad, end_pad)

        if 0:
            print '-----indices:  '
            print start_index, end_index
            print end_index - start_index
            print mag_full.shape

        mag_frag = mag_full[start_index:end_index, :]
        real_frag = real_full[start_index:end_index, :]
        imag_frag = imag_full[start_index:end_index, :]
        f0_frag = f0_interp[start_index:end_index, :]
        # f0_frag = f0_full[start_index:end_index, :]  ## !!!!!!!!!!!!!!!!!!!!!!!!!!
        vuv_frag = vuv[start_index:end_index, :]

        ### add zero padding where :
        mag_frag = zero_pad_matrix(mag_frag, start_pad, end_pad)
        real_frag = zero_pad_matrix(real_frag, start_pad, end_pad)
        imag_frag = zero_pad_matrix(imag_frag, start_pad, end_pad)
        f0_frag = zero_pad_matrix(f0_frag, start_pad, end_pad)
        vuv_frag = zero_pad_matrix(vuv_frag, start_pad, end_pad)
        # print mag_frag.shape

        # print '======'
        # print extra_frames
        
        # print new_start_index
        # print new_end_index

        # print start_pad
        # print end_pad


        ## sanity check dimensions
        m,n = mag_frag.shape
        if 0:
            print multiepoch
            print extra_frames
            print m
        assert m == multiepoch + (extra_frames*2)


        ### add taper (weighting for cross-fade):
        if extra_frames > 0:
            mag_frag = taper_matrix(mag_frag, extra_frames*2)
            real_frag = taper_matrix(real_frag, extra_frames*2)
            imag_frag = taper_matrix(imag_frag, extra_frames*2)
            #pylab.plot(f0_frag)            
            f0_frag = taper_matrix(f0_frag, extra_frames*2) 
            #print 'welvinrbo90'
            #pylab.plot(f0_frag)
            #pylab.show()
            vuv_frag = taper_matrix(vuv_frag, extra_frames*2)            


        return (mag_frag, real_frag, imag_frag, f0_frag, vuv_frag)


    def concatenateMagPhaseEpoch(self, path, fname, fzero=np.zeros(0)):

        print path
        print fzero
        print '------'
        mag = self.mp_mag[path,:] 
        imag = self.mp_imag[path,:] 
        real = self.mp_real[path,:] 
        fz = self.mp_fz[path,:].reshape((-1,1))

        if fzero.size > 0:
            fz = fzero

        # import pylab
        # pylab.plot(mag)
        # pylab.show()
        # sys.exit('aevsdb0000s')

        syn_wave = magphase.synthesis_from_lossless(mag, real, imag, fz, 48000)
        la.write_audio_file(fname, syn_wave, 48000)


    def concatenateMagPhaseEpoch_sep_files(self, path, fname, fzero=np.zeros(0), overlap=0):
        assert overlap % 2 == 0, 'frame overlap should be even number'

        multiepoch = self.config.get('multiepoch', 1)
        nframes = len(path) * multiepoch
        nframes += overlap ## beginning and ending fade in and out -- can trim these after

        mag = np.zeros((nframes, FFTHALFLEN))
        real = np.zeros((nframes, FFTHALFLEN))
        imag = np.zeros((nframes, FFTHALFLEN))
        fz = np.zeros((nframes, 1))
        vuv = np.zeros((nframes, 1))

        write_start = 0
        OFFSET = 0
        for ix in path:
            
            write_end = write_start + multiepoch + overlap
            (mag_frag, real_frag, imag_frag, fz_frag, vuv_frag) = self.retrieve_magphase_frag(ix, extra_frames=overlap/2)
            mag[write_start:write_end, :] += mag_frag
            real[write_start:write_end, :] += real_frag
            imag[write_start:write_end, :] += imag_frag
            #fz[write_start+(overlap/2):write_end-(overlap/2), :] += fz_frag[(overlap/2):-(overlap/2),:]
            fz[write_start:write_end, :] += fz_frag

            if 0:
                import pylab
                this_fz = np.zeros((nframes, 1))
                this_fz[write_start:write_end, :] += fz_frag
                pylab.plot(this_fz + OFFSET)
                OFFSET += 100

            vuv[write_start:write_end, :] += vuv_frag

            write_start += multiepoch

        if 0:
            pylab.show()
            sys.exit('sdcn89v9egvb')

        ## trim beginning fade in and end fade out:
        if overlap > 0:
            taper = overlap / 2
            mag = mag[taper:-taper, :]
            real = real[taper:-taper, :]
            imag = imag[taper:-taper, :]
            fz = fz[taper:-taper, :]
            vuv = vuv[taper:-taper, :]

        if fzero.size > 0:
            fz = fzero
        else:
            unvoiced = np.where(vuv < 0.5)[0]
            fz[unvoiced, :] = 0.0

        if 0:
            import pylab
            pylab.imshow( mag)
            pylab.show()



        if 0:
            import pylab
            pylab.plot(fz)
            pylab.show()
            sys.exit('evevwev9999')

        sample_rate = self.config.get('sample_rate', 48000)
        syn_wave = magphase.synthesis_from_lossless(mag, real, imag, fz, sample_rate)
        la.write_audio_file(fname, syn_wave, sample_rate)

        #speech_manip.put_speech(fz, fname + '.f0')
  


    ##### ========== utility methods ==========

    ## TODO_ verbosity level -- logging?
    def report(self, msg):
        if self.verbose:
            print msg

    def start_clock(self, comment):
        if self.verbose:
            print '%s... '%(comment),
        return (timeit.default_timer(), comment)

    def stop_clock(self, (start_time, comment), width=40):
        padding = (width - len(comment)) * ' '
        if self.verbose:
            print '%s--> took %.2f seconds' % (padding, (timeit.default_timer() - start_time))  ##  / 60.)  ## min




    ##### ========== reconfigure methods ==========


    def reconfigure_settings(self, changed_config_values):
        '''
        Currently used by weight tuning script -- adjust configs and rebuild trees etc as 
        necessary

        Return True if anything has changed in config, else False
        '''
        print 'reconfiguring synthesiser...'
        assert self.config['target_representation'] == 'epoch'
        assert self.config['greedy_search']

        for key in ['join_stream_weights', 'target_stream_weights', 'join_cost_weight', 'search_epsilon', 'multiepoch', 'magphase_use_target_f0', 'magphase_overlap']:
            assert key in changed_config_values, key

        rebuild_tree = False
        small_change = False ## dont need to rebuild, but register a change has happened
        
        if self.config['join_cost_weight'] != changed_config_values['join_cost_weight']:
            self.config['join_cost_weight'] = changed_config_values['join_cost_weight']
            rebuild_tree = True

        if self.config['join_stream_weights'] != changed_config_values['join_stream_weights']:
            self.config['join_stream_weights'] = changed_config_values['join_stream_weights']
            rebuild_tree = True

        if self.config['target_stream_weights'] != changed_config_values['target_stream_weights']:
            self.config['target_stream_weights'] = changed_config_values['target_stream_weights']
            rebuild_tree = True

        if self.config['multiepoch'] != changed_config_values['multiepoch']:
            self.config['multiepoch'] = changed_config_values['multiepoch']
            rebuild_tree = True

        if self.config.get('search_epsilon', 1.0) != changed_config_values['search_epsilon']:
            self.config['search_epsilon'] = changed_config_values['search_epsilon']
            small_change = True

        if self.config.get('magphase_use_target_f0', True) != changed_config_values['magphase_use_target_f0']:
            self.config['magphase_use_target_f0'] = changed_config_values['magphase_use_target_f0']
            small_change = True

        if self.config.get('magphase_overlap', 0) != changed_config_values['magphase_overlap']:
            self.config['magphase_overlap'] = changed_config_values['magphase_overlap']
            small_change = True

        if rebuild_tree:
            print 'set join weights after reconfiguring'
            self.set_join_weights(np.array(self.config['join_stream_weights']) * self.config['join_cost_weight'])
            print 'set target weights after reconfiguring'
            self.set_target_weights(np.array(self.config['target_stream_weights']) * (1.0 - self.config['join_cost_weight']))   
            print 'tree...'           
            self.get_tree_for_greedy_search()
            print 'done'

        return (rebuild_tree or small_change)



    def reconfigure_from_config_file(self):
        '''
        Currently used by weight tuning script -- adjust configs and rebuild trees etc as 
        necessary
        '''
        print 'Refresh config...'
        refreshed_config = {}
        execfile(self.config_file, refreshed_config)
        del refreshed_config['__builtins__']

        keys = ['join_stream_weights', 'target_stream_weights', 'join_cost_weight', 'search_epsilon', 'multiepoch', 'magphase_use_target_f0', 'magphase_overlap']
            
        changed_config_values = [refreshed_config[key] for key in keys]
        changed_config_values = dict(zip(keys, changed_config_values))
        anything_changed = self.reconfigure_settings(changed_config_values)

        return anything_changed

        


    ##### ========== diagnostic methods ==========

    def get_path_information_epoch(self, target_features, best_path):
        '''
        Store information about what was selected, where the joins are, what the costs
        were, etc. etc. to file
        '''
        data = []
        multiepoch = self.config.get('multiepoch', 1)
        for index in best_path:
            start_index = self.unit_index_within_sentence[index]
            end_index = start_index + multiepoch
            data.append( '%s %s %s'%(self.train_filenames[index], start_index, end_index) )
        return data
        ## !TODO: non-epoch version has more here which can be tailored





        
    #####================= questionable methods: keep or not? ========================

    def resynth_training_chunk(self, chunksize, outfile, seed=-1, natural=False, noisy=False):
        '''
        Resynthesise a randomly chosen chunk of training data, optionally holding out data occurring within that chunk
        '''

        assert self.config.get('target_representation') == 'epoch'
        assert self.config.get('greedy_search', False)

        if seed > -1:
            random.seed(seed)

        # find chunk location:
        start = random.randint(0,self.number_of_units-chunksize-1)
        unit_features = self.train_unit_features[start: start+chunksize, :]
        original_units = range(start, start+chunksize)

        ### setting start_state=start and holdout=[] will synthesise natural training 
        ### speech, where unit_features are consective target features in training data
        ### and start is index of first unit. (This might not hold if approximate search is used...)
        if natural:
            holdout_units = []
        else:
            holdout_units = original_units
        best_path = self.greedy_joint_search(unit_features, start_state=start, holdout=holdout_units)
        self.concatenate(best_path, outfile)


        if noisy:
            print 'Original units:'
            print original_units
            print
            print 'Path found:'
            print best_path

            if natural:
                assert best_path == original_units

    def __call__(self, fname):
        '''
        To enable parallelisation
        '''
        return self.synth_utt(fname)


    def random_walk(self, start_ix, outfname):
        #### TODO: move this to separate scriptfor tidying?
        import random
        #print self.train_unit_features.shape
        #sys.exit('vsdrbv')

        #walkdata = self.train_unit_features
        m,n = self.train_unit_features.shape
        left_left_context = np.vstack([np.zeros((20,n)), self.train_unit_features[:-20]])
        left_context = np.vstack([np.zeros((10,n)), self.train_unit_features[:-10]])
        walkdata = np.hstack([left_left_context, left_context, self.train_unit_features])

        print 'make tree'
        tree = StashableKDTree.StashableKDTree(walkdata, leaf_size=100, metric='euclidean')
        print 'done'
        assert start_ix < len(self.train_unit_features)
        path = [start_ix]
        ix = start_ix
        while len(path) < 1000:
            #print self.train_unit_features[ix,:].reshape((1,-1))
            #print self.train_unit_features[ix,:].reshape((1,-1)).shape
            (dists, indexes) = tree.query(walkdata[ix,:].reshape((1,-1)), k=10)
            cand = indexes.flatten()
            cand = [item for item in cand if item not in range(ix-5, ix+1)]
            ix = random.choice(cand)
            ix = ix + 1 
            path.append(ix)
        self.concatenate(path, outfname)
        print path


    #### reinstate?
    def get_selection_vector(self, stream_list, stream_dims, truncation_values):
        '''Return index list for selecting features by advanced indexing corresponding to the required truncation of streams'''
        assert len(truncation_values) == len(stream_list), (truncation_values, stream_list)
        selection_vector = []
        start = 0
        #print 'get_selection_vector'
        for (stream, trunc) in zip(stream_list, truncation_values):
            stream_dim = stream_dims[stream] 
            #print (stream_dim, trunc)
            if trunc == -1:
                trunc = stream_dim
            assert trunc <= stream_dim, 'stream %s has only %s dims, cannot truncate to %s'%(stream, stream_dim, trunc)
            selection_vector.extend(range(start, start+trunc))
            start += stream_dim  
        #print len(selection_vector)
        #print selection_vector
        return selection_vector

    #### reinstate?
    def truncate_join_streams(self, truncation_values):
        selection_vector = self.get_selection_vector(self.stream_list_join, self.datadims_join, truncation_values)

        if self.config['target_representation'] == 'epoch':
            ### for join streams, double up selection vector:
            dim = sum([self.datadims_join[stream] for stream in self.stream_list_join])
            selection_vector = selection_vector + [val + dim for val in selection_vector]

        self.unit_end_data = self.unit_end_data[:, selection_vector]
        self.unit_start_data = self.unit_start_data[:, selection_vector]
        
    #### reinstate?
    def truncate_target_streams(self, truncation_values):
        selection_vector = self.get_selection_vector(self.stream_list_target, self.datadims_target, truncation_values)
        self.train_unit_features = self.train_unit_features[:, selection_vector]
        if self.holdout_samples > 0:
            self.train_unit_features_dev = self.train_unit_features_dev[:, selection_vector]
        self.target_truncation_vector = selection_vector

    #! def test_concatenation_code(self):
    #     ofile = '/afs/inf.ed.ac.uk/user/o/owatts/temp/concat_test.wav'
    #     print 'concatenate the start of the training data, output here: %s'%(ofile)
    #     self.concatenate(np.arange(100, 150), ofile)    
        

    #### reinstate? (used by get_multiple_trees_for_greedy_search)
    def cluster(self, ncluster, limit=0):
        if limit == 0:
            data = self.train_unit_features
        else:
            data = random_subset_data(self.train_unit_features, train_frames=limit)
        t = self.start_clock('cluster')
        kmeans = KMeans(n_clusters=ncluster, init='k-means++', n_init=1, max_iter=300, tol=0.0001, precompute_distances=True, verbose=0, random_state=1234, copy_x=True, n_jobs=1, algorithm='auto')
        kmeans.fit(data)
        self.stop_clock(t)

        self.cbook = kmeans.cluster_centers_

        self.cbook_tree = scipy.spatial.cKDTree(self.cbook, leafsize=1, compact_nodes=True, balanced_tree=True)
        dists, cix = self.cbook_tree.query(self.train_unit_features)
        self.cluster_ixx = cix.flatten()


    #### reinstate?
    def get_multiple_trees_for_greedy_search(self):
        '''
        Partition data in hard way with k-means, build 1 KD-tree per partition
        '''

        #! m,n = self.unit_start_data.shape

        self.prev_join_rep = self.unit_start_data    ## !osw
        self.current_join_rep = self.unit_end_data   ## !osw

        start_time = self.start_clock('build multiple joint KD trees')
        ## Needs to be stored synthesis options specified (due to weights applied before tree construction):
        treefile = get_data_dump_name(self.config) + '_' + make_synthesis_condition_name(self.config) + '_joint_tree.pkl' 

        multiepoch = self.config.get('multiepoch', 1)
        overlap = 0
        if multiepoch > 1:
            overlap = multiepoch-1
            ### reshape target rep:
            m,n = self.train_unit_features.shape
            self.train_unit_features = segment_axis(self.train_unit_features, multiepoch, overlap=overlap, axis=0)
            self.train_unit_features = self.train_unit_features.reshape(m-overlap,n*multiepoch)

            if self.config.get('last_frame_as_target', False):
                print 'test -- take last frame only as target...'  ## TODO99
                # self.train_unit_features = self.train_unit_features[:,-n:]
                self.train_unit_features = np.hstack([self.train_unit_features[:,:n], self.train_unit_features[:,-n:]])

            ### alter join reps: -- first tried taking first vs. last
            m,n = self.current_join_rep.shape
            self.current_join_rep = self.current_join_rep[overlap:,:]
            self.prev_join_rep = self.prev_join_rep[:-overlap, :]

        #### ---- cluster self.train_unit_features
        self.cluster(self.config.get('multiple_search_trees', 1), limit=self.config.get('cluster_data_on_npoints', 10000))
        ### ^--- populates self.cluster_ixx

        combined_rep = np.hstack([self.prev_join_rep, self.train_unit_features])

        self.joint_trees = []
        self.node_maps = []

        for cluster_number in range(self.config.get('multiple_search_trees', 1)):
            t = self.start_clock('make joint join + target tree for cluster %s...'%(cluster_number))
            self.joint_trees.append(scipy.spatial.cKDTree(combined_rep[self.cluster_ixx==cluster_number, :], leafsize=100, balanced_tree=False, compact_nodes=False))
            self.node_maps.append(np.arange(self.number_of_units-overlap)[self.cluster_ixx==cluster_number])
            self.stop_clock(t)
        



if __name__ == '__main__':

    #################################################
      
    # ======== process command line ==========

    a = ArgumentParser()
    a.add_argument('-c', dest='config_fname', required=True)
    a.add_argument('-o', dest='output_dir', required=False, default='')
    opts = a.parse_args()

    synth = Synthesiser(opts.config_fname)

    if opts.output_dir:
        if not os.path.isdir(opts.output_dir):
            os.makedirs(opts.output_dir)
        os.system('cp %s %s'%(opts.config_fname, opts.output_dir))

    synth.synth_from_config(inspect_join_weights_only=False, synth_type='test', outdir=opts.output_dir)

