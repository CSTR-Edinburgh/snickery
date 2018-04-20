#!/usr/bin/env python
# -*- coding: utf-8 -*-
## Project:  
## Author: Oliver Watts - owatts@staffmail.ed.ac.uk

import sys
import os
import glob
import re
import timeit
import math
import copy
import random
from argparse import ArgumentParser

# Cassia added
import smoothing.fft_feats as ff
import smoothing.libwavgen as lwg
import smoothing.libaudio as la


# modify import path to obtain modules from the tools/magphase/src directory:
snickery_dir = os.path.split(os.path.realpath(os.path.abspath(os.path.dirname(__file__))))[0]+'/'
sys.path.append(os.path.join(snickery_dir, 'tool', 'magphase', 'src'))
import magphase
import libaudio as la


import numpy as np
import scipy

import h5py
import pywrapfst as openfst

# from sklearn.neighbors import KDTree as sklearn_KDTree
import StashableKDTree


from sklearn.cluster import KMeans

from util import safe_makedir, vector_to_string, basename, writelist
from speech_manip import read_wave, write_wave, weight, get_speech
from label_manip import break_quinphone, extract_monophone
from train_halfphone import get_data_dump_name, compose_speech, standardise, destandardise, \
        read_label, get_halfphone_stats, reinsert_terminal_silence, make_train_condition_name, \
        locate_stream_directories

DODEBUG=False ## print debug information?

from segmentaxis import segment_axis
from train_halfphone import debug

from const import VERY_BIG_WEIGHT_VALUE

import pylab

import speech_manip



WRAPFST=True # True: used python bindings (pywrapfst) to OpenFST; False: use command line interface

assert WRAPFST

if WRAPFST:
    from fst_functions_wrapped import compile_fst, make_target_sausage_lattice, cost_cache_to_text_fst, get_best_path_SIMP, compile_lm_fst, make_mapping_loop_fst, plot_fst, extract_path, compile_simple_lm_fst, sample_fst, make_sausage_lattice, cost_cache_to_compiled_fst
else:
    from fst_functions import compile_fst, make_t_lattice_SIMP, cost_cache_to_text_fst, get_best_path_SIMP, compile_lm_fst, make_mapping_loop_fst

import const
from const import label_delimiter


import cPickle as pickle

# import matplotlib.pyplot as plt; plt.rcdefaults()
# import matplotlib.pyplot as plt
import pylab

# verbose = False # True # False

APPLY_JCW_ON_TOP = True     ## for IS2018 -- scale weights by jcw

from const import FFTHALFLEN 
HALFFFTLEN = FFTHALFLEN

## TODO: where to put this?
def zero_pad_matrix(a, start_pad, end_pad):
    '''
    if start_pad and end_pad are both 0, do nothing
    '''
    if start_pad > 0:
        dim = a.shape[1] 
        a = np.vstack([np.zeros((start_pad, dim)), a])
    if end_pad > 0:
        dim = a.shape[1] 
        a = np.vstack([a, np.zeros((end_pad, dim))])
    return a

def taper_matrix(a, taper_length):
    m,n = a.shape
    assert taper_length * 2 <= m, 'taper_length (%s) too long for (padded) unit length (%s)'%(taper_length, m) 
    in_taper = np.hanning(((taper_length + 1)*2)+1)[1:taper_length+1].reshape(-1,1)
    out_taper = np.flipud(in_taper).reshape(-1,1)
    if 0:
        pylab.plot(in_taper)
        pylab.plot(out_taper)
        pylab.plot((in_taper + out_taper)-0.05)   ### check sum to 1
        pylab.show()
        sys.exit('wrvwsfrbesbr')
    a[:taper_length,:] *= in_taper
    a[-taper_length:,:] *= out_taper
    return a


def suppress_weird_festival_pauses(label, replace_list=['B_150'], replacement='pau'):
    outlabel = []
    for ((s,e), quinphone) in label:
        new_quinphone = []
        for phone in quinphone:
            if phone in replace_list:
                new_quinphone.append(replacement)
            else:
                new_quinphone.append(phone)
        outlabel.append(((s,e),new_quinphone))
    return outlabel


def random_subset_data(data, seed=1234, train_frames=0):
    '''
    shuffle and select subset of data; train_frames==0 : all
    '''
    shuff_data = copy.copy(data)
    np.random.seed(seed)
    np.random.shuffle(shuff_data)   
    m,n = np.shape(shuff_data)
    if train_frames == 0:
        train_frames = m
    if m < train_frames:
        train_frames = m
    shuff_data = shuff_data[:train_frames, :]
    print 'selected %s of %s frames for learning codebook(s)'%(train_frames, m)
    #put_speech(train_data, top_vq_dir + '/traindata_subset.cmp')
    return shuff_data    
        

class Synthesiser(object):

    def __init__(self, config_file, holdout_percent=0.0):



        self.mode_of_operation = 'normal'
        self.verbose = True

        print 'Load config...'
        self.config = {}
        execfile(config_file, self.config)
        del self.config['__builtins__']
        

        self.config_file = config_file   ## in case we need to refresh...


        self.stream_list_target = self.config['stream_list_target']
        self.stream_list_join = self.config['stream_list_join']

        print 'Prepare weights from config'
        ### TODO: check!
        self.datadims_target = self.config['datadims_target']
        self.datadims_join = self.config['datadims_join']

        self.target_representation = self.config['target_representation']

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
        self.train_cutpoints = f["cutpoints"][:] 
        self.train_filenames = f["filenames"][:]                 
        self.mean_vec_target = f["mean_target"][:] 
        self.std_vec_target = f["std_target"][:] 
        self.mean_vec_join = f["mean_join"][:] 
        self.std_vec_join = f["std_join"][:] 

        self.join_contexts_unweighted = f["join_contexts"][:,:]


        if self.config.get('store_full_magphase', False):
            self.mp_mag = f["mp_mag"][:] 
            self.mp_imag = f["mp_imag"][:] 
            self.mp_real = f["mp_real"][:] 
            self.mp_fz = f["mp_fz"][:]                                     


        if 'unit_index_within_sentence_dset' in f:
            self.unit_index_within_sentence = f['unit_index_within_sentence_dset'][:]

        f.close()

        self.number_of_units, _ = self.train_unit_features_unweighted.shape

        self.holdout_percent = holdout_percent
        self.holdout_samples = 0
        if holdout_percent > 0.0:
            holdout_samples = int(self.number_of_units * (holdout_percent/100.0))
            print 'holdout_samples:'
            print holdout_samples
            #holdout_indices = np.random.choice(m, size=npoint, replace=False)

            if 0:
                print 'check data all present...'
                rwsum = self.train_unit_features_unweighted.sum(axis=1)
                pylab.plot(rwsum)
                pylab.show()
                sys.exit('asdbvsrfbsfrb0000')

            
            self.train_unit_features_unweighted_dev = self.train_unit_features_unweighted[-holdout_samples:,:]
            self.train_unit_features_unweighted = self.train_unit_features_unweighted[:-holdout_samples,:]
        
            self.train_unit_names_dev = self.train_unit_names[-holdout_samples:]
            self.train_unit_names = self.train_unit_names[:-holdout_samples]

            self.number_of_units -= holdout_samples
            #sys.exit('evwservgwsrbv')
            self.holdout_samples = holdout_samples






        if APPLY_JCW_ON_TOP:
            if self.config.get('weight_target_data', True):
                self.set_target_weights(np.array(self.config['target_stream_weights']) * (1.0 - self.config['join_cost_weight'])) 
            if self.config.get('weight_join_data', True):
                self.set_join_weights(np.array(self.config['join_stream_weights']) * self.config['join_cost_weight'])

        else:
            if self.config.get('weight_target_data', True):
                self.set_target_weights(self.config['target_stream_weights'])
            if self.config.get('weight_join_data', True):
                self.set_join_weights(self.config['join_stream_weights'])

        # if 'truncate_target_streams' in self.config:
        #     self.truncate_target_streams(self.config['truncate_target_streams'])
        # if 'truncate_join_streams' in self.config:
        #     self.truncate_join_streams(self.config['truncate_join_streams'])



        self.first_silent_unit = 0 ## assume first unit is a silence, for v naive backoff

        if self.config['target_representation'] != 'epoch':
            self.quinphone_regex = re.compile(self.config['quinphone_regex'])

            print 'prepare data for search...'
            self.unit_index = {}
            for (i,quinphone) in enumerate(self.train_unit_names):
                mono, diphone, triphone, quinphone = break_quinphone(quinphone)
                #extract_quinphone(quinphone)
                for form in [mono, diphone, triphone, quinphone]:
                    if form not in self.unit_index:
                        self.unit_index[form] = []
                    self.unit_index[form].append(i)
        else:
            print 'epochs -- no indexing by label. Prepare search tree instead...'
            print 'Set preselection to acoustic'
            self.config['preselection_method'] = 'acoustic'

        ## set up some shorthand:-
        self.tool = self.config.get('openfst_bindir', '')

        self.waveforms = {}

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

        assert self.config['preselection_method'] in ['acoustic', 'quinphone', 'monophone_then_acoustic']





        if self.config.get('greedy_search', False):

            assert self.config['target_representation'] == 'epoch'
            
            if self.config.get('multiple_search_trees', 1) > 1:
                sys.exit('multiple_search_trees not implemented yet -- try adjusting search_epsilon instead to speed up search')
                self.get_multiple_trees_for_greedy_search()
            else:
                self.get_tree_for_greedy_search()


        elif self.config['preselection_method'] == 'acoustic':

            start_time = self.start_clock('build/reload KD tree')
            treefile = get_data_dump_name(self.config, searchtree=True)
            if False: # os.path.exists(treefile):   ##### <---- for now, just rebuild tree at synthesis time
                print 'Tree file found -- reload from %s'%(treefile)
                self.tree = StashableKDTree.resurrect_tree(treefile)
            else:
                print 'Seems like this is first time synthesis has been run on this data.'
                print 'Build a search tree which will be saved for future use'

                
                
                # if config['kdt_implementation'] == 'sklearn':
                #train = weight(self.train_unit_features, self.target_weight_vector)
                train = self.train_unit_features
                #self.tree = sklearn_KDTree(train, leaf_size=1, metric='euclidean')   
                # elif config['kdt_implementation'] == 'scipy':
                #     tree = scipy_cKDTree(train, leafsize=1) 
                # elif config['kdt_implementation'] == 'stashable':

                ### This was just to test build time of different KD tree implementations.
                ### TODO: move to separate script/note. 
                test_tree_build_times = False
                if test_tree_build_times:

                    np.random.shuffle(train)
                    for datasize in range(10000, 1000000, 10000):
                        sub_start_time = self.start_clock('build/reload KD tree: %s'%(datasize))
                        tree = StashableKDTree.StashableKDTree(train[:datasize,:], leaf_size=100, metric='euclidean')
                        #tree = scipy.spatial.cKDTree(train[:datasize,:], leafsize=100, compact_nodes=False, balanced_tree=False)
                        self.stop_clock(sub_start_time)

                #self.tree = StashableKDTree.StashableKDTree(train, leaf_size=100, metric='euclidean')
                self.tree = scipy.spatial.cKDTree(train, leafsize=100, compact_nodes=False, balanced_tree=False)
                print '...'
                #self.tree.save_hdf(treefile)  ##### <---- for now, just rebuild tree at synthesis time
            self.stop_clock(start_time)


        elif self.config['preselection_method'] == 'monophone_then_acoustic':

            start_time = self.start_clock('build KD trees for search by phone')
            self.phonetrees = {}
            self.phonetrees_index_converters = {}
            monophones = np.array([quinphone.split(const.label_delimiter)[2] for quinphone in self.train_unit_names])
            monophone_inventory = dict(zip(monophones,monophones))

            for phone in monophone_inventory:

                train = self.train_unit_features[monophones==phone, :]
                # print phone
                # print (monophones==phone)
                # print train.shape
                tree = scipy.spatial.cKDTree(train, leafsize=10, compact_nodes=False, balanced_tree=False)
                self.phonetrees[phone] = tree
                self.phonetrees_index_converters[phone] = np.arange(self.number_of_units)[monophones==phone]
            self.stop_clock(start_time)
            # print self.phonetrees
            # sys.exit('aedvsb')
        print 'Database loaded'
        print '\n\n----------\n\n'


        self.test_data_target_dirs = locate_stream_directories(self.config['test_data_dirs'], self.stream_list_target)

        print 'Found target directories: %s'%(self.test_data_target_dirs)
        print 
        print 
        if self.config.get('tune_data_dirs', ''):
            self.tune_data_target_dirs = locate_stream_directories(self.config['tune_data_dirs'], self.stream_list_target)

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

        


    def get_tree_for_greedy_search(self):

        m,n = self.unit_start_data.shape

        ## Prev and current frames for join cost -- this is obtained in non-obvious way from
        ## data written in training. TODO: consider storing this data differently in training?

        ## copy the data (not reference it) here so the original join_contexts_unweighted 
        ## is unaffected and we can later have other weights applied:-
#            self.prev_join_rep = copy.copy(self.join_contexts_unweighted[:-1,:n/2]) ## all but last frame         
#            self.current_join_rep = copy.copy(self.join_contexts_unweighted[:-1,n/2:]) ## all but last frame 

        ## this should do same thing using weights applied from config:--
        self.prev_join_rep = self.unit_start_data[:,:n/2]       
        self.current_join_rep = self.unit_start_data[:,n/2:]



        #start_time = self.start_clock('build/reload joint KD tree')
        ## Needs to be stored synthesis options specified (due to weights applied before tree construction):
        treefile = get_data_dump_name(self.config) + '_' + self.make_synthesis_condition_name() + '_joint_tree.pkl' 
        if False: # os.path.exists(treefile):  ## never reload!
            print 'Tree file found -- reload from %s'%(treefile)
            self.joint_tree = pickle.load(open(treefile,'rb'))
        else:
            #print 'Seems like this is first time synthesis has been run on this data.'
            #print 'Build a search tree which will be saved for future use'

            multiepoch = self.config.get('multiepoch', 1)
            if multiepoch > 1:
                t = self.start_clock('reshape data for multiepoch...')
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

                ### then, whole comparison for join:
                # m,n = self.current_join_rep.shape
                # self.current_join_rep = segment_axis(self.current_join_rep, multiepoch, overlap=overlap, axis=0).reshape(m-overlap,n*multiepoch)
                # self.prev_join_rep = segment_axis(self.prev_join_rep, multiepoch, overlap=overlap, axis=0).reshape(m-overlap,n*multiepoch)
                self.stop_clock(t)

            #print self.prev_join_rep.shape
            #print self.train_unit_features.shape
            t = self.start_clock('stack data to train joint tree...')
            combined_rep = np.hstack([self.prev_join_rep, self.train_unit_features])
            self.stop_clock(t)
            

            #self.report('make joint join + target tree...')
            t = self.start_clock('make joint join + target tree...')
            ### scipy.spatial.cKDTree is used for joint tree instead of sklearn one due to
            ### speed of building. TODO: Also change in standard acoustic distance case (non-greedy).
            ### TODO: check speed and reliability of pickling, could look at HDF storage as 
            ### we did for StashableKDTree? Compare resurrection time with rebuild time.
            self.joint_tree = scipy.spatial.cKDTree(combined_rep, leafsize=100, balanced_tree=False) # , compact_nodes=False)
            #print 'done -- now pickle...'
            ### TODO: seems rebuilding is much quicker than reloading (at least -> 3000 sentences).
            #pickle.dump(self.joint_tree,open(treefile,'wb'))
            self.stop_clock(t)
            


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



    def get_multiple_trees_for_greedy_search(self):
        '''
        Partition data in hard way with k-means, build 1 KD-tree per partition
        '''

        m,n = self.unit_start_data.shape

        self.prev_join_rep = self.unit_start_data[:,:n/2]       
        self.current_join_rep = self.unit_start_data[:,n/2:]

        start_time = self.start_clock('build multiple joint KD trees')
        ## Needs to be stored synthesis options specified (due to weights applied before tree construction):
        treefile = get_data_dump_name(self.config) + '_' + self.make_synthesis_condition_name() + '_joint_tree.pkl' 

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
        



    def set_join_weights(self, weights):
        assert len(weights) == len(self.stream_list_join)
        
        ## get from per-stream to per-coeff weights:
        join_weight_vector = []
        for (i,stream) in enumerate(self.stream_list_join):
            # if stream in const.vuv_stream_names:
            #     join_weight_vector.extend([weights[i]]*2)
            # else:
                join_weight_vector.extend([weights[i]]*self.datadims_join[stream])

        if self.target_representation == 'epoch':
            ## then natural2 type cost -- double up:
            join_weight_vector = join_weight_vector + join_weight_vector

        join_weight_vector = np.array(join_weight_vector)
        ## TODO: be more explicit about how this copies and does NOT weight original self.join_contexts_unweighted
        join_contexts_weighted = weight(self.join_contexts_unweighted, join_weight_vector)   

        ## This should not copy:
        self.unit_end_data = join_contexts_weighted[1:,:]
        self.unit_start_data = join_contexts_weighted[:-1,:]

        if self.holdout_samples > 0:
            self.unit_end_data = self.unit_end_data[:-self.holdout_samples,:]
            self.unit_start_data = self.unit_start_data[:-self.holdout_samples,:]


        # print 'applied join_weight_vector'
        # print join_weight_vector

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

        # print 'applied taget_weight_vector'
        # print target_weight_vector

    # def get_selection_vector(stream_list, stream_dims, truncation_values):
    #     '''Return binary vector for selecting features by advanced indexing corresponding to the required truncation of streams'''
    #     assert len(truncation_values) == len(stream_list), (truncation_values, stream_list)
    #     dim = sum([stream_dims[stream] for stream in stream_list])
    #     selection_vector = np.zeros(dim)
    #     start = 0
    #     for (stream, trunc) in zip(stream_list, truncation_values):
    #         stream_dim = stream_dims[stream] 
    #         if trunc != -1:                
    #             assert trunc <= stream_dim
    #             selection_vector[start:start+trunc] = 1
    #         start += stream_dim  
    #     return selection_vector     

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

    def truncate_join_streams(self, truncation_values):
        selection_vector = self.get_selection_vector(self.stream_list_join, self.datadims_join, truncation_values)

        if self.config['target_representation'] == 'epoch':
            ### for join streams, double up selection vector:
            dim = sum([self.datadims_join[stream] for stream in self.stream_list_join])
            selection_vector = selection_vector + [val + dim for val in selection_vector]

        self.unit_end_data = self.unit_end_data[:, selection_vector]
        self.unit_start_data = self.unit_start_data[:, selection_vector]
        

    def truncate_target_streams(self, truncation_values):
        selection_vector = self.get_selection_vector(self.stream_list_target, self.datadims_target, truncation_values)
        self.train_unit_features = self.train_unit_features[:, selection_vector]
        if self.holdout_samples > 0:
            self.train_unit_features_dev = self.train_unit_features_dev[:, selection_vector]
        self.target_truncation_vector = selection_vector

    def test_concatenation_code(self):
        ofile = '/afs/inf.ed.ac.uk/user/o/owatts/temp/concat_test.wav'
        print 'concatenate the start of the training data, output here: %s'%(ofile)
        self.concatenate(np.arange(100, 150), ofile)    
        

    def get_sentence_set(self, set_name): 
        assert set_name in ['test', 'tune']

        first_stream = self.stream_list_target[0]

        if set_name == 'test':
            data_dirs = self.test_data_target_dirs[first_stream]
            name_patterns = self.config['test_patterns']
            limit = self.config['n_test_utts']
        elif set_name == 'tune':
            data_dirs = self.tune_data_target_dirs[first_stream]
            name_patterns = self.config['tune_patterns']
            limit = self.config['n_tune_utts']            
        else:
            sys.exit('Set name unknown: "%s"'%(set_name))

        flist = sorted(glob.glob(data_dirs + '/*.' + first_stream))
        flist = [basename(fname) for fname in flist]

        ## find all files containing one of the patterns in test_patterns
        L = len(flist)
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


    def get_test_sentences(self):

        first_stream = self.stream_list_target[0]
        test_flist = sorted(glob.glob(self.test_data_target_dirs[first_stream] + '/*.' + first_stream))

        ## find all files containing one of the patterns in test_patterns
        
        L = len(test_flist)
        selected_test_flist = []
        for (i,fname) in enumerate(test_flist):
            for pattern in self.config['test_patterns']:
                if pattern in fname:
                    if fname not in selected_test_flist:
                        selected_test_flist.append(fname)
                    

        test_flist = selected_test_flist 

        ### Only synthesise n sentences:
        test_flist = test_flist[:self.config['n_test_utts']]

        ntest = len(test_flist)
        if ntest == 0:
            print 'No test files found based on configured test_data_dir and test_pattern'
        else:
            self.report('get_test_sentences: synthesising %s utternances based on config'%(ntest))
        return test_flist



    def synth_from_config(self, inspect_join_weights_only=False, synth_type='test', outdir=''):

        self.report('synth_from_config')


        test_flist = self.get_sentence_set(synth_type)



        # if self.mode_of_operation == 'stream_weight_balancing':
        #     all_tscores = []
        #     all_jscores = []


        # all_distances = []

        for fname in test_flist:
            # if inspect_join_weights_only:
            #     all_distances.append(self.inspect_join_weights_on_utt(fname))
            #     # print 'single utt -- break'
            #     # break 
            # else:

            # if self.mode_of_operation == 'stream_weight_balancing':
            #     tscores, jscores = self.synth_utt(fname)    
            #     all_tscores.append(tscores)
            #     all_jscores.append(jscores)
            # else:    


            self.synth_utt(fname, synth_type=synth_type, outdir=outdir)    

        # if self.mode_of_operation == 'stream_weight_balancing':
        #     all_tscores = np.vstack(all_tscores)
        #     all_jscores = np.vstack(all_jscores)

        #     mean_tscores = np.mean(all_tscores, axis=0)
        #     mean_jscores = np.mean(all_jscores, axis=0)

        #     # self.report( '------------- join and target cost summaries by stream -----------')
        #     # self.report('')

        #     # for (stream, mu) in zip (self.stream_list_join, mean_jscores ):
        #     #     self.report( 'join   %s -- mean: %s   '%(stream.ljust(10), mu))
        #     # self.report('')
        #     # for (stream, mu) in zip (self.stream_list_target, mean_tscores ):
        #     #     self.report( 'target %s -- mean: %s   '%(stream.ljust(10), mu))
        #     # self.report('')
        #     # self.report( '--------------------------------------------------------------------')

        #     if verbose:
        #         print( '------------- join and target cost summaries by stream -----------')
        #         print('')

        #         for (stream, mu) in zip (self.stream_list_join, mean_jscores ):
        #             print( 'join   %s -- mean: %s   '%(stream.ljust(10), mu))
        #         print('')
        #         for (stream, mu) in zip (self.stream_list_target, mean_tscores ):
        #             print( 'target %s -- mean: %s   '%(stream.ljust(10), mu))
        #         print('')
        #         print( '--------------------------------------------------------------------')


        #     return (all_tscores, all_jscores)
        #     #return ((all_tscores.mean(axis=0), all_jscores.mean(axis=0)),  (all_tscores.std(axis=0), all_jscores.std(axis=0)))



    def junk(self):
        if inspect_join_weights_only:
            # import scipy
            all_distances = np.vstack(all_distances)

            ## this is per coeff square error terms; now split into streams:
            start = 0
            distance_by_stream = []
            for stream_name in self.stream_list_join: 
                stream_width = self.datadims_join[stream_name]
                distance_by_stream.append(all_distances[:,start:start+stream_width])
                start += stream_width
            stream_sums = [stream_dist.sum() for stream_dist in distance_by_stream] ## sum over diff joins and over coeffs

            print stream_sums
            #sys.exit('stop here for now')


            maxi = max(stream_sums)
            #factors = [(maxi / stream_sum) / self.datadims_join[stream] for (stream, stream_sum) in zip(self.stream_list_join, stream_sums)]
            factors = [(maxi / stream_sum) for (stream, stream_sum) in zip(self.stream_list_join, stream_sums)]


            normed_distance_by_stream = [stream_vals * factor for (stream_vals, factor) in zip(distance_by_stream, factors)]
            for thing in normed_distance_by_stream:
                print thing.shape
            normed_stream_sums = [stream_dist.sum() for stream_dist in normed_distance_by_stream]

            print stream_sums
            print factors
            print normed_stream_sums


            print 'stream weights'
            print np.sqrt(np.array(factors))

            summed_normed_distance_by_stream = [stream_vals.sum(axis=1).reshape((-1,1)) for stream_vals in normed_distance_by_stream]
            print [t.shape for t in summed_normed_distance_by_stream]
            summed_normed_distance_by_stream = np.hstack(summed_normed_distance_by_stream)
            m,n = summed_normed_distance_by_stream.shape
            ## make shared bins for all superimposed histogram plots:
            nbins = 100
            (mini, maxi) = (summed_normed_distance_by_stream.min(), summed_normed_distance_by_stream.max())
            binwidth = (maxi-mini) / nbins
            bins = np.arange(mini, maxi + binwidth, binwidth)
            for stream in range(n):
                pylab.hist(summed_normed_distance_by_stream[:,stream], bins=bins, alpha=0.3) # , label=self.stream_list_join[stream])
            pylab.legend()
            pylab.show()
            print summed_normed_distance_by_stream.sum(axis=0) # [ 194773.71157401   93592.93011832  175219.98631917]




            sys.exit('---5555----')
            #all_distances *= np.array([ 1.  ,        2.08107291 , 1.11159529])


            # diffs = all_distances[:,0] - all_distances[:,1]
            # pylab.hist(diffs)
            # pylab.show()
            # print scipy.stats.wilcoxon(diffs)
            # sys.exit('wesv')


            m,n = all_distances.shape
            ## make shared bins for all superimposed histogram plots:
            nbins = 100
            (mini, maxi) = (all_distances.min(), all_distances.max())
            binwidth = (maxi-mini) / nbins
            bins = np.arange(mini, maxi + binwidth, binwidth)
            for stream in range(n):
                pylab.hist(all_distances[:,stream], bins=bins, alpha=0.3) # , label=self.stream_list_join[stream])
            pylab.legend()
            pylab.show()
            print all_distances.sum(axis=0) # [ 194773.71157401   93592.93011832  175219.98631917]



    '''
import numpy as np
sums = np.array([ 13811.11409336 , 78233.73970166 , 11202.56575783])
maxi = sums.max()
factors = maxi / sums
nrmed = sums * factors
print nrmed
print factors


def get_facts(vals):
    import numpy as np
    sums = np.array(vals)
    maxi = sums.max()
    factors = maxi / sums
    nrmed = sums * factors
    print nrmed
    print factors

    '''


    def make_synthesis_condition_name(self):
        '''
        Return string encoding all variables which can be ... 
        '''
        #### Previous version: compressed representtion of coeff per dimension
        # target_weights = vector_to_string(self.config['feature_weights_target'])
        # join_weights = vector_to_string(self.config['feature_weights_join'])

        if self.config.get('synth_smooth', False):
            smooth='smooth_'
        else:
            smooth=''

        if self.config.get('greedy_search', False): 
            greedy = 'greedy-yes_'
        else:
            greedy = 'greedy-no_'

        ##### Current version: weight per stream.
        target_weights = '-'.join([str(val) for val in self.config['target_stream_weights']])
        if self.config['target_representation'] == 'sample':
            name = 'sample_target-%s'%(target_weights)
        else:
            join_weights = '-'.join([str(val) for val in self.config['join_stream_weights']])
            jcw = self.config['join_cost_weight']
            jct = self.config.get('join_cost_type', 'natural2')  ## TODO: not relevant to epoch, provided default consistent with IS2018 exp
            nc = self.config.get('n_candidates', 30) ## TODO: not relevant to epoch, provided default consistent with IS2018 exp
            tl = self.config.get('taper_length', 50) ## TODO: not relevant to epoch, provided default consistent with IS2018 exp
            name = '%s%starget-%s_join-%s_scale-%s_presel-%s_jmetric-%s_cand-%s_taper-%s'%(
                        greedy, smooth,
                        target_weights, join_weights, jcw,
                        self.config['preselection_method'],
                        jct,
                        nc,
                        tl
                    )    
            name += 'multiepoch-%s'%(self.config.get('multiepoch', 1))        
        return name


    def __call__(self, fname):
        '''
        To enable parallelisation
        '''
        return self.synth_utt(fname)


    # def synth_utts_bulk(self, fnames, synth_type='test'): 
    #     ## TODO: does not behave the same as synth_utt ... why not?
    #     '''
    #     Share a single join lattice to save time---
    #     '''
    #     if synth_type == 'test':
    #         data_dirs = self.test_data_target_dirs
    #         lab_dir = self.config['test_lab_dir']
    #     elif synth_type == 'tune':
    #         data_dirs = self.tune_data_target_dirs
    #         lab_dir = self.config['tune_lab_dir']            
    #     else:
    #         sys.exit('Unknown synth_type  9058324378375')


    #     train_condition = make_train_condition_name(self.config)
    #     synth_condition = self.make_synthesis_condition_name()
    #     synth_dir = os.path.join(self.config['workdir'], 'synthesis', train_condition, synth_condition)
    #     safe_makedir(synth_dir)
        
    #     candidates_per_utt = []
    #     distances_per_utt = []
    #     T_per_utt = []
    #     target_features_per_utt = []

    #     for base in fnames:
            
    #         self.report('               ==== SYNTHESISE %s ===='%(base))
            
    #         outstem = os.path.join(synth_dir, base)       

    #         start_time = self.start_clock('Get speech ')
    #         speech = compose_speech(data_dirs, base, self.stream_list_target, \
    #                                 self.config['datadims_target']) 

    #         m,dim = speech.shape

    #         if (self.config['standardise_target_data'], True):                                
    #             speech = standardise(speech, self.mean_vec_target, self.std_vec_target)         
            
    #         #fshift_seconds = (0.001 * self.config['frameshift_ms'])
    #         #fshift = int(self.config['sample_rate'] * fshift_seconds)        

    #         labfile = os.path.join(lab_dir, base + '.' + self.config['lab_extension'])
    #         labs = read_label(labfile, self.quinphone_regex)

    #         if self.config.get('untrim_silence_target_speech', False):
    #             speech = reinsert_terminal_silence(speech, labs)

    #         if self.config.get('suppress_weird_festival_pauses', False):
    #             labs = suppress_weird_festival_pauses(labs)

    #         unit_names, unit_features, unit_timings = get_halfphone_stats(speech, labs)
           
    #         if self.config['weight_target_data']:                                
    #             unit_features = weight(unit_features, self.target_weight_vector)       

    #         #print unit_features
    #         #print unit_names

    #         n_units = len(unit_names)
    #         self.stop_clock(start_time)


    #         if self.config['preselection_method'] == 'acoustic':

    #             start_time = self.start_clock('Acoustic select units ')
    #             ## call has same syntax for sklearn and scipy KDTrees:--
    #             distances, candidates = self.tree.query(unit_features, k=self.config['n_candidates'])
    #             self.stop_clock(start_time) 

    #         elif self.config['preselection_method'] == 'quinphone':

    #             start_time = self.start_clock('Preselect units ')
    #             #candidates = np.ones((n_units, self.config['n_candidates'])) * -1
    #             candidates = []
    #             for quinphone in unit_names:
    #                 current_candidates = []
    #                 mono, diphone, triphone, quinphone = break_quinphone(quinphone) 
    #                 for form in [quinphone, triphone, diphone, mono]:
    #                     for unit in self.unit_index.get(form, []):
    #                         current_candidates.append(unit)
    #                         if len(current_candidates) == self.config['n_candidates']:
    #                             break
    #                     if len(current_candidates) == self.config['n_candidates']:
    #                         break
    #                 if len(current_candidates) == 0:
    #                     print 'Warning: no cands in training data to match %s! Use v naive backoff to silence...'%(quinphone)
    #                     current_candidates = [self.first_silent_unit]
    #                     ## TODO: better backoff?

    #                 if len(current_candidates) != self.config['n_candidates']:
    #                     # 'W', TODO -- warning
    #                     #print 'Warning: only %s candidates for %s (%s)' % (len(current_candidates), quinphone, current_candidates)
    #                     difference = self.config['n_candidates'] - len(current_candidates) 
    #                     current_candidates += [-1]*difference
    #                 candidates.append(current_candidates)
    #             candidates = np.array(candidates)
    #             self.stop_clock(start_time)          


    #             start_time = self.start_clock('Compute target distances...')
    #             zero_target_cost = False
    #             if zero_target_cost:
    #                 distances = np.ones(candidates.shape)
    #             else:
    #                 distances = []
    #                 for (i,row) in enumerate(candidates):
    #                     candidate_features = self.train_unit_features[row]
    #                     target_features = unit_features[i].reshape((1,-1))
    #                     dists = np.sqrt(np.sum(((candidate_features - target_features)**2), axis=1))
    #                     distances.append(dists)
    #                 distances = np.array(distances)
    #             self.stop_clock(start_time)          
           

    #         else:
    #             sys.exit('preselection_method unknown')


        
    #         start_time = self.start_clock('Make target FST')
    #         if WRAPFST:
    #             T = make_target_sausage_lattice(distances, candidates)        
    #         else:
    #             comm('rm -f /tmp/{target,join,comp,output}.*') ## TODO: don't rely on /tmp/ !
    #             make_t_lattice_SIMP(distances, candidates, '/tmp/target.fst.txt')
    #         self.stop_clock(start_time)          

    #         distances_per_utt.append(distances)
    #         candidates_per_utt.append(candidates)
    #         target_features_per_utt.append(unit_features)
    #         T_per_utt.append(T)


    #     ## ==================
    #     ### Make shared J lattice:


    #     direct = True   
    #     assert self.precomputed_joincost == False ## TODO: still need to debug this
    #     if self.precomputed_joincost:
    #         print 'FORCE: Use existing join cost loaded from %s'%(self.join_cost_file)
    #         sys.exit('not implemented fully')
    #     else:
    #         ### compile J directly without writing to text then compiling. In fact doesn't save much time...
    #         J = self.make_on_the_fly_join_lattice_BLOCK_DIRECT(candidates_per_utt, multiple_sentences=True)
            

    #     best_path_per_utt = []
    #     for T in T_per_utt:
    #         start_time = self.start_clock('Compose and find shortest path')  
    #         if WRAPFST:
    #             if True: # not self.precomputed_joincost:
    #                 if not direct:
    #                     J = openfst.Fst.read(self.join_cost_file + '.bin')
    #                 #self.stop_clock(start_time)     
    #                 #start_time = self.start_clock('Compose and find shortest path 2')     
    #                 best_path = get_best_path_SIMP(T, J, \
    #                                                 join_already_compiled=True, \
    #                                                 add_path_of_last_resort=False)                        
    #             else:
    #                 J = self.J ## already loaded into memory
    #                 best_path = get_best_path_SIMP(T, J, \
    #                                                 join_already_compiled=True, \
    #                                                 add_path_of_last_resort=True)        
    #         else:
    #             best_path = get_best_path_SIMP(self.tool, '/tmp/target.fst.txt', self.join_cost_file, \
    #                                             join_already_compiled=self.precomputed_joincost, \
    #                                             add_path_of_last_resort=True)

    #         best_path_per_utt.append(best_path)
    #         self.stop_clock(start_time)          


    #         self.report( 'got shortest path:')
    #         self.report( best_path)
    #         # print len(best_path)
    #         # for i in best_path:
    #         #     print self.train_unit_names[i]



    #     if self.mode_of_operation == 'stream_weight_balancing':
    #         self.report('' )
    #         self.report( 'balancing stream weights -- skip making waveform')
    #         self.report('' )
    #     else:
    #         sys.exit('TODO: bulk synth for plain synthesis')
    #         start_time = self.start_clock('Extract and join units')
    #         self.concatenate(best_path, outstem + '.wav')    
    #         self.stop_clock(start_time)          
    #         self.report( 'Output wave: %s.wav'%(outstem ))
    #         self.report('')
    #         self.report('')


    #     if self.mode_of_operation == 'stream_weight_balancing':
    #         scores_per_utt = []
    #         for (target_features, best_path) in zip(target_features_per_utt, best_path_per_utt):
    #             tscores = self.get_target_scores_per_stream(target_features, best_path)
    #             jscores = self.get_join_scores_per_stream(best_path)

    #             #print self.get_njoins(best_path)

    #             scores_per_utt.append( (tscores, jscores) )
    #         return scores_per_utt


    #     if self.config['get_selection_info']:
    #         sys.exit('TODO: bulk synth for plain synthesis 2')
    #     #     self.get_path_information(target_features, best_path)


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


    def preselect_units_quinphone(self, unit_features, unit_names):
        '''
        NB: where candidates are too few, returned matrices are padded with
        -1 entries 
        '''
        start_time = self.start_clock('Preselect units ')
        #candidates = np.ones((n_units, self.config['n_candidates'])) * -1
        candidates = []
        for quinphone in unit_names:
            current_candidates = []
            mono, diphone, triphone, quinphone = break_quinphone(quinphone) 
            #print mono, diphone, triphone, quinphone
            for form in [quinphone, triphone, diphone, mono]:
                for unit in self.unit_index.get(form, []):
                    current_candidates.append(unit)
                    if len(current_candidates) == self.config['n_candidates']:
                        break
                if len(current_candidates) == self.config['n_candidates']:
                    break
            if len(current_candidates) == 0:
                print 'Warning: no cands in training data to match %s! Use v naive backoff to silence...'%(quinphone)
                current_candidates = [1] # [self.first_silent_unit]
                ## TODO: better backoff
                #sys.exit('no cands in training data to match %s! TODO: add backoff...'%(quinphone))

            if len(current_candidates) != self.config['n_candidates']:
                # 'W', TODO -- warning
                #print 'Warning: only %s candidates for %s (%s)' % (len(current_candidates), quinphone, current_candidates)
                difference = self.config['n_candidates'] - len(current_candidates) 
                current_candidates += [-1]*difference
            candidates.append(current_candidates)
        candidates = np.array(candidates)
        self.stop_clock(start_time)          


        start_time = self.start_clock('Compute target distances...')
        zero_target_cost = False
        if zero_target_cost:
            distances = np.ones(candidates.shape)
        else:
            distances = []
            for (i,row) in enumerate(candidates):
                candidate_features = self.train_unit_features[row]
                target_features = unit_features[i].reshape((1,-1))
                dists = np.sqrt(np.sum(((candidate_features - target_features)**2), axis=1))
                distances.append(dists)
            distances = np.array(distances)
        self.stop_clock(start_time)          
   
        return (candidates, distances)




    def preselect_units_acoustic(self, unit_features):


        start_time = self.start_clock('Acoustic select units ')
        ## call has same syntax for sklearn and scipy KDTrees:--
        distances, candidates = self.tree.query(unit_features, k=self.config['n_candidates'])
        self.stop_clock(start_time) 
        return (candidates, distances)


    def preselect_units_monophone_then_acoustic(self, unit_features, unit_names):
        '''
        NB: where candidates are too few, returned matrices are padded with
        -1 entries 
        '''
        start_time = self.start_clock('Preselect units ')
    
        m,n = unit_features.shape
        candidates = np.ones((m, self.config['n_candidates']), dtype=int) * -1
        distances = np.ones((m, self.config['n_candidates'])) * const.VERY_BIG_WEIGHT_VALUE

        monophones = np.array([quinphone.split(const.label_delimiter)[2] for quinphone in unit_names])
        assert len(monophones) == m, (len(monophones), m)
        for (i,phone) in enumerate(monophones):
            assert phone in self.phonetrees, 'unseen monophone %s'%(phone)
            current_distances, current_candidates = self.phonetrees[phone].query(unit_features[i,:], k=self.config['n_candidates'])
            mapped_candidates = self.phonetrees_index_converters[phone][current_candidates]
            candidates[i,:current_distances.size] = mapped_candidates
            distances[i,:current_distances.size] = current_distances

            # current_distances = current_distances.flatten()
            # current_candidates = current_candidates.flatten()
            # if len(current_candidates) != self.config['n_candidates']:
            #     difference = self.config['n_candidates'] - len(current_candidates) 
            #     current_candidates = np.concatenate([ current_candidates , np.ones(difference)*-1.0])
            #     current_distances = np.concatenate([ current_distances , np.zeros(difference)])

        return (candidates, distances)


    def viterbi_search(self, candidates, distances):

        start_time = self.start_clock('Make target FST')
        T = make_target_sausage_lattice(distances, candidates)        
        self.stop_clock(start_time)          

        self.precomputed_joincost = False
        if self.precomputed_joincost:
            print 'FORCE: Use existing join cost loaded from %s'%(self.join_cost_file)
            sys.exit('precomputed join cost not fully implemented - 87867')
        else:
            ### compile J directly without writing to text. In fact doesn't save much time...
            J = self.make_on_the_fly_join_lattice_BLOCK_DIRECT(candidates)
        
        if 0:
            T.draw('/tmp/T.dot')
            J.draw('/tmp/J.dot')
            sys.exit('stop here 9893487t3')

        start_time = self.start_clock('Compose and find shortest path')  
        if not self.precomputed_joincost:   
            best_path = get_best_path_SIMP(T, J, \
                                            join_already_compiled=True, \
                                            add_path_of_last_resort=False)                        
        else:
            sys.exit('precomputed join cost not fully implemented - 2338578')
            J = self.J ## already loaded into memory
            best_path = get_best_path_SIMP(T, J, \
                                            join_already_compiled=True, \
                                            add_path_of_last_resort=True)        
        self.stop_clock(start_time)          

        ### TODO:
        # if self.config.get('WFST_pictures', False):

        self.report( 'got shortest path:')
        self.report( best_path)
        return best_path


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
            synth_condition = self.make_synthesis_condition_name()
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


        #speech = speech[20:80,:]

        m,dim = unnorm_speech.shape

        if self.config.get('standardise_target_data', True):                                
            speech = standardise(unnorm_speech, self.mean_vec_target, self.std_vec_target)         
        else:
            speech = unnorm_speech


        #fshift_seconds = (0.001 * self.config['frameshift_ms'])
        #fshift = int(self.config['sample_rate'] * fshift_seconds)        

        if self.config['target_representation'] == 'epoch':
            unit_features = speech[1:-1, :]  ### TODO??
        else:
            labfile = os.path.join(lab_dir, base + '.' + self.config['lab_extension'])
            print 'reading %s'%(labfile)
            labs = read_label(labfile, self.quinphone_regex)

            if self.config.get('untrim_silence_target_speech', False):
                speech = reinsert_terminal_silence(speech, labs)

            if self.config.get('suppress_weird_festival_pauses', False):
                labs = suppress_weird_festival_pauses(labs)

            unit_names, unit_features, unit_timings = get_halfphone_stats(speech, labs, representation_type=self.target_representation)

            if 0: ## debug -- take first few units only 989
                N= 20
                unit_names = unit_names[15:20]
                unit_features = unit_features[15:20, :]
                unit_timings = unit_timings[15:20]


        if self.config['weight_target_data']:                                
            unit_features = weight(unit_features, self.target_weight_vector)       

        # if hasattr(self, 'target_truncation_vector'):
        #     print 'truncate target streams...'
        #     print unit_features.shape
        #     unit_features = unit_features[:, self.target_truncation_vector]
        #     print unit_features.shape
        #     sys.exit('wewevws000')

        n_units, _ = unit_features.shape

        # print unit_features.shape
        # print unit_names
        # print unit_names.shape
        # sys.exit('efvsedv000')

        self.stop_clock(start_time)

        if self.config.get('debug_with_adjacent_frames', False):
            print 'Concatenate naturally contiguous units to debug concatenation!'
            assert not self.config.get('magphase_use_target_f0', True), 'set magphase_use_target_f0 to False for using debug_with_adjacent_frames'
            multiepoch = self.config.get('multiepoch', 1)
            if multiepoch > 1:
                best_path = np.arange(0,500, multiepoch)
            else:
                best_path = np.arange(500)


        elif self.config.get('greedy_search', False):
            print '.'
            assert self.config.get('target_representation') == 'epoch'
            #### =-------------
            ##### For greedy version, skip preselection and full Viterbi search
            #### =-------------
            best_path = self.greedy_joint_search(unit_features)
        else:

            if self.config['preselection_method'] == 'acoustic':
                (candidates, distances) = self.preselect_units_acoustic(unit_features)
            elif self.config['preselection_method'] == 'quinphone':
                (candidates, distances) = self.preselect_units_quinphone(unit_features, unit_names)
            elif self.config['preselection_method'] == 'monophone_then_acoustic':
                (candidates, distances) = self.preselect_units_monophone_then_acoustic(unit_features, unit_names)                
            else:
                sys.exit('preselection_method unknown')

            if 0:
                print candidates
                print distances
                sys.exit('aefegvwrbetbte98456549')

            if self.mode_of_operation == 'find_join_candidates':
                print 'mode_of_operation == find_join_candidates: return here'
                ## TODO: shuffle above operations so we can return this before looking at target features
                return candidates          

            # print candidates.shape
            # np.save('/tmp/cand', candidates)
            # sys.exit('wevwrevwrbv')


            best_path = self.viterbi_search(candidates, distances)
            



        if self.mode_of_operation == 'stream_weight_balancing':
            self.report('' )
            self.report( 'balancing stream weights -- skip making waveform')
            self.report('' )
        else:

            PRELOAD_UTTS = False
            if PRELOAD_UTTS:
                start_time = self.start_clock('Preload magphase utts for sentence')
                self.preload_magphase_utts(best_path)
                self.stop_clock(start_time) 

            start_time = self.start_clock('Extract and join units')
            # if self.config['target_representation'] == 'epoch':
            #     self.concatenate_epochs(best_path, outstem + '.wav')
            #     #self.make_epoch_labels(best_path, outstem + '.lab')   ### !!!!

            if self.config.get('store_full_magphase_sep_files', False):
                assert self.config['target_representation'] == 'epoch'
                target_fz = unnorm_speech[:,-1]
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
                if self.config.get('synth_smooth', False) and not (self.config['target_representation'] == 'epoch'):
                    print "Smooth output"
                    self.concatenateMagPhase(best_path, outstem + '.wav')
                else:
                    print "Does not smooth output"
                    self.concatenate(best_path, outstem + '.wav')
            self.stop_clock(start_time)          
            self.report( 'Output wave: %s.wav'%(outstem ))
            self.report('')
            self.report('')

        #print 'path info:'
        #print self.train_unit_names[best_path].tolist()


        target_features = unit_features ## older nomenclature?
        if self.mode_of_operation == 'stream_weight_balancing':
            tscores = self.get_target_scores_per_stream(target_features, best_path)
            jscores = self.get_join_scores_per_stream(best_path)
            return (tscores, jscores)

        if self.config['get_selection_info']:
            if self.config['target_representation'] == 'epoch':
                trace_lines = self.get_path_information_epoch(target_features, best_path)
                writelist(trace_lines, outstem + '.trace.txt')
                print 'Wrote trace file %s'%(outstem + '.trace.txt')
            else:
                self.get_path_information(target_features, best_path)




    # def synth_utt_greedy_epoch(self, base, synth_type='tune'): 
    #     ### TODO: refactor to deduplicate large parts of this and synth_utt()

    #     if synth_type == 'test':
    #         data_dirs = self.test_data_target_dirs
    #         lab_dir = self.config['test_lab_dir']
    #     elif synth_type == 'tune':
    #         data_dirs = self.tune_data_target_dirs
    #         lab_dir = self.config['tune_lab_dir']            
    #     else:
    #         sys.exit('Unknown synth_type  9489384')

    #     train_condition = make_train_condition_name(self.config)
    #     synth_condition = self.make_synthesis_condition_name()
    #     synth_dir = os.path.join(self.config['workdir'], 'synthesis_%s'%(synth_type), train_condition, synth_condition)
    #     safe_makedir(synth_dir)
            
    #     self.report('               ==== GREEDILY SYNTHESISE %s ===='%(base))
    #     outstem = os.path.join(synth_dir, base)       

    #     start_time = self.start_clock('Get speech ')
    #     speech = compose_speech(data_dirs, base, self.stream_list_target, \
    #                             self.config['datadims_target']) 


    #     #speech = speech[10:80,:]

    #     m,dim = speech.shape

    #     if (self.config['standardise_target_data'], True):                                
    #         speech = standardise(speech, self.mean_vec_target, self.std_vec_target)         
        
    #     #fshift_seconds = (0.001 * self.config['frameshift_ms'])
    #     #fshift = int(self.config['sample_rate'] * fshift_seconds)        

    #     if self.config['target_representation'] == 'epoch':
    #         unit_features = speech[1:-1, :]
    #     else:
    #         labfile = os.path.join(lab_dir, base + '.' + self.config['lab_extension'])
    #         labs = read_label(labfile, self.quinphone_regex)

    #         if self.config.get('untrim_silence_target_speech', False):
    #             speech = reinsert_terminal_silence(speech, labs)

    #         if self.config.get('suppress_weird_festival_pauses', False):
    #             labs = suppress_weird_festival_pauses(labs)

    #         unit_names, unit_features, unit_timings = get_halfphone_stats(speech, labs, representation_type=self.target_representation)
           
    #     if self.config['weight_target_data']:                                
    #         unit_features = weight(unit_features, self.target_weight_vector)       

    #     n_units, _ = unit_features.shape
    #     self.stop_clock(start_time)




    #     #### =-------------
    #     ##### For greedy version, skip preselection and full Viterbi search
    #     #### =-------------
    #     best_path = self.greedy_joint_search(unit_features)

    #     # if self.config['preselection_method'] == 'acoustic':

    #     #     start_time = self.start_clock('Acoustic select units ')
    #     #     ## call has same syntax for sklearn and scipy KDTrees:--
    #     #     distances, candidates = self.tree.query(unit_features, k=self.config['n_candidates'])
    #     #     self.stop_clock(start_time) 

    #     # elif self.config['preselection_method'] == 'quinphone':

    #     #     start_time = self.start_clock('Preselect units ')
    #     #     #candidates = np.ones((n_units, self.config['n_candidates'])) * -1
    #     #     candidates = []
    #     #     for quinphone in unit_names:
    #     #         current_candidates = []
    #     #         mono, diphone, triphone, quinphone = break_quinphone(quinphone) 
    #     #         #print mono, diphone, triphone, quinphone
    #     #         for form in [quinphone, triphone, diphone, mono]:
    #     #             for unit in self.unit_index.get(form, []):
    #     #                 current_candidates.append(unit)
    #     #                 if len(current_candidates) == self.config['n_candidates']:
    #     #                     break
    #     #             if len(current_candidates) == self.config['n_candidates']:
    #     #                 break
    #     #         if len(current_candidates) == 0:
    #     #             print 'Warning: no cands in training data to match %s! Use v naive backoff to silence...'%(quinphone)
    #     #             current_candidates = [self.first_silent_unit]
    #     #             ## TODO: better backoff
    #     #             #sys.exit('no cands in training data to match %s! TODO: add backoff...'%(quinphone))

    #     #         if len(current_candidates) != self.config['n_candidates']:
    #     #             # 'W', TODO -- warning
    #     #             #print 'Warning: only %s candidates for %s (%s)' % (len(current_candidates), quinphone, current_candidates)
    #     #             difference = self.config['n_candidates'] - len(current_candidates) 
    #     #             current_candidates += [-1]*difference
    #     #         candidates.append(current_candidates)
    #     #     candidates = np.array(candidates)
    #     #     self.stop_clock(start_time)          


    #     #     start_time = self.start_clock('Compute target distances...')
    #     #     zero_target_cost = False
    #     #     if zero_target_cost:
    #     #         distances = np.ones(candidates.shape)
    #     #     else:
    #     #         distances = []
    #     #         for (i,row) in enumerate(candidates):
    #     #             candidate_features = self.train_unit_features[row]
    #     #             target_features = unit_features[i].reshape((1,-1))
    #     #             dists = np.sqrt(np.sum(((candidate_features - target_features)**2), axis=1))
    #     #             distances.append(dists)
    #     #         distances = np.array(distances)
    #     #     self.stop_clock(start_time)          
       

    #     # else:
    #     #     sys.exit('preselection_method unknown')



    #     # # print candidates.shape
    #     # # np.save('/tmp/cand', candidates)
    #     # # sys.exit('wevwrevwrbv')

    #     # if self.mode_of_operation == 'find_join_candidates':
    #     #     print 'mode_of_operation == find_join_candidates: return here'
    #     #     ## TODO: shuffle above operations so we can return this before looking at target features
    #     #     return candidates          


    #     # start_time = self.start_clock('Make target FST')
    #     # T = make_target_sausage_lattice(distances, candidates)        
    #     # self.stop_clock(start_time)          

    #     # self.precomputed_joincost = False
    #     # if self.precomputed_joincost:
    #     #     print 'FORCE: Use existing join cost loaded from %s'%(self.join_cost_file)
    #     #     sys.exit('precomputed join cost not fully implemented - 87867')
    #     # else:
    #     #     ### compile J directly without writing to text. In fact doesn't save much time...
    #     #     J = self.make_on_the_fly_join_lattice_BLOCK_DIRECT(candidates)
            

    #     # start_time = self.start_clock('Compose and find shortest path')  
    #     # if not self.precomputed_joincost:   
    #     #     best_path = get_best_path_SIMP(T, J, \
    #     #                                     join_already_compiled=True, \
    #     #                                     add_path_of_last_resort=False)                        
    #     # else:
    #     #     sys.exit('precomputed join cost not fully implemented - 2338578')
    #     #     J = self.J ## already loaded into memory
    #     #     best_path = get_best_path_SIMP(T, J, \
    #     #                                     join_already_compiled=True, \
    #     #                                     add_path_of_last_resort=True)        
    #     # self.stop_clock(start_time)          

    #     if self.config.get('debug_with_adjacent_frames', False):
    #         print 'Concatenate naturally contiguous units to debug concatenation!'
    #         best_path = np.arange(500)


    #     ### TODO:
    #     # if self.config.get('WFST_pictures', False):

    #     self.report( 'got shortest path:')
    #     self.report( best_path)
 
    #     if self.mode_of_operation == 'stream_weight_balancing':
    #         self.report('' )
    #         self.report( 'balancing stream weights -- skip making waveform')
    #         self.report('' )
    #     else:
    #         start_time = self.start_clock('Extract and join units')
    #         # if self.config['target_representation'] == 'epoch':
    #         #     self.concatenate_epochs(best_path, outstem + '.wav')
    #         #     #self.make_epoch_labels(best_path, outstem + '.lab')   ### !!!!

    #         if self.config.get('synth_smooth', False) and not (self.config['target_representation'] == 'epoch'):
    #             print "Smooth output"
    #             self.concatenateMagPhase(best_path, outstem + '.wav')
    #         else:
    #             print "Does not smooth output"
    #             self.concatenate(best_path, outstem + '.wav')
    #         self.stop_clock(start_time)          
    #         self.report( 'Output wave: %s.wav'%(outstem ))
    #         self.report('')
    #         self.report('')

    #     if self.mode_of_operation == 'stream_weight_balancing':
    #         tscores = self.get_target_scores_per_stream(target_features, best_path)
    #         jscores = self.get_join_scores_per_stream(best_path)
    #         return (tscores, jscores)

    #     if self.config['get_selection_info'] and self.config['target_representation'] != 'epoch':
    #         self.get_path_information(target_features, best_path)


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


    def get_target_scores_per_stream(self, target_features, best_path):
        chosen_features = self.train_unit_features[best_path]
        dists = np.sqrt(np.sum(((chosen_features - target_features)**2), axis=1))
        sq_errs = (chosen_features - target_features)**2
        stream_errors_target = self.aggregate_squared_errors_by_stream(sq_errs, 'target')
        return stream_errors_target

    def get_join_scores_per_stream(self, best_path):
        if self.config.get('greedy_search', False):
            best_path = np.array(best_path)
            sq_diffs_join = (self.prev_join_rep[best_path[1:],:] - self.current_join_rep[best_path[:-1],:])**2
            #sq_diffs_join = (self.current_join_rep[best_path[:-1],:] - self.current_join_rep[best_path[1:],:])**2
            stream_errors_join = self.aggregate_squared_errors_by_stream(sq_diffs_join, 'join')
            #print stream_errors_join
        else:
            sq_diffs_join = (self.unit_end_data[best_path[:-1],:] - self.unit_start_data[best_path[1:],:])**2
            stream_errors_join = self.aggregate_squared_errors_by_stream(sq_diffs_join, 'join')
        return stream_errors_join


    def get_njoins(self, best_path):

        njoins = 0
        for (a,b) in zip(best_path[:-1], best_path[1:]):                
            if b != a+1:
                njoins += 1
        percent_joins = (float(njoins) / (len(best_path)-1)) * 100
        return (njoins, percent_joins)
        #print '%.1f%% of junctures (%s) are joins'%(percent_joins, n_joins)


    def get_path_information(self, target_features, best_path):
        '''
        Print out some information about what was selected, where the joins are, what the costs
        were, etc. etc.
        '''

        print '============'
        print 'Display some information about the chosen path -- turn this off with config setting get_selection_info'
        print 
        output = []
        for (a,b) in zip(best_path[:-1], best_path[1:]):                
            output.append(extract_monophone(self.train_unit_names[a]))
            if b != a+1:
                output.append('|')
        output.append(extract_monophone(self.train_unit_names[best_path[-1]]))
        print ' '.join(output)
        print
        n_joins = output.count('|')
        percent_joins = (float(n_joins) / (len(best_path)-1)) * 100
        print '%.1f%% of junctures (%s) are joins'%(percent_joins, n_joins)


        print 
        print ' --- Version with unit indexes ---'
        print 
        for (a,b) in zip(best_path[:-1], best_path[1:]):
            output.append( extract_monophone(self.train_unit_names[a]) + '-' + str(a))
            if b != a+1:
                output.append('|')

        output.append('\n\n\n')
        output.append(extract_monophone(self.train_unit_names[best_path[-1]]) + '-' + str(best_path[-1]))
        print ' '.join(output)            

        # print
        # print 'target scores'
        
        stream_errors_target =  self.get_target_scores_per_stream(target_features, best_path)

        # print stream_errors_target

        # print dists
        #mean_dists = np.mean(dists)
        #std_dists = np.std(dists)
        # print dists
        # print (mean_dists, std_dists)

        # print 
        # print 'join scores'

        stream_errors_join = self.get_join_scores_per_stream(best_path)

        # print stream_errors_join


        #### TODO: remove zeros from stream contrib scores below
        print 
        print '------------- join and target cost summaries by stream -----------'
        print

        ## take nonzeros only, but avoid division errors:
        # stream_errors_join = stream_errors_join[stream_errors_join>0.0]
        # if stream_errors_join.size == 0:
        #     stream_errors_join = np.zeros(stream_errors_join.shape) ## avoid divis by 0
        # stream_errors_target = stream_errors_target[stream_errors_target>0.0]
        # if stream_errors_target.size == 0:
        #     stream_errors_target = np.zeros(stream_errors_target.shape) ## avoid divis by 0

        for (stream, mu, sig) in zip (self.stream_list_join,
            np.mean(stream_errors_join, axis=0),
            np.std(stream_errors_join, axis=0) ):
            print 'join   %s -- mean: %s   std:  %s'%(stream.ljust(10), str(mu).ljust(15), str(sig).ljust(1))
        print 
        for (stream, mu, sig) in zip (self.stream_list_target,
            np.mean(stream_errors_target, axis=0),
            np.std(stream_errors_target, axis=0) ):
            print 'target %s -- mean: %s   std:  %s'%(stream.ljust(10), str(mu).ljust(15), str(sig).ljust(1))
        print 
        print '--------------------------------------------------------------------'



        print 'Skip plots for now and return' ### TODO: optionally plot
        return

        ## plot scores per unit 
         
        ##### TARGET ONLY         
        # units = [extract_monophone(self.train_unit_names[a]) for a in best_path]
        # y_pos = np.arange(len(units))
        # combined_t_cost = np.sum(stream_errors_target, axis=1)
        # nstream = len(self.stream_list_target)
        # print self.stream_list_target
        # for (i,stream) in enumerate(self.stream_list_target):
        #     plt.subplot('%s%s%s'%((nstream+1, 1, i+1)))
        #     plt.bar(y_pos, stream_errors_target[:,i], align='center', alpha=0.5)
        #     plt.xticks(y_pos, ['']*len(units))
        #     plt.ylabel(stream)
        # plt.subplot('%s%s%s'%(nstream+1, 1, nstream+1))
        # plt.bar(y_pos, combined_t_cost, align='center', alpha=0.5)
        # plt.xticks(y_pos, units)
        # plt.ylabel('combined')


        ## TARGWET AND JOIN
        units = [extract_monophone(self.train_unit_names[a]) for a in best_path]
        y_pos = np.arange(len(units))
        combined_t_cost = np.sum(stream_errors_target, axis=1)
        nstream = len(self.stream_list_target) + len(self.stream_list_join)
        i = 0
        i_graphic = 1
        for stream in self.stream_list_target:
            #print stream
            plt.subplot('%s%s%s'%((nstream+2, 1, i_graphic)))
            plt.bar(y_pos, stream_errors_target[:,i], align='center', alpha=0.5)
            plt.xticks(y_pos, ['']*len(units))
            plt.ylabel(stream)
            i += 1
            i_graphic += 1
        plt.subplot('%s%s%s'%(nstream+2, 1, i_graphic))
        plt.bar(y_pos, combined_t_cost, align='center', alpha=0.5)
        plt.xticks(y_pos, units)
        plt.ylabel('combined')         
        i_graphic += 1
        i = 0  ## reset for join streams

        combined_j_cost = np.sum(stream_errors_join, axis=1)
        y_pos_join = y_pos[:-1] + 0.5
        for stream in self.stream_list_join:
            print stream
            plt.subplot('%s%s%s'%((nstream+2, 1, i_graphic)))
            plt.bar(y_pos_join, stream_errors_join[:,i], align='center', alpha=0.5)
            plt.xticks(y_pos_join, ['']*len(units))
            plt.ylabel(stream)
            i += 1
            i_graphic += 1
        plt.subplot('%s%s%s'%(nstream+2, 1, i_graphic))
        plt.bar(y_pos_join, combined_j_cost, align='center', alpha=0.5)
        plt.xticks(y_pos, units)
        plt.ylabel('combined')            


        plt.show()        




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




        '''
        print '============'
        print 'Display some information about the chosen path -- turn this off with config setting get_selection_info'
        print 
        output = []
        for (a,b) in zip(best_path[:-1], best_path[1:]):                
            output.append(extract_monophone(self.train_unit_names[a]))
            if b != a+1:
                output.append('|')
        output.append(extract_monophone(self.train_unit_names[best_path[-1]]))
        print ' '.join(output)
        print
        n_joins = output.count('|')
        percent_joins = (float(n_joins) / (len(best_path)-1)) * 100
        print '%.1f%% of junctures (%s) are joins'%(percent_joins, n_joins)


        print 
        print ' --- Version with unit indexes ---'
        print 
        for (a,b) in zip(best_path[:-1], best_path[1:]):
            output.append( extract_monophone(self.train_unit_names[a]) + '-' + str(a))
            if b != a+1:
                output.append('|')

        output.append('\n\n\n')
        output.append(extract_monophone(self.train_unit_names[best_path[-1]]) + '-' + str(best_path[-1]))
        print ' '.join(output)            

        # print
        # print 'target scores'
        
        stream_errors_target =  self.get_target_scores_per_stream(target_features, best_path)

        # print stream_errors_target

        # print dists
        #mean_dists = np.mean(dists)
        #std_dists = np.std(dists)
        # print dists
        # print (mean_dists, std_dists)

        # print 
        # print 'join scores'

        stream_errors_join = self.get_join_scores_per_stream(best_path)

        # print stream_errors_join


        #### TODO: remove zeros from stream contrib scores below
        print 
        print '------------- join and target cost summaries by stream -----------'
        print

        ## take nonzeros only, but avoid division errors:
        # stream_errors_join = stream_errors_join[stream_errors_join>0.0]
        # if stream_errors_join.size == 0:
        #     stream_errors_join = np.zeros(stream_errors_join.shape) ## avoid divis by 0
        # stream_errors_target = stream_errors_target[stream_errors_target>0.0]
        # if stream_errors_target.size == 0:
        #     stream_errors_target = np.zeros(stream_errors_target.shape) ## avoid divis by 0

        for (stream, mu, sig) in zip (self.stream_list_join,
            np.mean(stream_errors_join, axis=0),
            np.std(stream_errors_join, axis=0) ):
            print 'join   %s -- mean: %s   std:  %s'%(stream.ljust(10), str(mu).ljust(15), str(sig).ljust(1))
        print 
        for (stream, mu, sig) in zip (self.stream_list_target,
            np.mean(stream_errors_target, axis=0),
            np.std(stream_errors_target, axis=0) ):
            print 'target %s -- mean: %s   std:  %s'%(stream.ljust(10), str(mu).ljust(15), str(sig).ljust(1))
        print 
        print '--------------------------------------------------------------------'
        '''





    def inspect_join_weights_on_utt(self, fname):

        # if self.inspect_join_weights:
        #     self.config['preselection_method'] = 'quinphone'
        #     self.config['n_candidates'] = 10000 # some very large number


        # train_condition = make_train_condition_name(self.config)
        # synth_condition = self.make_synthesis_condition_name()
        # synth_dir = os.path.join(self.config['workdir'], 'synthesis', train_condition, synth_condition)
        # safe_makedir(synth_dir)
            
        junk,base = os.path.split(fname)
        print '               ==== SYNTHESISE %s ===='%(base)
        base = base.replace('.mgc','')
        #outstem = os.path.join(synth_dir, base)       

        # start_time = start_clock('Get speech ')
        speech = compose_speech(self.test_data_target_dirs, base, self.stream_list_target, \
                                self.config['datadims_target']) 

        # m,dim = speech.shape

        # if (self.config['standardise_target_data'], True):                                
        #     speech = standardise(speech, self.mean_vec_target, self.std_vec_target)         
        
        #fshift_seconds = (0.001 * self.config['frameshift_ms'])
        #fshift = int(self.config['sample_rate'] * fshift_seconds)        

        labfile = os.path.join(self.config['test_lab_dir'], base + '.' + self.config['lab_extension'])
        labs = read_label(labfile, self.quinphone_regex)

        if self.config.get('untrim_silence_target_speech', False):
            speech = reinsert_terminal_silence(speech, labs)

        if self.config.get('suppress_weird_festival_pauses', False):
            labs = suppress_weird_festival_pauses(labs)

        unit_names, unit_features, unit_timings = get_halfphone_stats(speech, labs)
       
        # if self.config['weight_target_data']:                                
        #     unit_features = weight(unit_features, self.target_weight_vector)       

        #print unit_features
        #print unit_names

        # n_units = len(unit_names)
        # stop_clock(start_time)


        # if self.config['preselection_method'] == 'acoustic':

        #     start_time = start_clock('Acoustic select units ')
        #     ## call has same syntax for sklearn and scipy KDTrees:--
        #     distances, candidates = self.tree.query(unit_features, k=self.config['n_candidates'])
        #     stop_clock(start_time) 





        ##### self.config['preselection_method'] == 'quinphone':
        #self.config['n_candidates'] = 100 ### large number
        start_time = start_clock('Preselect units (quinphone criterion) ')
        candidates = []
        for quinphone in unit_names:
            current_candidates = []
            mono, diphone, triphone, quinphone = break_quinphone(quinphone) 
            for form in [mono]: # [quinphone, triphone, diphone, mono]:
                for unit in self.unit_index.get(form, []):
                    current_candidates.append(unit)
                    if len(current_candidates) == self.config['n_candidates']:
                        break
                if len(current_candidates) == self.config['n_candidates']:
                    break
            if len(current_candidates) == 0:
                sys.exit('no cands in training data to match %s! TODO: add backoff...'%(quinphone))
            if len(current_candidates) != self.config['n_candidates']:
                print 'W',
                #print 'Warning: only %s candidates for %s (%s)' % (len(current_candidates), quinphone, current_candidates)
                difference = self.config['n_candidates'] - len(current_candidates) 
                current_candidates += [-1]*difference
            candidates.append(current_candidates)
        candidates = np.array(candidates)
        stop_clock(start_time)         



        print 'get join costs...'
        self.join_cost_file = '/tmp/join.fst'  ## TODO: don't rely on /tmp/ !           
        
        print 
        j_distances = self.make_on_the_fly_join_lattice(candidates, self.join_cost_file, by_stream=True)
        j_distances = np.array(j_distances.values())

        # pylab.hist(j_distances.values(), bins=30)
        # pylab.show()
        #print distances
        print 'Skip full synthesis -- only want to look at the weights...'
        return j_distances





    def retrieve_speech_OLD(self, index):
        #if self.config['hold_waves_in_memory']:  TODO work out config
        if self.train_filenames[index] in self.waveforms:
            wave = self.waveforms[self.train_filenames[index]]  
        else:     
            wavefile = os.path.join(self.config['wav_datadir'], self.train_filenames[index] + '.wav')
            wave, sample_rate = read_wave(wavefile)
        T = len(wave)        
        (start,end) = self.train_cutpoints[index]
        end += 1 ## non-inclusive end of slice
        #print (start,end)
        taper = self.config['taper_length']
        halftaper = taper / 2        
        if taper > 0:
            #start = max(0, (start - halftaper))
            #end = min(T, end + halftaper)
            start = (start - halftaper)
            end = end + halftaper
            if start < 0:
                pad = np.zeros(math.fabs(start))
                wave = np.concatenate([pad, wave])
                end += math.fabs(start)
                start = 0
                T = len(wave) 
            if end > T:
                pad = np.zeros(end - T)
                wave = np.concatenate([wave, pad])
                

        frag = wave[start:end]
        if taper > 0:
            hann = np.hanning(taper)
            open_taper = hann[:halftaper]
            close_taper = hann[halftaper:]
            frag[:halftaper] *= open_taper
            frag[-halftaper:] *= close_taper


        return frag

    def retrieve_speech(self, index):

        if self.train_filenames[index] in self.waveforms:
            wave = self.waveforms[self.train_filenames[index]]  
        else:     
            wavefile = os.path.join(self.config['wav_datadir'], self.train_filenames[index] + '.wav')
            wave, sample_rate = read_wave(wavefile)
        T = len(wave)        
        (start,end) = self.train_cutpoints[index]
        end += 1 ## non-inclusive end of slice
        
        taper = self.config['taper_length']
        
        # Overlap happens at the pitch mark + taper/2 (extend segment by a taper in the end)
        # if taper > 0:
        #     end = end + taper
        #     if end > T:
        #         pad = np.zeros(end - T)
        #         wave = np.concatenate([wave, pad])

        # Overlap happens at the pitch mark (extend segment by half taper in each end)
        if taper > 0:
            end = end + taper/2
            if end > T:
                pad = np.zeros(end - T)
                wave = np.concatenate([wave, pad])
            start = start - taper/2
            if start < 0:
                pad   = np.zeros(-start)
                wave  = np.concatenate([pad, wave])
                start = 0
                
        frag = wave[start:end]
        if taper > 0:
            hann = np.hanning(taper*2)
            open_taper = hann[:taper]
            close_taper = hann[taper:]
            frag[:taper] *= open_taper
            frag[-taper:] *= close_taper

        if DODEBUG:
            orig = (self.train_cutpoints[index][1] - self.train_cutpoints[index][0])
            print('orig length: %s' %  orig)
            print('length with taper: %s '%(frag.shape))
            print (frag.shape - orig)
        return frag



    def retrieve_speech_epoch(self, index):

        if self.config['hold_waves_in_memory']:
            wave = self.waveforms[self.train_filenames[index]]  
        else:     
            wavefile = os.path.join(self.config['wav_datadir'], self.train_filenames[index] + '.wav')
            wave, sample_rate = read_wave(wavefile)
        T = len(wave)        
        (start,middle,end) = self.train_cutpoints[index]
        end += 1 ## non-inclusive end of slice

        left_length = middle - start
        right_length = end - middle 

        frag = wave[start:end]

        ### scale with non-symmetric hanning:
        win = np.concatenate([np.hanning(left_length*2)[:left_length], np.hanning(right_length*2)[right_length:]])
        frag *= win



        return (frag, left_length)
 


    def retrieve_speech_epoch_new(self, index):

        ## TODO: see copy.copy below --- make sure copy with other configureations, otherwise 
                                            ## in the case hold_waves_in_memory we disturb original audio which is reused -- TODO -- use this elsewhere too

        if self.train_filenames[index] in self.waveforms:
            orig_wave = self.waveforms[self.train_filenames[index]]  
        else:     
            wavefile = os.path.join(self.config['wav_datadir'], self.train_filenames[index] + '.wav')
            print wavefile
            orig_wave, sample_rate = read_wave(wavefile)
            self.waveforms[self.train_filenames[index]]  = orig_wave
        T = len(orig_wave)        
        (start,middle,end) = self.train_cutpoints[index]


        multiepoch = self.config.get('multiepoch', 1)
        if multiepoch > 1:
            (start_ii,middle,end_ii) = self.train_cutpoints[index + (multiepoch-1)]


        end = middle  ## just use first half of fragment (= 1 epoch)





        wave = copy.copy(orig_wave)              

        taper = self.config['taper_length']

        # Overlap happens at the pitch mark (extend segment by half taper in each end)
        if taper > 0:
            end = end + taper/2
            if end > T:
                pad = np.zeros(end - T)
                wave = np.concatenate([wave, pad])
            start = start - taper/2
            if start < 0:
                pad   = np.zeros(-start)
                wave  = np.concatenate([pad, wave])
                start = 0
                
        frag = wave[start:end]
        if taper > 0:
            hann = np.hanning(taper*2)
            open_taper = hann[:taper]
            close_taper = hann[taper:]
            frag[:taper] *= open_taper
            frag[-taper:] *= close_taper


        return frag
 

    def preload_magphase_utts(self, path):
        '''
        preload utts used for a given path
        '''
        #HALFFFTLEN = 513  ## TODO
        for index in path:
            if self.train_filenames[index] in self.waveforms: # self.config['hold_waves_in_memory']:  ### i.e. waves or magphase FFT spectra
                (mag_full, real_full, imag_full, f0_interp, vuv) = self.waveforms[self.train_filenames[index]]  
            else:     
                mag_full = get_speech(os.path.join(self.config['full_magphase_dir'], 'mag', self.train_filenames[index] + '.mag'), HALFFFTLEN)
                real_full = get_speech(os.path.join(self.config['full_magphase_dir'], 'real',  self.train_filenames[index] + '.real'), HALFFFTLEN)
                imag_full = get_speech(os.path.join(self.config['full_magphase_dir'], 'imag',  self.train_filenames[index] + '.imag'), HALFFFTLEN)
                f0_full = get_speech(os.path.join(self.config['full_magphase_dir'], 'f0',  self.train_filenames[index] + '.f0'), 1)            
                f0_interp, vuv = speech_manip.lin_interp_f0(f0_full)
                self.waveforms[self.train_filenames[index]] = (mag_full, real_full, imag_full, f0_interp, vuv)


    def preload_all_magphase_utts(self):
        #HALFFFTLEN = 513  ## TODO
        start_time = self.start_clock('Preload magphase utts for corpus')
        for base in np.unique(self.train_filenames):
            print base
            mag_full = get_speech(os.path.join(self.config['full_magphase_dir'], 'mag', base + '.mag'), HALFFFTLEN)
            real_full = get_speech(os.path.join(self.config['full_magphase_dir'], 'real',  base + '.real'), HALFFFTLEN)
            imag_full = get_speech(os.path.join(self.config['full_magphase_dir'], 'imag',  base + '.imag'), HALFFFTLEN)
            f0_full = get_speech(os.path.join(self.config['full_magphase_dir'], 'f0', base + '.f0'), 1)            
            f0_interp, vuv = speech_manip.lin_interp_f0(f0_full)
            self.waveforms[base] = (mag_full, real_full, imag_full, f0_interp, vuv)
        self.stop_clock(start_time) 




    def retrieve_magphase_frag(self, index, extra_frames=0):
        #HALFFFTLEN = 513  ## TODO

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
                mag_full = get_speech(os.path.join(self.config['full_magphase_dir'], 'mag', self.train_filenames[index] + '.mag'), HALFFFTLEN)
                real_full = get_speech(os.path.join(self.config['full_magphase_dir'], 'real',  self.train_filenames[index] + '.real'), HALFFFTLEN)
                imag_full = get_speech(os.path.join(self.config['full_magphase_dir'], 'imag',  self.train_filenames[index] + '.imag'), HALFFFTLEN)
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

 


        # print mag_frag.shape

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


    def concatenate(self, path, fname):

        if self.config['target_representation'] == 'epoch':
            NEW_METHOD = True
            if NEW_METHOD:
                self.concatenate_epochs_new(path, fname)
            else:
                self.concatenate_epochs(path, fname)
        else:
            frags = []
            for unit_index in path:
                frags.append(self.retrieve_speech(unit_index))

            if self.config['taper_length'] == 0:
                synth_wave = np.concatenate(frags)
            else:
                synth_wave = self.overlap_add(frags)
            write_wave(synth_wave, fname, 48000, quiet=True)


    def concatenate_epochs(self, path, fname):

        frags = []
        for unit_index in path:
            frags.append(self.retrieve_speech_epoch(unit_index))

        synth_wave = self.epoch_overlap_add(frags)
        write_wave(synth_wave, fname, 48000, quiet=True)



    def concatenate_epochs_new(self, path, fname):
        # print '===== NEW METHOD: concatenate_epochs_new ======='
        frags = []
        for unit_index in path:
            frags.append(self.retrieve_speech_epoch_new(unit_index))
        synth_wave = self.epoch_overlap_add_new(frags)
        write_wave(synth_wave, fname, 48000, quiet=True)



    # def make_epoch_labels(self, path, fname):
    #     start_points = []
    #     start = 0
    #     for (i,unit_index) in enumerate(path):
    #         (start,middle,end) = self.train_cutpoints[unit_index]
    #         left_length = middle - start
    #         right_length = end - middle 

    #         start += 
    #         start_points.append(start)

    #     frag = wave[start:end]

    #     ### scale with non-symmetric hanning:
    #     win = np.concatenate([np.hanning(left_length*2)[:left_length], np.hanning(right_length*2)[right_length:]])
    #     frag *= win



    #     return (frag, left_length)




    def overlap_add(self, frags):
        taper = self.config['taper_length']
        length = sum([len(frag)-taper for frag in frags]) + 1000 # taper
        wave = np.zeros(length)
        start = 0
        for frag in frags:
            #print start 

            ## only for visualiseation:
            # padded = np.zeros(length)
            # padded[start:start+len(frag)] += frag
            # pylab.plot(padded)


            wave[start:start+len(frag)] += frag
            start += (len(frag) - taper) #+ 1

        return wave 


    def epoch_overlap_add(self, frags):
        
        length = sum([halflength for (wave, halflength) in frags[:-1]])
        lastwave, _ = frags[-1]
        length += len(lastwave) 
        wave = np.zeros(length)
        start = 0
        for (frag, halflength) in frags:
            wave[start:start+len(frag)] += frag
            start += halflength
        return wave 



    def epoch_overlap_add_new(self, frags):
        taper = self.config['taper_length']
        length = sum([len(frag)-taper for frag in frags])
        length += taper
        wave = np.zeros(length)
        start = 0
        for frag in frags:
            wave[start:start+len(frag)] += frag
            start += len(frag)-taper
        return wave 


    def concatenateMagPhase(self,path,fname,prosody_targets=[],prosody_target_confidences=[]):

        '''
        prosody_targets: list like: [(dur,ene,f0),(dur,ene,f0),...]  
           where dur = predicted dur in msec for halfphone
           ene = mean of predicted straight c0 for half phone
           f0 = mean of predicted lf0 for half phone (neg for segments where all frames unvoiced )

        prosody_target_confidences: 1.0 means impose target completely, 0.0 not at all, 
                  inbetween -- linearly interpoalate?
        '''
        if prosody_targets:
            if not prosody_target_confidences:
                prosody_target_confidences = [1.0] * len(prosody_targets)
            assert len(prosody_targets) == len(prosody_target_confidences) == len(path)

        fs     = 48000 # in Hz
        nfft   = 4096

        pm_reaper_dir = self.config['pm_datadir']
        wav_dir = self.config['wav_datadir']

        # Initializing fragments
        frags = {}
        frags['srcfile'] = []
        frags['src_strt_sec'] = []
        frags['src_end_sec'] = []

        '''
        for (index, pros_target) in zip(path, prosody_targets):
            target_dur = pros_target[0]

        for (i,index) in enumerate(path):
            target_dur = prosody_targets[i][0]
        '''

        for index in path:

            (start,end) = self.train_cutpoints[index]
            frags['srcfile'].append(self.train_filenames[index])
            frags['src_strt_sec'].append(start / float(fs))
            frags['src_end_sec'].append(end / float(fs))

        synth_wave = lwg.wavgen_improved_just_slope(frags, wav_dir, pm_reaper_dir, nfft, fs, npm_margin=3, diff_mf_tres=25, f0_trans_nfrms_btwn_voi=8)
        la.write_audio_file(fname, synth_wave, fs, norm=True)

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
  
        

    def get_natural_distance(self, first, second, order=2):
        '''
        first and second: indices of left and right units to be joined
        order: number of frames of overlap
        '''
        sq_diffs = (self.unit_end_data[first,:] - self.unit_start_data[second,:])**2
        ## already weighted, skip next line:
        #sq_diffs *= self.join_weight_vector
        distance = (1.0 / order) * math.sqrt(np.sum(sq_diffs))   
        return distance



    def get_natural_distance_vectorised(self, first, second, order=2):
        '''
        first and second: indices of left and right units to be joined
        order: number of frames of overlap
        '''
        sq_diffs = (self.unit_end_data[first,:] - self.unit_start_data[second,:])**2
        ## already weighted, skip next line:
        #sq_diffs *= self.join_weight_vector
        distance = (1.0 / order) * np.sqrt(np.sum(sq_diffs, axis=1))   
        return distance


    def get_natural_distance_by_stream(self, first, second, order=2):
        '''
        first and second: indices of left and right units to be joined
        order: number of frames of overlap
        '''
        sq_diffs = (self.unit_end_data[first,:] - self.unit_start_data[second,:])**2
        ## already weighted, skip next line:
        #sq_diffs *= self.join_weight_vector
        start = 0
        distance_by_stream = []
        for stream_name in self.stream_list_join:  #  [(1,'energy'),(12,'mfcc')]:
            stream_width = self.datadims_join[stream_name]
            distance_by_stream.append((1.0 / order) * math.sqrt(np.sum(sq_diffs[start:start+stream_width])) )

        # for (stream_width, stream_name) in [(1,'energy'),(12,'mfcc')]:
        #     distance_by_stream.append((1.0 / order) * math.sqrt(np.sum(sq_diffs[start:start+stream_width])) )
            start += stream_width

        distance = (1.0 / order) * math.sqrt(np.sum(sq_diffs))   
        #return (distance, distance_by_stream)
        return (distance, np.sqrt(sq_diffs))  ### experikent by per coeff


    def aggregate_squared_errors_by_stream(self, squared_errors, cost_type):
        '''
        NB: do not take sqrt!
        '''
        assert not (self.config.get('greedy_search', False)  and  self.config['target_representation'] != 'epoch')


        if cost_type == 'target':
            streams = self.stream_list_target
            stream_widths = self.datadims_target
        elif cost_type == 'join':
            streams = self.stream_list_join
            stream_widths = self.datadims_join
        else:
            sys.exit('cost type must be one of {target, join}')

        nstream = len(streams)
        
        m,n = squared_errors.shape
        stream_distances = np.ones((m,nstream)) * -1.0

        # print squared_errors.shape
        # print stream_distances.shape
        # print '----'

        start = 0
        for (i, stream) in enumerate(streams): 
            stream_width = stream_widths[stream]
            #stream_distances[:,i] = np.sqrt(np.sum(squared_errors[:, start:start+stream_width], axis=1)) 
            stream_distances[:,i] = np.sum(squared_errors[:, start:start+stream_width], axis=1)
            start += stream_width
        return stream_distances




    def make_on_the_fly_join_lattice(self, ind, outfile, join_cost_weight=1.0, by_stream=False):

        ## These are irrelevant when using halfphones -- suppress them:
        forbid_repetition = False # self.config['forbid_repetition']
        forbid_regression = False # self.config['forbid_regression']

        ## For now, force join cost to be natural2
        join_cost_type = self.config['join_cost_type']
        join_cost_type = 'pitch_sync'
        assert join_cost_type in ['pitch_sync']

        start = 0
        frames, cands = np.shape(ind)
    
        data_frames, dim = self.unit_end_data.shape
        
        ## cache costs for joins which could be used in an utterance.
        ## This can save  computing things twice, 52 seconds -> 33 (335 frames, 50 candidates) 
        ## (Probably no saving with half phones?)
        cost_cache = {} 
        
        cost_cache_by_stream = {}

        ## set limits to not go out of range -- unnecessary given new unit_end_data and unit_start_data?
        if join_cost_type == 'pitch_sync':
            mini = 1 
            maxi = data_frames - 1              
        else:
            sys.exit('dvsdvsedv098987897')
        
        t = start_clock('     DISTS')
        for i in range(frames-1): 
            for first in ind[i,:]:
                if first < mini or first >= maxi:
                    continue
                for second in ind[i+1,:]:
                    if second < mini or second >= maxi:
                        continue
                    #print (first, second)
                    if (first == -1) or (second == -1):
                        continue
                    if (first, second) in cost_cache:
                        continue
                    
                    if  join_cost_type == 'pitch_sync' and by_stream:
                        weight, weight_by_stream = self.get_natural_distance_by_stream(first, second, order=1)
                        cost_cache_by_stream[(first, second)] = weight_by_stream
                    elif  join_cost_type == 'pitch_sync':
                        weight = self.get_natural_distance(first, second, order=1)
                    else:
                        sys.exit('Unknown join cost type: %s'%(join_cost_type))

                    weight *= self.config['join_cost_weight']


                    if forbid_repetition:
                        if first == second:
                            weight = VERY_BIG_WEIGHT_VALUE
                    if forbid_regression > 0:
                        if (first - second) in range(forbid_regression+1):
                            weight = VERY_BIG_WEIGHT_VALUE
                    cost_cache[(first, second)] = weight
                    

        stop_clock(t)

        t = start_clock('      WRITE')
        ## 2nd pass: write it to file
        if False: ## VIZ: show join histogram
            print len(cost_cache)
            pylab.hist([v for v in cost_cache.values() if v < VERY_BIG_WEIGHT_VALUE], bins=60)
            pylab.show()
        ### pruning:--
        #cost_cache = dict([(k,v) for (k,v) in cost_cache.items() if v < 3000.0])
        cost_cache_to_text_fst(cost_cache, outfile, join_cost_weight=join_cost_weight)
        stop_clock(t)

        if by_stream:
            return cost_cache_by_stream




    def make_on_the_fly_join_lattice_BLOCK(self, ind, outfile, join_cost_weight=1.0, by_stream=False, direct=False):

        '''
        Get distances in blocks, not singly
        '''

        assert by_stream == False ## TODO: remove if unused

        ## These are irrelevant when using halfphones -- suppress them:
        forbid_repetition = False # self.config['forbid_repetition']
        forbid_regression = False # self.config['forbid_regression']

        ## For now, force join cost to be natural2
        join_cost_type = self.config['join_cost_type']
        join_cost_type = 'pitch_sync'
        assert join_cost_type in ['pitch_sync']

        start = 0
        frames, cands = np.shape(ind)
    
        data_frames, dim = self.unit_end_data.shape
        
        ## cache costs for joins which could be used in an utterance.
        ## This can save  computing things twice, 52 seconds -> 33 (335 frames, 50 candidates) 
        ## (Probably no saving with half phones?)
        cost_cache = {} 
        
        cost_cache_by_stream = {}

        ## set limits to not go out of range -- unnecessary given new unit_end_data and unit_start_data?
        if join_cost_type == 'pitch_sync':
            mini = 1 
            maxi = data_frames - 1              
        else:
            sys.exit('dvsdvsedv098987897')
        
        first_list = []
        second_list = []
        t = start_clock('     COST LIST')
        for i in range(frames-1): 
            for first in ind[i,:]:
                if first < mini or first >= maxi:
                    continue
                for second in ind[i+1,:]:
                    if second < mini or second >= maxi:
                        continue
                    #print (first, second)
                    if (first == -1) or (second == -1):
                        continue
                    if (first, second) in cost_cache:
                        continue
                    
                    # if  join_cost_type == 'pitch_sync' and by_stream:
                    #     weight, weight_by_stream = self.get_natural_distance_by_stream(first, second, order=1)
                    #     cost_cache_by_stream[(first, second)] = weight_by_stream
                    # elif  join_cost_type == 'pitch_sync':
                    #     weight = self.get_natural_distance(first, second, order=1)
                    # else:
                    #     sys.exit('Unknown join cost type: %s'%(join_cost_type))
                    # weight *= self.config['join_cost_weight']


                    if forbid_repetition:
                        if first == second:
                            weight = VERY_BIG_WEIGHT_VALUE
                    if forbid_regression > 0:
                        if (first - second) in range(forbid_regression+1):
                            weight = VERY_BIG_WEIGHT_VALUE
                    #cost_cache[(first, second)] = weight
                    first_list.append(first)
                    second_list.append(second)
        stop_clock(t)



        t = start_clock('     DISTS')
        dists = self.get_natural_distance_vectorised(first_list, second_list, order=1)
        #print dists
        stop_clock(t)

        t = start_clock('     make cost cache')
        cost_cache = dict([((l,r), weight) for (l,r,weight) in zip(first_list, second_list, dists)])
        stop_clock(t)


    
        
        if direct:
            t = start_clock('      WRITE compiled')
            J = cost_cache_to_compiled_fst(cost_cache, join_cost_weight=join_cost_weight)
        else:
            t = start_clock('      WRITE txt')
            ## 2nd pass: write it to file
            if False: ## VIZ: show join histogram
                print len(cost_cache)
                pylab.hist([v for v in cost_cache.values() if v < VERY_BIG_WEIGHT_VALUE], bins=60)
                pylab.show()
            ### pruning:--
            #cost_cache = dict([(k,v) for (k,v) in cost_cache.items() if v < 3000.0])
            cost_cache_to_text_fst(cost_cache, outfile, join_cost_weight=join_cost_weight)
        stop_clock(t)


        if direct:
            return J





    def make_on_the_fly_join_lattice_BLOCK_DIRECT(self, ind, join_cost_weight=1.0, multiple_sentences=False):

        '''
        Get distances in blocks, not singly
        '''
        direct = True
        #assert by_stream == False ## TODO: remove if unused

        if self.config['target_representation'] == 'epoch':
            forbid_repetition = self.config.get('forbid_repetition', False)
            forbid_regression = self.config.get('forbid_regression', 0)
        else:
            ## These are irrelevant when using halfphones -- suppress them:
            forbid_repetition = False # self.config['forbid_repetition']
            forbid_regression = False # self.config['forbid_regression']

        ## For now, force join cost to be natural2
        join_cost_type = self.config['join_cost_type']
        join_cost_type = 'pitch_sync'
        assert join_cost_type in ['pitch_sync']


        data_frames, dim = self.unit_end_data.shape
        
        ## cache costs for joins which could be used in an utterance.
        ## This can save  computing things twice, 52 seconds -> 33 (335 frames, 50 candidates) 
        ## (Probably no saving with half phones?)
        cost_cache = {} 
        
        cost_cache_by_stream = {}

        ## set limits to not go out of range -- unnecessary given new unit_end_data and unit_start_data?
        if join_cost_type == 'pitch_sync':
            mini = 1 
            maxi = data_frames - 1              
        else:
            sys.exit('dvsdvsedv098987897')
        
        ###start = 0
        if not multiple_sentences:
            inds = [ind]
        else:
            inds = ind


        first_list = []
        second_list = []

        t = self.start_clock('     COST LIST')

        for ind in inds:
            frames, cands = np.shape(ind)

            for i in range(frames-1): 
                for first in ind[i,:]:
                    if first < mini or first >= maxi:
                        continue
                    for second in ind[i+1,:]:
                        if second < mini or second >= maxi:
                            continue
                        #print (first, second)
                        if (first == -1) or (second == -1):
                            continue
                        if (first, second) in cost_cache:
                            continue
                        
                        # if  join_cost_type == 'pitch_sync' and by_stream:
                        #     weight, weight_by_stream = self.get_natural_distance_by_stream(first, second, order=1)
                        #     cost_cache_by_stream[(first, second)] = weight_by_stream
                        # elif  join_cost_type == 'pitch_sync':
                        #     weight = self.get_natural_distance(first, second, order=1)
                        # else:
                        #     sys.exit('Unknown join cost type: %s'%(join_cost_type))
                        # weight *= self.config['join_cost_weight']


                        if forbid_repetition:
                            if first == second:
                                weight = VERY_BIG_WEIGHT_VALUE
                        if forbid_regression > 0:
                            if (first - second) in range(forbid_regression+1):
                                weight = VERY_BIG_WEIGHT_VALUE
                        #cost_cache[(first, second)] = weight
                        first_list.append(first)
                        second_list.append(second)
        self.stop_clock(t)



        t = self.start_clock('     DISTS')
        dists = self.get_natural_distance_vectorised(first_list, second_list, order=1)
        #print dists
        self.stop_clock(t)

        t = self.start_clock('     make cost cache')
        cost_cache = dict([((l,r), weight) for (l,r,weight) in zip(first_list, second_list, dists)])
        self.stop_clock(t)


    
        t = self.start_clock('      WRITE')
        if direct:
            J = cost_cache_to_compiled_fst(cost_cache, join_cost_weight=join_cost_weight)
        else:
            ## 2nd pass: write it to file
            if False: ## VIZ: show join histogram
                print len(cost_cache)
                pylab.hist([v for v in cost_cache.values() if v < VERY_BIG_WEIGHT_VALUE], bins=60)
                pylab.show()
            ### pruning:--
            #cost_cache = dict([(k,v) for (k,v) in cost_cache.items() if v < 3000.0])
            cost_cache_to_text_fst(cost_cache, outfile, join_cost_weight=join_cost_weight)
        self.stop_clock(t)


        if direct:
            return J





    def make_on_the_fly_join_lattice_PDIST(self, ind, outfile, join_cost_weight=1.0):

        '''
        pdist -- do all actual distance calculation with pdist
        '''

        forbid_repetition = False # self.config['forbid_repetition']
        forbid_regression = False # self.config['forbid_regression']

        join_cost_type = self.config['join_cost_type']

        join_cost_type = 'natural'
        assert join_cost_type in ['natural']

        data = self.train_unit_features

        start = 0
        frames, cands = np.shape(ind)
    

        data_frames, dim = data.shape
    
        #frames = 2
        
        ## cache costs for joins which could be used in an utterance.
        ## This can save  computing things twice, 52 seconds -> 33 (335 frames, 50 candidates) 
        cost_cache = {} 
        
        
        if join_cost_type == 'natural4':
            #contexts = self.get_contexts_for_natural_joincost(4, time_domain=False, weighted=True, debug=False)
            mini = 2 # 0-self.context_padding
            maxi = data_frames - 3 # (self.context_padding + 1 )
        elif join_cost_type == 'ps_natural':
                mini = 1
                maxi = data_frames - 2
        elif join_cost_type == 'natural':
            #contexts = self.get_contexts_for_natural_joincost(4, time_domain=False, weighted=True, debug=False)
            mini = 1 # 0-self.context_padding
            maxi = data_frames - 1 # (self.context_padding + 1 )                
        else:
            sys.exit('dvsdvsedv1222')
        

        t = start_clock('  ---> DISTS ')
        for i in range(frames-1): # (frames+2):

#            end = start+(cands*cands)+1        
            for first in ind[i,:]:
                if first < mini or first >= maxi:
                    continue
                for second in ind[i+1,:]:
                    if second < mini or second >= maxi:
                        continue
                    #print (first, second)
 
                    if (first == -1) or (second == -1):
                        continue

                    if (first, second) in cost_cache:
                        continue
                    if join_cost_type == 'distance_across':
                        sq_diffs = (data[first,:] - data[second,:])**2
                        sq_diffs *= self.join_weight_vector
                        weight = math.sqrt(np.sum(sq_diffs))
                    
                    elif  join_cost_type == 'natural':
                        first_list.append(first)
                        second_list.append(second)
                        
                        # sq_diffs = (data[first:first+2,:] - data[second-1:second+1,:])**2
                        # # print sq_diffs.shape
                        # # sq_diffs *= self.join_weight_vector
                        # #print '++++'
                        # #print sq_diffs.shape
                        # #print np.vstack([self.join_weight_vector]).shape
                        # sq_diffs *= np.vstack([self.join_weight_vector]*2)
                        # weight = 0.5 * math.sqrt(np.sum(sq_diffs))
                        
                    
                    elif  join_cost_type == 'natural4':
                        weight = self.get_natural4_distance(first, second)
#                         weighted_diffs = contexts[first+self.left_context_offset] - \
#                                          contexts[second+self.right_context_offset]
#                         weight = math.sqrt(np.sum(weighted_diffs ** 2))
                        #print weight
                        
                    elif  join_cost_type == 'natural8':
                        sq_diffs = (data[first-2:first+3,:] - data[second-3:second+2,:])**2
                        sq_diffs *= np.vstack([self.join_weight_vector]*8)
                        weight = 0.125 * math.sqrt(np.sum(sq_diffs))       
                    elif join_cost_type == 'cross_correlation':
                    
                        first_vec = wave_data[first,:]
                        second_vec = wave_data[second,:]
                        triframelength = first_vec.shape[0]
                        fr_len = triframelength / 3
                        weight = self.get_best_lag(first_vec[:fr_len*2], second_vec, \
                                        'cross_correlation', return_distance=True)
                        ##print 'CC weight'
                        ##print weight
                    elif join_cost_type == 'ps_distance_across_waves': 
                        first_data = ps_wave_data[first,:]
                        second_data = ps_wave_data[second,:]
                        sq_diffs = (first_data - second_data)**2
                        weight = math.sqrt(np.sum(sq_diffs))

                    elif join_cost_type == 'ps_natural':
                        first_data = self.ps_wave_frags[first:first+2,:]
                        second_data = self.ps_wave_frags[second-1:second+1,:]                    
                        sq_diffs = (first_data - second_data)**2
                        weight = math.sqrt(np.sum(sq_diffs))

                    # elif join_cost_type == 'ps_natural':
                    #     first_data = ps_wave_data[first:first+2,:]
                    #     second_data = ps_wave_data[second-1:second+1,:]                    
                    #     sq_diffs = (first_data - second_data)**2
                    #     weight = math.sqrt(np.sum(sq_diffs))
                                                                                                
                    else:
                        sys.exit('Unknown join cost type: %s'%(join_cost_type))

                    weight *= self.config['join_cost_weight']
                    if forbid_repetition:
                        if first == second:
                            weight = VERY_BIG_WEIGHT_VALUE
                    if forbid_regression > 0:
                        if (first - second) in range(forbid_regression+1):
                            weight = VERY_BIG_WEIGHT_VALUE
                    cost_cache[(first, second)] = weight


 #           start = end
            
        stop_clock(t)

        t = start_time('WRITE')
        ## 2nd pass: write it to file
        #print ' WRITE ',
        if False: ## VIZ: show join histogram
            print len(cost_cache)
            pylab.hist([v for v in cost_cache.values() if v < VERY_BIG_WEIGHT_VALUE], bins=60)
            pylab.show()
        ### pruning:--
        #cost_cache = dict([(k,v) for (k,v) in cost_cache.items() if v < 3000.0])
        
        #print len(cost_cache)
        cost_cache_to_text_fst(cost_cache, outfile, join_cost_weight=join_cost_weight)
        #print ' COMPILE ',
        stop_clock(t)


    def oracle_synthesis_holdout(self, outfname, start, length):
        t = self.start_clock('oracle_synthesis_holdout')

        assert start >= 0
        assert start + length < self.holdout_samples

        assert self.config['store_full_magphase_sep_files']

        magphase_overlap = self.config.get('magphase_overlap', 0)

        unit_features = self.train_unit_features_dev[start:start+length, :]
        
                
        # recover target F0:
        unit_features_no_weight = self.train_unit_features_unweighted_dev[start:start+length, :]
        unnorm_speech = destandardise(unit_features_no_weight, self.mean_vec_target, self.std_vec_target)   
        target_fz = unnorm_speech[:,-1] ## TODO: do not harcode F0 postion
        target_fz = np.exp(target_fz).reshape((-1,1))
        ### TODO: nUV is : 88.62008057.      This breaks resynthesis for some reason...
        
        target_fz[target_fz<90] = 0.0
        #target_fz *= 20.0
        #print target_fz

        #target_fz = np.ones((unit_features.shape[0], 1)) * 50 # 88.62  ## monotone 300 hz

        best_path = self.greedy_joint_search(unit_features)

        if self.config.get('magphase_use_target_f0', True):
            self.concatenateMagPhaseEpoch_sep_files(best_path, outfname, fzero=target_fz, overlap=magphase_overlap)                
        else:
            self.concatenateMagPhaseEpoch_sep_files(best_path, outfname, overlap=magphase_overlap) 

        self.stop_clock(t)     

        # print 'path info:'
        # print self.train_unit_names[best_path].tolist()





    def natural_synthesis_holdout(self, outfname, start, length):
        if 0:
            print outfname
            print start
            print length
            print 
        t = self.start_clock('natural_synthesis_holdout')
        assert start >= 0
        assert start + length < self.holdout_samples
        assert self.config['store_full_magphase_sep_files']
        magphase_overlap = 0
        multiepoch = self.config.get('multiepoch', 1)
        natural_path = np.arange(start, start+length, multiepoch) + self.number_of_units ## to get back to pre-hold-out indexing
        self.concatenateMagPhaseEpoch_sep_files(natural_path, outfname, overlap=0)                
        self.stop_clock(t)     

   
    def get_heldout_frag_starts(self, sample_pool_size, frag_length, filter_silence=''):
        n_frag_frames = sample_pool_size * frag_length
        assert n_frag_frames <= self.holdout_samples, 'not enough held out data to generate frags, try incresing holdout_percent or decreasing sample_pool_size'
        
        if filter_silence:
            sys.exit('Still to implement filter_silence')
            frags = segment_axis(self.train_unit_names_dev[:n_frag_frames], frag_length, overlap=0, axis=0)
            pause_sums = (frags==filter_silence) # , dtype=int).sum(axis=1)
            percent_silent = pause_sums / frag_length
            print percent_silent

        starts = np.arange(0, n_frag_frames, frag_length)
        selected_starts = np.random.choice(starts, sample_pool_size, replace=False)
        return selected_starts


if __name__ == '__main__':

    #################################################
      
    # ======== process command line ==========

    a = ArgumentParser()
    a.add_argument('-c', dest='config_fname', required=True)
    a.add_argument('-o', dest='output_dir', required=False, default='')
    opts = a.parse_args()


    synth = Synthesiser(opts.config_fname)
    #synth.test_concatenation_code()
    
    #synth.synth_from_config()
    if opts.output_dir:
        if not os.path.isdir(opts.output_dir):
            os.makedirs(opts.output_dir)
        os.system('cp %s %s'%(opts.config_fname, opts.output_dir))

    synth.synth_from_config(inspect_join_weights_only=False, synth_type='test', outdir=opts.output_dir)

