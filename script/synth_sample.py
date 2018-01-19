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

# # Cassia added
# import smoothing.fft_feats as ff
# import smoothing.libwavgen as lwg
# import smoothing.libaudio as la

import numpy as np
import scipy


import pylab

import h5py
# import pywrapfst as openfst

# # from sklearn.neighbors import KDTree as sklearn_KDTree
# import StashableKDTree

from util import safe_makedir # , vector_to_string, basename
from speech_manip import write_wave, weight #  read_wave,
# from label_manip import break_quinphone, extract_monophone
from train_halfphone import get_data_dump_name, locate_stream_directories, make_train_condition_name, compose_speech, standardise # , \
#         read_label, get_halfphone_stats, reinsert_terminal_silence, , \
#         
import resample
from mulaw2 import mu2lin, lin2mu
from segmentaxis import segment_axis
# DODEBUG=False ## print debug information?


# from train_halfphone import debug

# from const import VERY_BIG_WEIGHT_VALUE

# # import pylab

# WRAPFST=True # True: used python bindings (pywrapfst) to OpenFST; False: use command line interface

# assert WRAPFST

# if WRAPFST:
#     from fst_functions_wrapped import compile_fst, make_target_sausage_lattice, cost_cache_to_text_fst, get_best_path_SIMP, compile_lm_fst, make_mapping_loop_fst, plot_fst, extract_path, compile_simple_lm_fst, sample_fst, make_sausage_lattice, cost_cache_to_compiled_fst
# else:
#     from fst_functions import compile_fst, make_t_lattice_SIMP, cost_cache_to_text_fst, get_best_path_SIMP, compile_lm_fst, make_mapping_loop_fst

import const
# from const import label_delimiter


# import cPickle as pickle
# from pykdtree.kdtree import KDTree as pKDTree

from synth_halfphone import Synthesiser
import varying_filter

NORMWAVE=False # False

class SampleSelectionSynthesiser(Synthesiser):

    def __init__(self, config_file, holdout_percent=0.0, build_tree=True):



        self.mode_of_operation = 'normal'
        self.verbose = True

        print 'Load config...'
        self.config = {}
        execfile(config_file, self.config)
        del self.config['__builtins__']
        
        self.stream_list_target = self.config['stream_list_target']
        # self.stream_list_join = self.config['stream_list_join']

        print 'Prepare weights from config'
        ### TODO: check!
        self.datadims_target = self.config['datadims_target']
        # self.datadims_join = self.config['datadims_join']

        self.target_representation = self.config['target_representation']
        assert self.target_representation == 'sample'

        self.config['joincost_features'] = False

        self.rate = self.config['sample_rate']
        self.fshift_seconds = (0.001 * self.config['frameshift_ms'])

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


        '''
        Loading hybrid voice data:
        <HDF5 dataset "mean_target": shape (60,), type "<f4">
        <HDF5 dataset "nextsample": shape (151840, 1), type "<i4">
        <HDF5 dataset "std_target": shape (1, 60), type "<f4">
        <HDF5 dataset "train_unit_features": shape (151840, 70), type "<f4">
        '''

        self.train_unit_features_unweighted = f["train_unit_features"][:,:]
        self.mean_vec_target = f["mean_target"][:] 
        self.std_vec_target = f["std_target"][:] 
        self.nextsample = f["nextsample"][:,:]
        
        print self.train_unit_features_unweighted[:,:10].mean()

        print self.train_unit_features_unweighted[:,:10].std()

        # pylab.plot( self.train_unit_features_unweighted[221000,:10].transpose())
        # pylab.show()
        # sys.exit('avasdv')


        if NORMWAVE:
            wave_mu_sigma = f["wave_mu_sigma"][:]
            self.wave_mean, self.wave_std = wave_mu_sigma

        f.close()

        self.number_of_units, _ = self.train_unit_features_unweighted.shape

        # self.extend_weights_with_wavecontext()

        self.holdout_percent = holdout_percent
        if holdout_percent > 0.0:
            holdout_samples = int(self.number_of_units * (holdout_percent/100.0))
            print 'holdout_samples:'
            print holdout_samples
            #holdout_indices = np.random.choice(m, size=npoint, replace=False)

            self.train_unit_features_unweighted_dev = self.train_unit_features_unweighted[-holdout_samples:,:]
            self.nextsample_dev = self.nextsample[-holdout_samples:,:]

            self.train_unit_features_unweighted = self.train_unit_features_unweighted[:-holdout_samples,:]
            self.nextsample = self.nextsample[:-holdout_samples,:]

            self.number_of_units -= holdout_samples
            #sys.exit('evwservgwsrbv')


        if build_tree:
            self.set_weights(self.config['target_stream_weights'])            
            self.get_tree_for_greedy_search()






        print 'Database loaded'
        print '\n\n----------\n\n'


        self.test_data_target_dirs = locate_stream_directories(self.config['test_data_dirs'], self.stream_list_target)
        self.tune_data_target_dirs = locate_stream_directories(self.config['tune_data_dirs'], self.stream_list_target)
        print 'Found target directories: %s'%(self.test_data_target_dirs)
        print 
        print 
        

    # def extend_weights_with_wavecontext(self):

    #     self.stream_list_target_only = copy.copy(self.stream_list_target)
    #     self.stream_list_target = ['wavecontext'] + self.stream_list_target
    #     self.datadims_target['wavecontext'] = self.config['wave_context_length']
    #     self.config['target_stream_weights'] = [1.0] + self.config['target_stream_weights']

    def set_weights(self, weights):
        assert len(weights) == len(self.stream_list_target), (weights, self.stream_list_target)
        ## get from per-stream to per-coeff weights:
        target_weight_vector = []
        for (i,stream) in enumerate(self.stream_list_target):
            target_weight_vector.extend([weights[i]]*self.datadims_target[stream])
        nrepetitions = const.target_rep_widths[self.target_representation]
        target_weight_vector = np.array(target_weight_vector * nrepetitions)   
        ## save this so we can weight incoming predicted acoustics: 
        self.target_weight_vector = target_weight_vector

        ### extend with arbitrary weights (1s) for waveform history:-
        extended_weight_vector = np.concatenate([np.ones(self.config['wave_context_length']), target_weight_vector])
        self.train_unit_features = weight(self.train_unit_features_unweighted, extended_weight_vector)  
        if self.holdout_percent > 0.0:
            self.train_unit_features_dev = weight(self.train_unit_features_unweighted_dev, extended_weight_vector)   

    def set_weights_per_feature(self, weights):
        
        self.target_weight_vector = np.array(weights[self.config['wave_context_length']:])
        extended_weight_vector = np.array(weights)
        self.train_unit_features = weight(self.train_unit_features_unweighted, extended_weight_vector)  
        if self.holdout_percent > 0.0:
            self.train_unit_features_dev = weight(self.train_unit_features_unweighted_dev, extended_weight_vector)   




    def get_tree_for_greedy_search(self):

        start_time = self.start_clock('build/reload joint KD tree')
        ## Needs to be stored synthesis options specified (due to weights applied before tree construction):
        treefile = get_data_dump_name(self.config) + '_' + self.make_synthesis_condition_name() + '_joint_tree.pkl' 
        if False: # os.path.exists(treefile):  ## never reload!
            print 'Tree file found -- reload from %s'%(treefile)
            self.joint_tree = pickle.load(open(treefile,'rb'))
        else:
            #print 'Seems like this is first time synthesis has been run on this data.'
            #print 'Build a search tree which will be saved for future use'

            combined_rep = self.train_unit_features
            t = self.start_clock('make joint join + target tree...')
            self.joint_tree = scipy.spatial.cKDTree(combined_rep, leafsize=100, balanced_tree=False, compact_nodes=False)
            #self.joint_tree = scipy.spatial.cKDTree(combined_rep, leafsize=100)
            #self.joint_tree = scipy.spatial.KDTree(combined_rep, leafsize=100)

            # np.save('/tmp/dat', combined_rep)
            # sys.exit('llll')

            #self.joint_tree = pKDTree(combined_rep, leafsize=100)
            self.stop_clock(t)


            # print dir(self.joint_tree)
            # print dir(self.joint_tree.tree)
            # tr = self.joint_tree.tree
            # print tr.children
            # print tr.data_points.shape

            # print tr.greater
            # print tr.indices.shape
            # print tr.lesser
            # print tr.level
            # print tr.split
            # print tr.split_dim


            # print id(tr.data_points[0,0])
            # print id(combined_rep[0,0])

            # print tr.data_points[0,:3]
            # print combined_rep[tr.indices[0],:3]


            # sys.exit('adevaev')
            ## 368640 x 14:    3963520
            ### a) 0.15         2.51
            ### b) ages... 
            ### c) 2.11         40.1
            ## d) 0.16


            ### 


    def sample(self, nprevious, nsamples, fname):
        '''
        Only use join part of data to build a tree, then draw n samples of speech
        '''

        context_length = self.config['wave_context_length']

        ## nprevious lets us vary context length on the fly:
        if nprevious == 0:
            #start=0
            pass
        else:
            assert nprevious < context_length
            #start = context_length - nprevious
            self.config['wave_context_length'] = nprevious
            context_length = self.config['wave_context_length']

        self.train_unit_features = self.train_unit_features_unweighted[:,:context_length]
        self.get_tree_for_greedy_search()
        best_path, gen_wave = self.greedy_joint_search(nsamples, k=1, eps=0, initial_state=-1) # 30000)
        if NORMWAVE:
            ### denormalise:-
            gen_wave =  (gen_wave * self.wave_std) + self.wave_mean  # gen_wave + self.wave_mean # 
        if self.config['nonlin_wave']:
            gen_wave = mu2lin(gen_wave)
        #print gen_wave.tolist()
        #print best_path.tolist()
        write_wave(gen_wave, fname, self.rate)





    def synth_utt(self, base, synth_type='tune', outstem=''): 

        if synth_type == 'test':
            data_dirs = self.test_data_target_dirs
            # lab_dir = self.config['test_lab_dir']
        elif synth_type == 'tune':
            data_dirs = self.tune_data_target_dirs
            # lab_dir = self.config['tune_lab_dir']            
        else:
            sys.exit('Unknown synth_type  943957011')

        if not outstem:
            train_condition = make_train_condition_name(self.config)
            synth_condition = self.make_synthesis_condition_name()
            synth_dir = os.path.join(self.config['workdir'], 'synthesis_%s'%(synth_type), train_condition, synth_condition)
            safe_makedir(synth_dir)
                
            self.report('               ==== SYNTHESISE %s ===='%(base))
            outstem = os.path.join(synth_dir, base)       
        else:
            self.report('               ==== SYNTHESISE %s ===='%(outstem))

        start_time = self.start_clock('Get speech ')
        speech = compose_speech(data_dirs, base, self.stream_list_target, \
                                self.config['datadims_target']) 


        ### upsample before standardisation (inefficient, but standardisation rewrites uv values?? TODO: check this)
        nframes, dim = speech.shape
        len_wave = int(self.rate * self.fshift_seconds * nframes)
        speech = resample.upsample(len_wave, self.rate, self.fshift_seconds, speech, f0_dim=-1, convention='world')



        if (self.config['standardise_target_data'], True):                                
            speech = standardise(speech, self.mean_vec_target, self.std_vec_target)         
        
        #fshift_seconds = (0.001 * self.config['frameshift_ms'])
        #fshift = int(self.config['sample_rate'] * fshift_seconds)        

        unit_features = speech
           
        unit_features = weight(unit_features, self.target_weight_vector)       

        #### TEMp!!!!!!
        #unit_features = unit_features[2000:3000, :]

        n_units, _ = unit_features.shape
        self.stop_clock(start_time)

        ### always do greedy search for sample-based selection
        best_path, gen_wave = self.greedy_joint_search(unit_features)

        #print best_path
        #print gen_wave
       
        if NORMWAVE:

            print 'predenorm stats:'
            print (gen_wave.mean(), gen_wave.std())
            ### denormalise:-
            gen_wave =  (gen_wave * self.wave_std) + self.wave_mean  # gen_wave + self.wave_mean # 
            print 'denorm stats:'
            print (gen_wave.mean(), gen_wave.std())


        if self.config['nonlin_wave']:
            gen_wave = mu2lin(gen_wave)

            # print 'linear stats:'
            # print (gen_wave.mean(), gen_wave.std())


        # pylab.plot(gen_wave)
        # pylab.show()

        if self.mode_of_operation == 'stream_weight_balancing':
            self.report('' )
            self.report( 'balancing stream weights -- skip making waveform')
            self.report('' )
        else:
            start_time = self.start_clock('Wrtie wave')            
            write_wave(gen_wave, outstem + '.wav', self.rate)
            self.stop_clock(start_time)          
            self.report( 'Output wave: %s.wav'%(outstem ))
            self.report('')
            self.report('')

        ##### for now, do not print path information: -- TODO: make this configuratble

        # self.get_path_information(unit_features, best_path, gen_wave)

        # sys.exit('kcjabekcvbabekv')





        # target_features = unit_features ## older nomenclature?
        # if self.mode_of_operation == 'stream_weight_balancing':
        #     tscores = self.get_target_scores_per_stream(target_features, best_path)
        #     jscores = self.get_join_scores_per_stream(best_path)
        #     return (tscores, jscores)

        # if self.config['get_selection_info'] and self.config['target_representation'] != 'epoch':
        #     self.get_path_information(target_features, best_path)


    def get_path_information(self, target_features, path, waveform):
        context = self.config['wave_context_length']
        gen_wave = self.nextsample[path,:]
        print gen_wave.shape
        print '===='
        padded_wave = np.concatenate([np.zeros((context, 1)), gen_wave]) 
        print padded_wave.shape       
        wavefrags = segment_axis(padded_wave.flatten(), context+1, overlap=context, axis=0)
        join_features = wavefrags[:,:-1]
        nextsamples = wavefrags[:,-1]

        print join_features.shape
        print target_features.shape
        combined_features = np.hstack([join_features, target_features])
        dists, samples = self.joint_tree.query(combined_features, k=1, eps=2)
        print samples.shape
        print dists.shape
        print dists

        selected = self.train_unit_features[samples, :]
        dists2 = np.sqrt(((combined_features - selected)**2).sum(axis=1))

        ### stream contributions...
        raw_dists = (combined_features - selected)**2
        history_contrib = np.sqrt(raw_dists[:,:context].sum())
        print 'history'
        print history_contrib
        start = context
        for stream in self.stream_list_target:
            width = self.datadims_target[stream]
            end = start + width
            stream_contrib = np.sqrt(raw_dists[:,start:end].sum())
            print stream
            print stream_contrib
            start = end



        ### natural joins:
        pairs = copy.copy(segment_axis(path, 2, 1, axis=0))
        pairs[:,0] += 1
        pairs[:,0] *= -1
        diff = pairs.sum(axis=1)
        breaks = (diff != 0)
        breaks = np.array(breaks, dtype=int)
        breaks = np.concatenate([np.ones(1), breaks])
        print breaks



        sys.exit('sesrbsfrb')
        # pylab.plot(dists)
        # pylab.plot(dists2)
        # pylab.show()


        ### density:
        distance_thresh = dists.mean() * 2

        ## 1) how many points within twice average distance from targets?
        neighbours = self.joint_tree.query_ball_point(combined_features, distance_thresh, eps=2)
        n_neighbours_target = [len(thing) for thing in neighbours]

        ## 2) how many points within twice average distance from selected things?
        neighbours = self.joint_tree.query_ball_point(selected, distance_thresh, eps=2)
        n_neighbours_selected = [len(thing) for thing in neighbours]




        #print path



        pylab.subplot(411)
        pylab.plot(dists)
        pylab.subplot(412)
        pylab.plot(n_neighbours_target)
        pylab.plot(n_neighbours_selected)
        pylab.subplot(413)
        pylab.plot(breaks)
        pylab.subplot(414)        
        pylab.plot(waveform)
        
        pylab.show()        




    def greedy_joint_search(self, target_feats, k=1, eps=1, make_plot=False, initial_state=-1):   # , start_state=-1, holdout=[]

        ## if target_feats is integer, sample this many samples without conditioning on target:
        condition_on_target=True
        if type(target_feats) == int:
            target_feats = range(target_feats)
            condition_on_target=False


        wave_context_type = self.config.get('wave_context_type',0)
        context_length = self.config['wave_context_length']
        if wave_context_type == 0:
            wavefrag_length = context_length
        elif wave_context_type == 1:
            DILATION_FACTOR = 1.2
            filter_matrix = varying_filter.make_filter_01(DILATION_FACTOR, context_length)
            wavefrag_length, nfeats = filter_matrix.shape
            assert nfeats == context_length
        else:
            sys.exit('unknown wave_context_type: %s'%(wave_context_type))


        start_time = self.start_clock('Greedy search')
        path = []

        if initial_state < 0: ### then start with zero vector
            transformed_zero = 0.0
            if self.config.get('nonlin_wave', False):
                transformed_zero = lin2mu(transformed_zero) 
            if NORMWAVE:
                transformed_zero = (transformed_zero - self.wave_mean) / self.wave_std
            gen_wave = [transformed_zero] * wavefrag_length             # initialise state with zero vector
        else:
            assert initial_state < self.number_of_units
            assert wave_context_type == 0
            gen_wave = self.train_unit_features[initial_state, :context_length].tolist()
            pylab.plot(gen_wave)
            pylab.show()

        if make_plot:
            points_x = []
            points_y = []
            k = 6
        for (i,frame) in enumerate(target_feats):
            if i % 500 == 0:
                print 'Selecting the %sth sample...'%(i)
            # if i == 5000:
            #     break
            curr = np.array(gen_wave[-wavefrag_length:], dtype=int)
            if wave_context_type > 0:
                curr = np.dot(curr, filter_matrix)
            if condition_on_target:
                input = np.concatenate([curr, frame]).reshape((1,-1))
            else:
                input = curr.reshape((1,-1))
            dists, cands = self.joint_tree.query(input, k=k, eps=eps)
            # if i % 500 == 0:
            #     print dists            
            if make_plot:
                ix = cands.flatten()[0] # random.choice(cands.flatten())
                print self.nextsample[cands.flatten(),:].flatten()
                print self.nextsample[cands.flatten()[0],:].flatten()
                points_x.extend([i]*6)
                points_y.extend(self.nextsample[cands.flatten(),:].flatten().tolist())
                #next= self.nextsample[cands.flatten(),:].mean()            
            elif k > 1:
                ix = random.choice(cands.flatten())
                #next= self.nextsample[cands.flatten(),:].mean()
            else:
                ix = cands[0]
            #print '----'
            #print input
            #print self.train_unit_features[ix,:]

            path.append(ix)
            gen_wave.extend(self.nextsample[ix,:].tolist())
            #gen_wave.append(next.tolist())

        self.stop_clock(start_time)




        gen_wave = np.array(gen_wave[wavefrag_length:]) # trim off initial state (zero vector)
        if make_plot:
            pylab.scatter(points_x, points_y   )        
            pylab.plot(gen_wave, color='r')
            pylab.show()
        path = np.array(path)
        return path, gen_wave


    # def oracle_synthesis_training(self, start, end, eps=0 ):
    #     target_feats = self.train_unit_features[start:end, :]
    #     dists, cands = self.joint_tree.query(target_feats, k=2, eps=eps)
    #     assert dists[:,0].sum() == 0
    #     chosen_ixx = cands[:,1].flatten()
    #     synth_wave = self.nextsample[chosen_ixx,:]
    #     orig_wave = self.nextsample[start:end,:]
    #     error = math.sqrt(  ((synth_wave-orig_wave)*(synth_wave-orig_wave)).sum()  ) 
    #     print error
    #     return error

    def oracle_synthesis_training(self, npoint, eps=0 ):
        t = self.start_clock('oracle_synthesis_training')
        m,n = self.train_unit_features.shape
        points = np.random.choice(m, size=npoint, replace=False)
        target_feats = self.train_unit_features[points, :]
        dists, cands = self.joint_tree.query(target_feats, k=1, eps=eps, n_jobs=4)
        # assert dists[:,0].sum() == 0
        # chosen_ixx = cands[:,1].flatten()
        chosen_ixx = cands
        synth_wave = self.nextsample[chosen_ixx,:]
        orig_wave = self.nextsample[points,:]
        error = math.sqrt(  ((synth_wave-orig_wave)*(synth_wave-orig_wave)).sum()  ) 
        self.stop_clock(t)
        print error

        return error

    def oracle_synthesis_holdout(self, eps=2 ):
        t = self.start_clock('oracle_synthesis_holdout')
        dists, chosen_ixx = self.joint_tree.query(self.train_unit_features_dev, k=1, eps=eps, n_jobs=14)
        synth_wave = self.nextsample[chosen_ixx,:]
        orig_wave = self.nextsample_dev
        error = math.sqrt(  ((synth_wave-orig_wave)*(synth_wave-orig_wave)).sum()  ) 
        self.stop_clock(t)
        #print error

        return error



if __name__ == '__main__':

    #################################################
      
    # ======== process command line ==========

    a = ArgumentParser()
    a.add_argument('-c', dest='config_fname', required=True)
    opts = a.parse_args()

    synth = SampleSelectionSynthesiser(opts.config_fname, holdout_percent=5) # , build_tree=False)

    # scores = []
    # for wt in range(1000, 20000, 1000):
    #     synth.set_weights([wt, 1000.0])    
    #     synth.get_tree_for_greedy_search()
    #     s = synth.oracle_synthesis_holdout()  ## 1 second
    #     scores.append(s)
    # pylab.plot(scores)
    # pylab.show()

    synth.synth_from_config(inspect_join_weights_only=False, synth_type='test')

    #synth.sample(25, 16000, '/tmp/samples.wav')
