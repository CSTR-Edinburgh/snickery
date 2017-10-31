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
from argparse import ArgumentParser

import numpy as np
# import scipy

import h5py
import pywrapfst as openfst

from sklearn.neighbors import KDTree as sklearn_KDTree

from util import safe_makedir, vector_to_string, basename
from speech_manip import read_wave, write_wave, weight
from label_manip import break_quinphone, extract_monophone
from train_halfphone import get_data_dump_name, compose_speech, standardise, \
        read_label, get_halfphone_stats, reinsert_terminal_silence, make_train_condition_name, \
        locate_stream_directories

DODEBUG=False ## print debug information?


from train_halfphone import debug


# import pylab

WRAPFST=True # True: used python bindings (pywrapfst) to OpenFST; False: use command line interface

assert WRAPFST

if WRAPFST:
    from fst_functions_wrapped import compile_fst, make_target_sausage_lattice, cost_cache_to_text_fst, get_best_path_SIMP, compile_lm_fst, make_mapping_loop_fst, plot_fst, extract_path, compile_simple_lm_fst, sample_fst, make_sausage_lattice, cost_cache_to_compiled_fst
else:
    from fst_functions import compile_fst, make_t_lattice_SIMP, cost_cache_to_text_fst, get_best_path_SIMP, compile_lm_fst, make_mapping_loop_fst

import const
from const import label_delimiter



# import matplotlib.pyplot as plt; plt.rcdefaults()
# import matplotlib.pyplot as plt


# verbose = False # True # False



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

    

class Synthesiser(object):

    def __init__(self, config_file):



        self.mode_of_operation = 'normal'
        self.verbose = True

        print 'Load config...'
        self.config = {}
        execfile(config_file, self.config)
        del self.config['__builtins__']
        
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
        f.close()

        self.number_of_units, _ = self.train_unit_features_unweighted.shape

        if self.config.get('weight_target_data', True):
            self.set_target_weights(self.config['target_stream_weights'])
        if self.config.get('weight_join_data', True):
            self.set_join_weights(self.config['join_stream_weights'])

        self.first_silent_unit = 0 ## assume first unit is a silence, for v naive backoff


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

        ## set up some shorthand:-
        self.tool = self.config['openfst_bindir']

        if self.config['hold_waves_in_memory']:
            print 'load waves into memory'
            self.waveforms = {}
            for base in np.unique(self.train_filenames):

                print '.',
                wavefile = os.path.join(self.config['wav_datadir'], base + '.wav')
                wave, sample_rate = read_wave(wavefile)
                self.waveforms[base] = wave
        print

        assert self.config['preselection_method'] in ['acoustic', 'quinphone']
        if self.config['preselection_method'] == 'acoustic':

            start_time = self.start_clock('build KD tree')
            
            # if config['kdt_implementation'] == 'sklearn':
            #train = weight(self.train_unit_features, self.target_weight_vector)
            train = self.train_unit_features
            self.tree = sklearn_KDTree(train, leaf_size=1, metric='euclidean')   
            # elif config['kdt_implementation'] == 'scipy':
            #     tree = scipy_cKDTree(train, leafsize=1) 
            # elif config['kdt_implementation'] == 'stashable':
            #     tree = StashableKDTree.StashableKDTree(train, leaf_size=1, metric='euclidean')
                
            self.stop_clock(start_time)




        print 'Database loaded'
        print '\n\n----------\n\n'


        self.test_data_target_dirs = locate_stream_directories(self.config['test_data_dirs'], self.stream_list_target)
        self.tune_data_target_dirs = locate_stream_directories(self.config['tune_data_dirs'], self.stream_list_target)
        print 'Found target directories: %s'%(self.test_data_target_dirs)
        print 
        print 
        

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
        join_contexts_weighted = weight(self.join_contexts_unweighted, join_weight_vector)   

        ## This should not copy:
        self.unit_end_data = join_contexts_weighted[1:,:]
        self.unit_start_data = join_contexts_weighted[:-1,:]

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

        ## save this so we can weight incoming predicted acoustics: 
        self.target_weight_vector = target_weight_vector

        # print 'applied taget_weight_vector'
        # print target_weight_vector

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

        ### Only synthesise n sentences:
        if limit > 0:
            flist = flist[:limit]

        nfiles = len(flist)
        if nfiles == 0:
            print 'No files found for set "%s" based on configured test_data_dir and test_pattern'%(set_name)
        else:
            self.report('synthesising %s utternances based on config'%(nfiles))
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
            self.report('synthesising %s utternances based on config'%(ntest))
        return test_flist



    def synth_from_config(self, inspect_join_weights_only=False, synth_type='test'):

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


            self.synth_utt(fname, synth_type=synth_type)    

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

        ##### Current version: weight per stream.
        target_weights = '-'.join([str(val) for val in self.config['target_stream_weights']])
        join_weights = '-'.join([str(val) for val in self.config['join_stream_weights']])
        name = 'target-%s_join-%s_scale-%s_presel-%s_jmetric-%s_cand-%s_taper-%s'%(
                    target_weights, join_weights, self.config['join_cost_weight'],
                    self.config['preselection_method'],
                    self.config['join_cost_type'],
                    self.config['n_candidates'],
                    self.config['taper_length']
                )
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



    def synth_utt(self, base, synth_type='tune'): 

        if synth_type == 'test':
            data_dirs = self.test_data_target_dirs
            lab_dir = self.config['test_lab_dir']
        elif synth_type == 'tune':
            data_dirs = self.tune_data_target_dirs
            lab_dir = self.config['tune_lab_dir']            
        else:
            sys.exit('Unknown synth_type  9489384')

        train_condition = make_train_condition_name(self.config)
        synth_condition = self.make_synthesis_condition_name()
        synth_dir = os.path.join(self.config['workdir'], 'synthesis_%s'%(synth_type), train_condition, synth_condition)
        safe_makedir(synth_dir)
            
        self.report('               ==== SYNTHESISE %s ===='%(base))
        outstem = os.path.join(synth_dir, base)       

        start_time = self.start_clock('Get speech ')
        speech = compose_speech(data_dirs, base, self.stream_list_target, \
                                self.config['datadims_target']) 

        m,dim = speech.shape

        if (self.config['standardise_target_data'], True):                                
            speech = standardise(speech, self.mean_vec_target, self.std_vec_target)         
        
        #fshift_seconds = (0.001 * self.config['frameshift_ms'])
        #fshift = int(self.config['sample_rate'] * fshift_seconds)        

        labfile = os.path.join(lab_dir, base + '.' + self.config['lab_extension'])
        labs = read_label(labfile, self.quinphone_regex)

        if self.config.get('untrim_silence_target_speech', False):
            speech = reinsert_terminal_silence(speech, labs)

        if self.config.get('suppress_weird_festival_pauses', False):
            labs = suppress_weird_festival_pauses(labs)

        unit_names, unit_features, unit_timings = get_halfphone_stats(speech, labs, representation_type=self.target_representation)
       
        if self.config['weight_target_data']:                                
            unit_features = weight(unit_features, self.target_weight_vector)       

        n_units = len(unit_names)
        self.stop_clock(start_time)

        if self.config['preselection_method'] == 'acoustic':

            start_time = self.start_clock('Acoustic select units ')
            ## call has same syntax for sklearn and scipy KDTrees:--
            distances, candidates = self.tree.query(unit_features, k=self.config['n_candidates'])
            self.stop_clock(start_time) 

        elif self.config['preselection_method'] == 'quinphone':

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
                    current_candidates = [self.first_silent_unit]
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
       

        else:
            sys.exit('preselection_method unknown')

        if self.mode_of_operation == 'find_join_candidates':
            print 'mode_of_operation == find_join_candidates: return here'
            ## TODO: shuffle above operations so we can return this before looking at target features
            return candidates          


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
 
        if self.mode_of_operation == 'stream_weight_balancing':
            self.report('' )
            self.report( 'balancing stream weights -- skip making waveform')
            self.report('' )
        else:
            start_time = self.start_clock('Extract and join units')
            self.concatenate(best_path, outstem + '.wav')    
            self.stop_clock(start_time)          
            self.report( 'Output wave: %s.wav'%(outstem ))
            self.report('')
            self.report('')

        if self.mode_of_operation == 'stream_weight_balancing':
            tscores = self.get_target_scores_per_stream(target_features, best_path)
            jscores = self.get_join_scores_per_stream(best_path)
            return (tscores, jscores)

        if self.config['get_selection_info']:
            self.get_path_information(target_features, best_path)

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
            for form in [quinphone, triphone, diphone, mono]:
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
        if self.config['hold_waves_in_memory']:
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

        if self.config['hold_waves_in_memory']:
            wave = self.waveforms[self.train_filenames[index]]  
        else:     
            wavefile = os.path.join(self.config['wav_datadir'], self.train_filenames[index] + '.wav')
            wave, sample_rate = read_wave(wavefile)
        T = len(wave)        
        (start,end) = self.train_cutpoints[index]
        end += 1 ## non-inclusive end of slice
        
        taper = self.config['taper_length']
        
        if taper > 0:
            end = end + taper
            if end > T:
                pad = np.zeros(end - T)
                wave = np.concatenate([wave, pad])
                
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

    def concatenate(self, path, fname):

        frags = []
        for unit_index in path:
            frags.append(self.retrieve_speech(unit_index))

        if self.config['taper_length'] == 0:
            synth_wave = np.concatenate(frags)
        else:
            synth_wave = self.overlap_add(frags)
        write_wave(synth_wave, fname, 48000, quiet=True)


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



if __name__ == '__main__':

    #################################################
      
    # ======== process command line ==========

    a = ArgumentParser()
    a.add_argument('-c', dest='config_fname', required=True)
    opts = a.parse_args()


    synth = Synthesiser(opts.config_fname)
    #synth.test_concatenation_code()
    
    #synth.synth_from_config()

    synth.synth_from_config(inspect_join_weights_only=False, synth_type='test')

