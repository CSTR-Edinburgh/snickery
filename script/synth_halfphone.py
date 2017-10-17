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

from util import safe_makedir, vector_to_string
from speech_manip import read_wave, write_wave, weight
from label_manip import break_quinphone, extract_monophone
from train_halfphone import get_data_dump_name, compose_speech, standardise, \
        read_label, get_halfphone_stats, reinsert_terminal_silence, make_train_condition_name, \
        locate_stream_directories

DODEBUG=False ## print debug information?


from train_halfphone import debug


import pylab

WRAPFST=True # True: used python bindings (pywrapfst) to OpenFST; False: use command line interface

if WRAPFST:
    from fst_functions_wrapped import compile_fst, make_target_sausage_lattice, cost_cache_to_text_fst, get_best_path_SIMP, compile_lm_fst, make_mapping_loop_fst, plot_fst, extract_path, compile_simple_lm_fst, sample_fst, make_sausage_lattice
else:
    from fst_functions import compile_fst, make_t_lattice_SIMP, cost_cache_to_text_fst, get_best_path_SIMP, compile_lm_fst, make_mapping_loop_fst


from const import label_delimiter


def start_clock(comment):
    print '%s... '%(comment),
    return (timeit.default_timer(), comment)

def stop_clock((start_time, comment), width=40):
    padding = (width - len(comment)) * ' '
    print '%s--> took %.2f seconds' % (padding, (timeit.default_timer() - start_time))  ##  / 60.)  ## min

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

        print 'Load config...'
        self.config = {}
        execfile(config_file, self.config)
        del self.config['__builtins__']
        
        self.stream_list_target = self.config['stream_list_target']
        self.stream_list_join = self.config['stream_list_join']

        print 'Prepare weights from config'
        ### TODO: check!
        datadims_target = self.config['datadims_target']
        datadims_join = self.config['datadims_join']

        self.target_weight_vector = np.array(self.config['feature_weights_target'] + self.config['feature_weights_target'])
        self.join_weight_vector   = np.array(self.config['feature_weights_join'] )    ### TODO: currently hardcoded for pitch sync cost
        assert len(self.target_weight_vector) == sum(datadims_target.values()) * 2

        print 'load database...'        
        datafile = get_data_dump_name(self.config)
        if not os.path.isfile(datafile):
            sys.exit('data: \n   %s   \ndoes not exist -- try other?'%(datafile))
            
        f = h5py.File(datafile, "r")
        self.train_unit_features = f["train_unit_features"][:,:]
        self.train_unit_names = f["train_unit_names"][:] 
        self.train_cutpoints = f["cutpoints"][:] 
        self.train_filenames = f["filenames"][:]                 
        self.mean_vec_target = f["mean_target"][:] 
        self.std_vec_target = f["std_target"][:] 
        self.mean_vec_join = f["mean_join"][:] 
        self.std_vec_join = f["std_join"][:] 

        join_contexts = f["join_contexts"][:,:]
        f.close()

        self.number_of_units, _ = self.train_unit_features.shape

        if self.config.get('weight_target_data', True):
            self.train_unit_features = weight(self.train_unit_features, self.target_weight_vector)   

        if self.config.get('weight_join_data', True):
            join_contexts = weight(join_contexts, self.join_weight_vector)   

        ## This should not copy:
        self.unit_end_data = join_contexts[1:,:]
        self.unit_start_data = join_contexts[:-1,:]

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

            start_time = start_clock('build KD tree')
            
            # if config['kdt_implementation'] == 'sklearn':
            #train = weight(self.train_unit_features, self.target_weight_vector)
            train = self.train_unit_features
            self.tree = sklearn_KDTree(train, leaf_size=1, metric='euclidean')   
            # elif config['kdt_implementation'] == 'scipy':
            #     tree = scipy_cKDTree(train, leafsize=1) 
            # elif config['kdt_implementation'] == 'stashable':
            #     tree = StashableKDTree.StashableKDTree(train, leaf_size=1, metric='euclidean')
                
            stop_clock(start_time)




        print 'Database loaded'
        print '\n\n----------\n\n'

        self.test_data_target_dirs = locate_stream_directories(self.config['test_data_dirs'], self.stream_list_target)
        print 'Found target directories: %s'%(self.test_data_target_dirs)
        print 
        print 
        

    def test_concatenation_code(self):
        ofile = '/afs/inf.ed.ac.uk/user/o/owatts/temp/concat_test.wav'
        print 'concatenate the start of the training data, output here: %s'%(ofile)
        self.concatenate(np.arange(100, 150), ofile)    
        


    def synth_from_config(self, inspect_join_weights_only=False):

        print 'synth_from_config'

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
            print 'synthesising %s utternances based on config'%(ntest)

        all_distances = []
        for fname in test_flist:
            if inspect_join_weights_only:
                all_distances.append(self.inspect_join_weights_on_utt(fname))
                print 'single utt -- break'
                break 
            else:
                self.synth_utt(fname)    


        if inspect_join_weights_only:
            all_distances = np.vstack(all_distances)
            m,n = all_distances.shape
            for stream in range(n):
                pylab.hist(all_distances[:,stream], bins=30, alpha=0.7)
            pylab.show()



    def make_synthesis_condition_name(self):
        '''
        Return string encoding all variables which can be ... 
        '''
        target_weights = vector_to_string(self.config['feature_weights_target'])
        join_weights = vector_to_string(self.config['feature_weights_join'])
        name = 'target-%s_join-%s_scale-%s_presel-%s_jmetric-%s_cand-%s_taper-%s'%(
                    target_weights, join_weights, self.config['join_cost_weight'],
                    self.config['preselection_method'],
                    self.config['join_cost_type'],
                    self.config['n_candidates'],
                    self.config['taper_length']
                )
        return name



    def synth_utt(self, fname):

        train_condition = make_train_condition_name(self.config)
        synth_condition = self.make_synthesis_condition_name()
        synth_dir = os.path.join(self.config['workdir'], 'synthesis', train_condition, synth_condition)
        safe_makedir(synth_dir)
            
        junk,base = os.path.split(fname)
        print '               ==== SYNTHESISE %s ===='%(base)
        base = base.replace('.mgc','')
        outstem = os.path.join(synth_dir, base)       

        start_time = start_clock('Get speech ')
        speech = compose_speech(self.test_data_target_dirs, base, self.stream_list_target, \
                                self.config['datadims_target']) 

        m,dim = speech.shape

        if (self.config['standardise_target_data'], True):                                
            speech = standardise(speech, self.mean_vec_target, self.std_vec_target)         
        
        #fshift_seconds = (0.001 * self.config['frameshift_ms'])
        #fshift = int(self.config['sample_rate'] * fshift_seconds)        

        labfile = os.path.join(self.config['test_lab_dir'], base + '.' + self.config['lab_extension'])
        labs = read_label(labfile, self.quinphone_regex)

        if self.config.get('untrim_silence_target_speech', False):
            speech = reinsert_terminal_silence(speech, labs)

        if self.config.get('suppress_weird_festival_pauses', False):
            labs = suppress_weird_festival_pauses(labs)

        unit_names, unit_features, unit_timings = get_halfphone_stats(speech, labs)
       
        if self.config['weight_target_data']:                                
            unit_features = weight(unit_features, self.target_weight_vector)       

        #print unit_features
        #print unit_names

        n_units = len(unit_names)
        stop_clock(start_time)


        if self.config['preselection_method'] == 'acoustic':

            start_time = start_clock('Acoustic select units ')
            ## call has same syntax for sklearn and scipy KDTrees:--
            distances, candidates = self.tree.query(unit_features, k=self.config['n_candidates'])
            stop_clock(start_time) 

        elif self.config['preselection_method'] == 'quinphone':

            start_time = start_clock('Preselect units ')
            #candidates = np.ones((n_units, self.config['n_candidates'])) * -1
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


            start_time = start_clock('Compute target distances...')
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
            stop_clock(start_time)          
       

        else:
            sys.exit('preselection_method unknown')


    
        start_time = start_clock('Make target FST')
        if WRAPFST:
            T = make_target_sausage_lattice(distances, candidates)        
        else:
            comm('rm -f /tmp/{target,join,comp,output}.*') ## TODO: don't rely on /tmp/ !
            make_t_lattice_SIMP(distances, candidates, '/tmp/target.fst.txt')
        stop_clock(start_time)          




        #start_time = start_clock('Make join FST with distances...')
        if False: # self.precomputed_joincost:
            print 'FORCE: Use existing join cost loaded from %s'%(self.join_cost_file)
        else:
            self.join_cost_file = '/tmp/join.fst'  ## TODO: don't rely on /tmp/ !           
            ## TODO: WRAPFST  

            self.make_on_the_fly_join_lattice(candidates, self.join_cost_file)


            t = start_clock('    COMPILE')
            compile_fst(self.tool, self.join_cost_file, self.join_cost_file + '.bin')
            stop_clock(t)


            #print 'sleep 1...'
 #           os.system('sleep 1')  
        #stop_clock(start_time)          

    




        start_time = start_clock('Compose and find shortest path')  
        if WRAPFST:
            if True: # not self.precomputed_joincost:
                J = openfst.Fst.read(self.join_cost_file + '.bin')
                stop_clock(start_time)     
                start_time = start_clock('Compose and find shortest path 2')     
                best_path = get_best_path_SIMP(T, J, \
                                                join_already_compiled=True, \
                                                add_path_of_last_resort=False)                        
            else:
                J = self.J ## already loaded into memory
                best_path = get_best_path_SIMP(T, J, \
                                                join_already_compiled=True, \
                                                add_path_of_last_resort=True)        
        else:
            best_path = get_best_path_SIMP(self.tool, '/tmp/target.fst.txt', self.join_cost_file, \
                                            join_already_compiled=self.precomputed_joincost, \
                                            add_path_of_last_resort=True)
        stop_clock(start_time)          


        # print 'got shortest path:'
        # print best_path
        # print len(best_path)
        # for i in best_path:
        #     print self.train_unit_names[i]




        start_time = start_clock('Extract and join units')
        self.concatenate(best_path, outstem + '.wav')    
        stop_clock(start_time)          


        print 'Output wave: %s.wav'%(outstem )
        print 
        print 

        output = []
        if self.config['get_selection_info']:
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

            print
            print 'target scores'
            
            chosen_features = self.train_unit_features[best_path]
            dists = np.sqrt(np.sum(((candidate_features - target_features)**2), axis=1))
            mean_dists = np.mean(distances, axis=1)
            std_dists = np.std(distances, axis=1)
            print zip(dists, mean_dists, std_dists)




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
        speech = compose_speech(self.config['test_data_dir'], base, self.stream_list_target, \
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
        self.config['n_candidates'] = 100 ### large number
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
        for (stream_width, stream_name) in [(1,'energy'),(12,'mfcc')]:
            distance_by_stream.append((1.0 / order) * math.sqrt(np.sum(sq_diffs[start:start+stream_width])) )
            start += stream_width

        distance = (1.0 / order) * math.sqrt(np.sum(sq_diffs))   
        return (distance, distance_by_stream)


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
    
    synth.synth_from_config()

    #synth.synth_from_config(inspect_join_weights_only=True)

