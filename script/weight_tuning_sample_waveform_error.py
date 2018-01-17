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

from synth_sample import SampleSelectionSynthesiser

import copy
import random
# import pylab


from speech_manip import read_wave, write_wave


precision = 100 # 0.01



import logging
logger = logging.getLogger('weight_tuning')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('weight_tuning.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') # logging.Formatter('%(message)s')
fh.setFormatter(file_formatter)
ch.setFormatter(console_formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)





def ternary_search(synth, weights, dim, low, high, tau, verbose=False):
    if verbose:
        print 'ternary_search on interval: %s'%(str((low, high)))
    gap = (high - low)
    if gap <= tau:
        return low + (gap / 2.0)

    a = low + (gap * 0.33333)
    b = low + (gap * 0.66666)
    weights_a = copy.copy(weights)
    weights_b = copy.copy(weights)
    weights_a[dim] = a
    weights_b[dim] = b

    #print 'weights a: %s'%(a)
    synth.set_weights(weights_a)    
    #print 'tree a'    
    synth.get_tree_for_greedy_search()
    #print 'score a'
    score_a = synth.oracle_synthesis_holdout()  

    #print 'weights b: %s'%(b)
    synth.set_weights(weights_b)  
    #print 'tree b'      
    synth.get_tree_for_greedy_search()
    #print 'score b'    
    score_b = synth.oracle_synthesis_holdout()   

    print '              scored: %s : %s'%(str(weights_a), score_a)
    print '              scored: %s : %s'%(str(weights_b), score_b)


    if score_b < score_a:
        return ternary_search(synth, weights, dim, a, high, tau)
    else: 
        return ternary_search(synth, weights, dim, low, b, tau)






if __name__ == '__main__':

    #################################################
      
    # ======== process command line ==========


    a = ArgumentParser()
    a.add_argument('-c', dest='config_fname', required=True)
    opts = a.parse_args()

    synth = SampleSelectionSynthesiser(opts.config_fname, holdout_percent=5)

    CUSTOM = False
    STREAM = False
    if CUSTOM:

        #logger.debug('Duplicating synth...')
        #synth_b = copy.deepcopy(synth)
        #logger.debug('Done')


        #assert synth.config['join_cost_weight'] == 1.0
        #synth.mode_of_operation = 'stream_weight_balancing'
        synth.verbose = False

        ## initially, unweighted streams -- all 1.0:
        njoin = 0 # len(synth.stream_list_join)
        ntarget = len(synth.stream_list_target)

        weight_size = njoin + ntarget
        
        hist = []

        
        
        #current_weights = np.random.uniform(low=0.0, high=1.0, size=weight_size)
        #current_weights = np.array([ 0.28776146,  0.06687811,  0.50752005,  0.92932127,  0.5148148 ])


        search_ranges = [(0, 20000), (0, 20000)]

        ### start at centre of search space:
        current_weights = [(s+e)/2.0 for (s,e) in search_ranges]




        print 'Start search per dimension...'

        while True:

            for dim in range(weight_size):
                print '====== DIM %s ====='%(dim)
                #current_weights = line_search(synth, current_weights, dim, njoin)
                w = ternary_search(synth, current_weights, dim, search_ranges[dim][0], search_ranges[dim][1], precision)
                current_weights[dim] = w
                print current_weights

    elif STREAM:
        synth.verbose = False
        from scipy.optimize import minimize

        def display_current(vals):
            print vals

        def objective(x):
            print random.choice('abcdefghijk')
            synth.set_weights(x)    
            synth.get_tree_for_greedy_search()
            score = synth.oracle_synthesis_holdout() 
            # nevals += 1
            return score 
        x0 = np.array([10000, 10000])
        res = minimize(objective, x0, method='nelder-mead', options={'xtol': 100, 'disp': True}, callback=display_current) # BFGS
        #res = minimize(objective, x0, method='powell', options={'xtol': 100, 'disp': True})        
        #res = minimize(objective, x0, method='BFGS', options={'xtol': 100, 'disp': True}) # BFGS
        
        print res

    else:
        synth.verbose = False
        from scipy.optimize import minimize

        def display_current(vals):
            print vals

        def objective(x):
            print random.choice('abcdefghijk')
            synth.set_weights_per_feature(x)    
            synth.get_tree_for_greedy_search()
            score = synth.oracle_synthesis_holdout() 
            # nevals += 1
            return score 

        target_weight_vector = []
        for (i,stream) in enumerate(synth.stream_list_target):
            target_weight_vector.extend([synth.config['target_stream_weights'][i]]*synth.datadims_target[stream])
        ### extend with arbitrary weights (1s) for waveform history:-
        extended_weight_vector = np.concatenate([np.ones(synth.config['wave_context_length']), target_weight_vector])


        x0 = np.array(extended_weight_vector)
        print x0
        res = minimize(objective, x0, method='nelder-mead', options={'xtol': 100, 'disp': True}, callback=display_current) # BFGS
        #res = minimize(objective, x0, method='powell', options={'xtol': 100, 'disp': True})        
        #res = minimize(objective, x0, method='BFGS', options={'xtol': 100, 'disp': True}) # BFGS
        
        print res