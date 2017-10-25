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

from synth_halfphone import Synthesiser

import copy
import pylab


if __name__ == '__main__':

    #################################################
      
    # ======== process command line ==========

    a = ArgumentParser()
    a.add_argument('-c', dest='config_fname', required=True)
    opts = a.parse_args()


    synth = Synthesiser(opts.config_fname)
    synth.mode_of_operation = 'stream_weight_balancing'
    synth.verbose = False

    ## initially, unweighted streams -- all 1.0:
    njoin = len(synth.stream_list_join)
    ntarget = len(synth.stream_list_target)

    ### multiply errors by these factors, to ensure target and join contribute equally.
    ##  The per-stream subcosts will contribute equally within each stream.
    stream_number_correction = np.array( [0.5/njoin]*njoin  +  [0.5/ntarget]*ntarget )

    weights = np.ones(njoin + ntarget)

    #weights[1] = 1000000000.0

    hist = []

    best_weights = copy.copy(weights)
    best_score = previous_score = float('inf')
    epochs_without_improvement = 0
    thresh = 0.001 ## when we are this close to zero, stop 
    patience = 5
    max_epochs=1000
    losses = []


    weight_floor = 0.0 # 0.01

    eta = 0.1 # 0.001

    lrates = np.ones(weights.shape)*eta
    directions = np.zeros(weights.shape)

    prev_directions = np.ones(weights.shape)

    amplifier = 1.2
    attenuator = 0.5

    dmax = 50.0
    dmin = 0.000001

    # smooth_over = 5

    flist = synth.get_test_sentences() # [:1]

    MP = False
    if MP:
        import multiprocessing, logging
        mpl = multiprocessing.log_to_stderr()
        mpl.setLevel(logging.DEBUG) # INFO)


        p = multiprocessing.Manager().Pool(4)

    global_mean_dummy_initial_value = -10000000
    global_mean = global_mean_dummy_initial_value

    


    contribs = []
    for i in range(max_epochs):

        print '===== %s ====='%(i)

        synth.set_join_weights(weights[:njoin])
        synth.set_target_weights(weights[njoin:])


        #all_scores = p.map(synth, flist)
        all_scores = []
        
        if MP:
            results = [p.apply_async(synth, args = (fname, )) for fname in flist]
            output = [r.get() for r in results] 
            p.close()
            p.join()
            print output

            # print all_scores
            sys.exit('0000000000')
        #print 'Get scores on %s'%fname

        # cache = []

        cache = synth.synth_utts_bulk(flist)

        # for fname in flist:
        #     print fname

        #     tscores, jscores = synth.synth_utt(fname)

        #     cache.append((tscores, jscores))
           

        mean_scores = []

        #for (jscores, tscores) in cache:
        cached_jscores = np.vstack([jscores for (tscores, jscores) in cache])
        cached_tscores = np.vstack([tscores for (tscores, jscores) in cache]) 
        for scores in [cached_jscores, cached_tscores]:
            m,n = scores.shape
            for column in xrange(n):
                vals = scores[:,column]
                vals = vals[vals>0.0]
                total = vals.shape[0]
                if total == 0:  ## There are no nonzeros where e.g. there are no joins.
                    ### This is unlikey to ever happen when using a batch of many utterances.
                    mean_without_zeros = 0.0 ### Avoid divis by zero -- TODO: choose better value than this hack...
                else:
                    mean_without_zeros = vals.sum() / total
                mean_scores.append(mean_without_zeros)
        mean_scores = np.array(mean_scores)

      

        # ((mean_tscores, mean_jscores),(std_tscores, std_jscores)) = synth.synth_from_config()
        # mean_scores = np.concatenate([mean_jscores, mean_tscores])
        # std_scores = np.concatenate([std_jscores, std_tscores])



        #print mean_scores


        # if i==0:  ## work out max on first iteration and keep same as out target
        #     goal = np.argmax(mean_scores)
            
#        goal = np.argmax(mean_scores)
#        errors = mean_scores[goal] - mean_scores

        if global_mean==global_mean_dummy_initial_value:
            global_mean = mean_scores.mean()

        errors = global_mean - mean_scores
        #errors = mean_scores -  global_mean 


        #errors *= stream_number_correction

        print errors

        loss = np.abs(errors).sum() 

        print '=======LOSS========='
        print loss
        # print epochs_without_improvement
        losses.append(loss)
        if loss < previous_score:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1 

        if loss < best_score:
            best_score = loss
            best_weights = copy.copy(weights)

        if epochs_without_improvement == patience:
            print 'converged'
            break

        if loss < thresh:
            print 'loss approaching 0 -- stop here'
            break

        # print 'a'
        # print('means')
        # print(mean_scores)
        
        # print('error')
        # print( errors)
        #print j_errors

        directions = np.sign(errors)
        
        # change: 1 means same direction, -1 measns change direction
        direction_change = directions * prev_directions

        lrates[direction_change>0] *= amplifier
        #lrates_j[direction_change_j>0] *= amplifier
        lrates[direction_change<0] *= attenuator
        #lrates_j[direction_change_j<0] *= attenuator

        lrates = np.clip(lrates, dmin, dmax)

        prev_directions = copy.copy(directions)
        # prev_directions_j = copy.copy(directions_j)


        #update = errors * lrates
        update = directions * lrates


        # j_update = j_errors * lrates_j

        # print( 'update')
        # print( update)
        # #print j_update

        # print( 'old weigths')
        # #print join_weights
        # print( str(weights))

        # print 'b'
        #weights += update

        print    'prev weights: ' + ' '.join(['%f'%(val) for val in weights.tolist()])

        weights += update
        weights = np.maximum(weights, weight_floor)

        # print( 'new weigths')
        # #print join_weights
        # print( str(weights))

        previous_score = loss
        # print 'c'


        ### print report:
        print 
        print ' '.join( synth.stream_list_join + synth.stream_list_target )
        report = 'mean contrib: ' + ' '.join(['%f'%(val) for val in mean_scores.tolist()])
        print report

        # print    '         std: ' + ' '.join(['%f'%(val) for val in std_scores.tolist()])
        print    '      errors: ' + ' '.join(['%f'%(val) for val in errors.tolist()])
        print    '    glob mean ' + str(global_mean)
        print    '      update: ' + ' '.join(['%f'%(val) for val in update.tolist()])
        print    '     weights: ' + ' '.join(['%f'%(val) for val in weights.tolist()])

        contribs.append(mean_scores)
    if i==max_epochs-1:
        print 'max epochs reached'

    print 
    print '========='
    print 
    #print best_score
    print 
    print '## Weights found to best balance stream contributions -- you can copy these to config:.'
    print '## Found using X utterances (from x to y)'
    print '##'


    print 'join_stream_weights = %s'%(best_weights.tolist()[:njoin])
    print 'target_stream_weights = %s'%(best_weights.tolist()[njoin:])

    contribs = np.vstack(contribs)
    pylab.subplot(211)
    pylab.plot(losses)
    pylab.subplot(212)
    pylab.plot(contribs)
    pylab.show()

    # for line in hist:
    #     print line