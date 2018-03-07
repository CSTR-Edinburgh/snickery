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
# import pylab


if __name__ == '__main__':

    #################################################
      
    # ======== process command line ==========

    a = ArgumentParser()
    a.add_argument('-c', dest='config_fname', required=True)
    opts = a.parse_args()


    synth = Synthesiser(opts.config_fname)
    assert synth.config['join_cost_weight'] == 1.0
    synth.mode_of_operation = 'stream_weight_balancing'
    synth.verbose = False

    ## initially, unweighted streams -- all 1.0:
    njoin = len(synth.stream_list_join)
    ntarget = len(synth.stream_list_target)

    #stream_number_correction = np.array( [0.5/njoin]*njoin  +  [0.5/ntarget]*ntarget )

    weights = np.ones(njoin + ntarget)
    hist = []

    best_weights = copy.copy(weights)
    best_score = previous_score = float('inf')
    epochs_without_improvement = 0
    thresh = 0.001 ## when we are this close to zero, stop 
    patience = 5
    max_epochs=1000
    n_tune = 10 # 100
    n_valid = 10 # 100
    losses = []
    weight_floor = 0.0 

    eta = 0.1

    lrates = np.ones(weights.shape)*eta
    directions = np.zeros(weights.shape)

    prev_directions = np.ones(weights.shape)

    amplifier = 1.2
    attenuator = 0.5

    dmax = 50.0
    dmin = 0.000001

    flist = synth.get_sentence_set('tune')
    if n_tune < len(flist):
        tune_flist = flist[:n_tune]
    else:
        tune_flist = flist
    
    valid_flist = flist[n_tune:n_tune+n_valid]
    flist = tune_flist

    contribs = []
    
    for i in range(max_epochs):

        synth.set_join_weights(weights[:njoin])
        synth.set_target_weights(weights[njoin:])

        if synth.config.get('greedy_search', False):
            synth.get_tree_for_greedy_search()

        ### TODO: check why the following, which should give the same results, do not (also below on validation set):--
        #cache = synth.synth_utts_bulk(flist, synth_type='tune')
        cache = [synth.synth_utt(fname, synth_type='tune') for fname in flist]


        mean_scores = []

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

        if i==0:
            ### Take weighted average, to ensure target and join contribute equally.
            ##  The per-stream subcosts will contribute equally within each stream.            
            goal_join = (mean_scores.sum()/2.0) / njoin
            goal_target = (mean_scores.sum()/2.0) / ntarget
            goals = np.array([goal_join]*njoin    +   [goal_target]*ntarget)

        errors = mean_scores - goals 
        loss = np.abs(errors).sum() 

        print
        print '=== iteration %s | loss %s ==='%(i+1, loss)

        losses.append(loss)
        if loss < previous_score:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1 

        if loss < best_score:
            best_score = loss
            best_weights = copy.copy(weights)

        if epochs_without_improvement == patience:
            print '\n   ----> converged (or diverged and ran out of patience)\n'
            break

        if loss < thresh:
            print '\n   ----> loss approaching 0: stop here\n'
            break

        directions = np.sign(-1.0 * errors)   ### change weights in opposite direction of errors
        
        # change: 1 means same direction, -1 means change direction
        direction_change = directions * prev_directions

        lrates[direction_change>0] *= amplifier
        lrates[direction_change<0] *= attenuator
        lrates = np.clip(lrates, dmin, dmax)

        prev_directions = copy.copy(directions)
        update = directions * lrates

        ### print report:
        print 
        print '     Streams: ' +   ' '.join( [item.ljust(8) for item in synth.stream_list_join + synth.stream_list_target] )
        print    'Prev weights: ' + ' '.join(['%f'%(val) for val in weights.tolist()])

        weights += update
        weights = np.maximum(weights, weight_floor)

        previous_score = loss

        print 'mean contrib: ' + ' '.join(['%f'%(val) for val in mean_scores.tolist()])
        print    '       goals: ' + ' '.join(['%f'%(val) for val in goals.tolist()])        
        print    '      errors: ' + ' '.join(['%f'%(val) for val in errors.tolist()])
        print    '      update: ' + ' '.join(['%f'%(val) for val in update.tolist()])
        print    '     weights: ' + ' '.join(['%f'%(val) for val in weights.tolist()])

        contribs.append(mean_scores)

    if i==max_epochs-1:
        print '\n   ----> max epochs reached: stop here\n'

    print 
    print 
    print '# ============================================'
    print '# validate found weights on %s sentences (%s ... %s)'%(len(valid_flist), valid_flist[0], valid_flist[-1])
    print
    

    synth.set_join_weights(best_weights[:njoin])
    synth.set_target_weights(best_weights[njoin:])
    #cache = synth.synth_utts_bulk(valid_flist, synth_type='tune')
    cache = [synth.synth_utt(fname, synth_type='tune') for fname in flist]
    mean_scores = []
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
    print    '# mean contribution (validation): ' + ' '.join(['%f'%(val) for val in mean_scores.tolist()])


    print 
    print '# ============================================'
    print 
    print 
    print '## Weights found to best balance stream contributions -- you can copy these to config.'
    print '## Weights were found using %s utterances (%s ... %s)'%(len(flist), flist[0], flist[-1])
    print '##'


    print 'join_stream_weights = %s'%(best_weights.tolist()[:njoin])
    print 'target_stream_weights = %s'%(best_weights.tolist()[njoin:])

    contribs = np.vstack(contribs)
    # pylab.subplot(211)
    # pylab.plot(losses)
    # pylab.subplot(212)
    # pylab.plot(contribs)
    # pylab.show()

    # for line in hist:
    #     print line