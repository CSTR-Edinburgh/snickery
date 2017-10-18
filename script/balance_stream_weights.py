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
    join_weights = np.ones(len(synth.stream_list_join))
    target_weights = np.ones(len(synth.stream_list_target))


    ### first try -- take max each interation -- unstable
    # tmax = []
    # for i in range(5):
    #     synth.set_join_weights(join_weights)
    #     synth.set_target_weights(target_weights)
    #     (mean_tscores, mean_jscores) = synth.synth_from_config()

    #     t_errors = mean_tscores.max() - mean_tscores
    #     j_errors = mean_jscores.max() - mean_jscores

    #     tmax.append(np.argmax(t_errors))


    #     print t_errors
    #     print j_errors

    #     t_update = t_errors * eta
    #     j_update = j_errors * eta

    #     print t_update
    #     print j_update

    #     join_weights += j_update
    #     target_weights += t_update

    #     print join_weights
    #     print target_weights

    # print tmax



    hist = []



    best_weights = [copy.copy(join_weights), copy.copy(target_weights)]
    best_score = previous_score = float('inf')
    epochs_without_improvement = 0
    thresh = 0.001 ## when we are this close to zero, stop 
    patience = 5
    max_epochs=100
    losses = []



    eta = 0.001

    lrates_j, lrates_t = (np.ones(join_weights.shape)*eta, np.ones(target_weights.shape)*eta)
    directions_j, directions_t = (np.zeros(join_weights.shape), np.zeros(target_weights.shape))

    prev_directions_t = np.ones(target_weights.shape)
    prev_directions_j =  np.ones(join_weights.shape)


    amplifier = 1.1
    attenuator = 0.9


    for i in range(max_epochs):
        # print 'aaa'
        synth.set_join_weights(join_weights)
        synth.set_target_weights(target_weights)
        (mean_tscores, mean_jscores) = synth.synth_from_config()

        if i==0:  ## work out max on first iteration and keep same as out target
            t_goal = np.argmax(mean_tscores)
            j_goal = np.argmax(mean_jscores)

        print 'target goal : %s'%(synth.stream_list_target[t_goal])
        print 'join goal : %s'%(synth.stream_list_join[j_goal])


        t_errors = mean_tscores[t_goal] - mean_tscores
        j_errors = mean_jscores[j_goal] - mean_jscores

        loss = t_errors.sum() + j_errors.sum()

        print '=======LOSS========='
        print loss
        print epochs_without_improvement
        losses.append(loss)
        if loss < previous_score:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1 

        if loss < best_score:
            best_score = loss
            best_weights = [copy.copy(join_weights), copy.copy(target_weights)]

        if epochs_without_improvement == patience:
            print 'converged'
            break

        if loss < thresh:
            print 'loss approaching 0 -- stop here'
            break

        # print 'a'
        print('means')
        print(mean_tscores)
        
        print('error')
        print( t_errors)
        #print j_errors

        directions_t = np.sign(t_errors)
        directions_j = np.sign(j_errors)

        # change: 1 means same direction, -1 measns change direction
        direction_change_t = directions_t * prev_directions_t
        direction_change_j = directions_j * prev_directions_j

        lrates_t[direction_change_t>0] *= amplifier
        lrates_j[direction_change_j>0] *= amplifier
        lrates_t[direction_change_t<0] *= attenuator
        lrates_j[direction_change_j<0] *= attenuator

        prev_directions_t = copy.copy(directions_t)
        prev_directions_j = copy.copy(directions_j)


        t_update = t_errors * lrates_t
        j_update = j_errors * lrates_j

        print( 'update')
        print( t_update)
        #print j_update

        print( 'old weigths')
        #print join_weights
        print( str(target_weights))

        # print 'b'
        join_weights += j_update
        target_weights += t_update

        print( 'new weigths')
        #print join_weights
        print( str(target_weights))

        previous_score = loss
        # print 'c'
    if i==max_epochs-1:
        print 'max epochs reached'

    print 
    print '========='
    print 
    #print best_score
    print 
    print '## Weights found to best balance stream contributions -- you can copy these to config:'
    print '##'


    print 'join_stream_weights = %s'%(best_weights[0].tolist())
    print 'target_stream_weights = %s'%(best_weights[1].tolist())

    pylab.plot(losses)
    pylab.show()

    # for line in hist:
    #     print line