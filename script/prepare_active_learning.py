#!/usr/bin/env python
# -*- coding: utf-8 -*-
## Project: 
## Contact: Oliver Watts - owatts@staffmail.ed.ac.uk
  
import sys
import os
import glob
import os
import fileinput
from argparse import ArgumentParser

import shelve
import h5py
import numpy
import numpy as np

from train_halfphone import get_data_dump_name
from synth_halfphone import Synthesiser

# import pylab



from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.callbacks import EarlyStopping

def train_autoencoder(data):

    m,n = data.shape

    model = Sequential()

    model.add(Dense(units=1000, input_dim=n))
    model.add(Activation('relu'))
    model.add(Dense(units=1000))
    model.add(Activation('relu'))

    model.add(Dense(units=60))
#    model.add(Activation('relu'))

    model.add(Dense(units=1000))
    model.add(Activation('relu'))
    model.add(Dense(units=n))

    model.compile(loss='mean_squared_error', optimizer='adam')

    earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')

    model.fit(data, data, epochs=30, batch_size=64, callbacks=[earlyStopping], validation_split=0.10, shuffle=True)

    return model


def get_prediction_errors(data, autoencoder):
    predictions = autoencoder.predict(data)
    sq_errs = (predictions - data)**2
    scores = sq_errs.mean(axis=1)    
    return scores

def main_work(config, synth):
    print 
    print get_data_dump_name(config)
    join_data_dump = get_data_dump_name(config, joindata=True)
    


    f = h5py.File(join_data_dump, "r")
    start_join_feats = f["start_join_feats"][:,:]
    end_join_feats = f["end_join_feats"][:,:]    
    f.close()

    '''
    print synth.join_weight_vector.shape

    s = start_join_feats[100].reshape((5,87))
    e = end_join_feats[99].reshape((5,87))
    es = np.vstack([e,s])
    print e.shape

    pylab.plot(es)
    pylab.show()
    sys.exit('wewev')
    '''


    
    


    #print synth.unit_index

    ## Work out the minimal linguistic cues which should hold to consider a join:-
    # lc_inventory = {}
    # for ll_l_c_r_rr in synth.train_unit_names:
    #     (ll,l,c,r,rr) = ll_l_c_r_rr.split('/')
    #     lc_inventory[(l,c.split('_')[0])] = 0
    # lc_inventory = lc_inventory.keys()

    ### build index of units by l phone only (c phones are already handled by synth.unit_index)
    print 'build l index...'
    l_index = {}
    seen = {}
    for ll_l_c_r_rr in synth.train_unit_names:
        (ll,l,c,r,rr) = ll_l_c_r_rr.split('/')
        if c.endswith('_L'):
            #assert '%s/%s'%(l,c) not in l_index
            if '%s/%s'%(l,c) not in seen:
                seen['%s/%s'%(l,c)] = 0
                if l not in l_index:
                    l_index[l] = []
                l_index[l].extend(synth.unit_index['%s/%s'%(l,c)])
    print '...done'

    lengths = [(len(v), k) for (k,v) in l_index.items()]
    # for k,v in l_index.items():
    #     print k
    #     print len(v)
    #     print 
    lengths.sort()
    print lengths[-10:]
    # sys.exit('evwevwsrb222')


    #transitions = [] ## list of lists
    transitions = shelve.open('/tmp/transitions2')

    nlinks = 0
    for (i,name) in enumerate(synth.train_unit_names):
        if i % 100 == 0:
            print '%s of %s'%(i, synth.number_of_units)
        (ll,l,c,r,rr) = name.split('/')
        if c.endswith('_L'):
            next_unit = c.replace('_L','_R')
            #transitions.append(synth.unit_index[next_unit])
            transitions_from_here = synth.unit_index[next_unit]
            
        else:
            ## find all following units compatible with it:
            this_current = c.split('_')[0]
            this_right = r
            transitions_from_here = l_index.get(this_current, [])  +   synth.unit_index.get(this_right + '_L', []) 


            # for (l,c) in lc_inventory:
            #     if this_current == l or this_right == c:
            #         next_unit = '%s/%s_L'%(l,c)
            #         #print '%s %s   %s'%(this_current, this_right, next_unit)
            #         transitions_from_here.extend(synth.unit_index[next_unit])


            #transitions.append(transitions_from_here)
        



        transitions[str(i)] = numpy.array(transitions_from_here, dtype=int)
        nlinks += len(transitions_from_here)

    #print transitions
    print transitions['101']
    #nlinks = sum([len(sublist) for sublist in transitions])
    print nlinks
    print synth.number_of_units
    sys.exit('dvwsv0000000')

    negative_scores = []
    for i in xrange(synth.number_of_units):
        if i % 100 == 0:
            print i
        to_index = transitions[str(i)] 

        ## Remove natural transitions:-
        to_index = to_index[to_index!=i+1]

        from_ixx = np.ones(to_index.shape, dtype=int) * i
        features = np.hstack([end_join_feats[from_ixx,:], start_join_feats[to_index,:]])
        negative_scores.append(get_prediction_errors(features, autoencoder))

    negative_scores = np.concatenate(negative_scores)

    #pylab.subplot(211)
    pylab.hist(positive_scores, bins=30, alpha=0.6, color='g', normed=1)

    #pylab.subplot(212)
    pylab.hist(negative_scores, bins=30, alpha=0.6, color='b', normed=1)

    pylab.show()


    transitions.close()


    ### 23 blizzard utts:
    # 53909
    # 1408

    ### all blizzard utts:




    # for i in xrange(synth.number_of_units-1):
    #     j = i + 1
    #     distance = synth.get_natural_distance(i,j,order=2)
    #     print distance

    
if __name__=="__main__":


    #################################################
      
    # ======== process command line ==========

    a = ArgumentParser()
    a.add_argument('-c', dest='config_fname', required=True)
    opts = a.parse_args()
    
    config = {}
    execfile(opts.config_fname, config)
    del config['__builtins__']
    print config

    from active_learning_join import JoinDatabaseForActiveLearning
    o = JoinDatabaseForActiveLearning(opts.config_fname)


    # sys.exit('wvwrvwrv----------')

    # synth = Synthesiser(opts.config_fname)

    # main_work(config, synth)
    
    

