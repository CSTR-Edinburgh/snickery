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
import random
# import pylab


from speech_manip import read_wave, write_wave


from segmentaxis import segment_axis

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

from train_halfphone import destandardise




from sklearn.cluster import KMeans
import scipy



from synth_halfphone import Synthesiser



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
    

class Patcher(Synthesiser):

    def __init__(self, config_file, holdout_percent=0.0):
        super(Patcher, self).__init__(config_file, holdout_percent=holdout_percent)



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
        #self.cluster_ixx = kmeans.labels_

        ## find examples nearest centroids:
        # tree = scipy.spatial.cKDTree(data, leafsize=100, compact_nodes=False, balanced_tree=False)
        # db_dists, db_indexes = tree.query(cbook, k=1)
        # db_indexes = db_indexes.flatten()
        # print db_indexes

        self.cbook_tree = scipy.spatial.cKDTree(self.cbook, leafsize=1, compact_nodes=True, balanced_tree=True)
        dists, cix = self.cbook_tree.query(self.train_unit_features)
        self.cluster_ixx = cix.flatten()

    def multi_cluster(self, maxcluster, limit=0):
        if limit == 0:
            data = self.train_unit_features
        else:
            data = random_subset_data(self.train_unit_features, train_frames=limit)
        t = synth.start_clock('multi_cluster')
        ncluster = maxcluster
        cluster_ixx = []
        self.cbooks = []
        self.cbook_trees = []
        while ncluster >=2:
            print ncluster
            kmeans = KMeans(n_clusters=ncluster, init='k-means++', n_init=1, max_iter=300, tol=0.0001, precompute_distances=True, verbose=0, random_state=1234, copy_x=True, n_jobs=1, algorithm='auto')
            kmeans.fit(data)
            self.cbooks.append( kmeans.cluster_centers_)
            cbook_tree = scipy.spatial.cKDTree(kmeans.cluster_centers_, leafsize=1, compact_nodes=True, balanced_tree=True)
            self.cbook_trees.append( cbook_tree )
            dists, cix = cbook_tree.query(self.train_unit_features)
            cluster_ixx.append(cix.reshape(-1,1))
            ncluster /= 2
        synth.stop_clock(t)

        self.cluster_ixx = np.hstack(cluster_ixx)

        



    def index_patches(self, max_patch_length):

        self.patches = {}
        for i in range(max_patch_length):
            length = i+1
            if length == 1:
                for (ix, val) in enumerate(self.cluster_ixx):
                    if (val,) not in self.patches:
                        self.patches[(val,)] = []
                    self.patches[(val,)].append(ix)
            else:
                data = segment_axis(self.cluster_ixx, length, overlap=length-1, axis=0)
                for (ix, vals) in enumerate(data):
                    key = tuple(vals.tolist())
                    if key not in self.patches:
                        self.patches[key] = []
                    self.patches[key].append(ix)
        #print self.patches


    def multi_index_patches(self, patch_length):

        assert patch_length >= 2
        self.patches = []
        self.patch_length = patch_length
        m,n_resolutions = self.cluster_ixx.shape
        for i in xrange(n_resolutions):
            res_patches = {}
            data = segment_axis(self.cluster_ixx[:,i].flatten(), patch_length, overlap=patch_length-1, axis=0)
            for (ix, vals) in enumerate(data):
                key = tuple(vals.tolist())
                if key not in res_patches:
                    res_patches[key] = []
                res_patches[key].append(ix)
            self.patches.append(res_patches)




    def patch_over(self, cb_path):
        i = 0
        db_path = []
        while i < len(cb_path):
            assert (cb_path[i],) in self.patches  ## sanity check
            left = len(cb_path) - i
            for patch_len in reversed(range(left)):
                key = tuple(cb_path[i:i+patch_len])
                if key in self.patches:
                    start = self.patches[key][0] ## take first item
                    db_path.extend(range(start, start+len(key)))
                    # if len(key) == 1:
                    #     db_path.append(start)
                    # else:
                    #     db_path.extend(range(start, start+len(key)))
                    i += patch_len
                    print patch_len
        return db_path




    def multi_patch_over(self, cb_path):
        cb_path = segment_axis(cb_path, self.patch_length, overlap=0, axis=0)
        ## gives: npatch x patchlength + nres

        db_path = []
        for chunk in cb_path:
             
            matched = False
            for (res, res_patch) in enumerate(chunk.transpose()):
                
                key = tuple(res_patch.tolist())
                if key in self.patches[res]:
                    start = self.patches[res][key][0] # take first!
                    end = start + self.patch_length
                    db_path.extend(range(start, end))
                    matched = True
                    print 'res: %s'%(res)
                    break
                
            if not matched:
                sys.exit('need back off strategy!')

        return db_path

    def multi_synth_from_contin(self, odir):

        ## convert new data:
        test_data = self.train_unit_features_dev[:800,:]
        ixx = []
        for cb_tree in self.cbook_trees:
            dists, cb_ixx = cb_tree.query(test_data)
            cb_ixx = cb_ixx.reshape((-1,1))
            ixx.append(cb_ixx)
        ixx = np.hstack(ixx)

        path = self.multi_patch_over(ixx)


        olap = 0 # 2
        # if self.patch_length < 4:
        #     olap = 0



        #db_ixx = db_indexes[cb_ixx]
        synth.concatenateMagPhaseEpoch_sep_files(path, odir + '/output_%s.wav'%(self.patch_length), overlap=olap)


        # recover target F0:
        unit_features_no_weight = synth.train_unit_features_unweighted_dev[:800, :]
        unnorm_speech = destandardise(unit_features_no_weight, synth.mean_vec_target, synth.std_vec_target)   
        target_fz = unnorm_speech[:,-1] ## TODO: do not harcode F0 postion
        target_fz = np.exp(target_fz).reshape((-1,1))
        target_fz[target_fz<90] = 0.0
        synth.concatenateMagPhaseEpoch_sep_files(path, odir + '/output_%s_F0.wav'%(self.patch_length), overlap=olap, fzero=target_fz)





if __name__ == '__main__':

    #################################################
      
    # ======== process command line ==========

    a = ArgumentParser()
    a.add_argument('-c', dest='config_fname', required=True)
    opts = a.parse_args()

    synth = Patcher(opts.config_fname, holdout_percent=5)

    assert synth.config['multiepoch'] == 1

    if 1:
        synth.cluster(32, limit=60000) # 600)
        print 'p'
        synth.index_patches(20)
        print 'done'
        synth.synth_from_contin()

    if 0:
        synth.multi_cluster(1024, limit=10000) # 600)    ## 1024, 10 works OK
        for csize in [2,3,4,5,6,7,8,9,10]:
            print 'p'
            synth.multi_index_patches(csize)
            print 'done'
            synth.multi_synth_from_contin('/tmp/')
