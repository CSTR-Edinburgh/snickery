#!/usr/bin/env python
# -*- coding: utf-8 -*-
## Project: ... - February 2017 - ...
## Contact: Oliver Watts - owatts@staffmail.ed.ac.uk
  
import sys
import os
import glob
import os
import random
import numpy as np

class DataProvider(object):
    '''
    To use, provide method get_file_data_from_one_file and optionally get_number_points_in_one_file
    '''

    def __init__(self, input_dir_list, input_extensions, batch_size=32, partition_size=100000, \
            shuffle=True, testpattern='', validation=False, limit_files=0):

        self.partition_size = partition_size
        self.input_dir_list = input_dir_list
        self.filelist = self.get_filelist(input_dir_list, input_extensions)
        self.input_extensions = input_extensions

        if limit_files > 0:
            self.filelist = self.filelist[:min(limit_files, len(self.filelist))]

        if testpattern:
            train_only_filelist = [name for name in self.filelist if testpattern not in name]
            print 'Remove %s sentences containing pattern %s'%(len(self.filelist) - len(train_only_filelist), testpattern)
            self.filelist = train_only_filelist
            
        valid_len = int(len(self.filelist)*0.05)  ##  95% for training, 5% for validation
        if valid_len == 0:
            valid_len = 1
        train_len = len(self.filelist) - valid_len
        assert train_len > 0
        # print train_len
        # print valid_len
        # sys.exit('wefwsrvwsrbg')

        if validation:
            self.filelist = self.filelist[train_len:]
            self.operation='validation'
        else:
            self.filelist = self.filelist[:train_len]
            self.operation='training'            

        self.list_size = len(self.filelist)
        
        print 'Created DataProvider with %s files for %s' % (len(self.filelist), self.operation)
        
        self.shuffle = shuffle
        if self.shuffle:
            random.seed(271638)
        self.batch_size = batch_size

        #self.points_in_partition = 0

        self.reset()

        self.n_points = self.get_n_examples()
        self.n_batches = self.n_points / self.batch_size

        if self.partition_size > self.n_points:
            print 'reducing partition_size -- too few points!'
            self.partition_size = self.n_points


    def get_filelist(self, input_dir_list, input_extensions):
        filelist = []
        for fname in glob.glob(input_dir_list[0] + '/*.' + input_extensions[0]):
            path,base = os.path.split(fname)
            base = base.replace('.'+input_extensions[0], '')
            utterance_list = [fname]
            for (other_dir, other_ext) in zip(input_dir_list, input_extensions)[1:]:
                other_fname = os.path.join(other_dir, base + '.' + other_ext)
                if os.path.isfile(other_fname):
                    utterance_list.append(other_fname)
            if len(utterance_list) == len(input_dir_list):    ## if we've found a file for each input directory
                filelist.append(tuple(utterance_list))
        return sorted(filelist)


    def reset(self):
        """reset for a new pass through the data (new epoch)"""
        print '\n\n\nreset!\n\n\n'
        self.file_index = 0
        self.points_in_partition = 0
        if self.shuffle:
            random.shuffle(self.filelist)

    def populate_partition(self):
        #print '======populate_partition (%s)'%(self.operation)
        new_partition_data = [[] for i in self.input_dir_list] 

        ## keep any existing partition data:
        if self.points_in_partition > 0:
            for (i, data) in enumerate(self.partition):
                new_partition_data[i].append(data)

        while self.points_in_partition < self.partition_size:
            file_data_list = self.get_file_data_from_one_file()
            self.file_index += 1
            if self.file_index==len(self.filelist):
                self.reset()
            if file_data_list[0].size == 0:
                print 'bad data -- continue'
                continue
            for (i, data) in enumerate(file_data_list):
                new_partition_data[i].append(data)
            self.points_in_partition += file_data_list[0].shape[0]

        self.partition = []
        for datalist in new_partition_data:
            self.partition.append(np.vstack(datalist))

        if self.shuffle:
            shuffle_ixx = np.arange(self.partition[0].shape[0])
            np.random.shuffle(shuffle_ixx)
            for (i,data) in enumerate(self.partition):
                self.partition[i] == data[shuffle_ixx, :]

    def get_next_batch(self):
        #print '====get_next_batch'
        #print '             points %s  |  batch %s'%(self.points_in_partition, self.batch_size)
        if self.points_in_partition < self.batch_size:
            self.populate_partition()

        batch_data = []
        for i in xrange(len(self.partition)):
            batch_data.append(self.partition[i][-self.batch_size:, :])
            self.partition[i] = self.partition[i][:-self.batch_size, :]

        self.points_in_partition -= self.batch_size         

        return tuple(batch_data)

    def batch_generator(self):
        while True:
            yield self.get_next_batch()
 
    def get_n_examples(self):
        self.reset()
        pointcount = 0
        while self.file_index < len(self.filelist):
            pointcount += self.get_number_points_in_one_file()
            self.file_index += 1
        self.reset()   

        # print 'get_n_batches'
        # print pointcount
        # print self.batch_size
        # return pointcount / self.batch_size

        return pointcount

    def get_number_points_in_one_file(self):
        '''
        This will work OK, but can be overridden with something more efficient
        which just needs to work out data size without returning it.
        '''
        data = self.get_file_data_from_one_file()
        return data[0].shape[0]

    def get_file_data_from_one_file(self):
        '''
        Here is most of the database specific stuff.
        This one should be provided by subclasses to manipulate data appropriately.
        '''
        raise NotImplementedError

    def get_filename(self):
        fname = self.filelist[self.file_index][0]
        path,base = os.path.split(fname)
        base = base.replace('.'+self.input_extensions[0], '')
        return base





