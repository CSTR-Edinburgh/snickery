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
        assert not (self.config.get('greedy_search', False)  and  self.config['target_representation'] != 'epoch')


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

        if self.config['target_representation'] == 'epoch':
            forbid_repetition = self.config.get('forbid_repetition', False)
            forbid_regression = self.config.get('forbid_regression', 0)
        else:
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


    def oracle_synthesis_holdout(self, outfname, start, length):
        t = self.start_clock('oracle_synthesis_holdout')

        assert start >= 0
        assert start + length < self.holdout_samples

        assert self.config['store_full_magphase_sep_files']

        magphase_overlap = self.config.get('magphase_overlap', 0)

        unit_features = self.train_unit_features_dev[start:start+length, :]
        
                
        # recover target F0:
        unit_features_no_weight = self.train_unit_features_unweighted_dev[start:start+length, :]
        unnorm_speech = destandardise(unit_features_no_weight, self.mean_vec_target, self.std_vec_target)   
        target_fz = unnorm_speech[:,-1] ## TODO: do not harcode F0 postion
        target_fz = np.exp(target_fz).reshape((-1,1))
        ### TODO: nUV is : 88.62008057.      This breaks resynthesis for some reason...
        
        target_fz[target_fz<90] = 0.0
        #target_fz *= 20.0
        #print target_fz

        #target_fz = np.ones((unit_features.shape[0], 1)) * 50 # 88.62  ## monotone 300 hz

        best_path = self.greedy_joint_search(unit_features)

        if self.config.get('magphase_use_target_f0', True):
            self.concatenateMagPhaseEpoch_sep_files(best_path, outfname, fzero=target_fz, overlap=magphase_overlap)                
        else:
            self.concatenateMagPhaseEpoch_sep_files(best_path, outfname, overlap=magphase_overlap) 

        self.stop_clock(t)     

        # print 'path info:'
        # print self.train_unit_names[best_path].tolist()





    def natural_synthesis_holdout(self, outfname, start, length):
        if 0:
            print outfname
            print start
            print length
            print 
        t = self.start_clock('natural_synthesis_holdout')
        assert start >= 0
        assert start + length < self.holdout_samples
        assert self.config['store_full_magphase_sep_files']
        magphase_overlap = 0
        multiepoch = self.config.get('multiepoch', 1)
        natural_path = np.arange(start, start+length, multiepoch) + self.number_of_units ## to get back to pre-hold-out indexing
        self.concatenateMagPhaseEpoch_sep_files(natural_path, outfname, overlap=0)                
        self.stop_clock(t)     

   
    def get_heldout_frag_starts(self, sample_pool_size, frag_length, filter_silence=''):
        n_frag_frames = sample_pool_size * frag_length
        assert n_frag_frames <= self.holdout_samples, 'not enough held out data to generate frags, try incresing holdout_percent or decreasing sample_pool_size'
        
        if filter_silence:
            sys.exit('Still to implement filter_silence')
            frags = segment_axis(self.train_unit_names_dev[:n_frag_frames], frag_length, overlap=0, axis=0)
            pause_sums = (frags==filter_silence) # , dtype=int).sum(axis=1)
            percent_silent = pause_sums / frag_length
            print percent_silent

        starts = np.arange(0, n_frag_frames, frag_length)
        selected_starts = np.random.choice(starts, sample_pool_size, replace=False)
        return selected_starts



    def concatenate(self, path, fname):

        if self.config['target_representation'] == 'epoch':
            NEW_METHOD = True
            if NEW_METHOD:
                self.concatenate_epochs_new(path, fname)
            else:
                self.concatenate_epochs(path, fname)
        else:
            frags = []
            for unit_index in path:
                frags.append(self.retrieve_speech(unit_index))

            if self.config['taper_length'] == 0:
                synth_wave = np.concatenate(frags)
            else:
                synth_wave = self.overlap_add(frags)
            write_wave(synth_wave, fname, 48000, quiet=True)


    def concatenate_epochs(self, path, fname):

        frags = []
        for unit_index in path:
            frags.append(self.retrieve_speech_epoch(unit_index))

        synth_wave = self.epoch_overlap_add(frags)
        write_wave(synth_wave, fname, 48000, quiet=True)



    def concatenate_epochs_new(self, path, fname):
        # print '===== NEW METHOD: concatenate_epochs_new ======='
        frags = []
        for unit_index in path:
            frags.append(self.retrieve_speech_epoch_new(unit_index))
        synth_wave = self.epoch_overlap_add_new(frags)
        write_wave(synth_wave, fname, 48000, quiet=True)



    # def make_epoch_labels(self, path, fname):
    #     start_points = []
    #     start = 0
    #     for (i,unit_index) in enumerate(path):
    #         (start,middle,end) = self.train_cutpoints[unit_index]
    #         left_length = middle - start
    #         right_length = end - middle 

    #         start += 
    #         start_points.append(start)

    #     frag = wave[start:end]

    #     ### scale with non-symmetric hanning:
    #     win = np.concatenate([np.hanning(left_length*2)[:left_length], np.hanning(right_length*2)[right_length:]])
    #     frag *= win



    #     return (frag, left_length)




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


    def epoch_overlap_add(self, frags):
        
        length = sum([halflength for (wave, halflength) in frags[:-1]])
        lastwave, _ = frags[-1]
        length += len(lastwave) 
        wave = np.zeros(length)
        start = 0
        for (frag, halflength) in frags:
            wave[start:start+len(frag)] += frag
            start += halflength
        return wave 



    def epoch_overlap_add_new(self, frags):
        taper = self.config['taper_length']
        length = sum([len(frag)-taper for frag in frags])
        length += taper
        wave = np.zeros(length)
        start = 0
        for frag in frags:
            wave[start:start+len(frag)] += frag
            start += len(frag)-taper
        return wave 


    def concatenateMagPhase(self,path,fname):

        fs     = 48000 # in Hz
        nfft   = 4096

        pm_reaper_dir = self.config['pm_datadir']
        wav_dir = self.config['wav_datadir']

        # Initializing fragments
        frags = {}
        frags['srcfile'] = []
        frags['src_strt_sec'] = []
        frags['src_end_sec'] = []
        for index in path:
            (start,end) = self.train_cutpoints[index]
            frags['srcfile'].append(self.train_filenames[index])
            frags['src_strt_sec'].append(start / float(fs))
            frags['src_end_sec'].append(end / float(fs))

        synth_wave = lwg.wavgen_improved_just_slope(frags, wav_dir, pm_reaper_dir, nfft, fs, npm_margin=3, diff_mf_tres=25, f0_trans_nfrms_btwn_voi=8)
        la.write_audio_file(fname, synth_wave, fs, norm=True)





    def retrieve_speech(self, index):

        if self.train_filenames[index] in self.waveforms:
            wave = self.waveforms[self.train_filenames[index]]  
        else:     
            wavefile = os.path.join(self.config['wav_datadir'], self.train_filenames[index] + '.wav')
            wave, sample_rate = read_wave(wavefile)
        T = len(wave)        
        (start,end) = self.train_cutpoints[index]
        end += 1 ## non-inclusive end of slice
        
        taper = self.config['taper_length']
        
        # Overlap happens at the pitch mark + taper/2 (extend segment by a taper in the end)
        # if taper > 0:
        #     end = end + taper
        #     if end > T:
        #         pad = np.zeros(end - T)
        #         wave = np.concatenate([wave, pad])

        # Overlap happens at the pitch mark (extend segment by half taper in each end)
        if taper > 0:
            end = end + taper/2
            if end > T:
                pad = np.zeros(end - T)
                wave = np.concatenate([wave, pad])
            start = start - taper/2
            if start < 0:
                pad   = np.zeros(-start)
                wave  = np.concatenate([pad, wave])
                start = 0
                
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



    def retrieve_speech_epoch(self, index):

        if self.config['hold_waves_in_memory']:
            wave = self.waveforms[self.train_filenames[index]]  
        else:     
            wavefile = os.path.join(self.config['wav_datadir'], self.train_filenames[index] + '.wav')
            wave, sample_rate = read_wave(wavefile)
        T = len(wave)        
        (start,middle,end) = self.train_cutpoints[index]
        end += 1 ## non-inclusive end of slice

        left_length = middle - start
        right_length = end - middle 

        frag = wave[start:end]

        ### scale with non-symmetric hanning:
        win = np.concatenate([np.hanning(left_length*2)[:left_length], np.hanning(right_length*2)[right_length:]])
        frag *= win



        return (frag, left_length)
 


    def retrieve_speech_epoch_new(self, index):

        ## TODO: see copy.copy below --- make sure copy with other configureations, otherwise 
                                            ## in the case hold_waves_in_memory we disturb original audio which is reused -- TODO -- use this elsewhere too

        if self.train_filenames[index] in self.waveforms:
            orig_wave = self.waveforms[self.train_filenames[index]]  
        else:     
            wavefile = os.path.join(self.config['wav_datadir'], self.train_filenames[index] + '.wav')
            print wavefile
            orig_wave, sample_rate = read_wave(wavefile)
            self.waveforms[self.train_filenames[index]]  = orig_wave
        T = len(orig_wave)        
        (start,middle,end) = self.train_cutpoints[index]


        multiepoch = self.config.get('multiepoch', 1)
        if multiepoch > 1:




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
            for form in [mono]: # [quinphone, triphone, diphone, mono]:
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





            (start_ii,middle,end_ii) = self.train_cutpoints[index + (multiepoch-1)]


        end = middle  ## just use first half of fragment (= 1 epoch)





        wave = copy.copy(orig_wave)              

        taper = self.config['taper_length']

        # Overlap happens at the pitch mark (extend segment by half taper in each end)
        if taper > 0:
            end = end + taper/2
            if end > T:
                pad = np.zeros(end - T)
                wave = np.concatenate([wave, pad])
            start = start - taper/2
            if start < 0:
                pad   = np.zeros(-start)
                wave  = np.concatenate([pad, wave])
                start = 0
                
        frag = wave[start:end]
        if taper > 0:
            hann = np.hanning(taper*2)
            open_taper = hann[:taper]
            close_taper = hann[taper:]
            frag[:taper] *= open_taper
            frag[-taper:] *= close_taper


        return frag





    def get_target_scores_per_stream(self, target_features, best_path):
        chosen_features = self.train_unit_features[best_path]
        dists = np.sqrt(np.sum(((chosen_features - target_features)**2), axis=1))
        sq_errs = (chosen_features - target_features)**2
        stream_errors_target = self.aggregate_squared_errors_by_stream(sq_errs, 'target')
        return stream_errors_target

    def get_join_scores_per_stream(self, best_path):
        if self.config.get('greedy_search', False):
            best_path = np.array(best_path)
            sq_diffs_join = (self.prev_join_rep[best_path[1:],:] - self.current_join_rep[best_path[:-1],:])**2
            #sq_diffs_join = (self.current_join_rep[best_path[:-1],:] - self.current_join_rep[best_path[1:],:])**2
            stream_errors_join = self.aggregate_squared_errors_by_stream(sq_diffs_join, 'join')
            #print stream_errors_join
        else:
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




    def preselect_units_quinphone(self, unit_features, unit_names):
        '''
        NB: where candidates are too few, returned matrices are padded with
        -1 entries 
        '''
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
                current_candidates = [1] # [self.first_silent_unit]
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
   
        return (candidates, distances)




    def preselect_units_acoustic(self, unit_features):


        start_time = self.start_clock('Acoustic select units ')
        ## call has same syntax for sklearn and scipy KDTrees:--
        distances, candidates = self.tree.query(unit_features, k=self.config['n_candidates'])
        self.stop_clock(start_time) 
        return (candidates, distances)


    def preselect_units_monophone_then_acoustic(self, unit_features, unit_names):
        '''
        NB: where candidates are too few, returned matrices are padded with
        -1 entries 
        '''
        start_time = self.start_clock('Preselect units ')
    
        m,n = unit_features.shape
        candidates = np.ones((m, self.config['n_candidates']), dtype=int) * -1
        distances = np.ones((m, self.config['n_candidates'])) * const.VERY_BIG_WEIGHT_VALUE

        monophones = np.array([quinphone.split(const.label_delimiter)[2] for quinphone in unit_names])
        assert len(monophones) == m, (len(monophones), m)
        for (i,phone) in enumerate(monophones):
            assert phone in self.phonetrees, 'unseen monophone %s'%(phone)
            current_distances, current_candidates = self.phonetrees[phone].query(unit_features[i,:], k=self.config['n_candidates'])
            mapped_candidates = self.phonetrees_index_converters[phone][current_candidates]
            candidates[i,:current_distances.size] = mapped_candidates
            distances[i,:current_distances.size] = current_distances

            # current_distances = current_distances.flatten()
            # current_candidates = current_candidates.flatten()
            # if len(current_candidates) != self.config['n_candidates']:
            #     difference = self.config['n_candidates'] - len(current_candidates) 
            #     current_candidates = np.concatenate([ current_candidates , np.ones(difference)*-1.0])
            #     current_distances = np.concatenate([ current_distances , np.zeros(difference)])

        return (candidates, distances)


    def viterbi_search(self, candidates, distances):

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
        
        if 0:
            T.draw('/tmp/T.dot')
            J.draw('/tmp/J.dot')
            sys.exit('stop here 9893487t3')

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
        return best_path