'''
Functions for standardising and composing streams of speech features, and locating their directories
'''

def compose_speech(feat_dir_dict, base, stream_list, datadims, ignore_streams=['triphone']): 
    '''
    where there is trouble, signal this by returning a 1 x 1 matrix
    '''

    stream_list = [stream for stream in stream_list if stream not in ignore_streams]
    # mgc_fn = os.path.join(indir, 'mgc', base+'.mgc' ) 
    # f0_fn = os.path.join(indir, 'f0', base+'.f0' ) 
    # ap_fn = os.path.join(indir, 'ap', base+'.ap' ) 

    stream_data_list = []
    for stream in stream_list:
        stream_fname = os.path.join(feat_dir_dict[stream], base+'.'+stream ) 
        if not os.path.isfile(stream_fname):
            print stream_fname + ' does not exist'
            return np.zeros((1,1))
        stream_data = get_speech(stream_fname, datadims[stream])
        if stream == 'aef':
            stream_data = np.vstack([np.zeros((1,datadims[stream])), stream_data, np.zeros((1,datadims[stream]))])
        ### previously:        
        # if stream in vuv_stream_names:
        #     uv_ix = np.arange(stream_data.shape[0])[stream_data[:,0]<=0.0]
        #     vuv = np.ones(stream_data.shape)
        #     vuv[uv_ix, :] = 0.0
        #     ## set F0 to utterance's voiced frame mean in unvoiced frames:   
        #     voiced = stream_data[stream_data>0.0]
        #     if voiced.size==0:
        #         voiced_mean = 100.0 ### TODO: fix artibrary nnumber!
        #     else:
        #         voiced_mean = voiced.mean()
        #     stream_data[stream_data<=0.0] = voiced_mean 
        #     stream_data_list.append(stream_data)
        #     stream_data_list.append(vuv)

        ### Now, just set unvoiced frames to -1.0 (they will be specially weighted later):
        if stream in vuv_stream_names:
            # uv_ix = np.arange(stream_data.shape[0])[stream_data[:,0]<=0.0]
            # vuv = np.ones(stream_data.shape)
            # vuv[uv_ix, :] = 0.0
            ## set F0 to utterance's voiced frame mean in unvoiced frames:   
            # voiced = stream_data[stream_data>0.0]
            # if voiced.size==0:
            #     voiced_mean = 100.0 ### TODO: fix artibrary nnumber!
            # else:
            #     voiced_mean = voiced.mean()
            stream_data[stream_data<=0.0] = const.special_uv_value
            stream_data_list.append(stream_data)
            # stream_data_list.append(vuv)
        else:
            stream_data_list.append(stream_data)

    ## where data has different number of frames per stream, chop off the extra frames:
    frames = [np.shape(data)[0] for data in stream_data_list]
    nframe = min(frames)
    stream_data_list = [data[:nframe,:] for data in stream_data_list]
    
    speech = np.hstack(stream_data_list)

    return speech

def locate_stream_directories(directories, streams): 
    '''
    For each stream in streams, find a subdirectory for some directory in 
    directories, directory/stream. Make sure that there is only 1 such subdirectory
    named after the stream. Return dict mapping from stream names to directory locations. 
    '''
   
    stream_directories = {}
    for stream in streams:
        for directory in directories:
            candidate_dir = os.path.join(directory, stream)
            if os.path.isdir(candidate_dir):
                ## check unique:
                if stream in stream_directories:
                    sys.exit('Found at least 2 directories for stream %s: %s and %s'%(stream, stream_directories[stream], candidate_dir))
                stream_directories[stream] = candidate_dir
    ## check we found a location for each stream:
    for stream in streams:
        if stream not in stream_directories:
            sys.exit('No subdirectory found under %s for stream %s'%(','.join(directories), stream))
    return stream_directories



def get_mean(flist, dim, exclude_uv=False):
    '''
    Take mean over each coeff, to centre their trajectories around zero.
    '''
    frame_sum = np.zeros(dim)
    frame_count = 0
    for fname in flist:
        if not os.path.isfile(fname):
            continue    
        print 'mean: ' + fname
        
        speech = get_speech(fname, dim)
        if np.sum(np.isnan(speech)) + np.sum(np.isinf(speech)) > 0:
            print 'EXCLUDE ' + fname
            continue
        
        if exclude_uv:
            ## remove speech where first column is <= 0.0
            speech = speech[speech[:,0]>0.0, :]
        
        frame_sum += speech.sum(axis=0)
        m,n = np.shape(speech)
        frame_count += m



    mean_vec = frame_sum / float(frame_count)
    return mean_vec, frame_count
    
def get_std(flist, dim, mean_vec, exclude_uv=False):
    '''
    Unlike mean, use single std value over all coeffs in stream, to preserve relative differences in range of coeffs within a stream
    The value we use is the largest std across the coeffs, which means that this stream when normalised
    will have std of 1.0, and other streams decreasing. 
    Reduplicate this single value to vector the width of the stream.
    '''
    diff_sum = np.zeros(dim)
    frame_count = 0    
    for fname in flist:
        if not os.path.isfile(fname):
            continue
        print 'std: ' + fname
        
        speech = get_speech(fname, dim)
        if np.sum(np.isnan(speech)) + np.sum(np.isinf(speech)) > 0:
            print 'EXCLUDE ' + fname
            continue

        if exclude_uv:
            ## remove speech where first column is <= 0.0
            speech = speech[speech[:,0]>0.0, :]
                                
        m,n = np.shape(speech)
        #mean_mat = np.tile(mean_vec,(m,1))
        mean_vec = mean_vec.reshape((1,-1))
        sq_diffs = (speech - mean_vec) ** 2
        diff_sum += sq_diffs.sum(axis=0)
        frame_count += m

    max_diff_sum = diff_sum.max()
    print mean_vec.tolist()
    print max_diff_sum.tolist()
    std_val = (max_diff_sum / float(frame_count)) ** 0.5
    std_vec = np.ones((1,dim)) * std_val
    return std_vec
    
def standardise(speech, mean_vec, std_vec):

    m,n = np.shape(speech)
        
    ### record where unvoiced values are with Boolean array, so we can revert them later:
    uv_positions = (speech==const.special_uv_value)

    mean_vec = mean_vec.reshape((1,-1))
    
    ## standardise:-
    speech = (speech - mean_vec) / std_vec
    
    uv_values = std_vec * -1.0 * const.uv_scaling_factor

    for column in range(n):
        # print speech[:,column].shape
        # print uv_positions[:,column].shape
        # print speech[:,column]
        # print uv_positions[:,column]
        # print column
        #if True in uv_positions[:,column]:
        speech[:,column][uv_positions[:,column]] = uv_values[0, column]

    ## leave weighting till later!
    return speech

def destandardise(speech, mean_vec, std_vec):

    m,n = np.shape(speech)
        
    mean_vec = mean_vec.reshape((1,-1))
    #std_mat = np.tile(std_vec,(m,1))
    #weight_mat = np.tile(weight_vec,(m,1))
    
    ## standardise:-
    speech = (speech * std_vec) + mean_vec
    
    ## leave weighting till later!
    # speech = speech * weight_mat
    return speech
    


def get_mean_std(feat_dir_dict, stream_list, datadims, flist):

    means = {}
    stds = {}
    for stream in stream_list:
        stream_files = [os.path.join(feat_dir_dict[stream], base+'.'+stream) for base in flist]
        if stream in vuv_stream_names:
            means[stream], _ = get_mean(stream_files, datadims[stream], exclude_uv=True)
            stds[stream] = get_std(stream_files, datadims[stream], means[stream], exclude_uv=True)
        else:
            means[stream], nframe = get_mean(stream_files, datadims[stream])
            stds[stream] = get_std(stream_files, datadims[stream], means[stream])


    mean_vec = []
    for stream in stream_list:
        mean_vec.append(means[stream])
        # if stream in vuv_stream_names: ## add fake stats for VUV which will leave values unaffected
        #     mean_vec.append(numpy.zeros(means[stream].shape))

    std_vec = []
    for stream in stream_list:
        std_vec.append(stds[stream])
        # if stream in vuv_stream_names: ## add fake stats for VUV which will leave values unaffected
        #     std_vec.append(numpy.ones(stds[stream].shape))

    mean_vec = np.hstack(mean_vec)
    std_vec = np.hstack(std_vec)

    return mean_vec, std_vec    


