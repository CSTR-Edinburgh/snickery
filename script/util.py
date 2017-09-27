

import os
import numpy
import numpy as np
from speech_manip import get_speech, put_speech, rate2fftlength, rate2worldapsize


# [http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays]
def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out
    
    
    
    
def destandardise(norm_speech, config):
    ### TEMP -- copied from standardise.pu
    statsdir = os.path.join(config['workdir'], 'stats')
    bap_size = rate2worldapsize(config['sample_rate'])

    w = config['weights']
    weight_vec = [w['mgc']] * (config['mcc_order']+1) + \
                 [w['lf0']]                           + \
                 [w['vuv']]                           + \
                 [w['bap']] * bap_size 
    weight_vec = np.array(weight_vec)    
    
    dim = config['mcc_order'] + 3 + bap_size 
            ## 3 = f0, vuv, energy

    mean_vec = np.loadtxt(os.path.join(statsdir, 'mean.txt'))
    std_vec = np.loadtxt(os.path.join(statsdir, 'std.txt'))        

    vuv_dim = config['mcc_order'] + 2



    m,n = np.shape(norm_speech)
            
    mean_mat = np.tile(mean_vec,(m,1))
    std_mat = np.tile(std_vec,(m,1))
    weight_mat = np.tile(weight_vec,(m,1))

    speech = norm_speech / weight_mat
    
    speech *= std_mat
    speech += mean_mat 
    
    speech[:,vuv_dim] = norm_speech[:,vuv_dim]
    
    return speech
    
    
    
    
    
def safe_makedir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)
        
def writelist(seq, fname):
    f = open(fname, 'w')
    f.write('\n'.join(seq) + '\n')
    f.close()
    
def readlist(fname):
    f = open(fname, 'r')
    data = f.readlines()
    f.close()
    return [line.strip('\n') for line in data]
    
def read_norm_data(fname, stream_names):
    out = {}
    vals = np.loadtxt(fname)
    mean_ix = 0
    for stream in stream_names:
        std_ix = mean_ix + 1
        out[stream] = (vals[mean_ix], vals[std_ix])
        mean_ix += 2
    return out
    

def makedirecs(direcs):
    for direc in direcs:
        if not os.path.isdir(direc):
            os.makedirs(direc)

def split_into_streams(speech, stream_names, datadims):
    
#     if 'vuv' not in datadims:
#         dim += 1     ## add 1 for vuv
#    speech = get_speech(cmpfile, dim)
   

    
    start = 0
    outputs = {}
    for stream in stream_names:
        stream_dim = datadims[stream]
        if stream == 'mgc':
            stream_dim += 1
        end = start + stream_dim
        print stream
        outputs[stream] = speech[:,start:end]
        start = end

    return outputs
        
def world_synth(cmpfile, wavefile, config, denorm=False):

    stream_names = config['stream_names']
    datadims = dict(zip(stream_names, config['datadims_list']))
    
    
    datadims['vuv'] = 1
    speech = get_speech(cmpfile, sum(datadims.values())+1)
    
    #print config
    if denorm:
        
        speech = destandardise(speech, config)
        
            
    streams = split_into_streams(speech, stream_names, datadims)
    #print streams


        
    if 'lf0' in streams:
        fzero = numpy.exp(streams['lf0'])     
        
        vuv_thresh = 0.5
        if 'vuv' in streams:
            vuv = streams['vuv']
            lf0 = streams['lf0']
            fzero[vuv <= vuv_thresh] = 0.0
        
        #fzero *= fzero_scale
        
        streams['lf0'] = fzero
    

          

    streams2wav(streams, wavefile, config)
    

def denorm_data(streams, config):

    stream_names = config['stream_names']
    norm_data = read_norm_data(config['norm_data_file'] , stream_names)
    denorm_data = {}

    weights = config['weights']
        
    for (stream_name, stream_data) in streams.items():
    
        (mean_val, std_val) = norm_data[stream_name]
        
        stream_data /= weights[stream_name]
        
        stream_data *= std_val
        stream_data += mean_val
        
        denorm_data[stream_name] =  stream_data

    return denorm_data
    
    
def streams2wav(streams, outfile, config):

    bin_dir = config['bindir']

    alpha = config['mcc_alpha']
    order = config['mcc_order']
    sr = config['sample_rate']
    fftl = rate2fftlength(sr)
    

    ## TODO -- handle tmp better
    os.system('rm /tmp/tmp*')
        
    for (stream, data) in streams.items():   
        put_speech(data, '/tmp/tmp.%s'%(stream))  
        comm=bin_dir+"/x2x +fd /tmp/tmp."+stream+" >/tmp/tmp_d."+stream
        print comm
        os.system(comm)
    
    comm = "%s/mgc2sp -a %s -g 0 -m %s -l %s -o 2 /tmp/tmp.mgc | %s/sopr -d 32768.0 -P | %s/x2x +fd -o > /tmp/tmp.spec"%(bin_dir, alpha, order, fftl, bin_dir, bin_dir)
    print comm
    os.system(comm)

    '''Avoid:   x2x : error: input data is over the range of type 'double'!
           -o      : clip by minimum and maximum of output data            
             type if input data is over the range of               
             output data type.
    '''    

    comm = "%s/synth %s %s /tmp/tmp_d.lf0 /tmp/tmp.spec /tmp/tmp_d.bap %s"%(bin_dir, fftl, sr, outfile)
    print comm
    res = os.system(comm)
    if res != 0:
        print
        print 'trouble with resynth command:'
        print comm
        print
    else:
#     os.system("mv /tmp/tmp.resyn.wav "+outfile)
        print 'Produced %s'%(outfile)   


def splice_data(data, splice_weights):

    assert len(splice_weights) % 2 == 1, 'no. of weights should be odd'
    middle = (len(splice_weights) - 1) / 2
    assert splice_weights[middle] == 1.0, 'middle weight must be 1!'
    
    #print data
    #print '===='
    offset = len(splice_weights)-1
    stacked = []
    for (i,w) in enumerate(splice_weights):
        if offset == 0:
            #print data[i:, :] #* w  
            stacked.append(data[i:, :])
        else:
            #print data[i:-offset, :] #* w
            stacked.append(data[i:-offset, :])
        #print i
        #print offset
        offset -= 1
    stacked = np.hstack(stacked)
    #print stacked
    return stacked


def unsplice(data, splice_weights):

    assert len(splice_weights) % 2 == 1, 'no. of weights should be odd'
    middle = (len(splice_weights) - 1) / 2
#     print splice_weights
#     print splice_weights[middle]
#     print middle
    m,n = np.shape(data)
    dim = n / len(splice_weights)
    return data[:,(dim*middle):(dim*middle)+dim]


def comm(comm_line, quiet=False):
    if not quiet:
        print comm_line
    os.system(comm_line)    
    
    
def latex_matrix(X, integer=False):
    """
    Print a numpy array to a string that will compile when pasted into latex .
    Arbitary matric name M -- change this after.
    See http://selinap.com/2009/05/how-to-create-a-matrix-in-latex/ on latex matrices
    """
    m,n = X.shape
    align_pattern = "c" * n        
    
    outstring="""\\begin{pmatrix}\n"""

    for i in range(m):
        for j in range(n):
            if integer:
                outstring += str(int(X[i,j]))
            else:        
                outstring += "%.2f"%(X[i,j])  ### 2 decimal places for floats
            if j == (n-1):
                outstring += " \\\\ \n" 
            else:
                outstring += " & "        
    outstring += """\end{pmatrix}  \n"""
    return outstring

    
def vector_to_string(vect):
    '''
    Version of vector suitable for use in filenames. Try to drop values after decimal point if possible
    '''
    string_values = []
    for value in vect:
        if int(value) == value:
            string_values.append(str(int(value)))
        else:
            string_values.append(str(value))

    ## 'run length encode' repeated values to get sensible size string:
    unique_values = []
    counts = []
    prev_val = ''
    assert '' not in string_values
    for val in string_values:
        if val != prev_val:
            unique_values.append(val)
            if prev_val != '':
                counts.append(count)
            count = 1
        else:
            count += 1
        prev_val = val
    counts.append(count)
    assert len(unique_values) == len(counts)
    dedup_string_values = []
    for (value, count) in zip(unique_values, counts):
        dedup_string_values.append('%sx%s'%(count, value))
    dedup_string_values = '-'.join(dedup_string_values)
    return dedup_string_values