import sklearn.neighbors
from sklearn.neighbors import DistanceMetric
import numpy
import numpy as np
import h5py

class StashableKDTree(sklearn.neighbors.KDTree):
    '''
    Pickling slows and breaks with large trees -- use another mechanism (HDF5) for persistance
    '''
    '''
    KD tree state:
    
    data_arr = state[0]
    idx_array_arr = state[1]
    node_data_arr = state[2]
    node_bounds_arr = state[3]
    leaf_size = state[4]
    n_levels = state[5]
    n_nodes = state[6]
    n_trims = state[7]
    n_leaves = state[8]
    n_splits = state[9]
    n_calls = state[10]
    dist_metric = state[11]
    
    example sizes and types:
    
    <type 'numpy.ndarray'>          (687810, 20)
    <type 'numpy.ndarray'>          (687810,)
    <type 'numpy.ndarray'>          (1048575,)
    <type 'numpy.ndarray'>          (2, 1048575, 20)
    <type 'int'>
    <type 'int'>
    <type 'int'>
    <type 'int'>
    <type 'int'>
    <type 'int'>
    <type 'int'>
    <type 'sklearn.neighbors.dist_metrics.EuclideanDistance'>
    
    '''    
    def save_hdf(self, fname):
        state = self.__getstate__()
        self.check_state_compatibility(state)
        f = h5py.File(fname, "w")
        for (i,array_data) in enumerate(state[:4]):
            dset = f.create_dataset("state_%s"%(i), array_data.shape, dtype=array_data.dtype)
            dset[:] = array_data[:]
        int_array = np.asarray(state[4:11], dtype=int)
        dset = f.create_dataset("int_values", int_array.shape, dtype=int_array.dtype)
        dset[:] = int_array[:]
        f.close()
            

    def load_hdf(self, fname, copydata=None, noisy=False):
        if noisy:
            print 'start load_hdf'
        f = h5py.File(fname, "r")
        state = []
        for i in range(4):       
            array_data = f["state_%s"%(i)][:]
            state.append(array_data)
            if noisy:
                print "load state_%s"%(i)
                print array_data
        if noisy:
            print 'done'
        int_array = f["int_values"][:]
        for val in int_array:
            state.append(int(val)) ## ensure type int not 'numpy.int64' 
        euc_dist = DistanceMetric.get_metric('euclidean')
        state.append(euc_dist)
        state = tuple(state)
        self.check_state_compatibility(state)
        self.__setstate__(state)
        if noisy:
            print 'end load_hdf'
        
    def check_state_compatibility(self, state):
        '''
        Run some compatibility checks on the model's data
        '''
        assert len(state) == 12
        state_types = [type(thing) for thing in state]
        assert state_types[-1] == sklearn.neighbors.dist_metrics.EuclideanDistance, 'only EuclideanDistance supported for StashableKDTree'
        ## this is how things should look, else we are e.g. using incompatible version
        ## of SKLearn: 
        reference_state_types = [numpy.ndarray, numpy.ndarray, \
             numpy.ndarray, numpy.ndarray, int, int, int, int, int, \
             int, int, sklearn.neighbors.dist_metrics.EuclideanDistance]
        
        for (x,y) in zip( state_types, reference_state_types):
            assert x==y, 'mismatched types: %s and %s'%(x,y)
        
        
## helper function -- before restoring state, tree is __init__'d using dummy data
def resurrect_tree(fname):
    dummy_train = np.ones((5,5))
    tree = StashableKDTree(dummy_train, leaf_size=1, metric='euclidean')
    tree.load_hdf(fname)
    return tree
    
    