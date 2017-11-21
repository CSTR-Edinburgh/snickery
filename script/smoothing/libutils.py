# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 23:16:51 2016

@author: felipe
"""

#==============================================================================
# IMPORTS
#==============================================================================
import numpy as np
import os
from binary_io  import BinaryIOCollection
#import glob
import warnings

_io = BinaryIOCollection()  

#==============================================================================
# FUNCTIONS
#==============================================================================


# Add directories to the Python Path:
def add_rel_path(rel_path):
    import sys, os, inspect
    caller_file = inspect.stack()[1][1]
    caller_dir  = os.path.dirname(caller_file)
    dir_to_add  = os.path.realpath(caller_dir + rel_path)
    sys.path.append(dir_to_add)

def add_absolute_path(abs_path):
    import sys
    sys.path.append(abs_path)




#==============================================================================

class DimProtect(object):
    def __init__(self):
        self.b_unidim = False
            
    def start(self, m_data):
        if m_data.ndim==1:
            self.b_unidim = True
            m_data = m_data[None,:]
        return m_data
            
    def finish(self, m_data):
        if self.b_unidim==True:
            m_data = np.squeeze(m_data)
            self.b_unidim = False
        return m_data

#==============================================================================
# These two functions are useful to protect agains unitary dimension.
'''
def dim_protect_strt(m_data):
    b_udim = False
    if m_data.ndim == 1:
        m_data = m_data[None,:]
        b_udim = True
    return m_data, b_udim
    
def dim_protect_end(m_data):
    m_data = np.squeeze(m_data)
    return m_data
'''
# Write text file line by line:================================================
# If add_newline is True, adds "\n" to the end of each line.
def write_text_file(l_strings, filename, add_newline=False):    
    with open(filename, 'w') as f:
        if add_newline:
            f.writelines("%s\n" % l for l in l_strings)
        else:
            f.writelines(l_strings)
    f.close()

# Read scp file:===============================================================
def read_scp_file(filename):
    return read_text_file2(filename, dtype='string', comments='#') 
   
# Read text file2:=============================================================
# Uses numpy.genfromtxt to read files, and protects against the "bug" for data with only one element.
def read_text_file2(*args, **kargs):
    data = np.genfromtxt(*args, **kargs)
    data = np.atleast_1d(data)    
    return data

# Read text file:==============================================================
# output_mode: 'split'=each element as separated string, 
# 'lines'=each line separated.
# NOTE: For numbers, use the function numpy.loadtxt
def read_text_file(filename, output_mode='split', delimiter=' '):
    warnings.warn("Deprecated! Use read_text_file2, instead")
    '''   
    if output_mode == 'num':
        m_data = np.loadtxt(filename)
        return m_data
    '''     
    
    # read file:
    with open(filename) as f:
        l_lines = f.readlines() 
    f.close()

    if output_mode == 'lines':
        return l_lines
    
    elif output_mode == 'split':   
        ll_lines = []
        for line in l_lines:
            ll_lines.append(line.split(delimiter))
        return ll_lines
        


# Get file lit from path:======================================================
# e.g., files_path = "path/to/files/*.ext"
'''
def get_file_list(files_path):
    files_list = glob.glob(files_path)
    n_files    = len(files_list)
    return files_list, n_files
'''

# read and write binary files (wrappers):======================================
#dim = dimmension
def read_binfile(filename, dim=60):
    m_data = _io.load_binary_file(filename, dim).astype('float64') # This is to keep compatibility with numpy default dtype.
    m_data = np.squeeze(m_data)
    return m_data

def write_binfile(m_data, filename):
    _io.array_to_binary_file(m_data, filename)
    return

# Rounds a number and converts into int (e.g., to be used as an index):========
# before, this function was called "round_int"
def round_to_int(float_num):    
    float_num = np.round(float_num).astype(int)
    return float_num

# Extract path, name and ext of a full path:===================================    
def fileparts(fullpath):
    path_with_token, ext = os.path.splitext(fullpath)            
    path,  filename      = os.path.split(fullpath)            
    filetoken            = os.path.basename(path_with_token)
    return [path, filetoken, ext, path_with_token]      
     

# mkdir avoiding exception in case the directory already exists:===========
def mkdir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)


if __name__ == '__main__':
    hola = 45
    