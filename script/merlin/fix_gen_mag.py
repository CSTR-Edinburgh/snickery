#!/usr/bin/env python
## Project: SCRIPT
## Author: Cassia Valentini-Botinhao - cvbotinh@inf.ed.ac.uk

import os
import numpy as np
import sys

def main_work():

	## Usage: python fig_gen_mag.py mag_feat_dir fixed_mag_feat_dir
	indir  = sys.argv[1]
	outdir = sys.argv[2]

	print "Input directory:" + indir
	print "Output directory (fixed features will be saved here):" + outdir

	#### Creating output directory
	if not os.path.exists(outdir):
		os.makedirs(outdir)

	#### Find silence segments in each file according to generated mag
	silence_value = -1.0e+10
	new_silence_value = -10

	for filename in os.listdir(indir):
		gen = read_binfile(indir + '/' + filename, 60)
		[sil_start,sil_end] = find_silence(gen, silence_value)
		silence = new_silence_value * np.ones( gen.shape )
		gen_mod = replace_with_template(gen, silence, sil_start, sil_end)
		write_binfile(gen_mod, outdir +  '/' +  filename)

def replace_with_template(gen, silence, sil_start,sil_end):
	gen_mod = gen
	gen_mod[:sil_start] = silence[:sil_start]
	gen_mod[sil_end:]   = silence[:gen_mod.shape[0]-sil_end]
	return gen_mod

def find_silence(m_data, silence_value):
	num_frames, num_dim = m_data.shape
	silence = True
	for f in range(num_frames):
		if np.mean(m_data[f,:]) != silence_value :
			if silence:
				sil_start = f
			silence = False
		else:
			if not silence:
				sil_end = f
			silence = True
	return sil_start, sil_end

### from magphase
def write_binfile(m_data, filename):
	m_data = np.array(m_data, dtype=np.float32) # Ensuring float32 output
	fid    = open(filename, 'wb')
	m_data.tofile(fid)
	fid.close()
	return

### from magphase
def read_binfile(filename, dim=60):
	fid    = open(filename, 'rb')
	v_data = np.fromfile(fid, dtype=np.float32)
	fid.close()
	if np.mod(v_data.size, dim) != 0:
		raise ValueError('Dimension provided not compatible with file size.')
	m_data = v_data.reshape((-1, dim)).astype('float64') # This is to keep compatibility with numpy default dtype.
	if dim > 1: # Cassia modified here
		m_data = np.squeeze(m_data)
	return  m_data

###############################

if __name__=="__main__":
	main_work()