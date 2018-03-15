#### Smooth data following Tom Merritt's work: temporal smoothing and variance scaling
#### - Default values for win_size (Hanning window size for temporal smoothing) and 
#### - std_scale (scale factor for standard deviation) are the ones closest to perceived HMM-SPSS
#### - (see his thesis)
#### Cassia VB - 14/03/18

import os
import sys
import numpy as np
import glob
from argparse import ArgumentParser

def main():

	a = ArgumentParser()
	a.add_argument('-f', dest='input_dir',  required=True)
	a.add_argument('-o', dest='output_dir', required=True)    
	a.add_argument('-t', dest='data_ext',   required=True)
	a.add_argument('-m', dest='dimension',  type=int, default=60, help='dimension')  
	a.add_argument('-w', dest='win_size',   type=int, default=5, help='win_size - needs to be an ODD number')  
	a.add_argument('-s', dest='std_scale',  type=float, default=0.8, help='std_scale')  

	opts = a.parse_args()

	input_dir  = opts.input_dir
	output_dir = opts.output_dir
	data_ext   = opts.data_ext
	dimension  = opts.dimension
	win_size   = opts.win_size # needs to be an ODD number - window is centered in centre frame
	std_scale  = opts.std_scale

	if os.path.exists(output_dir) is False:
		os.mkdir(output_dir)

	if win_size % 2 == 0:
		print "-w option needs an ODD number"
		exit()

	files = sorted(glob.glob(input_dir + '/*.' + data_ext))
	
	print "Processing " + str(len(files)) + " files."
	for file in files:

		file_name     = os.path.basename(file)
		out_file_name = output_dir + '/' + file_name
		data, num_frames = load_binary_file_frame(file, dimension)

		smoothed_data = data
		smoothed_data = temporal_smoothing(smoothed_data, win_size)
		smoothed_data = variance_scaling(smoothed_data, std_scale)
		
		array_to_binary_file(smoothed_data, out_file_name)

def temporal_smoothing(data, win_size):

	half_win   = ( win_size -1 ) / 2
	num_frames = data.shape[0]
	num_coeff  = data.shape[1]

	smoothed_data = data*0.0
	window  = np.transpose ( np.tile( np.hanning( win_size ) , (num_coeff , 1) ) )
	
	for f in range(num_frames):

		if f-half_win < 0: 
			pad = np.zeros( ( np.abs(f-half_win) , num_coeff) )
			win_data = np.concatenate( ( pad , data[:f+half_win+1,:]))
		elif f+half_win+1 > num_frames:
			pad = np.zeros( ( f+half_win+1 - num_frames , num_coeff) )
			win_data = np.concatenate( ( data[f-half_win:,:] , pad ))
		else:
			win_data = data[f-half_win:f+half_win+1,:]

		smoothed_data[f, :] = np.mean( win_data * window , axis = 0 )
		
	return smoothed_data

def variance_scaling(data, std_scale):

	# Subtract the utterance-level mean for each coefficient 
	data_mean = np.mean(data,axis=0)
	data -= data_mean

	# print np.std(data, axis=0)

	# Scale variance
	data *= std_scale

	# print np.std(data, axis=0)

	# Add mean back
	data += data_mean

	return data

##################################################
def array_to_binary_file(data, output_file_name):

	data = np.array(data, 'float32')
	fid = open(output_file_name, 'wb')
	data.tofile(fid)
	fid.close()

def load_binary_file_frame(file_name, dimension):

	fid_lab = open(file_name, 'rb')
	features = np.fromfile(fid_lab, dtype=np.float32)
	fid_lab.close()
	assert features.size % float(dimension) == 0.0,'specified dimension not compatible with data'
	frame_number = features.size / dimension
	features = features[:(dimension * frame_number)]
	features = features.reshape((-1, dimension))

	return  features, frame_number

if __name__ == '__main__':
	main()