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
	a.add_argument('-l', dest='file_list',  required=False)
	a.add_argument('-m', dest='dimension',  type=int,   default=60,  help='dimension')  
	a.add_argument('-w', dest='twin_size',  type=int,   default=5,   help='temporal win_size in frames - needs to be an ODD number')  
	a.add_argument('-c', dest='cwin_size',  type=int,   default=1,   help='coefficient win_size in frames - needs to be an ODD number')  
	a.add_argument('-s', dest='std_scale',  type=float, default=0.8, help='std_scale')  

	opts = a.parse_args()

	input_dir  = opts.input_dir
	output_dir = opts.output_dir
	data_ext   = opts.data_ext
	dimension  = opts.dimension
	twin_size  = opts.twin_size # needs to be an ODD number - window is centered in centre frame
	cwin_size  = opts.cwin_size # needs to be an ODD number - window is centered in centre frame
	std_scale  = opts.std_scale
	file_list  = opts.file_list	

	if os.path.exists(output_dir) is False:
		os.makedirs(output_dir)

	if twin_size % 2 == 0:
		print "-w option needs an ODD number"
		exit()

	if cwin_size % 2 == 0:
		print "-c option needs an ODD number"
		exit()

	if file_list is not None:
		f = open(file_list)
		files = [newline.strip() for newline in f.readlines()]
		f.close()
	else:
		files = sorted(glob.glob(input_dir + '/*.' + data_ext))
	
	print "Processing " + str(len(files)) + " files."
	for file in files:

		if file_list is not None:
			file = input_dir + file + '.' + data_ext

		file_name = os.path.basename(file)
		
		out_file_name = output_dir + '/' + file_name
		data, num_frames = load_binary_file_frame(file, dimension)

		smoothed_data = data

                if twin_size != 1 or cwin_size != 1 or std_scale != 1.0:
			if 'f0' in data_ext: # interpolate F0
				smoothed_data , vuv = interpolate_f0(smoothed_data)
			if 'ap' in data_ext: # interpolate BAP
				smoothed_data, vuv = interpolate_bap(smoothed_data)
			if 'real' in data_ext or 'imag' in data_ext: # interpolate REAL/IMAG
				smoothed_data, vuv = interpolate_real_imag(smoothed_data)

		if twin_size != 1:
			smoothed_data = temporal_smoothing(smoothed_data, twin_size)
		if cwin_size != 1:
			smoothed_data = coefficient_smoothing(smoothed_data, cwin_size)
		if std_scale != 1.0:
			smoothed_data = variance_scaling(smoothed_data, std_scale)

		if twin_size != 1 or cwin_size != 1 or std_scale != 1.0:
			if 'f0' in data_ext: # inforce original V/UV
				smoothed_data[vuv==0.0] = -1.0000e+10
			if 'ap' in data_ext or 'real' in data_ext or 'imag' in data_ext: # inforce original V/UV
				smoothed_data[vuv==0.0,:] = 0.0

		array_to_binary_file(smoothed_data, out_file_name)

# more dimensions and ap values are always negative
def interpolate_real_imag(data): 

	num_frames = data.shape[0]
	num_coeff  = data.shape[1]
	ipdata     = data
	for n in range(num_coeff):
		data_coeff   = np.reshape( data[:,n] , ( num_frames , 1) )
		data_coeff[ data_coeff != 0.0 ] += 2.0
		ipdata_coeff , vuv = interpolate_f0(data_coeff)
		ipdata[:,n]  = np.squeeze(ipdata_coeff) - 2.0

	vuv = np.squeeze(vuv)

	return ipdata, vuv

# more dimensions and ap values are always negative
def interpolate_bap(data): 

	num_frames = data.shape[0]
	num_coeff  = data.shape[1]
	ipdata     = data
	for n in range(num_coeff):
		data_coeff   = np.reshape( data[:,n] , ( num_frames , 1) )
		data_coeff   = np.abs(data_coeff)
		ipdata_coeff , vuv = interpolate_f0(data_coeff)
		ipdata[:,n]  = np.squeeze(-ipdata_coeff) # add the negative

	vuv = np.squeeze(vuv)

	return ipdata, vuv

### from speech_manip.py
def interpolate_f0(data):
    
    #data = numpy.reshape(data, (datasize, 1))
    datasize,n = np.shape(data)

    vuv_vector = np.zeros((datasize, 1))
    vuv_vector[data > 0.0] = 1.0
    vuv_vector[data <= 0.0] = 0.0   

    ip_data = data        

    frame_number = datasize
    last_value = 0.0
    for i in xrange(frame_number):
        if data[i] <= 0.0:
            j = i+1
            for j in range(i+1, frame_number):
                if data[j] > 0.0:
                    break
            if j < frame_number-1:
                if last_value > 0.0:
                    step = (data[j] - data[i-1]) / float(j - i)
                    for k in range(i, j):
                        ip_data[k] = data[i-1] + step * (k - i + 1)
                else:
                    for k in range(i, j):
                        ip_data[k] = data[j]
            else:
                for k in range(i, frame_number):
                    ip_data[k] = last_value
        else:
            ip_data[i] = data[i]
            last_value = data[i]

    return  ip_data, vuv_vector   

def coefficient_smoothing(data, win_size):

	smoothed_data = np.transpose(data)
	smoothed_data = temporal_smoothing(smoothed_data, win_size)
	smoothed_data = np.transpose(smoothed_data)

	return smoothed_data

def temporal_smoothing(data, win_size):

	half_win   = ( win_size -1 ) / 2
	num_frames = data.shape[0]
	num_coeff  = data.shape[1]

	smoothed_data = data*0.0
	window  = np.hanning( win_size )
	win_w   = np.sum(window, axis=0)
	window  = np.transpose ( np.tile( window , (num_coeff , 1) ) )
	
	for f in range(num_frames):

		if f-half_win < 0: 
			#pad = np.zeros( ( np.abs(f-half_win) , num_coeff) )
			pad = np.repeat( data[0,:].reshape(1,-1), np.abs(f-half_win), axis=0 )
			win_data = np.concatenate( ( pad , data[:f+half_win+1,:]))
		elif f+half_win+1 > num_frames:
			#pad = np.zeros( ( f+half_win+1 - num_frames , num_coeff) )
			pad = np.repeat( data[-1,:].reshape(1,-1), f+half_win+1 - num_frames, axis=0 )
			win_data = np.concatenate( ( data[f-half_win:,:] , pad ))
		else:
			win_data = data[f-half_win:f+half_win+1,:]

		smoothed_data[f, :] = np.sum( win_data * window , axis = 0 ) / win_w
		
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
