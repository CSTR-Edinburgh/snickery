

VERY_BIG_WEIGHT_VALUE = 1000000000000000.0

label_delimiter = '/'  ## used in internal quinphone format

## names of data streams which include V/UV decision; unvoiced frames have value <= 0.0
vuv_stream_names = ['f0', 'lf0']

label_length_diff_tolerance = 5


special_uv_value = -1000.0  ## value assigned to unvoiced frames of F0 etc after when composed but before standardisation (at which point it is swapped for -1 * F * std(voiced) )

uv_scaling_factor = 20.0 ## number of times std of voice values unvoiced frames are set below 0.

target_rep_widths = {'onepoint': 1, 'twopoint': 2, 'threepoint': 3, 'epoch': 1} 
