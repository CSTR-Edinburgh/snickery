import numpy as np

## use mulaw nonlinearity, but do not quantise
# ## Mu-law functions from sampleRNN code: sampleRNN_ICLR2017-master/datasets/dataset.py
def lin2mu(x):
    sample_width = 16
    wrange = 2**sample_width
    mu = wrange
    half_range = wrange / 2.0
    x = np.float32(x) - (- half_range)
    x = x / wrange
    x = (x - 0.5) * 2
    x_mu = np.sign(x) * np.log(1 + mu*np.abs(x))/np.log(1 + mu)

    ### return from -1 -> 1 to the 16 bit range (for weighting):
    x_mu *= (half_range / 2)

    return x_mu


def mu2lin(x):
    sample_width = 16
    mu = 2**sample_width
    mu = float(mu)
    half_range = mu / 2.0


    ###  16 bit range to -1 -> 1 
    x = np.array(x)
    x /= (half_range / 2)


    #x = x.astype('float32')
    #y = 2. * (x - (mu+1.)/2.) / (mu+1.)
    y = x
    ylin = np.sign(y) * (1./mu) * ((1. + mu)**np.abs(y) - 1.)
    ylin *= half_range
    return ylin