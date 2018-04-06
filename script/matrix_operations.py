
import numpy as np

def zero_pad_matrix(a, start_pad, end_pad):
    '''
    if start_pad and end_pad are both 0, do nothing
    '''
    if start_pad > 0:
        dim = a.shape[1] 
        a = np.vstack([np.zeros((start_pad, dim)), a])
    if end_pad > 0:
        dim = a.shape[1] 
        a = np.vstack([a, np.zeros((end_pad, dim))])
    return a

def taper_matrix(a, taper_length):
    m,n = a.shape
    assert taper_length * 2 <= m, 'taper_length (%s) too long for (padded) unit length (%s)'%(taper_length, m) 
    in_taper = np.hanning(((taper_length + 1)*2)+1)[1:taper_length+1].reshape(-1,1)
    out_taper = np.flipud(in_taper).reshape(-1,1)
    if 0:
        import pylab
        pylab.plot(in_taper)
        pylab.plot(out_taper)
        pylab.plot((in_taper + out_taper)-0.05)   ### check sum to 1
        pylab.show()
        sys.exit('wrvwsfrbesbr')
    a[:taper_length,:] *= in_taper
    a[-taper_length:,:] *= out_taper
    return a

