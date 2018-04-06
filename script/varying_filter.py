
import numpy as np
# import pylab
def make_filter_01(dilation_factor, noutputs, verbose=False):

    ## first pass to determine length of filtered region:
    stepsize = 1.0
    centre = 0
    for step in range(noutputs):
        winlength = (int(round(stepsize)) * 2) - 1 ## choose odd number
        #output += ('|' + '_'*(int(round(stepsize))-1))
        stepsize *= dilation_factor
        centre += int(round(stepsize))
    centre -= int(round(stepsize)) ## because the final iteration's increment is not used
    stepsize /= dilation_factor    ## because the final iteration's increment is not used

    filter_length = centre + int(round(stepsize))  
    filter_matrix = np.zeros((filter_length, noutputs))

    if verbose:
        print filter_length
        print centre
        print stepsize

    ## second pass -- add the windows:
    stepsize = 1.0
    centre = 0
    for step in range(noutputs):
        winlength = (int(round(stepsize)) * 2) - 1 ## choose odd number
        if step==0:
            assert winlength==1
        if verbose:
            print '--'
            print winlength
        win = np.hanning(winlength)
        start = centre - (int(round(stepsize)) - 1)
        end = centre + int(round(stepsize)) 
        if verbose: 
            print (centre, int(round(stepsize)) , start, end)
        filter_matrix[start:end, step] = win

        # print win
        stepsize *= dilation_factor
        centre += int(round(stepsize))

    ## Normalise, so resulting extracted features will be commensurate:
    filter_matrix /= filter_matrix.sum(axis=0).reshape((1,-1))

    ### Finally, flip the filter!
    filter_matrix = np.flipud(filter_matrix)

    if verbose:
        pylab.subplot( 211 )
        pylab.imshow( filter_matrix, interpolation='nearest')
        pylab.gray()
        ax = pylab.subplot( 212 )    
        pylab.plot(filter_matrix, color='k')
        ax.set_xlim([0, filter_length])
        #ax.set_xticks(ax.get_xticks() / 16000)
        pylab.show()
        sys.exit('asdv')
    return filter_matrix
    
if __name__=="__main__":

    print 'test code script/varying_filter.py'
    #make_filter_01(1.6, 10, verbose=True)
    make_filter_01(1.2, 20, verbose=True)


#111_1_1__1__1___1___1____1____1_____1_____    