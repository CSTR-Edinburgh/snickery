"""
My personal plotting functions library (used for debugging).
@author: Felipe Espic
"""

import numpy as np
import __builtin__ as builtins
import matplotlib
matplotlib.use("Qt4Agg")
from matplotlib import pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
pl.ion()

matplotlib.rcParams['lines.antialiased'] = False
matplotlib.rcParams['lines.linewidth']   = 1.0


# FOR GENERAL USE:====================================================
def plotm(m_data):
    ret = pl.imshow(m_data.T, cmap=pl.cm.jet, aspect='auto', origin='lower', interpolation='nearest')
    #ret = pl.imshow(m_data.T, cmap=pl.cm.jet, aspect='auto', origin='lower', interpolation='spline16')
    pl.colorbar(ret)
    return
pl.plotm = plotm

def plotg(*args, **kargs):
    '''
    Plot with grid
    '''
    pl.plot(*args, **kargs)
    pl.grid()
pl.plotg = plotg


def plots(m_data):
    '''
    Plot 3D surface
    '''
    #v_x = np.arange(-5, 5, 0.25)
    #v_x = np.arange(m_data.shape[0],0,-1)
    v_x = np.arange(m_data.shape[0])
    v_y = np.arange(m_data.shape[1])
    m_y, m_x = np.meshgrid(v_y, v_x) # this order is intentional
    ax = pl.axes(projection='3d')
    ax.plot_surface(m_x, m_y, m_data, cmap=pl.cm.jet, rstride=1, cstride=1, linewidth=0)
pl.plots = plots

# FOR INTERACTIVE PLOTTING (Default mode):===========================
def plot_int(*args, **kargs):
    pl.clf() # clean current figure (avoid overlap)
    pl.plot(*args, **kargs)
    pl.grid()
    pl.show()
    return
builtins.plot = plot_int

# Just a wrapper for now:
def plot_matrix_int(m_data):
    plotm(m_data)
    return
builtins.plotm = plot_matrix_int




'''
def hola():
    print('HOLA')

#if __name__ == '__main__':

plt.hola = hola
'''
