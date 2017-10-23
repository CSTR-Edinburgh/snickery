

import sys
import os
import glob

# modify import path to obtain modules from the script/ directory:
snickery_dir = os.path.split(os.path.realpath(os.path.abspath(os.path.dirname(__file__))))[0]+'/'
sys.path.append(os.path.join(snickery_dir, 'script'))

from speech_manip import  get_speech, put_speech

datadir = sys.argv[1]  ## we expect mfcc to be a subdir of this, and will make energy and mfcc12 as its sisters

mfcc_dir = os.path.join(datadir, 'mfcc')
mfcc12 = os.path.join(datadir, 'mfcc12')
energy = os.path.join(datadir, 'energy')

if not os.path.isdir(mfcc_dir):
    sys.exit('%s does not exist'%(mfcc_dir))

if os.path.isdir(mfcc12):
    sys.exit('%s already exists'%(mfcc12))

if os.path.isdir(energy):
    sys.exit('%s already exists'%(energy))

os.makedirs(mfcc12)
os.makedirs(energy)


for mfcc_fname in sorted(glob.glob(mfcc_dir + '/*.mfcc')):
    _,base = os.path.split(mfcc_fname)
    base = base.replace('.mfcc','')
    print base
    speech = get_speech(mfcc_fname, 13)

    ## remove outlying values which make later standardisation of the data crazy:
    speech[speech<-100.0] = 0.0
    speech[speech>100.0] = 0.0
    
    e = speech[:,0].reshape(-1,1)
    m = speech[:,1:]
    put_speech(e, os.path.join(energy, base+'.energy'))
    put_speech(m, os.path.join(mfcc12, base+'.mfcc12'))








