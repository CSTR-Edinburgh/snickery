#!/usr/bin/env python
# -*- coding: utf-8 -*-
## Project:  
## Author: Oliver Watts - owatts@staffmail.ed.ac.uk

import sys
import os
import glob
import re
import timeit
import math
import copy
import random
import subprocess

from argparse import ArgumentParser


import Tkinter as tk
import tkMessageBox

from synth_simple import Synthesiser

from Tkinter import *
import re

from util import safe_makedir
import make_internal_webchart



def shall_we_continue():
   
    raw_reponse = raw_input("  Type q to quit or any other key to continue ... ")            
    if raw_reponse == "q":
        return False
    else:
        return True


def record_config(synth, fname):

    keys = ['join_stream_weights', 'target_stream_weights', 'join_cost_weight', 'search_epsilon', 'multiepoch']
         

    f = open(fname, 'w')

    for key in keys:
        val = synth.config[key]
        f.write('%s = %s\n'%(key, str(val)))
    f.close()


if __name__ == '__main__':




    #################################################
      
    # ======== process command line ==========

    a = ArgumentParser()
    a.add_argument('-c', dest='config_fname', required=True)
    a.add_argument('-o', dest='output_dir', required=True)
    opts = a.parse_args()


    synth = Synthesiser(opts.config_fname)
    
    trial = 1

    html_file = os.path.join(opts.output_dir, 'listen.html')

    while shall_we_continue():

        anything_changed = synth.reconfigure_from_config_file()
        if anything_changed:

            current_setting_dir = os.path.join(opts.output_dir, 't'+str(trial).zfill(5))
            safe_makedir(current_setting_dir)

            synth.synth_from_config(outdir=current_setting_dir)
            record_config(synth, current_setting_dir + '/tuned_settings.cfg')

            trial += 1
            voice_dirs = glob.glob(os.path.join(opts.output_dir, 't*'))
            make_internal_webchart.main_work(voice_dirs, outfile=html_file)
            print 'Browse to %s to listen'%(html_file)
        else:
            print 'Nothing changed in config file %s'%(opts.config_fname)




