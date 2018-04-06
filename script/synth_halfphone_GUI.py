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

from synth_halfphone import Synthesiser

from Tkinter import *
import re

from util import safe_makedir


from functools import partial

class SynthesiserTuningInterface(tk.Tk):
    def __init__(self, parent, synthesiser, outdir):
        tk.Tk.__init__(self, parent)
        self.parent = parent
        self.outdir = outdir

        self.synthesiser = synthesiser

        self.testset = self.synthesiser.get_sentence_set('test')

        self.initialise_tunable_settings()
        self.initialize()



        self.current_config_number = 1
        while os.path.isfile(self.current_setting_file()):
            self.current_config_number += 1

        self.current_config_number -= 1
        self.next_new_config_number = self.current_config_number + 1

        self.refresh()

        if not os.path.isfile(self.current_setting_file()):
            self.synthesise_condition()


        # ### load config from previous settings, if they exist:
        # if self.current_config_number > 1:
        #     self.current_config_number -= 1
        #     self.set_config_from_file()
        #     self.current_config_number += 1
        #     print 'Loaded config from iteration %s'%(self.current_config_number-1)
        # else:
        #     print 'Use average initial config'


        #self.reconfigure_synthesiser()



    def initialise_tunable_settings(self):
        '''
        Ignore input config except for stream names -- initialise everything to 
        average/sensible initial settings
        '''

        ## ignore initial config values -- set stream weights to average:
        self.target_stream_weight_variables = {}
        streams = self.synthesiser.config['stream_list_target']
        print streams
        for stream in streams:
            self.target_stream_weight_variables[stream] = tk.DoubleVar()
            self.target_stream_weight_variables[stream].set(1.0 / len(streams))
        
        self.join_stream_weight_variables = {}
        streams = self.synthesiser.config['stream_list_join']
        print streams
        for stream in streams:
            self.join_stream_weight_variables[stream] = tk.DoubleVar()
            self.join_stream_weight_variables[stream].set(1.0 / len(streams))

        self.join_cost_weight = tk.DoubleVar()
        self.join_cost_weight.set(0.5)

        self.search_epsilon = tk.DoubleVar() # 'Return approximate nearest neighbors; the kth returned value is guaranteed to be no further than (1+eps) times the distance to the real kth nearest neighbor.'
        self.search_epsilon.set(10.0)

        self.magphase_use_target_f0 = tk.BooleanVar()
        self.magphase_use_target_f0.set(True)


        self.multiepoch = tk.IntVar()
        self.multiepoch.set(4)
        self.magphase_overlap = tk.IntVar()
        self.magphase_overlap.set(2)

        # truncate_join_streams = [-1, -1, -1]
        # truncate_target_streams = [-1, -1] 

        self.tk_current_config_number = tk.StringVar()

    def get_current_settings(self):

        '''
        tk variable values -> python dict
        '''
      
        streams = self.synthesiser.config['stream_list_target']
        target_weights = []
        for stream in streams:        
            target_weights.append(self.target_stream_weight_variables[stream].get())
             
        streams = self.synthesiser.config['stream_list_join']
        join_weights = []
        for stream in streams:        
            join_weights.append(self.join_stream_weight_variables[stream].get())


        if self.magphase_overlap.get() >  self.multiepoch.get():
            print 'magphase_overlap cannot be larger than multiepoch -- reset it!'
            if self.multiepoch.get() % 2 == 0:
                self.magphase_overlap.set(self.multiepoch.get())
            else:
                self.magphase_overlap.set(self.multiepoch.get()-1)
        if self.magphase_overlap.get() % 2 != 0:
            print 'magphase_overlap must be even valued -- reset it!'
            self.magphase_overlap.set(self.magphase_overlap.get()-1)

        self.refresh()

        return {
               'join_cost_weight': self.join_cost_weight.get() ,
               'join_stream_weights': join_weights,
               'target_stream_weights': target_weights,
               
               'multiepoch': self.multiepoch.get(),
               'search_epsilon': self.search_epsilon.get(),
               'magphase_use_target_f0': self.magphase_use_target_f0.get(),
               'magphase_overlap': self.magphase_overlap.get()}


    def read_config(self):
        config = {}
        execfile(self.current_setting_file(), config)
        del config['__builtins__']        
        return config

    def set_config_from_file(self):
        
        '''
        config file -> tk variable values
        '''
        config = self.read_config()

        streams = self.synthesiser.config['stream_list_target']
        for (i,stream) in enumerate(streams):
            weight = config['join_stream_weights'][i]
            self.target_stream_weight_variables[stream].set(weight)
        
        streams = self.synthesiser.config['stream_list_join']
        for (i,stream) in enumerate(streams):
            weight = config['join_stream_weights'][i]
            self.join_stream_weight_variables[stream].set(weight)
        
        self.join_cost_weight.set(config['join_cost_weight'])
        self.search_epsilon.set(config['search_epsilon'])
        self.magphase_use_target_f0.set(config['magphase_use_target_f0'])
        self.multiepoch.set(config['multiepoch'])
        self.magphase_overlap.set(config['magphase_overlap'])



    def set_sliders_to_current_values(self):




        self.epsilon_slider.set_value()
        self.join_cost_slider.set_value()

        self.target_weight_sliders.set_value()

        self.join_weight_sliders.set_value()

        self.multiepoch_slider.set_value()


        self.overlap_slider.set_value()

        self.button_target_fzero_true.set(self.magphase_use_target_f0.get())
        self.button_target_fzero_false.set(self.magphase_use_target_f0.get())







    def record_config(self):
        '''
        tk variable values -> config file
        '''
        
        safe_makedir(self.current_setting_dir())
        
        f = open(self.current_setting_file(), 'w')
        for (k,v) in self.get_current_settings().items():
            f.write('%s = %s\n'%(k, str(v)))
        f.close()



    def reconfigure_synthesiser(self):

        s = self.get_current_settings()
        self.synthesiser.reconfigure_settings(s)

    def go_to_previous(self):
        self.current_config_number -= 1
        self.set_config_from_file()
        self.refresh()

    def go_to_next(self):
        self.current_config_number += 1
        self.set_config_from_file()
        self.refresh()


    def initialize(self):
        self.grid()

        main_frame = tk.Frame(self)
        main_frame.grid()
        self.main_frame = main_frame



        ## frame for controls:
        synth_frame = tk.Frame(main_frame)
        synth_frame.grid(row=0, column=0, columnspan=6, sticky=tk.W)



        ## navigation:
        #self.iteration_choice = [str(i) for i in range(self.next_new_config_number)]
        # tkvar.set('Pizza') # set the default option
         
        # popupMenu = OptionMenu(mainframe, tkvar, *choices)
        # Label(mainframe, text="Choose a dish").grid(row = 1, column = 1)
        # popupMenu.grid(row = 2, column =1)
         
        # # on change dropdown value
        # def change_dropdown(*args):
        #     print( tkvar.get() )
         
        # # link function to change dropdown
        # tkvar.trace('w', change_dropdown)        



        self.prev_button = tk.Button(synth_frame, text=u"Previous",
                                command=self.go_to_previous, state=DISABLED)
        self.prev_button.grid(column=0, row=0, sticky=tk.W)
        
        self.next_button = tk.Button(synth_frame, text=u"Next",
                                command=self.go_to_next, state=DISABLED)
        self.next_button.grid(column=1, row=0, sticky=tk.W)


        self.synth_button = tk.Button(synth_frame, text=u"Synthesise new",
                                command=self.synthesise_condition, state=NORMAL)
        self.synth_button.grid(column=2, row=0, sticky=tk.W)


        #self.position_indicator = tk.Label(synth_frame, text='position') # textvariable=self.tk_current_config_number,
        self.position_indicator = tk.Label(synth_frame, textvariable=self.tk_current_config_number)
        

        self.position_indicator.grid(column=3, row=0, sticky=tk.W)
        self.position_indicator.config(font=("Courier", 44))









        self.epsilon_slider = SingleSlider(main_frame, variable=self.search_epsilon, \
                name='Search approx', help_text='', limit=20.0)
        self.epsilon_slider.grid(column=0, row=1, sticky=tk.W, padx=10, pady=10)

        self.join_cost_slider = SingleSlider(main_frame, variable=self.join_cost_weight, \
                name='Join cost scaling', help_text='')
        self.join_cost_slider.grid(column=1, row=1, sticky=tk.W, padx=10, pady=10)


        # self.join_cost_slider = SlidersBox(main_frame, variables=self.join_cost_weight, name='Join cost scaling', force_sum_to_one=False, help_text='')
        # self.join_cost_slider.grid(column=0, row=1, sticky=tk.W, padx=10, pady=10)

        self.target_weight_sliders = SlidersBox(main_frame, variables=self.target_stream_weight_variables, \
                name='Target stream weights', force_sum_to_one=True)
        self.target_weight_sliders.grid(column=2, row=1, sticky=tk.W, padx=10, pady=10)

        self.join_weight_sliders = SlidersBox(main_frame, variables=self.join_stream_weight_variables, \
                name='Join stream weights', force_sum_to_one=True)
        self.join_weight_sliders.grid(column=3, row=1, sticky=tk.W, padx=10, pady=10)


        self.multiepoch_slider = SingleSlider(main_frame, variable=self.multiepoch, \
            name='Multiepoch', help_text='', limit=100, discrete=True)
        self.multiepoch_slider.grid(column=4, row=1, sticky=tk.W, padx=10, pady=10)



        self.overlap_slider = SingleSlider(main_frame, variable=self.magphase_overlap, \
            name='Crossfade window length', help_text='', limit=20, discrete=True)
        self.overlap_slider.grid(column=5, row=1, sticky=tk.W, padx=10, pady=10)

        button_target_frame = tk.Frame(main_frame)
        button_target_frame.grid(column=6, row=1)

        toplabel = tk.Label(button_target_frame, height=1, width=20, text='Impose target F0')
        toplabel.grid(row=0, column=0, sticky=tk.W)
        self.button_target_fzero_true = tk.Radiobutton(button_target_frame, text='Yes', variable=self.magphase_use_target_f0, value=True)
        self.button_target_fzero_true.grid(row=1, column=0, sticky=tk.W)
        self.button_target_fzero_false = tk.Radiobutton(button_target_frame, text='No', variable=self.magphase_use_target_f0, value=False)
        self.button_target_fzero_false.grid(row=2, column=0, sticky=tk.W)   



        ## frame for play buttons:
        play_frame = tk.Frame(main_frame)
        play_frame.grid(row=2, column=0, columnspan=6)

        self.play_buttons = {}
        for (i, base) in enumerate(self.testset):


            play_button = tk.Button(play_frame, text=base, state=DISABLED,\
                                    command=partial(self.playbase, base))
            play_button.grid(column=i, row=0, sticky=tk.W)
            self.play_buttons[base] = play_button



    def current_setting_dir(self):
        return os.path.join(self.outdir, str(self.current_config_number).zfill(5))


    def current_setting_file(self):
        return os.path.join(self.current_setting_dir(), 'tuned_settings.cfg')


    def synthesise_condition(self):

        if not self.settings_have_changed():
            print 'NO VALUES ALTERED'
            return


        self.current_config_number = self.next_new_config_number
        self.next_new_config_number += 1
        self.reconfigure_synthesiser()

        safe_makedir(self.current_setting_dir())

        self.update_play_buttons()
        self.synthesiser.synth_from_config(outdir=self.current_setting_dir()) # opts.output_dir) # (inspect_join_weights_only=False, synth_type='test', outdir=opts.output_dir)

        self.record_config()
        #self.synth_button['state'] = 'disabled'


    def update_play_buttons(self): # , iteration=0):
        for base in self.testset:
            self.play_buttons[base]['state'] = 'disabled' # grey all buttons to start with

        all_files_synthesised = True
        print 'update_play_buttons'
        #print iteration
        for base in self.testset:
            wave = os.path.join(self.current_setting_dir(), base + '.wav')
            print 'look for'
            print wave
            if os.path.isfile(wave):
                self.play_buttons[base]['state'] = 'normal' # un-grey the button
            else:
                all_files_synthesised = False
        if all_files_synthesised:
            return
        else: ## try again in 1 second:
            self.after(1000, self.update_play_buttons)

    def playbase(self,base):
        wave = os.path.join(self.current_setting_dir(), base + '.wav')
        self.play(wave)

    def play(self, wavfile, speed=1.0, first_seconds=0,  last_seconds=0):
        
        speed = str(speed) # self.speed.get()
        comm = ["play", wavfile, 'tempo', speed] #, 'trim', '0', str(first_seconds), '-'+str(last_seconds)]
        if first_seconds or last_seconds:
            comm.append('trim')
        if first_seconds:
            comm.extend(['0', str(first_seconds)])
        if first_seconds and last_seconds:
            comm.append('-'+str(last_seconds))
        elif last_seconds:
            comm.extend(['0', '0', '-'+str(last_seconds)])
        print comm
        print ' '.join(comm)
        return_code = subprocess.call(comm)
        
    def refresh(self):  
            
        if self.current_config_number == 1:
            self.prev_button['state'] = 'disabled'
        else:
            self.prev_button['state'] = 'normal'

    
        if self.current_config_number == self.next_new_config_number - 1:
            self.next_button['state'] = 'disabled'
        else:
            self.next_button['state'] = 'normal'

        self.tk_current_config_number.set('  Current trial: ' + str(self.current_config_number))

        self.update_play_buttons()
        self.main_frame.update()
        

    def settings_have_changed(self):
        if not os.path.isfile(self.current_setting_file()):
            return True
        changed = False
        stored_vals = self.read_config()
        modified_vals = self.get_current_settings()
        for (k,v) in stored_vals.items():
            if modified_vals[k] != v:
                changed = True
        return changed



# https://stackoverflow.com/questions/20399243/display-message-when-hovering-over-something-with-mouse-cursor-in-python
class HoverInfo(Menu):
    def __init__(self, parent, text, command=None):
       self._com = command
       Menu.__init__(self,parent, tearoff=0)
       if not isinstance(text, str):
          raise TypeError('Trying to initialise a Hover Menu with a non string type: ' + text.__class__.__name__)
       toktext=re.split('\n', text)
       for t in toktext:
          self.add_command(label = t)
       self._displayed=False
       self.master.bind("<Enter>",self.Display )
       self.master.bind("<Leave>",self.Remove )

    def __del__(self):
       self.master.unbind("<Enter>")
       self.master.unbind("<Leave>")

    def Display(self,event):
       if not self._displayed:
          self._displayed=True
          self.post(event.x_root, event.y_root)
       if self._com != None:
          self.master.unbind_all("<Return>")
          self.master.bind_all("<Return>", self.Click)

    def Remove(self, event):
     if self._displayed:
       self._displayed=False
       self.unpost()
     if self._com != None:
       self.unbind_all("<Return>")

    def Click(self, event):
       self._com()



class SingleSlider(tk.Frame):
    def __init__(self, parent, variable='', name='', force_sum_to_one=True, help_text='', limit=1.0, discrete=False):

        ## Frame is not a new-style class, but super requires new-style classes to work.
        ## https://stackoverflow.com/questions/43767988/typeerror-super-argument-1-must-be-type-not-classobj
        tk.Frame.__init__(self, parent, highlightbackground="black", highlightcolor="black", highlightthickness=5)
        # super(BoundSliders, self).__init__(parent)

        self.help_text = help_text
        self.variable = variable
        #self.sliders = {}
        toplabel = tk.Label(self, height=1, width=20, text=name)
        toplabel.grid(column=0, row=0, sticky=tk.W)

        if discrete:
            self.slider = tk.Scale(self, variable = self.variable, from_ = 0.0, to=limit, \
                                    length=300, digits=1, resolution=1)            
        else:
            self.slider = tk.Scale(self, variable = self.variable, from_ = 0.0, to=limit, \
                                    length=300, digits=3, resolution=0.01)
            

        self.slider.grid(column=0, row=1, sticky=tk.W)
            
        if self.help_text:
            self.hover = HoverInfo(self, self.help_text)

    def set_value(self):
        self.slider.set(self.variable.get())



class SlidersBox(tk.Frame):
    def __init__(self, parent, variables={}, name='', force_sum_to_one=True, help_text=''):

        ## bound: if True, force sliders in box to sum to 1

        ## Frame is not a new-style class, but super requires new-style classes to work.
        ## https://stackoverflow.com/questions/43767988/typeerror-super-argument-1-must-be-type-not-classobj
        tk.Frame.__init__(self, parent, highlightbackground="black", highlightcolor="black", highlightthickness=5)
        # super(BoundSliders, self).__init__(parent)

        self.help_text = help_text
        self.variables = variables
        #self.sliders = {}
        toplabel = tk.Label(self, height=1, width=20, text=name)
        toplabel.grid(column=0, row=0, columnspan=len(self.variables), sticky=tk.W)

        self.sliders = {}

        for (i, stream) in enumerate(variables.keys()):
           

            ### float value from slider: https://stackoverflow.com/questions/25361926/tkinter-scale-and-floats-when-resolution-1
                           # an MVC-trick an indirect value-holder
            if force_sum_to_one:
                slider = tk.Scale(self, variable = self.variables[stream], from_ = 0.0, to=1.0, \
                                length=300, digits=3, resolution=0.01, command = self.ensure_sliders_sum_to_one)
            else:
                slider = tk.Scale(self, variable = self.variables[stream], from_ = 0.0, to=1.0, \
                                length=300, digits=3, resolution=0.01)

            slider.grid(column=i, row=1, sticky=tk.W)
            label = tk.Label(self, height=1, width=10, text=stream)
            label.grid(column=i, row=2, sticky=tk.W)
            self.sliders[stream] = slider


        if self.help_text:
            self.hover = HoverInfo(self, self.help_text)

 

    def ensure_sliders_sum_to_one(self, dummy_val):
        ## work here:
        if 0:
            print 'TODO: fix weights...'
            print dummy_val
            print [(key, var.get()) for (key,var) in self.variables.items()]
            print 

        total = sum([var.get() for var in self.variables.values()])
        # print total
        total = max(total, 0.00001)
        for key in self.variables.keys():
            self.variables[key].set(self.variables[key].get() / total)

    def set_value(self):
        for (stream, var) in self.variables.items():
            self.sliders[stream].set(var.get())




if __name__ == '__main__':

    #################################################
      
    # ======== process command line ==========

    a = ArgumentParser()
    a.add_argument('-c', dest='config_fname', required=True)
    a.add_argument('-o', dest='output_dir', required=True)
    opts = a.parse_args()



    synth = Synthesiser(opts.config_fname)
    tuner = SynthesiserTuningInterface(None, synth, opts.output_dir)
    
    tuner.title('Tune synthesiser settings')
    tuner.mainloop()
    


    # if opts.output_dir:
    #     if not os.path.isdir(opts.output_dir):
    #         os.makedirs(opts.output_dir)
    #     os.system('cp %s %s'%(opts.config_fname, opts.output_dir))

    # synth.synth_from_config(inspect_join_weights_only=False, synth_type='test', outdir=opts.output_dir)












