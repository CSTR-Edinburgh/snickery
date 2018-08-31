#!/bin/bash
## Project: SCRIPT
## Author: Cassia Valentini-Botinhao - cvbotinh@inf.ed.ac.uk
## Based on: https://github.com/CSTR-Edinburgh/merlin/tree/master/egs/build_your_own_voice/s1/scripts/prepare_labels_from_txt.sh

if test "$#" -ne 1; then
    echo "Usage: ./scripts/remove_intermediate_files.sh conf/global_settings.cfg"
    exit 1
fi

if [ ! -f $1 ]; then
    echo "Global config file doesn't exist"
    exit 1
else
    source $1
fi

###################################################
######## remove intermediate synth files ##########
###################################################

current_working_dir=$(pwd)

synthesis_dir=${WorkDir}/experiments/${Voice}/test_synthesis
gen_lab_dir=${synthesis_dir}/gen-lab
gen_wav_dir=${synthesis_dir}/wav

shopt -s extglob

if [ -d "$gen_lab_dir" ]; then
    cd ${gen_lab_dir}
    rm -f *.!(lab)
fi

if [ -d "$gen_wav_dir" ]; then
    cd ${gen_wav_dir}
    rm -f weight
    rm -r *.cmp
    rm -r *.lab*
fi

cd ${current_working_dir}
