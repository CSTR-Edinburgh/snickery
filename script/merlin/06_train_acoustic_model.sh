#!/bin/bash -e
## Project: SCRIPT
## Author: Cassia Valentini-Botinhao - cvbotinh@inf.ed.ac.uk
## Based on: https://github.com/CSTR-Edinburgh/merlin/tree/master/egs/build_your_own_voice/s1/06_train_acoustic_model.sh

if test "$#" -ne 2; then
    echo "################################"
    echo "Usage:"
    echo "./06_train_acoustic_model.sh <work_dir> <path_to_acoustic_conf_file>"
    echo ""
    echo "Default path to acoustic conf file: conf/acoustic_Voice.conf"
    echo "################################"
    exit 1
fi

work_dir=$1
acoustic_conf_file=$2

global_config_file=${work_dir}/conf/global_settings.cfg
source $global_config_file

### Step 6: train acoustic model ###
echo "Step 6:"
echo "training acoustic model..."
${work_dir}/scripts/submit.sh ${MerlinDir}/src/run_merlin.py $acoustic_conf_file
