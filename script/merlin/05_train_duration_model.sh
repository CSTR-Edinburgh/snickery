#!/bin/bash -e
## Project: SCRIPT
## Author: Cassia Valentini-Botinhao - cvbotinh@inf.ed.ac.uk
## Based on: https://github.com/CSTR-Edinburgh/merlin/tree/master/egs/build_your_own_voice/s1/05_train_duration_model.sh

if test "$#" -ne 2; then
    echo "################################"
    echo "Usage:"
    echo "./05_train_duration_model.sh <work_dir> <path_to_duration_conf_file>"
    echo ""
    echo "Default path to duration conf file: conf/duration_Voice.conf"
    echo "################################"
    exit 1
fi

work_dir=$1
duration_conf_file=$2

global_config_file=${work_dir}/conf/global_settings.cfg
source $global_config_file

### Step 5: train duration model ###
echo "Step 5:"
echo "training duration model..."
${work_dir}/scripts/submit.sh ${MerlinDir}/src/run_merlin.py $duration_conf_file
