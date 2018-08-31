#!/bin/bash -e
## Project: SCRIPT
## Author: Cassia Valentini-Botinhao - cvbotinh@inf.ed.ac.uk
## Based on: https://github.com/CSTR-Edinburgh/merlin/tree/master/egs/build_your_own_voice/s1/04_prepare_conf_files.sh

if test "$#" -ne 1; then
    echo "################################"
    echo "Usage:"
    echo "./04_prepare_conf_files.sh <work_dir>"
    echo ""
    echo "path to global conf file: <work_dir>/conf/global_settings.cfg"
    echo "Config files will be prepared based on settings in global conf file"
    echo "################################"
    exit 1
fi

work_dir=$1
global_config_file=${work_dir}/conf/global_settings.cfg

### Step 4: prepare config files for acoustic, duration models and for synthesis ###
echo "Step 4:"

echo "preparing config files for acoustic, duration models..."
${work_dir}/scripts/prepare_config_files.sh $global_config_file

# echo "preparing config files for synthesis..."
${work_dir}/scripts/prepare_config_files_for_synthesis.sh $global_config_file

