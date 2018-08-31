#!/bin/bash
## Project: SCRIPT
## Author: Cassia Valentini-Botinhao - cvbotinh@inf.ed.ac.uk

if test "$#" -ne 1; then
    echo "################################"
    echo "Usage:"
    echo "./03_correct_labels.sh <work_dir>"
    echo ""
    echo "the path to lab dir is: <work_dir>/database/labels"
    echo "################################"
    exit 1
fi

work_dir=$1
lab_dir=${work_dir}/database/labels/

global_config_file=${work_dir}/conf/global_settings.cfg
source $global_config_file

# If MAGPHASE and variable length then modify state align labels here
# Modify for MAGPHASE
if [ ${Vocoder} == "MAGPHASE" ]
then

	# Modify label to deal with variable lenght frames
	python ${work_dir}/scripts/label_st_align_to_var_rate.py ${lab_dir}/${FileIDList} ${lab_dir}/label_state_align ${work_dir}/experiments/$Voice/acoustic_model/data/ ${SamplingFreq} ${lab_dir}/label_state_align_var_rate/

	# Copy new labels to experiment directory
	rm -f ${work_dir}/experiments/${Voice}/acoustic_model/data/label_state_align/*
	rm -f ${work_dir}/experiments/${Voice}/duration_model/data/label_state_align/*

	mkdir -p ${work_dir}/experiments/${Voice}/acoustic_model/data/label_state_align/
	mkdir -p ${work_dir}/experiments/${Voice}/duration_model/data/label_state_align/

	cp ${lab_dir}/label_state_align_var_rate/* ${work_dir}/experiments/${Voice}/acoustic_model/data/label_state_align/
	cp ${lab_dir}/label_state_align_var_rate/* ${work_dir}/experiments/${Voice}/duration_model/data/label_state_align/

fi
