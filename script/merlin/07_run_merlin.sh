#!/bin/bash -e
## Project: SCRIPT
## Author: Cassia Valentini-Botinhao - cvbotinh@inf.ed.ac.uk
## Based on: https://github.com/CSTR-Edinburgh/merlin/tree/master/egs/build_your_own_voice/s1/07_run_merlin.sh

if test "$#" -lt 4; then
    echo "################################"
    echo "Usage: "
    echo "./07_run_merlin.sh <work_dir> <path_to_text_dir> <path_to_test_dur_conf_file> <path_to_test_synth_conf_file> <genlab_dir> <predict_duration_flag>"
    echo ""
    echo "################################"
    exit 1
fi

work_dir=$1
inp_txt=$2
test_dur_config_file=$3
test_synth_config_file=$4
genlab_dir=$5
predict_duration=$6

global_config_file=${work_dir}/conf/global_settings.cfg
source $global_config_file

### Step 7: synthesize speech from text ###
echo "Step 7:" 
echo "synthesizing speech from labels..."

mkdir -p ${work_dir}/experiments/${Voice}/test_synthesis/txt/
mkdir -p ${work_dir}/experiments/${Voice}/test_synthesis/prompt-lab/
mkdir -p ${work_dir}/experiments/${Voice}/test_synthesis/gen-lab/

echo $genlab_dir
echo $predict_duration

if [ $predict_duration == 'False' ]; then

	echo "copying forced aligned state labels..."
	for file in `cat ${work_dir}/experiments/${Voice}/test_synthesis/test_id_list.scp`
	do
		cp $genlab_dir/$file.lab ${work_dir}/experiments/${Voice}/test_synthesis/gen-lab/
	done

else

	echo "copying labels..."
	for file in `cat ${work_dir}/experiments/${Voice}/test_synthesis/test_id_list.scp`
	do
		awk '{print $3}' ${genlab_dir}/$file.lab > ${work_dir}/experiments/${Voice}/test_synthesis/prompt-lab/$file.lab
	done

	echo "synthesizing durations..."
	${work_dir}/scripts/submit.sh ${MerlinDir}/src/run_merlin.py $test_dur_config_file
fi

echo "synthesizing speech..."
${work_dir}/scripts/submit.sh ${MerlinDir}/src/run_merlin.py $test_synth_config_file

echo "deleting intermediate synthesis files..."
${work_dir}/scripts/remove_intermediate_files.sh $global_config_file

echo "synthesized audio files are in: ${work_dir}experiments/${Voice}/test_synthesis/wav"
