#!/bin/bash
## Project: SCRIPT
## Author: Cassia Valentini-Botinhao - cvbotinh@inf.ed.ac.uk

if test "$#" -lt 3; then
    echo "################################"
    echo "Usage:"
    echo "./synthesize_merlin.sh <voice_name> <work_dir> <merlin_dir> <txtlist>(optional) <labelsdir>(optional)"
    echo ""
    echo "The text_dir location is: <work_dir>/experiments/<voice_name>/test_synthesis/txt/"
    echo ""
    echo "################################"
    exit 1
fi

voice_name=$1
work_dir=$2
merlin_dir=$3

if test "$#" -gt 3; then
    cp $4 ${work_dir}/experiments/${voice_name}/test_synthesis/test_id_list.scp
    if test "$#" -gt 4; then
        labdir=$5
        predict_duration=$6
    else
        labdir=''
        predict_duration=''
    fi
else
    txt_dir=${work_dir}/experiments/${voice_name}/test_synthesis/txt/
    # Create list of files to synthesize
    basename --suffix=.txt -- ${txt_dir}/* > ${work_dir}/experiments/${voice_name}/test_synthesis/test_id_list.scp
fi

./script/merlin/07_run_merlin.sh ${work_dir} ${work_dir}/experiments/${voice_name}/test_synthesis/txt ${work_dir}/conf/test_dur_synth_${voice_name}.conf ${work_dir}/conf/test_synth_${voice_name}.conf $labdir $predict_duration

# Create directory structure for generated parameters
for feat in mag imag real lf0
do
    feat_dir=${work_dir}/experiments/${voice_name}/test_synthesis/gen-feats/$feat'/'
    echo "Copy generated features to $feat_dir"
    mkdir -p $feat_dir
    cp ${work_dir}/experiments/${voice_name}/test_synthesis/wav/*.$feat $feat_dir
done

# Fix mag feature during silence
feat='mag'
feat_dir=${work_dir}/experiments/${voice_name}/test_synthesis/gen-feats/$feat'/'
feat_dir_fix=${work_dir}/experiments/${voice_name}/test_synthesis/gen-feats/$feat'_fixed/'
python ./script/merlin/fix_gen_mag.py $feat_dir $feat_dir_fix
rm -r $feat_dir
mv $feat_dir_fix $feat_dir
