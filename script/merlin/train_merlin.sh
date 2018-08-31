#!/bin/bash
## Project: SCRIPT
## Author: Cassia Valentini-Botinhao - cvbotinh@inf.ed.ac.uk

if test "$#" -ne 8; then
    echo "################################"
    echo "Usage:"
    echo "./train_merlin.sh <voice_name> <wav_dir> <txt_dir> <feats_dir> <work_dir> <merlin_dir> <accent> <nutts>"
    echo ""
    echo "################################"
    exit 1
fi

voice_name=$1
wav_dir=$2
txt_dir=$3
feats_dir=$4
work_dir=$5
merlin_dir=$6
accent=$7
nutts=$8

# Set question file
if [ $accent == 'edi' ] # Scottish accent
then
    echo "Using Scottish Combilex"
    question_file='questions-combilex-edi_dnn_601.hed'
elif [ $accent == 'rpx' ] # English accent
then
    echo "Using English RPX Combilex"
    question_file='questions-combilex-rpx_dnn_601.hed'
else
    echo "Using English radio question file"
    question_file='questions-radio_dnn_416.hed'
fi

#### Setting number of sentences to train/dev/test
if [ $voice_name == 'Nick' ] # SLT 2018 experiments
then
    nutts_train=1943
    nutts_dev=60
    nutts_test=1
elif [ $voice_name == 'Alba' ] # SLT 2018 experiments
then
    nutts_train=4217
    nutts_dev=100
    nutts_test=1
elif [ $voice_name == 'slt' ]
then
    if [ $nutts -eq 592 ]; then
        nutts_train=570
        nutts_dev=20
        nutts_test=2
    elif [ $nutts -eq 99 ]; then
        nutts_train=90
        nutts_dev=5
        nutts_test=4
    else
        echo "ERROR: Unknown number of total sentences, please set the number of train, dev and test utterances in train_merlin.sh !"
        exit
    fi
else
    echo "ERROR: Unknown voice, please set the number of train, dev and test utterances in train_merlin.sh !"
    exit
fi

# step 1: run setup
./script/merlin/01_setup.sh $voice_name $nutts_train $nutts_dev $nutts_test ${question_file} ${work_dir} ${merlin_dir}

### This step is done with VCTK
# step 2: prepare labels -- in ${work_dir}/database/labels/
./script/merlin/02_prepare_labels.sh ${wav_dir} ${txt_dir} ${work_dir}

# step 3: part 1 - feat extraction / copy from snickery feats low dimension
cp $feats_dir/low/{real,imag,mag,lf0}/* ${work_dir}/experiments/${voice_name}/acoustic_model/data/
cp $feats_dir/shift/* ${work_dir}/experiments/${voice_name}/acoustic_model/data/

# step 3: part 2 -- correct state align labels according to extracted accoustic features
./script/merlin/03_correct_labels.sh ${work_dir}

# step 4: prepare config files for training and testing
./script/merlin/04_prepare_conf_files.sh ${work_dir}

# step 5: train duration model
./script/merlin/05_train_duration_model.sh ${work_dir} ${work_dir}/conf/duration_${voice_name}.conf

# step 6: train acoustic model
./script/merlin/06_train_acoustic_model.sh ${work_dir} ${work_dir}/conf/acoustic_${voice_name}.conf

echo "done...!"
