#!/bin/bash
## Project: SCRIPT
## Author: Cassia Valentini-Botinhao - cvbotinh@inf.ed.ac.uk

MERLINWORKDIR=$1 
LAB=$2 
TRAINLIST=$3
SNICKERY=$4
voice=$5

# Create working directory
mkdir -p ${MERLINWORKDIR}/database/labels/label_state_align/
mkdir -p ${MERLINWORKDIR}/experiments/${voice}/duration_model/data/
mkdir -p ${MERLINWORKDIR}/experiments/${voice}/acoustic_model/data/

# Copy state label files, train and test lists
for file in `cat $TRAINLIST`
do
	cp $LAB'/'$file.lab ${MERLINWORKDIR}/database/labels/label_state_align/
done
cp $TRAINLIST ${MERLINWORKDIR}/database/labels/file_id_list.scp
cp $TRAINLIST ${MERLINWORKDIR}/experiments/${voice}/duration_model/data/file_id_list.scp 
cp $TRAINLIST ${MERLINWORKDIR}/experiments/${voice}/acoustic_model/data/file_id_list.scp 

# Copy conf and script directories from Snickery to Merlin work directory
cp -r ./script/merlin/conf ${MERLINWORKDIR}/
cp -r ./script/merlin/scripts ${MERLINWORKDIR}/

# Change location of Merlin MagPhase to Snickery Magphase
sed -i s#'MAGPHASE'#'"'${SNICKERY}/tool/magphase/src/'"'# ${MERLINWORKDIR}/scripts/label_st_align_to_var_rate.py
