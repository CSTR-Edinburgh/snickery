#!/bin/bash
## Project: SCRIPT
## Author: Cassia Valentini-Botinhao - cvbotinh@inf.ed.ac.uk
## Recipe to train a hybrid TTS system using Merlin (acoustic model) and Snickery (waveform generation; small unit and halfphone variants)

################## General Settings

## Getting line arguments
SNICKERY=$1
MERLIN=$2
WORKDIR=$3

## Tools location - CHANGE HERE!
#SNICKERY=/afs/inf.ed.ac.uk/group/cstr/projects/nst/cvbotinh/SCRIPT/snickery/
#MERLIN=/afs/inf.ed.ac.uk/group/cstr/projects/nst/cvbotinh/SCRIPT/tools/merlin/

## Workdir - CHANGE HERE!
#WORKDIR=/afs/inf.ed.ac.uk/group/cstr/projects/nst/cvbotinh/SCRIPT/hybrid/

## Flags
download_data=true
run_recipe=true
small_model=true # if true trains with 99 files / if false trains with 592

## Getting data
if [ $download_data = true ]; then
	mkdir -p $WORKDIR/data/
	cd $WORKDIR/data/
	wget http://104.131.174.95/slt_arctic_full_data.zip
	unzip slt_arctic_full_data.zip
fi

################## Experiment Settings

## Data
VOICE=slt
WAVTRAIN=${WORKDIR}/data/slt_arctic_full_data/wav/
LABTRAIN=${WORKDIR}/data/slt_arctic_full_data/merlin_baseline_practice/acoustic_data/label_state_align/
LABTEST=${WORKDIR}/data/slt_arctic_full_data/merlin_baseline_practice/acoustic_data/label_phone_align/
QUESTION='radio'
TRAINLIST=${WORKDIR}/train_list.scp
TESTLIST=${WORKDIR}/test_list.scp
if [ $small_model = true ]; then
	TRAINPATTERN='arctic_a00' # 'arctic_a00' 99 files
else
	TRAINPATTERN='arctic_a0' # 'arctic_a0' 592 files
fi
TESTPATTERN='arctic_b000' # 9 files
EXCLUDEPATTERN='arctic_a0461.lab' # this file poses an issue with MagPhase (0 frames for a phone error)
find $LABTRAIN -type f -not -name $EXCLUDEPATTERN -iname $TRAINPATTERN"*.lab" -execdir basename {} .lab ';' > ${TRAINLIST} 
find $LABTRAIN -type f -iname $TESTPATTERN"*.lab"  -execdir basename {} .lab ';' > ${TESTLIST}

## Merlin
MERLINWORKDIR=${WORKDIR}/merlin/
FEATSDIR=${WORKDIR}/feats/
NUMTRAINMERLIN=`wc -l $TRAINLIST | awk '{print $1}'`

## Snickery (general)
TARGETDIR=${MERLINWORKDIR}/experiments/${VOICE}/test_synthesis/gen-feats/
NUMTRAIN=0 # this means use all training sentences
NUMTEST=`wc -l $TESTLIST | awk '{print $1}'`
JOINWEIGHT=0.2
# Snickery - small units
SNICKERYWORKDIR=${WORKDIR}/snickery/
SNICKERYWAV=${WORKDIR}/snickery/synthesized/
NUMEPOCH=6
# Snickery - halfphone
SNICKERYHPWORKDIR=${WORKDIR}/snickery_halfphone/
SNICKERYHPWAV=${WORKDIR}/snickery_halfphone/synthesized/
LABTESTGEN=${MERLINWORKDIR}/experiments/${VOICE}/test_synthesis/gen-lab/ # magphase labels
LABTRAINMP=$LABTRAIN #${MERLINWORKDIR}/experiments/${VOICE}/acoustic_model/data/label_state_align/ # magphase labels
NUMCAND=50

## Create Snickery small unit conf file (based on a default one)
default_cfg=${SNICKERY}/config/hybrid_default.cfg
voice_cfg=${SNICKERY}/config/hybrid_${VOICE}.cfg
cp $default_cfg $voice_cfg
sed -i -- "s|SNICKERYWORKDIR|'$SNICKERYWORKDIR'|g" $voice_cfg
sed -i -- "s|WAV|'$WAVTRAIN'|g" $voice_cfg
sed -i -- "s|FEATS|'$FEATSDIR'|g" $voice_cfg
sed -i -- "s|TRAINLIST|'$TRAINLIST'|g" $voice_cfg
sed -i -- "s|TARGETDIR|'$TARGETDIR'|g" $voice_cfg
sed -i -- "s|NUMTRAIN|$NUMTRAIN|g" $voice_cfg
sed -i -- "s|NUMTEST|$NUMTEST|g" $voice_cfg
sed -i -- "s|TESTPATTERN|'$TESTPATTERN'|g" $voice_cfg
sed -i -- "s|NUMEPOCH|$NUMEPOCH|g" $voice_cfg
sed -i -- "s|JOINWEIGHT|$JOINWEIGHT|g" $voice_cfg

## Create Snickery halfphone conf file (based on a default one)
default_cfg=${SNICKERY}/config/hybrid_halfphone_default.cfg
voice_halfphone_cfg=${SNICKERY}/config/hybrid_halfphone_${VOICE}.cfg
cp $default_cfg $voice_halfphone_cfg
sed -i -- "s|SNICKERYWORKDIR|'$SNICKERYHPWORKDIR'|g" $voice_halfphone_cfg
sed -i -- "s|WAV|'$WAVTRAIN'|g" $voice_halfphone_cfg
sed -i -- "s|FEATS|'$FEATSDIR'|g" $voice_halfphone_cfg
sed -i -- "s|TRAINLIST|'$TRAINLIST'|g" $voice_halfphone_cfg
sed -i -- "s|TARGETDIR|'$TARGETDIR'|g" $voice_halfphone_cfg
sed -i -- "s|LABELDIR|'$LABTRAINMP'|g" $voice_halfphone_cfg
sed -i -- "s|LABELTESTDIR|'$LABTESTGEN'|g" $voice_halfphone_cfg
sed -i -- "s|NUMTRAIN|$NUMTRAIN|g" $voice_halfphone_cfg
sed -i -- "s|NUMTEST|$NUMTEST|g" $voice_halfphone_cfg
sed -i -- "s|TESTPATTERN|'$TESTPATTERN'|g" $voice_halfphone_cfg
sed -i -- "s|NUMEPOCH|$NUMEPOCH|g" $voice_halfphone_cfg
sed -i -- "s|JOINWEIGHT|$JOINWEIGHT|g" $voice_halfphone_cfg
sed -i -- "s|NUMCAND|$NUMCAND|g" $voice_halfphone_cfg
sed -i -- "s|UNTRIM|False|g" $voice_halfphone_cfg

################## Recipe

if [ $run_recipe = true ]; then

	cd $SNICKERY

	# Extract acoustic features using MagPhase: low (for target and join cost) and high (for waveform reconstruction)
	python ./script/extract_magphase_features.py -w ${WAVTRAIN} -o ${FEATSDIR} -ncores 4 -m 60 -p 45 -l ${TRAINLIST}

	## Set up Merlin (copy label files, file lists, conf and scripts)
	bash ./script/merlin/setup_merlin.sh ${MERLINWORKDIR} ${LABTRAIN} ${TRAINLIST} ${SNICKERY} ${VOICE}

	## Train Merlin using labels and low acoustic features
	bash ./script/merlin/train_merlin.sh ${VOICE} ${WAVTRAIN} ${LABTRAIN} ${FEATSDIR} ${MERLINWORKDIR} ${MERLIN} ${QUESTION} ${NUMTRAINMERLIN}

	## Synthesize train sentences using forced aligned state labels
	TXTLIST=$TRAINLIST
	PREDICTDURATION='False'
	LABELSDIR=${MERLINWORKDIR}/experiments/${VOICE}/acoustic_model/data/label_state_align/
	./script/merlin/synthesize_merlin.sh $VOICE $MERLINWORKDIR $MERLIN $TXTLIST $LABELSDIR $PREDICTDURATION

	## Train Snickery small units using synthesized targets, natural join and natural units
	python ${SNICKERY}/script/train_simple.py -c $voice_cfg -X

	# Train Snickery halfphone using synthesized targets, natural join and natural units
	python ${SNICKERY}/script/train_halfphone.py -c $voice_halfphone_cfg -X

	## Synthesize test sentences with Merlin and MagPhase
	TXTLIST=$TESTLIST
	LABELSDIR=${LABTEST}
	PREDICTDURATION='True'
	./script/merlin/synthesize_merlin.sh $VOICE $MERLINWORKDIR $MERLIN $TXTLIST $LABTEST $PREDICTDURATION

	## Synthesize test sentences with Merlin and Snickery small units
	python ${SNICKERY}/script/synth_simple.py -c $voice_cfg -o $SNICKERYWAV

	## Synthesize test sentences with Merlin and Snickery halfphone
	python ${SNICKERY}/script/synth_halfphone.py -c $voice_halfphone_cfg -o $SNICKERYHPWAV

	echo "----- DONE! -----"
	echo "To play synthesized TTS waveforms (MagPhase, Snickery small units, Snickery halfphone): "
	echo play \{${MERLINWORKDIR}/experiments/${VOICE}/test_synthesis/wav_wav_pf_magphase/,$SNICKERYWAV,$SNICKERYHPWAV\}/arctic_b0001.wav

fi
