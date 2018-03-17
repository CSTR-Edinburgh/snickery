# Usage:
# ./smooth_features.sh input_dir output_dir vocoder temporal_scaling variance_scaling file_list resynth
# For magphase:
# ./smooth_features.sh /group/project/cstr2/cvbotinh/SCRIPT/Nick/feats/magphase/low/ /afs/inf.ed.ac.uk/group/cstr/projects/nst/cvbotinh/SCRIPT/Smoothing/magphase/smoothed/ magphase 5 0.8 file_list.txt 1
# For world:
# ./smooth_features.sh /group/project/cstr2/cvbotinh/SCRIPT/Nick/feats/world/ /afs/inf.ed.ac.uk/group/cstr/projects/nst/cvbotinh/SCRIPT/Smoothing/WORLD/smoothed/ world 5 0.8 file_list.txt 1

indir=$1
outdir=$2
vocoder=$3
ts=$4
vs=$5
file_list=$6
resyn=$7

outdir=${outdir}'/'ts${ts}_vs${vs}'/'
num_files=($(wc $file_list))
num_files=${num_files[0]}

MerlinDir='/afs/inf.ed.ac.uk/group/cstr/projects/nst/cvbotinh/SCRIPT/tools/merlin/'
SamplingFreq=48000

echo "--- Number of files to smooth: " ${num_files}

if [ $vocoder == 'world' ]
then

	echo "--- Smoothing WORLD vocoder features"

	# Smooth features
	python ../script_data/smooth_data.py -f ${indir}'/mgc/'  -o ${outdir}'/mgc/'  -m 60 -t mgc  -w ${ts} -s ${vs} -l ${file_list}
	python ../script_data/smooth_data.py -f ${indir}'/bap/'  -o ${outdir}'/bap/'  -m 5  -t bap  -w ${ts} -s ${vs} -l ${file_list}
	python ../script_data/smooth_data.py -f ${indir}'/lf0/'  -o ${outdir}'/lf0/'  -m 1  -t lf0  -w ${ts} -s ${vs} -l ${file_list}

	# Resynthesize
	if [ $resyn == '1' ]
	then
		echo "--- Synthesizing waveforms"
		python ${MerlinDir}/misc/scripts/vocoder/${vocoder}/synthesis.py ${MerlinDir} ${outdir} ${outdir}'/resyn/' ${SamplingFreq} ${file_list}
	fi

elif [ $vocoder == 'magphase' ]
then

	echo "--- Smoothing MAGPHASE vocoder features"

	# Smooth features
	python ../script_data/smooth_data.py -f ${indir}'/mag/'  -o ${outdir}'/mag/'  -m 60 -t mag  -w ${ts} -s ${vs} -l ${file_list}
	python ../script_data/smooth_data.py -f ${indir}'/real/' -o ${outdir}'/real/' -m 45 -t real -w ${ts} -s ${vs} -l ${file_list}
	python ../script_data/smooth_data.py -f ${indir}'/imag/' -o ${outdir}'/imag/' -m 45 -t imag -w ${ts} -s ${vs} -l ${file_list}
	python ../script_data/smooth_data.py -f ${indir}'/lf0/'  -o ${outdir}'/lf0/'  -m 1  -t lf0  -w ${ts} -s ${vs} -l ${file_list}

	# Resynthesize
	if [ $resyn == '1' ]
	then
                echo "--- Synthesizing waveforms"
		python resynthesise_magphase.py -f ${outdir} -o ${outdir}'/resyn/' -N ${num_files} -m 60 -p 45 -fftlen 2048 -ncores 0 -fs ${SamplingFreq}
	fi

else 
	echo "Vocoder option not supported. Options are: world and magphase"

fi
