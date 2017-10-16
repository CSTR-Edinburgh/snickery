#!/bin/bash



## Location of this script:-
SCRIPTPATH=$( cd $(dirname $0) ; pwd -P )

## load config values:
source ${SCRIPTPATH}/data_config.txt


SRATE=48000




LABDIR='unused'

if [ $# -eq 2 ] ; then
    INFILE=$1
    OUTDIR=$2
elif [ $# -eq 3 ]; then
    INFILE=$1
    LABDIR=$2
    OUTDIR=$3
else
    echo "Wrong number of arguments supplied";
    exit 1 ;
fi




WOUTDIR=$OUTDIR/world_reaper

mkdir -p $WOUTDIR $OUTDIR/pitch_sync

mkdir -p $WOUTDIR/tmp $WOUTDIR/f0 $WOUTDIR/ap $WOUTDIR/mgc $WOUTDIR/pm $WOUTDIR/f0_reaper $WOUTDIR/mfcc

echo 'sox...'
### ------ SOX ------
BASE=`basename $INFILE .wav` ;  


if [ -e $WOUTDIR/pm/$BASE.pm ]; then
    echo "Looks like file already processed -- skip it" ;
    exit 0 ;
fi


$SOX $INFILE -r $SRATE -c 1 -b 16 $WOUTDIR/tmp/$BASE.wav


echo 'reaper...'
### ------ REAPER ------
    
$REAPER -i $WOUTDIR/tmp/$BASE.wav -p $WOUTDIR/pm/$BASE.pm -f $WOUTDIR/f0_reaper/$BASE.f0_reaper -u 0.005 -a 


echo 'world...'
### ------ WORLD ------
$WORLD_A $WOUTDIR/tmp/$BASE.wav $WOUTDIR/tmp/$BASE.f0.d $WOUTDIR/tmp/$BASE.sp $WOUTDIR/tmp/$BASE.ap.d

$SPTK/x2x +df $WOUTDIR/tmp/$BASE.f0.d > $WOUTDIR/f0/$BASE.f0
$SPTK/x2x +df $WOUTDIR/tmp/$BASE.ap.d > $WOUTDIR/ap/$BASE.ap
    
$SPTK/x2x +df $WOUTDIR/tmp/$BASE.sp | $SPTK/sopr -R -m 32768.0 | $SPTK/mcep -a 0.77 -m 59 -l 2048 -e 1.0E-8 -j 0 -f 0.0 -q 3 > $WOUTDIR/mgc/$BASE.mgc


### ------ [test world resynth...] ------
# $SPTK/mgc2sp -a 0.77 -g 0 -m 59 -l 2048 -o 2 $WOUTDIR/$BASE.mgc | $SPTK/sopr -d 32768.0 -P | $SPTK/x2x +fd > $WOUTDIR/$BASE.resyn.sp
#     
# $WORLD_S 2048 $SRATE $WOUTDIR/$BASE.f0.d $WOUTDIR/$BASE.resyn.sp $WOUTDIR/$BASE.ap $WOUTDIR/$BASE.resyn.wav


echo 'mfcc...'
### ----- Multisyn-style HTK MFCCs ------

MFCCFILE=$WOUTDIR/mfcc/$BASE.mfcc

## Convert wav to NIST and resample to 16000Hz
${CH_WAVE} -otype nist -F 16000 -o ${MFCCFILE}.wav $WOUTDIR/tmp/$BASE.wav > ${MFCCFILE}.log

## Add a little random noise -- too many 0s in a row leads to:
## WARNING [-7324]  StepBack: Bad data or over pruning
## Probably not necessary for join cost only?
${SOX} ${MFCCFILE}.wav ${MFCCFILE}.dithered.nist dither 

## Unlike Multisyn, store uncompressed, with no checksum, and natural endianness
cat > ${MFCCFILE}.conf<<EOF
    NATURALREADORDER = T
    NATURALWRITEORDER = T
    TARGETRATE = 20000.0
    TARGETKIND = MFCC_E
    SOURCEFORMAT = NIST
    ENORMALISE = F
    SAVECOMPRESSED = F
    SOURCEKIND = WAVEFORM
    SAVEWITHCRC = F
    USEHAMMING = T
    WINDOWSIZE = 100000.0
    CEPLIFTER = 22
    NUMCHANS = 26
    NUMCEPS = 12
    PREEMCOEF = 0.97
EOF

${HCOPY} -T 1 -C ${MFCCFILE}.conf ${MFCCFILE}.dithered.nist ${MFCCFILE} >> ${MFCCFILE}.log
rm ${MFCCFILE}.*



### ----- resample pitch synchronously ------
# if [ $LABDIR == 'unused' ] ; then
#     #$SCRIPTPATH/ps_resample.py -w $WOUTDIR/tmp/$BASE.wav -d $OUTDIR/world_reaper -o $OUTDIR/pitch_sync
#     echo 'resample 1...'
#     $SCRIPTPATH/ps_resample.py -w $WOUTDIR/tmp/$BASE.wav -p $OUTDIR/world_reaper/pm -f $OUTDIR/world_reaper/mgc -x mgc -d 60 -o $OUTDIR/pitch_sync/mgc
#     echo 'resample 2...'    
#     $SCRIPTPATH/ps_resample.py -w $WOUTDIR/tmp/$BASE.wav -p $OUTDIR/world_reaper/pm -f $OUTDIR/world_reaper/ap  -x ap  -d 5  -o $OUTDIR/pitch_sync/ap
#     echo 'resample 3...'    
#     $SCRIPTPATH/ps_resample.py -w $WOUTDIR/tmp/$BASE.wav -p $OUTDIR/world_reaper/pm -f $OUTDIR/world_reaper/f0  -x f0  -d 1  -o $OUTDIR/pitch_sync/f0
#     echo 'resample 4'
#     $SCRIPTPATH/ps_resample.py -w $WOUTDIR/tmp/$BASE.wav -p $OUTDIR/world_reaper/pm -f $OUTDIR/world_reaper/mfcc -x mfcc -d 13 -o $OUTDIR/pitch_sync/mfcc -s 0.002

# else
#     $SCRIPTPATH/ps_resample.py -w $WOUTDIR/tmp/$BASE.wav -d $OUTDIR/world_reaper -o $OUTDIR/pitch_sync -l $LABDIR
# fi


echo 'done'
