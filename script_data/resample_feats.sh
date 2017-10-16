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


### ------ SOX ------
BASE=`basename $INFILE .wav` ;  


if [ -e $OUTDIR/pitch_sync/mgc/$BASE.mgc ]; then
    echo $BASE
    echo "Looks like file already processed -- skip it" ;
    exit 0 ;
fi




### ----- resample pitch synchronously ------
if [ $LABDIR == 'unused' ] ; then
    #$SCRIPTPATH/ps_resample.py -w $WOUTDIR/tmp/$BASE.wav -d $OUTDIR/world_reaper -o $OUTDIR/pitch_sync
    echo 'resample 1...'
    $SCRIPTPATH/ps_resample.py -w $WOUTDIR/tmp/$BASE.wav -p $OUTDIR/world_reaper/pm -f $OUTDIR/world_reaper/mgc -x mgc -d 60 -o $OUTDIR/pitch_sync/mgc
    echo 'resample 2...'    
    $SCRIPTPATH/ps_resample.py -w $WOUTDIR/tmp/$BASE.wav -p $OUTDIR/world_reaper/pm -f $OUTDIR/world_reaper/ap  -x ap  -d 5  -o $OUTDIR/pitch_sync/ap
    echo 'resample 3...'    
    $SCRIPTPATH/ps_resample.py -w $WOUTDIR/tmp/$BASE.wav -p $OUTDIR/world_reaper/pm -f $OUTDIR/world_reaper/f0  -x f0  -d 1  -o $OUTDIR/pitch_sync/f0
    echo 'resample 4'
    $SCRIPTPATH/ps_resample.py -w $WOUTDIR/tmp/$BASE.wav -p $OUTDIR/world_reaper/pm -f $OUTDIR/world_reaper/mfcc -x mfcc -d 13 -o $OUTDIR/pitch_sync/mfcc -s 0.002

else
    $SCRIPTPATH/ps_resample.py -w $WOUTDIR/tmp/$BASE.wav -d $OUTDIR/world_reaper -o $OUTDIR/pitch_sync -l $LABDIR
fi


echo 'done'
