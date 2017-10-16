#!/bin/bash



## Location of this script:-
SCRIPTPATH=$( cd $(dirname $0) ; pwd -P )

## load config values:
source ${SCRIPTPATH}/data_config.txt



LABDIR=''

if [ $# -eq 2 ] ; then
    INDIR=$1
    OUTDIR=$2
elif [ $# -eq 3 ]; then
    INDIR=$1
    LABDIR=$2
    OUTDIR=$3
else
    echo "Wrong number of arguments supplied";
    exit 1 ;
fi



mkdir -p $OUTDIR

## just extract a few files for debugging:
if [ $N_FILES_TO_EXTRACT -gt 0 ] ; then
    ls $INDIR | head -${N_FILES_TO_EXTRACT} > $OUTDIR/flist.txt
else
    ## Extract all files for real:
    ls $INDIR > $OUTDIR/flist.txt
fi

$SCRIPTPATH/parallel -j $N_PARALLEL_JOBS -a $OUTDIR/flist.txt $SCRIPTPATH/extract_feats.sh $INDIR/{} $LABDIR $OUTDIR/  





