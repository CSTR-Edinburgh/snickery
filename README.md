## Setting up

Clone repository:

```
git clone https://github.com/oliverwatts/snickery.git
cd snickery
```

### Installation of Python dependencies with virtual environment

Make a directory to house vitrual environments if you don't already have one, and move to it:

```
cd ~/tool/virtual_python/
virtualenv --distribute --python=/usr/bin/python2.7 hybrid_synthesiser
source ./hybrid_synthesiser/bin/activate
```

On my machine, my prompt has now turned from ```[salton]owatts:``` to  ```(hybrid_synthesiser)[salton]owatts:```. With the virtual environment activated, you can now install the necessary packages:

```
pip install numpy
pip install scipy   ## required by sklearn
pip install h5py
pip install sklearn
```



### Local install of OpenFST binaries & OpenFST Python bindings

Try this one first, this definitely worked for me on Linux (DICE):-

<!-- 
## Oliver:
export MY_OPENFST_DIR=/afs/inf.ed.ac.uk/user/o/owatts/tool/openfst_for_hybrid
 -->

```
## Make location for downloading and compiling OpenFST
export MY_OPENFST_DIR=/your/chosen/location
mkdir $MY_OPENFST_DIR
cd $MY_OPENFST_DIR
wget http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.5.4.tar.gz
tar xvf openfst-1.5.4.tar.gz
cd openfst-1.5.4/
./configure --prefix=${MY_OPENFST_DIR} --enable-far --enable-mpdt --enable-pdt
make
make install  

## While still in virtual environment as above, install Python bindings (pywrapfst module) like this:
pip install --global-option=build_ext  --global-option="-I${MY_OPENFST_DIR}/include" --global-option="-L${MY_OPENFST_DIR}/lib" --global-option="-R${MY_OPENFST_DIR}/lib" openfst==1.5.4
```



### System-wide install OpenFST binaries & OpenFST Python bindings 

I think I needed to do this to get the tools compiled on Mac (please let me know if the above did in fact work OK for you):

```
## Make location for downloading and compiling OpenFST
export MY_OPENFST_DIR=/your/chosen/location
mkdir $MY_OPENFST_DIR
cd $MY_OPENFST_DIR
wget http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.5.4.tar.gz
tar xvf openfst-1.5.4.tar.gz
cd openfst-1.5.4/
./configure --enable-far --enable-mpdt --enable-pdt
make
sudo make install   # sudo for system-wide install to /usr/local/bin/ etc.

## While still in virtual environment as above, install Python bindings (pywrapfst module) like this:
pip install openfst==1.5.4
```


### Tools for feature extraction

Get GNU parallel to speed up feature extraction:

```
cd script_data/
wget http://ftp.gnu.org/gnu/parallel/parallel-20170922.tar.bz2
tar xvf parallel-20170922.tar.bz2
mv parallel-20170922/src/parallel .
rm -r parallel-20170922*
```

TODO: add notes on obtaining and compiling World & Reaper & other necessary things. For now, look at  `script_data/data_config.txt` to see the dependencies.

  
<!-- # 
# ### World and Reaper:
# # zip from https://github.com/CSTR-Edinburgh/merlin
# 
# cd ~/tool/merlin-master/
#   503  cd tools/WORLD_v2/
#   504  ls
#   505  more makefile 
#   506  make
#   507  make test
#   508  ls
#   511  less makefile 
#   512  make analysis synth
#   

# 
# cd ~/tool/merlin-master/
#   503  cd tools/WORLD/
#   504  ls
#   505  more makefile 
#   506  make
#   507  make test
#   512  make analysis synth
#   
#   
#   
#   # https://github.com/google/REAPER
#   
#   cd /afs/inf.ed.ac.uk/user/o/owatts/tool/REAPER-master
# mkdir build   # In the REAPER top-level directory
# cd build
# cmake ..
# make
# 
 -->



Extract  data -- edit `script_data/data_config.txt` and, using 30 cores on CSTR server zamora:

```
./script_data/extract_feats_parallel.sh /afs/inf.ed.ac.uk/group/cstr/projects/blizzard_entries/blizzard2017/data/segmented/wav/ /afs/inf.ed.ac.uk/group/cstr/projects/nst/oliver/hybrid_work/data/fls_data
```


... then using 4 cores on my DICE desktop salton:


```
./script_data/resample_feats_parallel.sh /afs/inf.ed.ac.uk/group/cstr/projects/blizzard_entries/blizzard2017/data/segmented/wav/ /afs/inf.ed.ac.uk/group/cstr/projects/nst/oliver/hybrid_work/data/fls_data
```

Note: initially, extraction and resampling were done with a single script. When using many cores on CSTR servers, the Python code for resampling really slowed things down. I separated the code until I have chance to debug this properly.

As a final step, split MFCCs into energy and 12 others. Also, check the data for some outlying values
like -10000000000.0 and -5000000000.0 which make standardisation of the data crazy. Do this crudely by setting values outside the range [-100, 100] to 0. These steps should probably be merged into the other scripts):

```
 python ./script_data/split_mfccs.py /afs/inf.ed.ac.uk/group/cstr/projects/nst/oliver/hybrid_work/data/fls_data/pitch_sync/
 ```

## Running the tools

### Example configuration files 

An example config file can be found in `./config`. This uses data prepared for the [CSTR entry to the Blizzard Challenge 2017](http://festvox.org/blizzard/bc2017/CSTR_Blizzard2017.pdf) with the intention of replicating those results.

If you have read access to the Blizzard data, you should be able to train and synthesise by changing only the first two values in the file:
- `openfst_bindir`: should point to the directory of OpenFst binaries you compiled above (`$MY_OPENFST_DIR/bin`)
- `workdir`: trained voices and synthesised speech will end up under here


### Training (=writing the database to the correct format)

The following directories mentioned in the config file give the locations of inputs required for training the system:
- `wav_datadir`: 48kHz, mono, 16-bit wav-headered audio for the training database .
- `pm_datadir`: pitchmark (glottal closure instance) files, in Edinburgh Speech Tools ASCII format (as output by [Reaper](https://github.com/google/REAPER)) 
- `join_datadir`: acoustic features which will be used to compute join cost. 
- `target_datadir`: as `join_datadir`, but for target cost.
- `label_datadir`: HTK/HTS format labels including state alignment. 

Both acoustic feature directories `join_datadir` and `target_datadir` contain one or more subdirectories, each containing files for a single stream of data.  In the examples given, `join_datadir` points to natural paramters extracted directly from speech, and `target_datadir` points to parameters for utterances in the training database which have been resynthesised from a model trained on that database, using natural durations. In principle, they can point to the same data, or entirely different data (consiting of different streams).

Train like this:

```
python script/train_halfphone.py -c ./config/blizzard_replication_03.cfg
```

Output is placed under the directory configured as `workdir`, in a subdirectory recording notable train-time hyperparameters, so that multiple runs can be done by editing a single config file, without overwriting previous results.


### Weight balancing

The config values `join_stream_weights` and `target_stream_weights` are used to scale join and target feature streams and so control their degree of influence on unit sequences selected. They can be set manually, however, a good starting point for adjusting them is the assumption that all the streams used for a subcost should have equal influence. Weights which lead to on average equal influence can be found be an iterative procedure on some held-out material (labels and predictions only -- no natural acoustics are needed, and so the set can be as large as desired). Make the following call to find appropriate stream weights:

```
python ./script/balance_stream_weights.py -c ./config/blizzard_replication_03.cfg
```

Weights will be printed to the terminal, and can be pasted into the config file before synthesis is done.


### Synthesis 

These directories give the locations of inputs required for synthesis with the system:
- `test_data_dir`: streams of data comparable to those in `target_datadir`, to be used for computing target cost at run time.
- `test_lab_dir`: labels files of same format as those in `label_datadir`.

Synthesise like this:

```
python script/synth_halfphone.py -c ./config/blizzard_replication_03.cfg
```

Again, output is placed under the directory configured as `workdir`, in a subdirectory recording notable synthesis_time hyperparameters, so that multiple synthesis runs can be done by editing the synthesis-time variables of a single config file, without overwriting previous results.

If this runs OK, and produces (lousy sounding) speech, you are ready to run on the full database. Change the value of `n_train_utts` from 50 to 0 in the config, and run training and synthesis again. Here, 0 means use all training sentences.

Check that the speech synthesised when using all the data is comparable to that found here:

```
https://www.dropbox.com/sh/lifsl831u0clyc3/AABC8XVLuhTL-ZY7PoyFN3M5a?dl=0
```

Quality is still not optimal, as none of the parameters in the config (in particular, the weights) have been tuned, and remain at pretty much arbitrary initial settings. 

<!-- 
## this can be any filename substring, selecting a portion of the data ('hvd') or a single file ('AMidsummerNightsDream_001_016')
test_patterns = ['PirateAdventures_00001_00010']
 -->




## Selection of single pitch epochs

The ability to use single pitch epochs (in fact, units consisting of 2 epochs centred on a glottal closure instant and windowed) has been added to `train_halfphone.py` and `synth_halfphone.py`; these scripts should now be renamed...

The example config `config/blizzard_replication_04_epoch.cfg` gives some clues on how to use it. Currently I've only tested using natural parameters for target. `target_representation` should be set to `epoch`.

To do:

- check overlap add code. Consider alternative approaches to windowing and placement. 
- experiment with phase-aware features
- add code to pitch synchronise fixed frame rate targets -- in the example config, the natural parameters are already pitch synchronous.
- store search tree instead of building from scratch each time we synthesise
- preselection is entirely acoustic -- add phonetic constraints or features back in?
- check speeds for building vs. loading KD trees, and effect of `leaf_size` parameter


And also:

- stream balancing (haven't tried yet)
- are separate join and target costs even needed?
- is full search even needed? Greedy methods?
- what should active learning try to tune in this set up?


## Selection of single samples

An example config is given at `config/micro_test.cfg`. To train and synthesise:

```
python ./script/train_halfphone.py -c config/micro_test.cfg
python ./script/synth_sample.py -c config/micro_test.cfg
```



## Install magphase


```
cd ./snickery/
mkdir tool
cd tool/
git clone https://github.com/CSTR-Edinburgh/magphase.git
cd magphase/
```

Edit `config.ini` to point to existing REAPER and SPTK installations, e.g.:

```
[TOOLS]
reaper=/afs/inf.ed.ac.uk/user/o/owatts/tool/REAPER-master/build/reaper
sptk_mcep=/afs/inf.ed.ac.uk/user/o/owatts/repos/dnn_swahili/dnn_tts/tools/SPTK-3.7/bin/mcep
```

or :

```
[TOOLS]
reaper=/Users/owatts/tool/reaper/REAPER/build/reaper
sptk_mcep=/Users/owatts/repos/simple4all/CSTRVoiceClone/trunk/bin/mcep

```


Modified magphase.py a little -- add to repo?


Also reaper:

def reaper(in_wav_file, out_est_file):
    print("Extracting epochs with REAPER...")
    global _reaper_bin
#    cmd =  _reaper_bin + " -s -x 400 -m 50 -a -u 0.005 -i %s -p %s" % (in_wav_file, out_est_file)
    cmd =  _reaper_bin + " -s -a -u 0.005 -i %s -p %s" % (in_wav_file, out_est_file)











python make_wave_patch_features.py -w /afs/inf.ed.ac.uk/group/cstr/projects/blizzard_entries/blizzard2017/data/segmented/wav/ -p /afs/inf.ed.ac.uk/group/cstr/projects/nst/oliver/hybrid_work/data/fls_data/world_reaper/pm/ -o /afs/inf.ed.ac.uk/group/cstr/projects/nst/oliver/hybrid_work/data/fls_data/pitch_sync/ -d 31 -r 8000 -warp mu
    



(hybrid_synthesiser)[zamora]owatts: python script/extract_magphse_oliver.py -w /afs/inf.ed.ac.uk/group/cstr/projects/blizzard_entries/blizzard2017/data/segmented/wav/ -o /afs/inf.ed.ac.uk/group/cstr/projects/nst/oliver/hybrid_work/data/fls_data/magphase_lo_hi -ncores 30


(hybrid_synthesiser)[zamora]owatts: python script/extract_magphse_oliver.py -w /afs/inf.ed.ac.uk/group/cstr/projects/blizzard_entries/blizzard2017/data/segmented/wav/ -o /afs/inf.ed.ac.uk/group/cstr/projects/nst/oliver/hybrid_work/data/fls_data/subset_magphase_lo_hi -N 100 -ncores 30




NICK

WAV=/Users/owatts/repos/ossian_git_gold/Ossian/corpus/en/speakers/nick/wav

MacBook-Air:snickery owatts$ python ./script/extract_magphase_features.py -w $WAV -o ~/working/hybrid/nicktest/data -N 20 -ncores 4

## TODO: cp f0 to full

        HALFFFTLEN = 513  ## TODO


### disables:
add taper (weighting for cross-fade):

python ./script/train_halfphone.py -c ./config/nick_01.cfg  -X




TODO:

train_halfphone: ADD_PHONETIC_EPOCH




sep stream contributions

```
import numpy as np

a = np.random.uniform(0,100, size=(30,3))
b = np.random.uniform(0,100, size=(30,3))
c = np.random.uniform(0,100, size=(30,3))
d = np.random.uniform(0,100, size=(30,3))

ascore = ((a - b) * (a - b)).sum(axis=1)
cscore = ((c - d) * (c - d)).sum(axis=1)

ac = np.hstack([a,c])
bd = np.hstack([b,d])
acscore = ((ac - bd) * (ac - bd)).sum(axis=1)

print acscore
print ascore + cscore
print ascore * cscore
```


## Setup and experiments for IS2018

### Setup tools

```
cd /afs/inf.ed.ac.uk/user/o/owatts/proj/slm-local/

git clone https://github.com/oliverwatts/snickery.git
cd snickery

mkdir tool
cd tool/
git clone https://github.com/CSTR-Edinburgh/magphase.git
```

Edit `config.ini` to point to existing REAPER and SPTK installations, e.g.:

```
[TOOLS]
reaper=/afs/inf.ed.ac.uk/user/o/owatts/tool/REAPER-master/build/reaper
sptk_mcep=/afs/inf.ed.ac.uk/user/o/owatts/repos/dnn_swahili/dnn_tts/tools/SPTK-3.7/bin/mcep
```

Patched `magphase.py` to avoid divide by 0 at runtime, pull request sent...

Virtual Python environment made as above, with addition of:

```
 pip install matplotlib
```

### Feature extraction (in parallel)

```
[lubbock]owatts:

cd /afs/inf.ed.ac.uk/user/o/owatts/proj/slm-local/snickery

WAV=/afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/data/nick/wav

python ./script/extract_magphase_features.py -w $WAV -o ~/sim2/oliver/hybrid_work/nick_data_01 -ncores 25
```


Made train list:

```
dhcp-90-053:snickery owatts$ find /afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/hybrid_work/nick_data_01/high/f0/ | while read fname ; do basename  $fname .f0  ; done > /afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/hybrid_work/nick_data_01/trainlist.txt

sort  /afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/hybrid_work/nick_data_01/trainlist.txt > /afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/hybrid_work/nick_data_01/trainlist2.txt

mv  /afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/hybrid_work/nick_data_01/trainlist2.txt  /afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/hybrid_work/nick_data_01/trainlist.txt
```

Edited, removed hvd, mrt and:

```
herald_980
herald_981
herald_982
herald_983
herald_984
herald_985
herald_986
herald_987
herald_988
herald_989
herald_990
herald_991
herald_992
herald_993
herald_994
herald_995
herald_996
herald_997
herald_998
herald_999
```

These added as synthesis and tuning patterns in config.

### Training and synthesis

```
python ./script/train_halfphone.py -c ./config/nick_01.cfg
python ./script/synth_halfphone.py -c ./config/nick_01.cfg
```

### (Note to self) older work moved here
mv slm_data_work/ /group/project/cstr2/oliver_AMT_2013-05-08/hybrid_work_backup_20180312/
mv  /group/project/cstr2/oliver_AMT_2013-05-08/hybrid_work_backup_20180312/  /group/project/cstr2/owatts/

### Temporal smoothing and variance scaling

Temporal smoothing and variance scaling. For no smoothing: -w 1 and -s 1.0

```
python ./script_data/smooth_data.py -f unsmoothed_feat_dir -o smoothed_feat_dir -m 60 -t mag -w 5 -s 0.8
```

To smooth WORLD or Magphase features:
```
./script/smooth_features.sh input_dir output_dir vocoder temporal_scaling variance_scaling file_list resynth

WORLD:
./script/smooth_features.sh /group/project/cstr2/cvbotinh/SCRIPT/Nick/feats/world/ /afs/inf.ed.ac.uk/group/cstr/projects/nst/cvbotinh/SCRIPT/Smoothing/WORLD/smoothed/ world 5 0.8 file_list.txt 1

MAGPHASE:
./script/smooth_features.sh /group/project/cstr2/cvbotinh/SCRIPT/Nick/feats/magphase/low/ /afs/inf.ed.ac.uk/group/cstr/projects/nst/cvbotinh/SCRIPT/Smoothing/magphase/smoothed/ magphase 5 0.8 file_list.txt 1
```



### extract cepstra:
```
cd /afs/inf.ed.ac.uk/user/o/owatts/proj/slm-local/snickery

WAV=/afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/data/nick/wav

python ./script/extract_magphase_features.py -w $WAV -o ~/sim2/oliver/hybrid_work/nick_data_03_cepstra -ncores 25 -m 60 -p 45 -N 20 -cepstra
```


### HDF format
```
python ./script/hdf_magphase_data.py -d ~/sim2/oliver/hybrid_work/nick_data_01/high/ -o ~/proj/hybrid/local_data_dumps/nick_01.h5 -f 1024
```



### Test at 16kHz (for compatibility with wp2)

cd /afs/inf.ed.ac.uk/user/o/owatts/proj/slm-local/snickery

WAV=/afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/wavepred_work/data/wav

python ./script/extract_magphase_features.py -w $WAV -o ~/sim2/oliver/hybrid_work/nick_data_02_16k -ncores 25 -m 60 -p 45

### with 1000 sentences:
/afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/hybrid_work/nick_work_02_16k/synthesis_test/1000_utts_jstreams-mag-real-imag_tstreams-mag-lf0_rep-epoch/greedy-yes_target-1.0-1.0_join-0.2-0.2-0.2_scale-1.0_presel-acoustic_jmetric-natural2_cand-30_taper-50multiepoch-4/hvd_001_1.wav


#### 16k residual:

NB use low dim feats from original wave files, only resid waves and hi-dim...

cd /afs/inf.ed.ac.uk/user/o/owatts/proj/slm-local/snickery

WAV=/afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/wp2/synched_from_hynek/B002DET/train/residuals_epoch_20

python ./script/extract_magphase_features.py -w $WAV -o ~/sim2/oliver/hybrid_work/nick_data_02_16k_resid -ncores 25 -m 26 -p 10 -pm_dir ~/sim2/oliver/hybrid_work/nick_data_02_16k/pm


#### 100 utts


  653  python script/train_halfphone.py -c  config/nick_03_16k_resid.cfg
  654  python script/synth_halfphone.py -c  config/nick_03_16k_resid.cfg

 ~/scripts/sum_waves.py /tmp/out.wav /afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/wp2/synched_from_hynek/B002DET/synthesis/epoch_20/hvd_001_1.wav  /afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/hybrid_work/nick_work_02_16k_resid/synthesis_test/100_utts_jstreams-mag-real-imag_tstreams-mag-lf0_rep-epoch/greedy-yes_target-1.0-1.0_join-0.2-0.2-0.2_scale-1.0_presel-acoustic_jmetric-natural2_cand-30_taper-50multiepoch-4/hvd_001_1.wav 0.9 0.1



#### 1000 utts:

 ~/scripts/sum_waves.py /tmp/out1000.wav /afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/wp2/synched_from_hynek/B002DET/synthesis/epoch_20/hvd_001_1.wav  /afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/hybrid_work/nick_work_02_16k_resid/synthesis_test/1000_utts_jstreams-mag-real-imag_tstreams-mag-lf0_rep-epoch/greedy-yes_target-1.0-1.0_join-0.2-0.2-0.2_scale-1.0_presel-acoustic_jmetric-natural2_cand-30_taper-50multiepoch-4/hvd_001_1.wav 0.8 0.2
