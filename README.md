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

import math
import numpy as np

jn = np.array([3,4,6])
jt = np.array([3.3,1,67])

tn = np.array([13,14,116,9])
tt = np.array([1.3,14.4,46,39])

def euc(a,b):
    return math.sqrt(((a - b) * (a - b)).sum(axis=0))

sep = euc(jn, jt) + euc(tn, tt)
comb = euc(np.concatenate([jn, tn]), np.concatenate([jt, tt]))
print sep
print comb






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



## IS2018 exps

### Test cep vs magphase on 100 sentences

find /group/project/cstr2/cvbotinh/SCRIPT/Nick/feats/magphase/high/f0/ | while read fname ; do basename  $fname .f0  ; done | grep herald | sort > /afs/inf.ed.ac.uk/user/o/owatts/sim2/oliver/hybrid_work/IS2018_exps/trainlist_nick.txt


Use snickey config:

./snickery/config/IS2018_nick_cep_vs_mp.cfg


cd /afs/inf.ed.ac.uk/user/o/owatts/proj/slm-local/snickery

python ./script/train_halfphone.py -c ./config/IS2018_nick_cep_vs_mp.cfg
source ~/sim2/oliver/tool/virtual_python/hybrid_synthesiser/bin/activate
python ./script/synth_halfphone.py -c ./config/IS2018_nick_cep_vs_mp.cfg


output here:

/afs/inf.ed.ac.uk/user/o/owatts/sim2/oliver/hybrid_work/IS2018_exps/voices/cep_vs_spec/synthesis_test/100_utts_jstreams-lf0-mag-real-imag-mag_cc-real_cc-imag_cc_tstreams-mag-lf0-mag_cc_rep-epoch/greedy-yes_target-0.333333333333-0.333333333333-0.333333333333_join-0.142857142857-0.142857142857-0.142857142857-0.142857142857-0.142857142857-0.142857142857-0.142857142857_scale-0.5_presel-acoustic_jmetric-natural2_cand-30_taper-50multiepoch-4/


Copy nick HDF data to laptop for tuning:

dice_scp /group/project/cstr2/cvbotinh/SCRIPT/Nick/feats/magphase/nick_01.h5
mv ~/Desktop/nick_01.h5 ~/proj/hybrid/local_data_dumps/


sep voices for MP and CEP:

  568  python ./script/train_halfphone.py -c ./config/IS2018_nick_cep_vs_mp_CEP.cfg
  571  python ./script/train_halfphone.py -c ./config/IS2018_nick_cep_vs_mp_MP.cfg 




have to copy low dim harvard for test set:

mkdir -p  ~/sim2/oliver/hybrid_work/data/nick_test_sets/magphase/low/
for STREAM in imag  imag_cc  lf0  mag  mag_cc  real  real_cc ; do 
   mkdir -p  ~/sim2/oliver/hybrid_work/data/nick_test_sets/magphase/low/$STREAM ; 
   cp  ~/pr/cvbotinh/SCRIPT/Nick/feats/magphase/low/$STREAM/hvd* ~/sim2/oliver/hybrid_work/data/nick_test_sets/magphase/low/$STREAM ; 
done


... and adjust test_data_dirs in config too



tune on laptop:

python ./script/synth_halfphone_GUI.py -c ./config/IS2018_nick_cep_vs_mp_CEP.cfg -o ~/sim2/oliver/hybrid_work/tuning/IS2018_nick_cep_vs_mp_CEP



### full nick voice, use MP rep:


cd /afs/inf.ed.ac.uk/user/o/owatts/proj/slm-local/snickery

source ~/sim2/oliver/tool/virtual_python/hybrid_synthesiser/bin/activate

python ./script/train_halfphone.py -c ./config/IS2018_nick_magphase60.cfg
python ./script/synth_halfphone.py -c ./config/IS2018_nick_magphase60.cfg


/afs/inf.ed.ac.uk/user/o/owatts/sim2/oliver/hybrid_work/IS2018_exps/voices/cep_vs_spec2/synthesis_test/0_utts_jstreams-mag-real-imag-lf0_tstreams-mag-lf0_rep-epoch/greedy-yes_target-0.5-0.5_join-0.25-0.25-0.25-0.25_scale-0.2_presel-acoustic_jmetric-natural2_cand-30_taper-50multiepoch-6/*wav


mv /afs/inf.ed.ac.uk/user/o/owatts/sim2/oliver/hybrid_work/IS2018_exps/voices/cep_vs_spec2/ /afs/inf.ed.ac.uk/user/o/owatts/sim2/oliver/hybrid_work/IS2018_exps/voices/nick_01/



### Norwegian only voice !!

cd /afs/inf.ed.ac.uk/user/o/owatts/proj/slm-local/snickery

source ~/sim2/oliver/tool/virtual_python/hybrid_synthesiser/bin/activate

python ./script/train_halfphone.py -c ./config/IS2018_norwegian_magphase60.cfg
python ./script/synth_halfphone.py -c ./config/IS2018_norwegian_magphase60.cfg



/afs/inf.ed.ac.uk/user/o/owatts/sim2/oliver/hybrid_work/IS2018_exps/voices/nor01/synthesis_test/0_utts_jstreams-mag-real-imag-lf0_tstreams-mag-lf0_rep-epoch/greedy-yes_target-0.5-0.5_join-0.25-0.25-0.25-0.25_scale-0.2_presel-acoustic_jmetric-natural2_cand-30_taper-50multiepoch-6/

### VCTK only voice !!




Softlink to top 6 speakers:

[lubbock]owatts: more /afs/inf.ed.ac.uk/group/cstr/projects/nst/cvbotinh/SCRIPT/SpeakerSelection/selected_speakers.txt
p304
p334
p374
p298
p246
p292
p347
p243
p227
p272
p258
p364
p226
p259
p252
p345
p260
p270




[salton]owatts: ls /group/project/cstr2/cvbotinh/SCRIPT/VCTK-Corpus/feats/p{304,334,374,298,246,292}/pm/*pm | wc -l
2444


mkdir ~/sim2/oliver/hybrid_work/data/VCTK_softlinks/top_6_speakers


FROM=/group/project/cstr2/cvbotinh/SCRIPT/VCTK-Corpus/feats/
TO=/afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/hybrid_work/data/VCTK_softlinks/top_6_speakers

for SUBDIR in high low pm shift high/f0/ high/imag/ high/mag/ high/real/ low/imag/ low/imag_cc/ low/lf0/ low/mag/ low/mag_cc/ low/real/ low/real_cc/ ; do 
  mkdir -p $TO/$SUBDIR ;
  cp -rs $FROM/p{304,334,374,298,246,292}/$SUBDIR/* $TO/$SUBDIR/ ;
done


Lubbock:

cd /afs/inf.ed.ac.uk/user/o/owatts/proj/slm-local/snickery

source ~/sim2/oliver/tool/virtual_python/hybrid_synthesiser/bin/activate

python ./script/train_halfphone.py -c ./config/IS2018_vctk6speakers_magphase60.cfg
python ./script/synth_halfphone.py -c ./config/IS2018_vctk6speakers_magphase60.cfg

/afs/inf.ed.ac.uk/user/o/owatts/sim2/oliver/hybrid_work/IS2018_exps/voices/vctk01/synthesis_test/0_utts_jstreams-mag-real-imag-lf0_tstreams-mag-lf0_rep-epoch/greedy-yes_target-0.5-0.5_join-0.25-0.25-0.25-0.25_scale-0.2_presel-acoustic_jmetric-natural2_cand-30_taper-50multiepoch-6/



#### VCTK separate speakers -- softlink data:

FROM=/group/project/cstr2/cvbotinh/SCRIPT/VCTK-Corpus/feats/
for SPKR in p304 p334 p374 p298 p246 p292 ; do
    TO=/afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/hybrid_work/data/VCTK_softlinks/VCTK_${SPKR}/
    mkdir -p $TO
    for SUBDIR in high low pm shift high/f0/ high/imag/ high/mag/ high/real/ low/imag/ low/imag_cc/ low/lf0/ low/mag/ low/mag_cc/ low/real/ low/real_cc/ ; do 
      mkdir -p $TO/$SUBDIR ;
      cp -rs $FROM/$SPKR/$SUBDIR/* $TO/$SUBDIR/ ;
    done
done


## prep configs
cp ./config/IS2018_vctk6speakers_magphase60.cfg ./config/IS2018_vctk1speaker_p246_magphase60.cfg

subl  ./config/IS2018_vctk1speaker_p246_magphase60.cfg


for SPKR in p304 p334 p374 p298 p292 ; do  cp ./config/IS2018_vctk1speaker_p246_magphase60.cfg ./config/IS2018_vctk1speaker_${SPKR}_magphase60.cfg; done



subl ./config/IS2018_vctk1speaker_*_magphase60.cfg

## lubbock

for SPKR in p304 p334 p374 p298 p246 p292 ; do
    python ./script/train_halfphone.py -c ./config/IS2018_vctk1speaker_${SPKR}_magphase60.cfg -X
done


for SPKR in p304 p334 p374 p298 p246 p292 ; do
    python ./script/synth_halfphone.py -c ./config/IS2018_vctk1speaker_${SPKR}_magphase60.cfg
done



cd /group/project/cstr2/owatts
SPKR=p304
python /afs/inf.ed.ac.uk/user/o/owatts/proj/slm-local/snickery/script/synth_halfphone.py -c /afs/inf.ed.ac.uk/user/o/owatts/proj/slm-local/snickery/config/IS2018_vctk1speaker_${SPKR}_magphase60.cfg


SNICKERY=/afs/inf.ed.ac.uk/user/o/owatts/proj/slm-local/snickery
for SPKR in p334 p374 p298 p246 p292 ; do
    python $SNICKERY/script/synth_halfphone.py -c $SNICKERY/config/IS2018_vctk1speaker_${SPKR}_magphase60.cfg
done







VOICES=/afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/hybrid_work/IS2018_exps/voices

COND=0_utts_jstreams-mag-real-imag-lf0_tstreams-mag-lf0_rep-epoch/greedy-yes_target-0.5-0.5_join-0.25-0.25-0.25-0.25_scale-0.2_presel-acoustic_jmetric-natural2_cand-30_taper-50multiepoch-6

python  ~/scripts/make_internal_webchart.py \
        natural /afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/data/nick/wav \
        NICK $VOICES/nick_01/synthesis_test/$COND/  \
        VCTK6 $VOICES/vctk01/synthesis_test/$COND/  \
        VCTK_p304 $VOICES/vctk02_p304/synthesis_test/$COND/  \
        VCTK_p334 $VOICES/vctk02_p334/synthesis_test/$COND/  \
        VCTK_p374 $VOICES/vctk02_p374/synthesis_test/$COND/  \
        VCTK_p298 $VOICES/vctk02_p298/synthesis_test/$COND/  \
        VCTK_p246 $VOICES/vctk02_p246/synthesis_test/$COND/  \
        VCTK_p292 $VOICES/vctk02_p292/synthesis_test/$COND/  \
        Norw_senn_pt1 $VOICES/nor01/synthesis_test/$COND/          >   $VOICES/index.html      



VOICES=/afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/hybrid_work/IS2018_exps/voices

SMWORLD=/afs/inf.ed.ac.uk/group/cstr/projects/nst/cvbotinh/SCRIPT/Smoothing/WORLD/

COND=0_utts_jstreams-mag-real-imag-lf0_tstreams-mag-lf0_rep-epoch/greedy-yes_target-0.5-0.5_join-0.25-0.25-0.25-0.25_scale-0.2_presel-acoustic_jmetric-natural2_cand-30_taper-50multiepoch-6

python  ~/scripts/make_internal_webchart.py \
        natural /afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/data/nick/wav \
        World $VOICES/WORLD/  \
        Magphase $VOICES/MAGPHASE \
        World_vs0.6 $SMWORLD/smoothed_ts5_vs0.6 \
        World_vs0.7 $SMWORLD/smoothed_ts5_vs0.7 \
        World_vs0.8 $SMWORLD/smoothed_ts5_vs0.8 \
        NICK $VOICES/nick_01/synthesis_test/$COND/  \
        VCTK6 $VOICES/vctk01/synthesis_test/$COND/  \
        VCTK_p304 $VOICES/vctk02_p304/synthesis_test/$COND/  \
        VCTK_p334 $VOICES/vctk02_p334/synthesis_test/$COND/  \
        VCTK_p374 $VOICES/vctk02_p374/synthesis_test/$COND/  \
        VCTK_p298 $VOICES/vctk02_p298/synthesis_test/$COND/  \
        VCTK_p246 $VOICES/vctk02_p246/synthesis_test/$COND/  \
        VCTK_p292 $VOICES/vctk02_p292/synthesis_test/$COND/  \
        Norw_senn_pt1 $VOICES/nor01/synthesis_test/$COND/          >   $VOICES/index.html      




~/sim2/oliver/hybrid_work/IS2018_exps/voices/WORLD





---- made magphse resynt:

 python ~/proj/slm-local/snickery/script/resynthesise_magphase.py -f /group/project/cstr2/cvbotinh/SCRIPT/Nick/feats/magphase/low/ -o ~/sim2/oliver/temp/nickout -fftlen 2048 -ncores 25 -pattern hvd

cp -r /afs/inf.ed.ac.uk/user/o/owatts/sim2/oliver/temp/nickout ~/sim2/oliver/hybrid_work/IS2018_exps/voices/MAGPHASE/





----- tune nick 01:

5:20


python ./script/synth_halfphone_GUI.py -c ./config/IS2018_nick_magphase60.cfg -o ~/sim2/oliver/hybrid_work/tuning/IS2018_nick_magphase60



test: 

python ~/proj/slm-local/snickery/script/synth_halfphone_NOGUI.py -c ~/proj/slm-local/snickery/config/IS2018_vctk1speaker_p246_magphase60.cfg -o ~/sim2/oliver/temp/test1




------- nick -- smooth data in various conditions:

Make train list:
```
ls /group/project/cstr2/cvbotinh/SCRIPT/Nick/feats/magphase/low/lf0/*.lf0 | grep herald | while read name ; do basename $name .lf0 ; done > /group/project/cstr2/cvbotinh/SCRIPT/Nick/trainlist.txt
```

cd ./script

./smooth_features.sh /group/project/cstr2/cvbotinh/SCRIPT/Nick/feats/magphase/low/ /group/project/cstr2/cvbotinh/SCRIPT/Nick/feats/magphase/low/smoothed/ magphase 5 0.8 /group/project/cstr2/cvbotinh/SCRIPT/Nick/trainlist.txt 0

./smooth_features.sh /group/project/cstr2/cvbotinh/SCRIPT/Nick/feats/magphase/low/ /group/project/cstr2/cvbotinh/SCRIPT/Nick/feats/magphase/low/smoothed/ magphase 5 0.6 /group/project/cstr2/cvbotinh/SCRIPT/Nick/trainlist.txt 0

./smooth_features.sh /group/project/cstr2/cvbotinh/SCRIPT/Nick/feats/magphase/low/ /group/project/cstr2/cvbotinh/SCRIPT/Nick/feats/magphase/low/smoothed/ magphase 11 0.6 /group/project/cstr2/cvbotinh/SCRIPT/Nick/trainlist.txt 0

----- 
and for test set (write to AFS):

ls /group/project/cstr2/cvbotinh/SCRIPT/Nick/feats/magphase/low/lf0/*.lf0 | grep hvd | while read name ; do basename $name .lf0 ; done > /group/project/cstr2/cvbotinh/SCRIPT/Nick/testlist.txt
```

OUT=/afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/hybrid_work/data/nick_test_sets

cd ./script

./smooth_features.sh /group/project/cstr2/cvbotinh/SCRIPT/Nick/feats/magphase/low/ $OUT/magphase/low/smoothed/ magphase 5 0.8 /group/project/cstr2/cvbotinh/SCRIPT/Nick/testlist.txt 0

./smooth_features.sh /group/project/cstr2/cvbotinh/SCRIPT/Nick/feats/magphase/low/ $OUT/magphase/low/smoothed/ magphase 5 0.6 /group/project/cstr2/cvbotinh/SCRIPT/Nick/testlist.txt 0

./smooth_features.sh /group/project/cstr2/cvbotinh/SCRIPT/Nick/feats/magphase/low/ $OUT/magphase/low/smoothed/ magphase 11 0.6 /group/project/cstr2/cvbotinh/SCRIPT/Nick/testlist.txt 
```
-----


## resynthesise all hvd test set:

OUT=/afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/hybrid_work/data/nick_test_sets

python ~/proj/slm-local/snickery/script/resynthesise_magphase.py -f $OUT/magphase/low/smoothed/ts5_vs0.8 -o $OUT/magphase/low/smoothed/ts5_vs0.8/resyn -m 60 -p 45 -fftlen 2048 -ncores 29 
  
python ~/proj/slm-local/snickery/script/resynthesise_magphase.py -f $OUT/magphase/low/smoothed/ts5_vs0.6 -o $OUT/magphase/low/smoothed/ts5_vs0.6/resyn -m 60 -p 45 -fftlen 2048 -ncores 29 
  


./smooth_features.sh 

./smooth_features.sh /group/project/cstr2/cvbotinh/SCRIPT/Nick/feats/magphase/low/ $OUT/magphase/low/smoothed/ magphase 5 0.6 /group/project/cstr2/cvbotinh/SCRIPT/Nick/testlist.txt 0





python ./script/train_halfphone.py -c ./config/IS2018_nick_smooth_ts11_vs6.cfg 
python ./script/train_halfphone.py -c ./config/IS2018_nick_smooth_ts5_vs6.cfg  
python ./script/train_halfphone.py -c ./config/IS2018_nick_smooth_ts5_vs8.cfg



python ./script/synth_halfphone.py -c ./config/IS2018_nick_smooth_ts11_vs6.cfg 
python ./script/synth_halfphone.py -c ./config/IS2018_nick_smooth_ts5_vs6.cfg  
python ./script/synth_halfphone.py -c ./config/IS2018_nick_smooth_ts5_vs8.cfg



----

mismatched train/test conditions:  SKIP this for IS2018!


cp config/IS2018_nick_smooth_ts11_vs6.cfg  config/IS2018_nick_smooth_ts11_vs6_mis.cfg
cp config/IS2018_nick_smooth_ts5_vs6.cfg  config/IS2018_nick_smooth_ts5_vs6_mis.cfg  
cp config/IS2018_nick_smooth_ts5_vs8.cfg config/IS2018_nick_smooth_ts5_vs8_mis.cfg

python ./script/train_halfphone.py -c ./config/IS2018_nick_smooth_ts11_vs6_mis.cfg 
python ./script/train_halfphone.py -c ./config/IS2018_nick_smooth_ts5_vs6_mis.cfg  
python ./script/train_halfphone.py -c ./config/IS2018_nick_smooth_ts5_vs8_mis.cfg



python ./script/synth_halfphone.py -c ./config/IS2018_nick_smooth_ts11_vs6_mis.cfg 
python ./script/synth_halfphone.py -c ./config/IS2018_nick_smooth_ts5_vs6_mis.cfg  
python ./script/synth_halfphone.py -c ./config/IS2018_nick_smooth_ts5_vs8_mis.cfg



----


VOICES=/afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/hybrid_work/IS2018_exps/voices

SMOO=/afs/inf.ed.ac.uk/group/cstr/projects/nst/cvbotinh/SCRIPT/Smoothing//

COND=0_utts_jstreams-mag-real-imag-lf0_tstreams-mag-lf0_rep-epoch/greedy-yes_target-0.5-0.5_join-0.25-0.25-0.25-0.25_scale-0.2_presel-acoustic_jmetric-natural2_cand-30_taper-50multiepoch-6

python  ~/scripts/make_internal_webchart.py \
        natural /afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/data/nick/wav \
        World $VOICES/WORLD/  \
        Magphase $VOICES/MAGPHASE \
        Snickery $VOICES/nick_01/synthesis_test/$COND/  \
        World_ts5_vs0.8 $SMOO/WORLD/smoothed/ts5_vs0.8/resyn/ \
        World_ts5_vs0.6 $SMOO/WORLD/smoothed/ts5_vs0.6/resyn/ \
        World_ts11_vs0.6 $SMOO/WORLD/smoothed/ts11_vs0.6/resyn/ \
        Mag_ts5_vs0.8 $SMOO/magphase/smoothed/ts5_vs0.8/resyn/ \
        Mag_ts5_vs0.6 $SMOO/magphase/smoothed/ts5_vs0.6/resyn/ \
        Mag_ts11_vs0.6 $SMOO/magphase/smoothed/ts11_vs0.6/resyn/ \
        Sni_ts5_vs0.8 $VOICES/nick_smooth_ts5_vs0.8/synthesis_test/$COND/  \
        Sni_ts5_vs0.6 $VOICES/nick_smooth_ts5_vs0.6/synthesis_test/$COND/  \
        Sni_ts11_vs0.6 $VOICES/nick_smooth_ts11_vs0.6/synthesis_test/$COND/ > $VOICES/index_smoothing.html      




### Experiments with adding noise

Listen to single dev sentence with different noises added. noise from:

http://homepages.inf.ed.ac.uk/cvbotinh/se/noises/

Downloaded to:

/Users/owatts/Downloads/babble_ssn



### Normalise stimuli for experiments

mkdir -p ~/sim2/oliver/hybrid_work/IS2018_exps/stimuli/experiment_1
mkdir ~/sim2/oliver/hybrid_work/IS2018_exps/stimuli/experiment_1/{N,W0,M0,S0,M1,S1,M2,S2,A}



PATH=$PATH:~/proj/slm-local/snickery/tool/


VOICES=/afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/hybrid_work/IS2018_exps/voices

SMOO=/afs/inf.ed.ac.uk/group/cstr/projects/nst/cvbotinh/SCRIPT/Smoothing//

COND=0_utts_jstreams-mag-real-imag-lf0_tstreams-mag-lf0_rep-epoch/greedy-yes_target-0.5-0.5_join-0.25-0.25-0.25-0.25_scale-0.2_presel-acoustic_jmetric-natural2_cand-30_taper-50multiepoch-6


STIM=/afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/hybrid_work/IS2018_exps/stimuli/experiment_1




python ~/proj/slm-local/snickery/script_experiment/normalise_level.py -i /afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/data/nick/wav/ -o $STIM/N/ -p hvd

python ~/proj/slm-local/snickery/script_experiment/normalise_level.py -i $VOICES/WORLD/ -o $STIM/W0/ -p hvd
python ~/proj/slm-local/snickery/script_experiment/normalise_level.py -i $VOICES/MAGPHASE/ -o $STIM/M0/ -p hvd
python ~/proj/slm-local/snickery/script_experiment/normalise_level.py -i $VOICES/nick_01/synthesis_test/$COND/ -o $STIM/S0 -p hvd

python ~/proj/slm-local/snickery/script_experiment/normalise_level.py -i $SMOO/magphase/smoothed/ts5_vs0.8/resyn/ -o $STIM/M1/ -p hvd
python ~/proj/slm-local/snickery/script_experiment/normalise_level.py -i $VOICES/nick_smooth_ts5_vs0.8/synthesis_test/$COND/ -o $STIM/S1 -p hvd

python ~/proj/slm-local/snickery/script_experiment/normalise_level.py -i $SMOO/magphase/smoothed/ts5_vs0.6/resyn/ -o $STIM/M2/ -p hvd
python ~/proj/slm-local/snickery/script_experiment/normalise_level.py -i  $VOICES/nick_smooth_ts5_vs0.6/synthesis_test/$COND/ -o $STIM/S2 -p hvd


### missing test: all but NAT and MAG

python ~/proj/slm-local/snickery/script/normalise_level.py 


## make html index, exclude anchor :

python ~/proj/slm-local/snickery/script/make_internal_webchart.py -o $STIM/index.html -d $STIM/{N,W0,M0,S0,M1,S1,M2,S2}




## Make anchor which attempts to combine artefacts found in all systems:

Resyn waves from magphase smoothing conition 2:

(hybrid_synthesiser)[lubbock]owatts: python ~/proj/slm-local/snickery/script/resynthesise_magphase.py -f /group/project/cstr2/cvbotinh/SCRIPT/Nick/feats/magphase/low/smoothed/ts5_vs0.6/ -o /group/project/cstr2/cvbotinh/SCRIPT/Nick/feats/magphase/low/smoothed/ts5_vs0.6/resyn -m 60 -p 45 -fftlen 2048 -ncores 29 -fs 48000 

Extraczt features for anchor voice:

python ~/proj/slm-local/snickery/script/extract_magphase_features.py -w /group/project/cstr2/cvbotinh/SCRIPT/Nick/feats/magphase/low/smoothed/ts5_vs0.6/resyn -o /group/project/cstr2/cvbotinh/SCRIPT/Nick/feats/magphase_for_anchor -m 60 -p 45 -fftlen 2048 -ncores 29 
   
Snickery:

python ~/proj/slm-local/snickery/script/train_halfphone.py -c  ~/proj/slm-local/snickery/config/IS2018_nick_anchor.cfg
python ~/proj/slm-local/snickery/script/synth_halfphone.py -c  ~/proj/slm-local/snickery/config/IS2018_nick_anchor.cfg


Doesn't work (Because of the smoothing, I think the same unit is selected many times in a row, so it becomes very periodic (in a  very bad way) Hence the chainsaw type noise instead of silence):

/afs/inf.ed.ac.uk/user/o/owatts/sim2/oliver/hybrid_work/IS2018_exps/voices/nick_anchor/synthesis_test/0_utts_jstreams-mag-real-imag-lf0_tstreams-mag-lf0_rep-epoch/greedy-yes_target-0.5-0.5_join-0.25-0.25-0.25-0.25_scale-0.2_presel-acoustic_jmetric-natural2_cand-30_taper-50multiepoch-6/

## Anchor attemp 2: Vocode and smooth snickery ouput (S2).

## Extract features for anchor voice 2:

VOICES=/afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/hybrid_work/IS2018_exps/voices
SMOO=/afs/inf.ed.ac.uk/group/cstr/projects/nst/cvbotinh/SCRIPT/Smoothing//
COND=0_utts_jstreams-mag-real-imag-lf0_tstreams-mag-lf0_rep-epoch/greedy-yes_target-0.5-0.5_join-0.25-0.25-0.25-0.25_scale-0.2_presel-acoustic_jmetric-natural2_cand-30_taper-50multiepoch-6

python ~/proj/slm-local/snickery/script/extract_magphase_features.py -w $VOICES/nick_smooth_ts5_vs0.6/synthesis_test/$COND/ -o /group/project/cstr2/cvbotinh/SCRIPT/Nick/feats/magphase_for_anchor2 -m 60 -p 45 -fftlen 2048 -ncores 29 
  

### smooth

OUT=/afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/hybrid_work/data/nick_test_sets

cd /group/project/cstr2/owatts/temp/snickery/script


./smooth_features.sh /group/project/cstr2/cvbotinh/SCRIPT/Nick/feats/magphase_for_anchor2/low/ /group/project/cstr2/cvbotinh/SCRIPT/Nick/feats/magphase_for_anchor2/low/smoothed/ magphase 5 0.6 /group/project/cstr2/cvbotinh/SCRIPT/Nick/testlist.txt 0

<!-- echo hvd_191 > /tmp/test.txt
./smooth_features.sh /group/project/cstr2/cvbotinh/SCRIPT/Nick/feats/magphase_for_anchor2/low/ /tmp/test magphase 5 0.6 /tmp/test.txt 0
 -->

### resynth:

python ~/proj/slm-local/snickery/script/resynthesise_magphase.py -f /group/project/cstr2/cvbotinh/SCRIPT/Nick/feats/magphase_for_anchor2/low/smoothed/ts5_vs0.6 -o /group/project/cstr2/cvbotinh/SCRIPT/Nick/feats/magphase_for_anchor2/low/smoothed/ts5_vs0.6/resyn  -m 60 -p 45 -fftlen 2048 -ncores 29 

rm -r  /afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/hybrid_work/IS2018_exps/voices/nick_anchor2
mkdir /afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/hybrid_work/IS2018_exps/voices/nick_anchor2
cp  /group/project/cstr2/cvbotinh/SCRIPT/Nick/feats/magphase_for_anchor2/low/smoothed/ts5_vs0.6/resyn/*   /afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/hybrid_work/IS2018_exps/voices/nick_anchor2


smoothing & snickery broke MP resynthesis in a few cases...




## Anchor attempt 3: LP filter natural speech

sox -R /afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/data/nick/wav/hvd_181.wav /tmp/test.wav lowpass 7000 





The set of processed signals consists of all the signals under test and at least two additional “anchor” signals. The standard anchor is a low-pass filtered version of the original signal with a cut-off frequency of 3.5 kHz; the mid quality anchor has a cut-off frequency of 7 kHz.
The bandwidths of the anchors correspond to the Recommendations for control circuits (3.5 kHz), used for supervision and coordination purpose in broadcasting, commentary circuits (7 kHz) and occasional circuits (10 kHz), according to Recommendations ITU-T G.711, G.712, G.722 and J.21, respectively.
The characteristics of the 3.5 kHz low-pass filter should be as follows: fc  3.5 kHz
Maximum pass band ripple  0.1 dB
Minimum attenuation at 4 kHz  25 dB
Minimum attenuation at 4.5 kHz  50 dB.
Additional anchors are intended to provide an indication of how the systems under test compare to
well-known audio quality levels and should not be used for rescaling results between different tests.







ls -d ~/sim2/oliver/hybrid_work/IS2018_exps/stimuli/experiment_1/* | while read DIREC ; do echo $DIREC ;  ls $DIREC/* | wc -l ; done





### Normalise stimuli for experiments -- rest of hvd




PATH=$PATH:~/proj/slm-local/snickery/tool/

VOICES=/afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/hybrid_work/IS2018_exps/voices
SMOO=/afs/inf.ed.ac.uk/group/cstr/projects/nst/cvbotinh/SCRIPT/Smoothing//
COND=0_utts_jstreams-mag-real-imag-lf0_tstreams-mag-lf0_rep-epoch/greedy-yes_target-0.5-0.5_join-0.25-0.25-0.25-0.25_scale-0.2_presel-acoustic_jmetric-natural2_cand-30_taper-50multiepoch-6

STIM=/afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/hybrid_work/IS2018_exps/stimuli/experiment_1


python ~/proj/slm-local/snickery/script_experiment/normalise_level.py -i $VOICES/nick_01/synthesis_test/$COND/ -o $STIM/S0 -p hvd

python ~/proj/slm-local/snickery/script_experiment/normalise_level.py -i $VOICES/nick_smooth_ts5_vs0.8/synthesis_test/$COND/ -o $STIM/S1 -p hvd

python ~/proj/slm-local/snickery/script_experiment/normalise_level.py -i  $VOICES/nick_smooth_ts5_vs0.6/synthesis_test/$COND/ -o $STIM/S2 -p hvd

python ~/proj/slm-local/snickery/script_experiment/normalise_level.py -i /afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/hybrid_work/IS2018_exps/voices/nick_anchor2 -o $STIM/A -p hvd


OUT=/afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/hybrid_work/data/nick_test_sets

python ~/proj/slm-local/snickery/script_experiment/normalise_level.py -i $OUT/magphase/low/smoothed/ts5_vs0.8/resyn/ -o $STIM/M1/ -p hvd
python ~/proj/slm-local/snickery/script_experiment/normalise_level.py -i $OUT/magphase/low/smoothed/ts5_vs0.6/resyn/ -o $STIM/M2/ -p hvd


python ~/proj/slm-local/snickery/script_experiment/normalise_level.py -i /group/project/cstr2/cvbotinh/SCRIPT/Nick/feats/world/resyn/ -o $STIM/W0/ -p hvd





## make html index, include world :

python ~/proj/slm-local/snickery/script/make_internal_webchart.py -o $STIM/index2.html -d $STIM/{N,W0,M0,S0,M1,S1,M2,S2,A}



python ~/proj/slm-local/snickery/script/make_internal_webchart.py -o $STIM/index3.html -d $STIM/{N,W0,M0,S0,M1,S1,M2,S2,A} -p hvd_30








#### -------- exp 2 :

NW
NV
NVW


TO=/afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/hybrid_work/data/nick_vctk6_softlinks

VCTK=/group/project/cstr2/cvbotinh/SCRIPT/VCTK-Corpus/feats/
NICK=/group/project/cstr2/cvbotinh/SCRIPT/Nick/feats/magphase/
for SUBDIR in high low pm shift high/f0/ high/imag/ high/mag/ high/real/ low/imag/ low/lf0/ low/mag/ low/real/ ; do 
  mkdir -p $TO/$SUBDIR/
  cp -rs $NICK/$SUBDIR/* $TO/$SUBDIR/ ;
  cp -rs $VCTK/p{304,334,374,298,246,292}/$SUBDIR/* $TO/$SUBDIR/ ;
done



TO=/afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/hybrid_work/data/nick_norw_softlinks

NICK=/group/project/cstr2/cvbotinh/SCRIPT/Nick/feats/magphase/
NORW=/group/project/cstr2/cvbotinh/SCRIPT/Norwegian/feats/sennheiser/part_1/
for SUBDIR in high low pm shift high/f0/ high/imag/ high/mag/ high/real/ low/imag/ low/lf0/ low/mag/ low/real/ ; do 
  mkdir -p $TO/$SUBDIR/
  cp -rs $NICK/$SUBDIR/* $TO/$SUBDIR/ ;
  cp -rs $NORW/$SUBDIR/* $TO/$SUBDIR/ ;
done




TO=/afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/hybrid_work/data/nick_vctk6_norw_softlinks

NICK=/group/project/cstr2/cvbotinh/SCRIPT/Nick/feats/magphase/
VCTK=/group/project/cstr2/cvbotinh/SCRIPT/VCTK-Corpus/feats/
NORW=/group/project/cstr2/cvbotinh/SCRIPT/Norwegian/feats/sennheiser/part_1/
for SUBDIR in high low pm shift high/f0/ high/imag/ high/mag/ high/real/ low/imag/ low/lf0/ low/mag/ low/real/ ; do 
  mkdir -p $TO/$SUBDIR/
  cp -rs $NICK/$SUBDIR/* $TO/$SUBDIR/ ;
  cp -rs $NORW/$SUBDIR/* $TO/$SUBDIR/ ;
  cp -rs $VCTK/p{304,334,374,298,246,292}/$SUBDIR/* $TO/$SUBDIR/ ;
done



source ~/sim2/oliver/tool/virtual_python/hybrid_synthesiser/bin/activate
python ~/proj/slm-local/snickery/script/train_halfphone.py -c ~/proj/slm-local/snickery/config/IS2018_NV.cfg  

source ~/sim2/oliver/tool/virtual_python/hybrid_synthesiser/bin/activate
python ~/proj/slm-local/snickery/script/train_halfphone.py -c ~/proj/slm-local/snickery/config/IS2018_NW.cfg  


source ~/sim2/oliver/tool/virtual_python/hybrid_synthesiser/bin/activate
python ~/proj/slm-local/snickery/script/train_halfphone.py -c ~/proj/slm-local/snickery/config/IS2018_NVW.cfg  



###


VOICES=/afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/hybrid_work/IS2018_exps/voices
COND=0_utts_jstreams-mag-real-imag-lf0_tstreams-mag-lf0_rep-epoch/greedy-yes_target-0.5-0.5_join-0.25-0.25-0.25-0.25_scale-0.2_presel-acoustic_jmetric-natural2_cand-30_taper-50multiepoch-6

STIM=/afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/hybrid_work/IS2018_exps/stimuli/experiment_1



python ~/proj/slm-local/snickery/script/make_internal_webchart.py -o $STIM/index_exp2_sample.html -d \
         $VOICES/nick_01/synthesis_test/$COND/  \
         $VOICES/vctk01/synthesis_test/$COND/  \
         $VOICES/nor01/synthesis_test/$COND/ \
        $VOICES/NV/synthesis_test/$COND/ \
        $VOICES/NW/synthesis_test/$COND/ \
        $VOICES/NVW/synthesis_test/$COND/ -n N V W NV NW NVW





#### try with Nick 200 + V W VW

## nick 200:


NICK=/group/project/cstr2/cvbotinh/SCRIPT/Nick/feats/magphase/
TO=/group/project/cstr2/cvbotinh/SCRIPT/Nick/feats/magphase_200utts
for SUBDIR in high low pm shift high/f0/ high/imag/ high/mag/ high/real/ low/imag/ low/lf0/ low/mag/ low/real/ ; do 
  mkdir -p $TO/$SUBDIR/
  cp -rs $NICK/$SUBDIR/herald_{0,1}??.* $TO/$SUBDIR/ ;
done



TO=/afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/hybrid_work/data/nick200_vctk6_softlinks

VCTK=/group/project/cstr2/cvbotinh/SCRIPT/VCTK-Corpus/feats/
NICK=/group/project/cstr2/cvbotinh/SCRIPT/Nick/feats/magphase_200utts
for SUBDIR in high low pm shift high/f0/ high/imag/ high/mag/ high/real/ low/imag/ low/lf0/ low/mag/ low/real/ ; do 
  mkdir -p $TO/$SUBDIR/
  cp -rs $NICK/$SUBDIR/* $TO/$SUBDIR/ ;
  cp -rs $VCTK/p{304,334,374,298,246,292}/$SUBDIR/* $TO/$SUBDIR/ ;
done



TO=/afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/hybrid_work/data/nick200_norw_softlinks

NICK=/group/project/cstr2/cvbotinh/SCRIPT/Nick/feats/magphase_200utts
NORW=/group/project/cstr2/cvbotinh/SCRIPT/Norwegian/feats/sennheiser/part_1/
for SUBDIR in high low pm shift high/f0/ high/imag/ high/mag/ high/real/ low/imag/ low/lf0/ low/mag/ low/real/ ; do 
  mkdir -p $TO/$SUBDIR/
  cp -rs $NICK/$SUBDIR/* $TO/$SUBDIR/ ;
  cp -rs $NORW/$SUBDIR/* $TO/$SUBDIR/ ;
done




TO=/afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/hybrid_work/data/nick200_vctk6_norw_softlinks

NICK=/group/project/cstr2/cvbotinh/SCRIPT/Nick/feats/magphase_200utts
VCTK=/group/project/cstr2/cvbotinh/SCRIPT/VCTK-Corpus/feats/
NORW=/group/project/cstr2/cvbotinh/SCRIPT/Norwegian/feats/sennheiser/part_1/
for SUBDIR in high low pm shift high/f0/ high/imag/ high/mag/ high/real/ low/imag/ low/lf0/ low/mag/ low/real/ ; do 
  mkdir -p $TO/$SUBDIR/
  cp -rs $NICK/$SUBDIR/* $TO/$SUBDIR/ ;
  cp -rs $NORW/$SUBDIR/* $TO/$SUBDIR/ ;
  cp -rs $VCTK/p{304,334,374,298,246,292}/$SUBDIR/* $TO/$SUBDIR/ ;
done




cp ~/proj/slm-local/snickery/config/IS2018_NV.cfg  ~/proj/slm-local/snickery/config/IS2018_N200.cfg
cp ~/proj/slm-local/snickery/config/IS2018_NV.cfg  ~/proj/slm-local/snickery/config/IS2018_NV200.cfg  
cp ~/proj/slm-local/snickery/config/IS2018_NW.cfg  ~/proj/slm-local/snickery/config/IS2018_NW200.cfg  
cp ~/proj/slm-local/snickery/config/IS2018_NVW.cfg  ~/proj/slm-local/snickery/config/IS2018_NVW200.cfg  




source ~/sim2/oliver/tool/virtual_python/hybrid_synthesiser/bin/activate
python ~/proj/slm-local/snickery/script/train_halfphone.py -c ~/proj/slm-local/snickery/config/IS2018_N200.cfg  

source ~/sim2/oliver/tool/virtual_python/hybrid_synthesiser/bin/activate
python ~/proj/slm-local/snickery/script/train_halfphone.py -c ~/proj/slm-local/snickery/config/IS2018_NV200.cfg  

source ~/sim2/oliver/tool/virtual_python/hybrid_synthesiser/bin/activate
python ~/proj/slm-local/snickery/script/train_halfphone.py -c ~/proj/slm-local/snickery/config/IS2018_NW200.cfg  

source ~/sim2/oliver/tool/virtual_python/hybrid_synthesiser/bin/activate
python ~/proj/slm-local/snickery/script/train_halfphone.py -c ~/proj/slm-local/snickery/config/IS2018_NVW200.cfg  

## last exhasted disk -- try again and stre to cstr2
mkdir -p /afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/hybrid_work/IS2018_exps/voices/NVW200/synthesis_test/0_utts_jstreams-mag-real-imag-lf0_tstreams-mag-lf0_rep-epoch/greedy-yes_target-0.5-0.5_join-0.25-0.25-0.25-0.25_scale-0.2_presel-acoustic_jmetric-natural2_cand-30_taper-50multiepoch-6/

cp -r ~/pr/owatts/IS2018_hybrid/NVW200/synthesis_test/0_utts_jstreams-mag-real-imag-lf0_tstreams-mag-lf0_rep-epoch/greedy-yes_target-0.5-0.5_join-0.25-0.25-0.25-0.25_scale-0.2_presel-acoustic_jmetric-natural2_cand-30_taper-50multiepoch-6/* /afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/hybrid_work/IS2018_exps/voices/NVW200/synthesis_test/0_utts_jstreams-mag-real-imag-lf0_tstreams-mag-lf0_rep-epoch/greedy-yes_target-0.5-0.5_join-0.25-0.25-0.25-0.25_scale-0.2_presel-acoustic_jmetric-natural2_cand-30_taper-50multiepoch-6/



VOICES=/afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/hybrid_work/IS2018_exps/voices
COND=0_utts_jstreams-mag-real-imag-lf0_tstreams-mag-lf0_rep-epoch/greedy-yes_target-0.5-0.5_join-0.25-0.25-0.25-0.25_scale-0.2_presel-acoustic_jmetric-natural2_cand-30_taper-50multiepoch-6
STIM=/afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/hybrid_work/IS2018_exps/stimuli/experiment_1
#
python ~/proj/slm-local/snickery/script/make_internal_webchart.py -o $STIM/index_exp2B_sample.html -d \
         $VOICES/nick_01/synthesis_test/$COND/  \
         $VOICES/vctk01/synthesis_test/$COND/  \
         $VOICES/nor01/synthesis_test/$COND/ \
        $VOICES/NV/synthesis_test/$COND/ \
        $VOICES/NW/synthesis_test/$COND/ \
        $VOICES/NVW/synthesis_test/$COND/   \
        $VOICES/N200/synthesis_test/$COND/ \
        $VOICES/NV200/synthesis_test/$COND/ \
        $VOICES/NW200/synthesis_test/$COND/ \
        $VOICES/NVW200/synthesis_test/$COND/ \
        $VOICES/N100/synthesis_test/$COND/ \
           -n N V W NV NW NVW N200 NV200 NW200 NVW200 N100









#### try with Nick 100 + V W VW

## nick 100:


NICK=/group/project/cstr2/cvbotinh/SCRIPT/Nick/feats/magphase/
TO=/group/project/cstr2/cvbotinh/SCRIPT/Nick/feats/magphase_100utts
for SUBDIR in high low pm shift high/f0/ high/imag/ high/mag/ high/real/ low/imag/ low/lf0/ low/mag/ low/real/ ; do 
  mkdir -p $TO/$SUBDIR/
  cp -rs $NICK/$SUBDIR/herald_0??.* $TO/$SUBDIR/ ;
done


cp ~/proj/slm-local/snickery/config/IS2018_N200.cfg ~/proj/slm-local/snickery/config/IS2018_N100.cfg


source ~/sim2/oliver/tool/virtual_python/hybrid_synthesiser/bin/activate
python ~/proj/slm-local/snickery/script/train_halfphone.py -c ~/proj/slm-local/snickery/config/IS2018_N100.cfg  

