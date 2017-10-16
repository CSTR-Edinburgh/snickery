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



<!-- 
./script_data/extract_feats_parallel.sh ~/sim2/oliver/slm_data_work/fls_hybrid/wav29/ ~/sim2/oliver/slm_data_work/fls_hybrid/feat_29/
 -->
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
python script/train_halfphone.py -c ./config/blizzard_replication_01.cfg
```

Output is placed under the directory configured as `workdir`, in a subdirectory recording notable train-time hyperparameters, so that multiple runs can be done by editing a single config file, without overwriting previous results.


### Synthesis 

These directories give the locations of inputs required for synthesis with the system:
- `test_data_dir`: streams of data comparable to those in `target_datadir`, to be used for computing target cost at run time.
- `test_lab_dir`: labels files of same format as those in `label_datadir`.

Synthesise like this:

```
python script/synth_halfphone.py -c ./config/blizzard_replication_01.cfg
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




## Operation of the scripts

### Training
[Add more detail on training here]

### Synthesis
[Add more detail on synthesis here]



