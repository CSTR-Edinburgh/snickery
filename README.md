# Snickery (minimal version)

This README is about use of scripts:

```
script/synth_simple.py 
script/train_simple.py
```

These are simplified version of the scripts:

```
script/synth_halfphone.py 
script/train_halfphone.py
```

and can only build a few restricted types of system (selection of epoch-based fragments, greedy search only). They can be used to replicate the IS2018 experiments. 

See README_FULL for other uses of the toolkit.

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
pip install matplotlib
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


