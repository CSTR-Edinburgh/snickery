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

Get magphase vocoder:
```
mkdir tool
cd tool/
git clone https://github.com/CSTR-Edinburgh/magphase.git
cd magphase
cd tools/
```
Now install tools required by magphase like this:

```
./download_and_compile_tools.sh
```
 ... or else create symbolic links to tools if you already have them compiled:

```
mkdir bin
cd bin
ln -s /afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/repos/Ossian/tools/bin/mcep ./mcep
ln -s /afs/inf.ed.ac.uk/user/o/owatts/tool/REAPER-master/build/reaper ./reaper
```





### Installation of Python dependencies with virtual environment

Make a directory to house virtual environments if you don't already have one, and move to it:

```
cd ~/tool/virtual_python/
virtualenv --distribute --python=/usr/bin/python2.7 hybrid_synthesiser
source ./hybrid_synthesiser/bin/activate
```

On my machine, my prompt has now turned from ```[salton]owatts:``` to  ```(hybrid_synthesiser)[salton]owatts:```. With the virtual environment activated, you can now install the necessary packages:

```
## first upgrade pip itself:
pip install -U pip

pip install numpy
pip install scipy   ## required by sklearn
pip install h5py
pip install sklearn
pip install matplotlib
pip install soundfile
```



## Running the tools

<!-- cd /group/project/cstr2/owatts/temp/slt_work -->

Move to a convenient location, which we'll call ```$WORK``` and download some data to work on:


```
WORK=`pwd`
mkdir -p experiment/data
cd experiment/data/
wget http://felipeespic.com/depot/databases/merlin_demos/slt_arctic_full_data.zip
unzip slt_arctic_full_data.zip
```

Extract magphase features from a small set of 120 sentences of the data like this, from the toplevel ```./snickery/``` directory of the repository you cloned:

```
WAV=$WORK/experiment/data/slt_arctic_full_data/exper/acoustic_model/data/wav/
FEATS=$WORK/experiment/data/magphase_features
python ./script/extract_magphase_features.py -w $WAV -o $FEATS -ncores 4 -m 60 -p 45 -N 120
```


Edit a single line of the configuration file ```config/slt_simplified_mini.cfg``` so that ```workdir``` points to the full path your used for $WORK:

```
workdir = '/path/to/your/work/dir/'
```

Train like this:

```
python script/train_simple.py -c config/slt_simplified_mini.cfg
```


Output is placed under the directory configured as `workdir`, in a subdirectory recording notable train-time hyperparameters, so that multiple runs can be done by editing a single config file, without overwriting previous results.

Synthesis:

```
python script/synth_simple.py -c config/slt_simplified_mini.cfg
```


Now you can try changing settings in the config. Important ones to look at:

- `n_train_utts`: number of sentences to use (you might have to run `extract_magphase_features.py` again to obtain features for more sentences). This is the only one where you will need to run train.py after changing it.

These can be changed without retraining:

- `target_stream_weights` and `join_stream_weights`: scale the importance of different streams in target and join cost. These are just python lists of floats.
- `join_cost_weight`: overall scaling factor for join stream. Must be between 0.0. and 1.0.
- `multiepoch`: how many consecutive epochs to select at each timestep
- `magphase_overlap`: how much to overlap selected units by.
- `magphase_use_target_f0`: whether to impose the target F0 on the concatenated units.
- `search_epsilon`: how much to approximate the search. 0 means no approximation, so find the largest setting where there is no perceptible difference from this.

Finally, this demo is training on all-natural speech and testing with natural targets. Clearly, natural targets will not be available to a real TTS system. Tweak the config to work with synthetic speech.
