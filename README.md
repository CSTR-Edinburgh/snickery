# Snickery

This repository contains code used to build the proposed systems presented in the following papers:



[<img src="https://github.com/oliverwatts/snickery/blob/master/media/IS2018_thumbnail_top2.png">](<https://www.isca-speech.org/archive/Interspeech_2018/pdfs/1857.pdf>)


```
@inproceedings{watts18examplar,
  title     = {Exemplar-based speech waveform generation},
  author    = {Oliver Watts and Cassia Valentini-Botinhao and Felipe Espic and Simon King},
  booktitle = {Interspeech},
  year      = {2018},
}

@inproceedings{cvb2018speech,
  title={Exemplar-based speech waveform generation for text-to-speech},
  author={Cassia Valentini-Botinhao and Oliver Watts and Felipe Espic and Simon King},
  booktitle={IEEE Workshop on Spoken Language Technology (submitted)},
  year={2018}
}
```

The first part of this README is about use of scripts:

```
script/train_simple.py
script/synth_simple.py 
```

... which can only build a few restricted types of system (selection of epoch-based fragments, greedy search only). They can be used to replicate the system proposed in the paper *Exemplar-based speech waveform generation*. 

Output of the systems evaluated in that paper can be heard [here](<http://homepages.inf.ed.ac.uk/owatts/papers/IS2018_snickery/>).

See section *Hybrid text-to-speech synthesis with Merlin* below for details on using other parts of the code - which is less well tested and documented - to build other types of systems. 

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

# Hybrid text-to-speech synthesis with Merlin

Snickery can be used in conjunction with Merlin to create a hybrid TTS system. We prepared a recipe that uses the slt arctic dataset (wavefiles, state and phone level labels), trains a Merlin model and two Snickery models (small unit and halfphone). The recipe synthesises waveforms from text using Merlin and three different waveform generation modules: MagPhase vocoder, Snickery small unit and Snickery halfphone.

## Requirements

- a version of Merlin (https://github.com/CSTR-Edinburgh/merlin.git) installed in your system
- a python environment with requirments from both Merlin and Snickery

OBS: Additionally from what Merlin requires, Snickery needs sklearn, sklearn and OpenFST (binaries and python bindings; this is only required for the Snickery halfphone variant, see README_FULL on how to install this).

## Running the tools

From the toplevel ```./snickery/``` directory of the repository you cloned:
```
SNICKERY=`pwd`
./script/merlin/hybrid_recipe.sh $SNICKERY $MERLIN $WORK/hybrid/
```
where $MERLIN points to the path where Merlin was cloned and $WORK/hybrid is the working directory where data, models and synthesized speech will be stored.
