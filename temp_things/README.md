
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



## Setup and experiments for IS2018 -- cuts



### (Note to self) older work moved here
mv slm_data_work/ /group/project/cstr2/oliver_AMT_2013-05-08/hybrid_work_backup_20180312/
mv  /group/project/cstr2/oliver_AMT_2013-05-08/hybrid_work_backup_20180312/  /group/project/cstr2/owatts/





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


