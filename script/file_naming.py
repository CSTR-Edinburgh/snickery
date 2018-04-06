
import os
from util import safe_makedir

def get_data_dump_name(config, joindata=False, joinsql=False, searchtree=False):
    safe_makedir(os.path.join(config['workdir'], 'data_dumps'))
    condition = make_train_condition_name(config)
    assert not (joindata and joinsql)
    if joindata:
        last_part = '.joindata.hdf5'
    elif joinsql:
        last_part = '.joindata.sql'
    elif searchtree:
        last_part = '.searchtree.hdf5'
    else:
        last_part = '.hdf5'
    database_fname = os.path.join(config['workdir'], "data_dumps", condition + last_part)
    return database_fname

def make_train_condition_name(config):
    '''
    condition name including any important hyperparams
    '''
    ### N-train_utts doesn't account for exclusions due to train_list, bad data etc. TODO - fix?
    if not config['target_representation'] == 'sample':
        jstreams = '-'.join(config['stream_list_join'])
        tstreams = '-'.join(config['stream_list_target'])
        return '%s_utts_jstreams-%s_tstreams-%s_rep-%s'%(config['n_train_utts'], jstreams, tstreams, config.get('target_representation', 'twopoint'))
    else:        
        streams = '-'.join(config['stream_list_target'])
        return '%s_utts_streams-%s_rep-%s'%(config['n_train_utts'], streams, config.get('target_representation', 'twopoint'))
    
def make_synthesis_condition_name(config):
    '''
    Return string encoding all variables which can be ... 
    '''

    if config.get('synth_smooth', False):
        smooth='smooth_'
    else:
        smooth=''

    if config.get('greedy_search', False): 
        greedy = 'greedy-yes_'
    else:
        greedy = 'greedy-no_'

    ##### Current version: weight per stream.
    target_weights = '-'.join([str(val) for val in config['target_stream_weights']])
    if config['target_representation'] == 'sample':
        name = 'sample_target-%s'%(target_weights)
    else:
        join_weights = '-'.join([str(val) for val in config['join_stream_weights']])
        jcw = config['join_cost_weight']
        jct = config.get('join_cost_type', 'natural2')  ## TODO: not relevant to epoch, provided default consistent with IS2018 exp
        nc = config.get('n_candidates', 30) ## TODO: not relevant to epoch, provided default consistent with IS2018 exp
        tl = config.get('taper_length', 50) ## TODO: not relevant to epoch, provided default consistent with IS2018 exp
        name = '%s%starget-%s_join-%s_scale-%s_presel-%s_jmetric-%s_cand-%s_taper-%s'%(
                    greedy, smooth,
                    target_weights, join_weights, jcw,
                    config['preselection_method'],
                    jct,
                    nc,
                    tl
                )    
        name += 'multiepoch-%s'%(config.get('multiepoch', 1))        
    return name
