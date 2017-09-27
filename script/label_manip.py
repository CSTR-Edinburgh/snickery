

import sys
import re
from const import label_delimiter

def extract_quinphone(label, quinphone_regex):
    m = re.match(quinphone_regex, label)
    #print quinphone_regex.name
    assert m, 'quinphone_regex does not match label %s'%(label)
    quinphone = m.groups()
    #print quinphone
    assert len(quinphone) == 5
    return quinphone

def break_quinphone(quinphone):
    '''
    Break internal representation of quinphone into (mono, diphone, triphone, quinphone).
    Only tricky thing is handling direction of diphone context.
    '''
    q = quinphone.split(label_delimiter)
    assert len(q) == 5
    mono = q[2]
    tri = label_delimiter.join(q[1:4])
    #quin = label_delimiter.join(q)
    if mono.endswith('_L'):
        di = label_delimiter.join(q[1:3])
    elif mono.endswith('_R'):
        di = label_delimiter.join(q[2:4])        
    else:
        sys.exit('efvaedvsdv')
    return (mono, di, tri, quinphone)

def extract_monophone(quinphone):
    '''
    Return only current phone from internal quinphone rep, stripping phone L/R position 
    '''
    q = quinphone.split(label_delimiter)
    assert len(q) == 5
    mono = q[2]           
    stripped_mono = mono.split('_')[0]    
    return stripped_mono