#!/usr/bin/env python
# -*- coding: utf-8 -*-
## Project:  
## Author: Oliver Watts - owatts@staffmail.ed.ac.uk

## replacements for functs in fst_functions.py using pywrapfst
import sys
import random

import pywrapfst as openfst
'''
All frame indexes input are Python indexes (starting 0). 1 is added internally for FST
format compatibility (0 is reserved for epsilon), and removed before path is returned
in get_shortest_path
'''
import timeit

import numpy as np
from util import comm

from const import VERY_BIG_WEIGHT_VALUE 


def compile_fst(tool, text_in, bin_out):
    #tool = self.config['openfst_bindir']
    comm('%s/fstcompile %s %s'%(tool, text_in, bin_out), quiet=True)

def make_target_sausage_lattice(dist, ind):
    '''
    sort on outsymbols
    '''
    fst = []
    start = 0
    frames, cands = np.shape(dist)

    for i in range(frames):
    
        end = start + 1
        for j in range(cands):

            frame_ix = ind[i,j]
            
            if frame_ix == -1:  ## this is a padding value because number of candidates was indufficient
                continue 

            weight = dist[i,j] 
            fst.append('%s %s %s %s %s'%(start, end, frame_ix+1, frame_ix+1, weight)) 
            
        start = end
    fst.append('%s'%(end)) 
    
    compiler = openfst.Compiler()    
    for line in fst:
        print >> compiler, line
        #compiler.write(line)   ## TODO test
    f = compiler.compile()
    f.arcsort(st="olabel")
    return f

## TODO: openfst.Weight.NoWeight(fst.weight_type())    gives NaN -> BandNUmber in pllot
def make_mapping_loop_fst_WRAP(mapping):
    fst = openfst.Fst()
    start = fst.add_state()
    mid = fst.add_state()
    end = fst.add_state()
    fst.set_start(start)
    fst.final(end)
    fst.add_arc(start, openfst.Arc(0, 0, openfst.Weight.NoWeight(fst.weight_type()), mid))
    for (fro, to) in mapping:
        fst.add_arc(mid, openfst.Arc(fro+1, to+1, openfst.Weight.NoWeight(fst.weight_type()), mid))
    fst.add_arc(mid, openfst.Arc(0, 0, openfst.Weight.NoWeight(fst.weight_type()), end))
    fst.arcsort(st="olabel")
    
    return fst
    
def make_mapping_loop_fst(mapping):
    lines = []
    start = '0'
    mid = '1'
    end = '2'
    
    lines.append('%s %s 0 0'%(start, mid))    
    for (fro, to) in mapping:
        lines.append('%s %s %s %s'%(mid, mid, fro+1, to+1))

    lines.append('%s %s 0 0'%(mid, end))    
    lines.append(str(end))
    
    compiler = openfst.Compiler()    
    for line in lines:
        print >> compiler, line
        #compiler.write(line)   ## TODO test
    f = compiler.compile()
    f.arcsort(st="olabel")
    return f
    
        
    
def make_t_lattice_SIMP2(dist, ind):
    '''
    version 2 -- don't go via strings
    '''
    
    #print 'target indexes'
    #print ind
    #print 
    #print 
    fst = openfst.Fst()
    start = fst.add_state()
    fst.set_start(start)
    
    frames, cands = np.shape(dist)

    #frames = 3

    for i in range(frames):
    
        end = start + 1
        for j in range(cands):
            
            frame_ix = ind[i,j] 
            weight = dist[i,j] 
            fst.append('%s %s %s %s %s'%(start, end, frame_ix+1, frame_ix+1, weight)) 
            
        start = end
    fst.append('%s'%(end)) 
    
    compiler = openfst.Compiler()    
    for line in fst:
        print >> compiler, line
        #compiler.write(line)   ## TODO test
    f = compiler.compile()
    f.arcsort(st="olabel")
    return f


def make_sausage_lattice(probs, weight_factor=1.0):
    '''
    similar to make_t_lattice_SIMP-- TODO: refactor
    '''
    
    fst = [] #  openfst.Fst()
    #start = fst.add_state()
    #fst.set_start(start)
    start = 0

    frames, classes = np.shape(probs)
    weights = probs * weight_factor    
    weights = -1.0 * np.log(weights) ## neg nat log prob
   
    for i in range(frames):
    
        end = start + 1
        for class_num in range(classes):
            
            weight = weights[i,class_num] 
            fst.append('%s %s %s %s %s'%(start, end, class_num+1, class_num+1, weight)) 
            
        start = end
    fst.append('%s'%(end)) 
    
    compiler = openfst.Compiler()    
    for line in fst:
        print >> compiler, line
        #compiler.write(line)   ## TODO test
    f = compiler.compile()
    f.arcsort(st="olabel")
    return f


## As cost_cache_to_text_fst but don't write anything -- compile directly
def cost_cache_to_compiled_fst(cost_cache, join_cost_weight=1.0, keep_weights=True, \
                                                                         final_frame_list=[]):
    '''
    Take dict-like cost_cache of form {(from_index, to_index): weight} 
    From and to indices index frames of data -- starting from zero (1 is added within
    function to satisfy FST format)
    '''
    fst = []
    frames = []
    for (fro, to) in cost_cache.keys():
        frames.extend([fro, to])
    frames = sorted(list(set(frames)))  # unique
    frame2state = dict(zip(frames, range(1,len(frames)+1)))
    
    if len(final_frame_list) == 0:
        # allow to jump in and out anywhere
        initial_frames = frames
        final_frames = frames
    else:
        initial_frames = (final_frame_list + 1)#[:-1]  ## could also prepend 0
        final_frames = final_frame_list

    ## ways in from start state 0 (unweighted, with epsilon in and out symbols):
    for frame in initial_frames:
        fst.append('0 %s 0 0'%(frame2state[frame]))  
    
    ## real transitions:
    for ((fro, to), weight) in cost_cache.items():
        if keep_weights:
            fst.append('%s %s %s %s %s'%(frame2state[fro], frame2state[to], fro+1, fro+1, weight))
        else:
            fst.append('%s %s %s %s'%(frame2state[fro], frame2state[to], fro+1, fro+1))
                        
    ## out transitions (labels, no wirghts)
    sink_state = len(frames)+1
    for frame in final_frames:
        fst.append('%s %s %s %s'%(frame2state[frame], sink_state, frame+1, frame+1))         
    
    fst.append('%s'%(sink_state))
    
    compiler = openfst.Compiler()    
    for line in fst:
        print >> compiler, line
    f = compiler.compile()
    #f.arcsort(st="olabel")
    return f

    

def cost_cache_to_text_fst(cost_cache, outfile, join_cost_weight=1.0, keep_weights=True, \
                                                                         final_frame_list=[]):
    '''
    Take dict-like cost_cache of form {(from_index, to_index): weight} 
    From and to indices index frames of data -- starting from zero (1 is added within
    function to satisfy FST format)
    '''
    fst = []
    
    frames = []
    for (fro, to) in cost_cache.keys():
        frames.extend([fro, to])
    frames = sorted(list(set(frames)))  # unique
    
    frame2state = dict(zip(frames, range(1,len(frames)+1)))
    

    
#     if len(final_frame_list) == 0:
#         # HACK -- allow to jump in and out every N frames
#         initial_frames = [f+1 for (i,f) in enumerate(frames[:-5]) if i%20==0]
#         final_frames = [f for (i,f) in enumerate(frames) if i%20==0]
    if len(final_frame_list) == 0:
        # allow to jump in and out anywhere
        initial_frames = frames
        final_frames = frames
    else:
        initial_frames = (final_frame_list + 1)#[:-1]  ## could also prepend 0
        final_frames = final_frame_list

    
    #print frame2state
    #print initial_frames
    #print final_frames
    
    ## ways in from start state 0 (unweighted, with epsilon in and out symbols):
    for frame in initial_frames:
        fst.append('0 %s 0 0'%(frame2state[frame]))  
    
    ## real transitions:
    for ((fro, to), weight) in cost_cache.items():
        if keep_weights:
            fst.append('%s %s %s %s %s'%(frame2state[fro], frame2state[to], fro+1, fro+1, weight))
        else:
            fst.append('%s %s %s %s'%(frame2state[fro], frame2state[to], fro+1, fro+1))
                        
    ## out transitions (labels, no wirghts)
    sink_state = len(frames)+1
    for frame in final_frames:
        fst.append('%s %s %s %s'%(frame2state[frame], sink_state, frame+1, frame+1))         
    
    
    
    fst.append('%s'%(sink_state))
    
    f = open(outfile, 'w')
    for line in fst:
        f.write(line + '\n')
    f.close()
    




def get_best_path_SIMP(t_latt, j_latt, join_already_compiled=False, add_path_of_last_resort=False):
    '''
    t_latt and j_latt: FST objects in memory
    '''

    ## TODO temp assertions:
    #assert join_already_compiled
    
    if not join_already_compiled:
        comm('%s/fstcompile %s %s.bin'%(tool, j_latt, j_latt))
        compiled_j_latt = j_latt + '.bin'
        sys.exit('complete porting to WRAP case wsoivnsovbnsfb34598t3h')
    else:
        compiled_j_latt = j_latt 
        if add_path_of_last_resort: ## only makes sense if join_already_compiled
            
            ## In order that composition with precomputed join won't remove all paths, add
            ## emergency path of last resort, which we arbitrarily take to be the best
            ## path through target lattice without regard to join cost:
            path_of_last_resort = get_shortest_path(t_latt)
        
            print
            print '----------------POLR----------------'
            print path_of_last_resort
            print '------------------------------------'
            print
        
            print 'convert back to FST indexing'
            path_of_last_resort = [val + 1 for val in path_of_last_resort]
            
            print 'edit J to add emergency arcs'
            POLR_transitions = {}
            for fro, to in zip(path_of_last_resort[:-1], path_of_last_resort[1:]):
                if fro not in POLR_transitions:
                    POLR_transitions[fro] = []
                POLR_transitions[fro].append(to)
                    
            
            #print j_latt
            
            print POLR_transitions
            # print 
#             
#             print j_latt.verify()
#             
#             print help(j_latt)
#             
#             for arc in j_latt.arcs():
#                 print arc

            print 'here b'


            for from_state in j_latt.states():
                if from_state in POLR_transitions:
                    for arc in j_latt.arcs(from_state):
                        to_state = arc.nextstate
                        if to_state in  POLR_transitions[from_state]:
                            POLR_transitions[from_state].remove(to_state)
                    ## what is left must be added:
                    #print POLR_transitions[from_state]

            print POLR_transitions
            for (fro, to_list) in POLR_transitions.items():
                if to_list == []:
                    del POLR_transitions[fro]

            print POLR_transitions
            
            print j_latt.weight_type()
            BIGWEIGHT = openfst.Weight(j_latt.weight_type(), 500000000)
            for from_state in POLR_transitions.keys():
                for to_state in POLR_transitions[from_state]:
                    j_latt.add_arc(from_state, openfst.Arc(from_state, from_state, BIGWEIGHT, to_state))
                
            assert j_latt.verify()    
            #print j_latt
            
            
            ### TODO -- remove added arcs after search to resuse J!!!!!!
            
    #comm('%s/fstarcsort --sort_type=olabel %s.bin %s.bin.srt'%(tool, t_latt, t_latt))

    c_latt = openfst.compose(t_latt, j_latt) 
    
    #print ' ---- CLATT ---'
    #print c_latt
    
    #'/tmp/comp.fst'  # 
    #comm('%s/fstcompose %s.bin.srt %s %s'%(tool, t_latt, compiled_j_latt, c_latt)) ## TODO check if comp is empty and report nicely

    shortest_path = get_shortest_path(c_latt)
    
    # print
    # print '----------------shortest path----------------'
    # print shortest_path
    # print '------------------------------------'
    # print
            
            
    return shortest_path

def get_shortest_path(fst_in, quiet=True):

    s = openfst.shortestpath(fst_in, weight=None) ## weight: 
             # A Weight or weight string indicating the desired weight threshold
             # below which paths are pruned; if omitted, no paths are pruned.

    if not quiet:
        print s

    ## reverse order when printing -- TODO investigate why and use proper operations to extract path
    data = [line.split('\t') for line in s.text().split('\n')]
    data = [line for line in data if len(line) in [4, 5]]
    data = [(int(line[0]), int(line[2])) for line in data] # (i,o,lab1,lab2,[weight])
    data.sort()
    data.reverse()
    
    data = [frame-1 for (index, frame) in data if frame != 0] ## remove epsilon 0
    #            ^--- back to python indices
    
#    print 'shortest path FST:'
#    print s
    return data



def test():

    print 'TARGET'
    
    ind = np.array([[4,8],[4,7],[2,5]])
    dist = np.array([[0.1, 0.3], [0.33, 0.25], [0.01, 0.41]])
    T = make_t_lattice_SIMP(dist, ind)
    print T
    sys.exit('sbsfrnsfrn')
    
    comm(TOOL + '/fstdraw /tmp/T_test.bin > /tmp/T_test.dot')
    comm('dot -Tpdf /tmp/T_test.dot > /tmp/T_test.pdf')
    
    from util import latex_matrix
    import random
    
    print latex_matrix(ind)
    print latex_matrix(dist)
    

    print 'JOIN'
    frames = sorted(list(set(ind.flatten().tolist())))
    print frames
    frames = range(4) # range(max(frames) + 2)
    join = {}
    for i in frames:
        for j in frames:
            val = random.randint(0, 2000)
            if val < 1000:
                join[(i,j)] = val
    
    cost_cache_to_text_fst(join,  '/tmp/J_test.txt' )
    compile_fst(TOOL, '/tmp/J_test.txt' , '/tmp/J_test.bin' )

    comm(TOOL + '/fstdraw /tmp/J_test.bin > /tmp/J_test.dot')
    comm('dot -Tpdf /tmp/J_test.dot > /tmp/J_test.pdf')
    comm('pdfcrop /tmp/J_test.pdf')
    
    
    
    print 'UNION'
#     frames = range(5,8) # range(max(frames) + 2)
#     join = {}
#     for i in frames:
#         for j in frames:
#             val = random.randint(0, 2000)
#             if val < 1000:
#                 join[(i,j)] = val
    
    join = {(2,3): 0.5, (3,2): 0.4}
    
    cost_cache_to_text_fst(join,  '/tmp/J_test2.txt' )
    compile_fst(TOOL, '/tmp/J_test2.txt' , '/tmp/J_test2.bin' )

    comm(TOOL + '/fstunion /tmp/J_test.bin /tmp/J_test2.bin /tmp/union.bin')

    comm(TOOL + '/fstdraw /tmp/union.bin > /tmp/union.dot')
    comm('dot -Tpdf /tmp/union.dot > /tmp/union.pdf')
    comm('pdfcrop /tmp/union.pdf')



def test2():
    cost_cache = {(0,1): 0.0, (0,2): 0.0, (0,3): 0.0, (1,0): 0.0, (1,1): 0.0, (3,0): 0.0, (3,4): 0.0, (4,4): 0.0}
    outfile = '/tmp/test.fst'
#    cost_cache_to_text_fst(cost_cache, outfile, keep_weights=False)
    cost_cache_to_text_fst(cost_cache, outfile, keep_weights=False, final_frame_list=np.array([2]))

#### -------- FST & LM -----------

def compile_lm_fst(kaldi, openfst, arpa_fname, compiled_fname):


    ## arpa -> wfst
    # http://vpanayotov.blogspot.co.uk/2012/06/kaldi-decoding-graph-construction.html
    
    symfile = arpa_fname + '.sym'
    
    arpa2fst = kaldi + '/src/bin/arpa2fst '
    eps2disambig = kaldi +'/egs/wsj/s5/utils/eps2disambig.pl '
    # This script replaces epsilon with #0 on the input side only, of the G.fst
    # acceptor.      
        
    s2eps = kaldi + '/egs/timit/s5/utils/s2eps.pl '
    # This script replaces <s> and </s> with <eps> (on both input and output sides),
    # for the G.fst acceptor.

    fstprint = openfst + '/fstprint '
    fstcompile = openfst + '/fstcompile '
    fstrmepsilon = openfst + '/fstrmepsilon '

#     cl =  'cat ' + arpa_fname + ' | '
#     cl += "grep -v '<s> <s>' | "
#     cl += "grep -v '</s> <s>' | "
#     cl += "grep -v '</s> </s>' | "
#     cl += arpa2fst + " - | "
#     cl += fstprint  + ' | '            
#     cl += eps2disambig + ' | '
#     cl += s2eps + ' | '
#     cl += fstcompile + ' --isymbols=' + symfile 
#     cl += ' --osymbols='+symfile           
#     cl += ' --keep_isymbols=false --keep_osymbols=false | '
#     cl += fstrmepsilon + ' > ' + compiled_fname
# 
# 

    cl =  'cat ' + arpa_fname + ' | '
    cl += "grep -v '<s> <s>' | "
    cl += "grep -v '</s> <s>' | "
    cl += "grep -v '</s> </s>' | "
    cl += arpa2fst + " - | "
    cl += fstprint 
#    cl += " | sed 's/<eps>\t<eps>/100000\t0/g' "  ## Use  100000 for \#0
    cl += " | sed 's/<eps>\t<eps>/0\t0/g' "  ## NO!!!!!!!
    cl += " | sed 's/<\/s>/0/g' "
    cl += " | sed 's/<s>/0/g' | "    
    cl += fstcompile + ' | '
    cl += fstrmepsilon + ' > ' + compiled_fname

    comm(cl)


def plot_fst(F):
    F.draw('/tmp/draw.dot')
    comm('dot -Tpdf /tmp/draw.dot > /tmp/draw.pdf') 
    comm('open /tmp/draw.pdf') 
    
    
def extract_path(p):    ## TODO -- rationale, necessary??

    lines = [line.split('\t') for line in p.text().split('\n')]
    lines = [line for line in lines if len(line) == 4]
    frames = [int(line[3]) for line in lines]
    frames = [f-1 for f in frames if f > 0] # remove eps; decrement to python indexing
    return frames
    
def compile_simple_lm_fst(tool, gramdict, outfile, twograms=None):
    '''
    write e.g. 3-grams without backoff to lower order grams
    TODO add 2 grams
    '''
    print 'in  compile_simple_lm_fst  ....'
    order = len(gramdict.keys()[0]) ## assume all entries of consistent order
    assert order > 1
    
    ## get unique list of symbols:
    vocab = {}
    for gram in gramdict.keys():
        vocab.update(dict(zip(gram,gram)))
    vocab = sorted(vocab.keys())
#     print gramdict
#     print vocab
#     print range(-2, max(vocab)+1)
    
    ## original vocab will be Python indices into data, plus -1 for start and -2 for stop
    assert vocab == range(-2, max(vocab)+1)
    
    ## map original vocab to one suitable for OpenFst, where 0 is reserved for epsilon symbol
    ## -- increment Python indices by one; map -2 and -1 to integers larger than vocab size.
    mapper = dict(zip(vocab, (np.array(vocab)+1).tolist()))    
    start_symbol = max(vocab)+2
    end_symbol = max(vocab)+3
    mapper[-1] = start_symbol
    mapper[-2] = end_symbol
    
    ## rewrite grams using these new symbols:
    mapped_gramdict = {}
    for (gram,score) in gramdict.items():
        mapped_gram = tuple([mapper[symbol] for symbol in gram])
        mapped_gramdict[mapped_gram] = score
    gramdict = mapped_gramdict
    
    ## get unique list of possible states
    states = {}
    for (gram,score) in gramdict.items():
        for state in [gram[:-1], gram[1:]]:
            states[state] = 0
    states = sorted(states.keys())
#     print states 
    
    ## map possible states to nonzero positive integer
    states = dict(zip(states, range(1,len(states)+1)))
#     print states
#     print type(states.keys()[0][0])
#     #sys.exit('ff')
#     print start_symbol
#     print end_symbol
#     print sorted(gramdict.keys())
#     
#     print states
   
    if twograms:  ## this is for debugging purposes only; actually, n-1 grams is a better name...

        ## then check all states are in 2-grams for sanity:
        twogramlist = {}
        for gram in twograms.keys():
            mapped_gram = tuple([mapper[symbol] for symbol in gram])
            twogramlist[mapped_gram] = 0
        
        for state in states.keys():
            if state not in twogramlist:
                print twogramlist
                print state
                sys.exit('sfbtndgnfrnfhn948934879')
    
#     sys.exit('rrfdtgn')
    
    
    print '         compile_simple_lm_fst A (time consuming...)'
    
    ## start state = 0; add transitions into all states beginning with start symbol
    ## Note that the 1st n-1 symbols of these states won't be emitted. TODO: option
    ## to allow start from any state?
    fst = []
    o = order
    for (state, state_number) in states.items():
        if state[0] == start_symbol:
 #            print state
#             print state_number
#             print '^^^^^-=-start'
            fst.append('0 %s 0 0'%(state_number))  ## epsilon symbols; no weight
    

    start_time = timeit.default_timer()
    if False: ## OLD method -- this is crazily inefficient -- iterates over many non-existant ngrams!
        if order == 2: ## then no overlap between consecutive states
            sys.exit('sedsrswrg')
        elif order >= 3:
            o = order-2  ## overlap between states which must match
            for (state_i, state_number_i) in states.items():
                for (state_j, state_number_j) in states.items():
                    if state_i[-o:] == state_j[:o]:
                        emitted = state_j[-1]
                        gram = tuple(list(state_i) + [emitted])

                        if gram in gramdict:               

                            score = gramdict[gram]
                            if emitted == end_symbol:
                                emitted = '0' ## epsilon -- end symbol is not a frame index
                            fst.append('%s %s %s %s %s'%(state_number_i, state_number_j, emitted, emitted, score))
        else:
            sys.exit('bad order 34983948')    

    else:  ## NEW METHOD
        if order == 2: ## then no overlap between consecutive states
            sys.exit('sedsrswrg')
        elif order >= 3:
            #o = order-2  ## overlap between states which must match
            for (gram, score) in gramdict.items():
                state_i = gram[:order-1] 
                state_j = gram[-(order-1):]
                #print gram
                #print state_i
                #print state_j                
                state_number_i = states[state_i]
                state_number_j = states[state_j]

                emitted = gram[-1]
                if emitted == end_symbol:
                    emitted = '0' ## epsilon -- end symbol is not a frame index
                fst.append('%s %s %s %s %s'%(state_number_i, state_number_j, emitted, emitted, score))
                            
        else:
            sys.exit('bad order 34983948')        

    end_time = timeit.default_timer()
    print 'gram training took %.2fm' % ((end_time - start_time) / 60.)

    #sys.exit('hedbdbd')
        
    end_state_number = max(states.values())+1
    for (state, state_number) in states.items():
        if state[-1] == end_symbol:
            fst.append('%s %s 0 0'%(state_number, end_state_number))  ## epsilon symbols; no weight 
               
    fst.append('%s'%(end_state_number)) 
    

    print '         compile_simple_lm_fst B '
        
            
    f = open(outfile + '.txt', 'w')
    for line in fst:
        f.write(line + '\n')
    f.close()
        

    print '         compile_simple_lm_fst C '
                
    
    compile_fst(tool, outfile + '.txt', outfile)


    print 'finished  compile_simple_lm_fst  ....'


def sample_fst(fst, nsequences=10, init_seed=1234, noisy=True):
    paths = []
    random.seed(init_seed)
    for i in range(nsequences):
        newseed = random.randint(0,100000)
        p = openfst.randgen(fst, max_length=100000000, npath=1, remove_total_weight=False, \
                seed=newseed, select="log_prob", weighted=True)        
        if noisy:
            print p   
        path = extract_path(p)
        paths.append(path)
    return paths




if __name__ == '__main__':

    test2()




