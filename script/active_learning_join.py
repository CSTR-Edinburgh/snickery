
import sys
import os

import numpy
import numpy as np
import h5py
import shelve
import sqlite3
import random

import pylab

from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.callbacks import EarlyStopping

from synth_halfphone import Synthesiser
from train_halfphone import get_data_dump_name

from util import cartesian

from sklearn.neighbors import KDTree as sklearn_KDTree

from speech_manip import read_wave, write_wave

class JoinDatabaseForActiveLearning(Synthesiser):

    ## enable use of this class with with statement, to close database connection: 
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.have_sql_db:
            self.connection.close()

    def train_classifier(self, data, labels, architecture=[512,512,512], activation='relu', batch_size=64, max_epoch=30, patience=5):

        # print data[:1,:]
        # mu,sig = (data.mean(axis=0), data.std(axis=0))
        # mu = mu.reshape((1,-1))
        # sig = sig.reshape((1,-1))
        # data = (data - mu) / sig
        # print data[:1,:]

        ## handle large negative values (unvoiced markers): TODO: fix this temporary hack!
        min_normal = data[data>-500.0].min()
        data[data<-500.0] = min_normal - 0.1


        m,n = data.shape
        outsize = int(np.max(labels)+1)

        model = Sequential()

        layer_size = architecture[0]
        model.add(Dense(units=layer_size, input_dim=n))
        model.add(Activation(activation))

        for layer_size in architecture[1:]:
            model.add(Dense(units=layer_size))
            model.add(Activation(activation))

        model.add(Dense(units=outsize))
        model.add(Activation('softmax'))

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

        earlyStopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='min')

        model.fit(data, labels, epochs=max_epoch, batch_size=batch_size, callbacks=[earlyStopping], validation_split=0.10, shuffle=True)

        self.classifier = model

        predictions = model.predict(data)
        
        np.save('/afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/temp/scores', predictions[:,0])


        ## TODO: save model to disk        



    def __init__(self, config_file):
        super(JoinDatabaseForActiveLearning, self).__init__(config_file)
        
        self.have_sql_db = False

###### for full AL -- sep method?!?        
#         join_data_dump = get_data_dump_name(self.config, joindata=True)
# ###        join_data_sql_fname = get_data_dump_name(self.config, joinsql=True)

#         if not os.path.isfile(join_data_dump):
#             sys.exit('Join data file %s does not exist -- try setting dump_join_data to True in config and rerunning train_halfphone.py')

#         f = h5py.File(join_data_dump, "r")
#         self.start_join_feats = f["start_join_feats"][:,:]
#         self.end_join_feats = f["end_join_feats"][:,:]    
#         f.close()


    def exhaustive_check_experiment(self, lattice_file):

        self.judgement_classes = ['good','bad']
        self.class_dict = dict([(thing[0], thing) for thing in self.judgement_classes])
        assert len(self.class_dict) == len(self.judgement_classes), 'Please use unique starting letters for judgement_classes'

        join_db = {}
        #join_db = {(109408, 20923): 'g', (421575, 416141): 'g', (1028819, 1747): 'b', (971825, 99548): 'b', (609357, 416357): 'g', (416357, 436180): 'g', (601693, 40023): 'g', (971825, 165641): 'b', (1028819, 892191): 'b', (1028819, 971825): 'b', (80306, 186235): 'g', (971825, 941687): 'b', (186235, 892191): 'b', (1315491, 1028819): 'g', (126669, 1178306): 'g', (1315491, 892192): 'b', (109319, 615853): 'g', (971825, 1749): 'b', (1028819, 73128): 'b', (1315491, 1315491): 'g', (1028819, 879261): 'b', (971825, 554068): 'b', (1178306, 885622): 'g', (879006, 879006): 'g', (971825, 1179002): 'b', (888070, 194028): 'g', (1028819, 1028819): 'g', (1173541, 19459): 'g', (879006, 879008): 'g', (1028819, 600461): 'b', (1126966, 99548): 'g', (48527, 188293): 'g', (99548, 901945): 'g', (1028819, 1746): 'b', (971825, 1747): 'b', (194028, 194029): 'g', (140202, 609357): 'g', (1028819, 58340): 'b', (38402, 879006): 'g', (416357, 436179): 'b', (19459, 38406): 'g', (127490, 127490): 'g', (1028819, 887531): 'b', (1028819, 892190): 'b', (1220901, 46399): 'g', (1315491, 892191): 'b', (971825, 941686): 'b', (1028819, 50253): 'g', (615853, 126669): 'g', (732938, 732938): 'g', (54102, 27845): 'g', (1315491, 1749): 'b', (971825, 1126967): 'b', (601693, 988698): 'b', (186235, 1315491): 'g', (732938, 1220901): 'g', (127966, 888070): 'g', (27845, 180708): 'g', (194029, 127490): 'g', (27845, 27845): 'g', (188293, 617393): 'g', (1026159, 885310): 'g', (80306, 80306): 'g', (1028819, 1749): 'b', (1028819, 532506): 'b', (609357, 613274): 'b', (617393, 593460): 'g', (416141, 127966): 'g', (416357, 427489): 'b', (1028819, 892193): 'b', (180708, 109408): 'g', (46399, 109319): 'g', (127490, 80306): 'g', (971825, 165642): 'b', (1315491, 892190): 'b', (593460, 421575): 'g', (38406, 54102): 'g', (609357, 609357): 'g', (436180, 601693): 'g', (80306, 589341): 'b', (971825, 1126966): 'g', (1315491, 892189): 'b', (901945, 140202): 'g', (40023, 584319): 'g', (879008, 1026159): 'g', (1315491, 892193): 'b', (1315491, 1747): 'b', (80306, 589342): 'b', (971825, 971825): 'g', (1028819, 904789): 'b', (1028819, 73129): 'b', (20923, 732938): 'g', (584319, 38402): 'g', (1028819, 1126967): 'b', (416357, 601693): 'b', (50253, 971825): 'g', (885310, 1173541): 'g', (1028819, 892192): 'b', (1315491, 532506): 'b'}

        lattice = np.load(lattice_file)

        #lattice = lattice[10:15,:]  #### !!!!!
        m, n = lattice.shape
        fro = lattice[0,0]
        path = [fro]

        
        for i in range(1,m):
            print '========'
            print i
            for to in lattice[i,:]:
                print (to, fro)
                if fro==to or fro-to in range(0,6):
                    print 'other constraint...'
                    continue
                if (fro, to) in join_db:
                    judgement = join_db[(fro, to)]
                else:
                    # judgement = self.present_candidate_joins_single_foreign_unit([(fro, to)], paired=True)
                    judgement = self.present_candidate_joins_nonsense([(fro, to)])
                    judgement = judgement[0][0]
                    join_db[(fro, to)] = judgement              
                if judgement == 'g':   
                    path.append(to)  
                    fro = to               
                    break
        print 
        print join_db
        print 
        print path
        print 
        self.concatenate(path, '/tmp/chosen_path.wav')


        # for i in range(1,m):
        #     print '========'
        #     print i
        #     transitions = cartesian([lattice[i,:], lattice[i+1,:]])
        #     for (fro,to) in transitions:
        #         judgement = self.present_candidate_joins_single_foreign_unit([(fro, to)])
        #         judgement = judgement[0]
        #         join_db[(fro, to)] = judgement[0]                
        #         if judgement == 'good':                    
        #             break
        # print join_db
        




 

    def analyse_candidate_types(self):
        self.mode_of_operation = 'find_join_candidates'
        flist = self.get_sentence_set('tune') #[:20]
        negative_sample_pool = {}
        data = []
        for fname in flist:
            candidates = self.synth_utt(fname, synth_type='tune')
            m,n = candidates.shape
            for i in xrange(m-1):
                for fro in candidates[i,:]:
                    for to in candidates[i+1,:]:
                        if fro != to-1 and fro >= 0 and to >= 0:
                            negative_sample_pool[(fro, to)] = 0      
            data.append(len(negative_sample_pool))      
        
        for (nutts, size) in enumerate(data):
            print '%s     %s'%(nutts+1, size)



    def initialise_join_table_with_knn(self, k):

        SQL=False
        if SQL:
            join_data_sql_fname = get_data_dump_name(self.config, joinsql=True)
            self.setup_join_database(join_data_sql_fname, connect_to_existing=False)  
                                            # creates self.connection & self.cursor
        else:
            self.join_db = {}

        unit_end_data = self.unit_end_data # [:200,:]
        unit_start_data = self.unit_start_data # [:200,:]

        t = self.start_clock('grow KD tree' ) ## c. 1 min on all FLS
        tree = sklearn_KDTree(unit_end_data, leaf_size=100, metric='euclidean')
        self.stop_clock(t)

        t = self.start_clock('get neighbours')
        dists, indices = tree.query(unit_start_data, k=k)
        self.stop_clock(t)

        t = self.start_clock('add neighbours to database')
        (njoin, nn) = unit_start_data.shape
        for i in xrange(njoin):
            for j in indices[i,:]:
                self.add_join_candidate_to_database(i,j,sql=SQL)
        if SQL:
            self.connection.commit()                
        self.stop_clock(t)                



    def initialise_join_table_with_heldout_data(self):

        t = self.start_clock('find candidate neighbours with heldout data')
        self.mode_of_operation = 'find_join_candidates'
        flist = self.get_sentence_set('tune')[:20]
        negative_sample_pool = {}
        for fname in flist:
            candidates = self.synth_utt(fname, synth_type='tune')
            #print candidates
            m,n = candidates.shape
            for i in xrange(m-1):
                for fro in candidates[i,:]:
                    for to in candidates[i+1,:]:
                        if fro != to-1 and fro >= 0 and to >= 0:
                            #print (self.train_unit_names[fro], self.train_unit_names[to])
                            if (fro, to) not in negative_sample_pool:
                                negative_sample_pool[(fro, to)] = 0 
                            negative_sample_pool[(fro, to)] += 1     
        self.stop_clock(t)

    

        SQL=False
        if SQL:
            join_data_sql_fname = get_data_dump_name(self.config, joinsql=True)
            self.setup_join_database(join_data_sql_fname, connect_to_existing=False)  
                                            # creates self.connection & self.cursor
        else:
            self.join_db = {}



        t = self.start_clock('add neighbours to database')
        # (njoin, nn) = unit_start_data.shape
        for (i,j) in negative_sample_pool.keys():
            self.add_join_candidate_to_database(i,j,sql=SQL)
        if SQL:
            self.connection.commit()                
        self.stop_clock(t)                


    def train_al_classifier_01(self):
        self.join_db



    def run_al_session(self, initial=False):
        if initial:
            ## get seed set:
            nseed = 5
            # self.cursor.execute("SELECT from_index, to_index FROM transitions ORDER BY RANDOM() LIMIT %s"%(nseed)) 
            # seedset = self.cursor.fetchall()

            # print seedset
            # self.cursor.execute("SELECT from_index, to_index FROM transitions ORDER BY RANDOM() LIMIT %s"%(nseed)) 
            # seedset = self.cursor.fetchall()

            cases = self.join_db.keys()[:3]   ### TODO: randomise properly

        else:
            ## get uncertain cases
            # cases = ...
            #self.classifier()
            pass 

        self.judgement_classes = ['good','bad']
        self.class_dict = dict([(thing[0], thing) for thing in self.judgement_classes])
        assert len(self.class_dict) == len(self.judgement_classes), 'Please use unique starting letters for judgement_classes'

        user_judgements = self.present_candidate_joins_single_foreign_unit(cases)

        ## add judgments to database:
        for ((fro, to), judgement) in zip(cases, user_judgements):
            points = [(fro, to), (to, fro+1)]    ### each stimulus gives TWO datapoints....

            for (a,b) in points:
                natural = False ### for now, never include natural...
                labelled=True
                labelled_ok={'good': True, 'bad':False}[judgement]
                trained_on=False
                confidence=0.5            

                self.join_db[(i,j)] = (natural, labelled, labelled_ok, trained_on, confidence)                
                

        #### train model:
        # train_data = 





    def get_noncontig_joins(self):

        self.mode_of_operation = 'find_join_candidates'
        flist = self.get_sentence_set('tune') #[:20]
        all_candidates = [self.synth_utt(fname, synth_type='tune') for fname in flist]
        negative_sample_pool = {}
        for candidates in all_candidates:
            m,n = candidates.shape
            for i in xrange(m-1):
                for fro in candidates[i,:]:
                    for to in candidates[i+1,:]:
                        if fro != to-1 and fro >= 0 and to >= 0:
                            negative_sample_pool[(fro, to)] = 0            
        negative_sample_pool = negative_sample_pool.keys()
        return negative_sample_pool

    def classifier_approach(self):

        ## find positive sample:
        positive_sample_pool = []        
        for fro in xrange(self.number_of_units-1):
            to = fro + 1
            if self.train_filenames[fro] == self.train_filenames[to]:
                positive_sample_pool.append((fro, to))

        ## find negative sample:
        negative_sample_pool = self.get_noncontig_joins()

        print len(negative_sample_pool)
        print len(positive_sample_pool)
        

        ### train initial classifier on subset:
        random.shuffle(positive_sample_pool)
        random.shuffle(negative_sample_pool)


        #sample_halfsize = 100  # default: len(positive_sample_pool)
        sample_halfsize = len(positive_sample_pool)
        positive_samples_subset = positive_sample_pool[:sample_halfsize]
        negative_samples_subset = negative_sample_pool[:sample_halfsize]
        ixx = np.array(positive_samples_subset + negative_samples_subset, dtype=int)
        labels = np.concatenate([np.zeros(sample_halfsize), np.ones(sample_halfsize)])
        from_ixx = ixx[:,0]
        to_ixx = ixx[:,1]


        train_examples = np.hstack([self.end_join_feats[from_ixx,:], self.start_join_feats[to_ixx,:]])
        self.train_classifier(train_examples, labels, architecture=[1024,1024,1024,1024,1024,1024], max_epoch=30, batch_size=512)
        np.save('/afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/temp/ixx', ixx)
        
        import pylab

        ixx = np.load('/afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/temp/ixx.npy')
        scores = np.load('/afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/temp/scores.npy')



        scores = zip(scores, ixx.tolist())
        # print scores
        negscores = scores[(len(scores)/2):]
        negscores.sort()
        # pylab.plot(negscores)
        # pylab.show()
        #print negscores

        ### synthesise joins at 100 points along whole negative scale:
        to_synth = np.linspace(0, len(negscores), 100, dtype=int)

        for ix in to_synth:
            print ix
            score, (fro, to) = negscores[ix]
            name = 'aaa_' + str(ix).zfill(20) + '-' + str(score)
            self.make_join_stimulus(fro, to, context=20, make_natural=False, name_prefix=name)


        # self.classifier.predict()


        sys.exit('wevwrv')





        
        ## get seed set:
        nseed = 20
        self.cursor.execute("SELECT from_index, to_index FROM transitions ORDER BY RANDOM() LIMIT %s"%(nseed)) 
        seedset = self.cursor.fetchall()

        print seedset
        self.cursor.execute("SELECT from_index, to_index FROM transitions ORDER BY RANDOM() LIMIT %s"%(nseed)) 
        seedset = self.cursor.fetchall()


        self.judgement_classes = ['good','bad']
        self.class_dict = dict([(thing[0], thing) for thing in self.judgement_classes])
        assert len(self.class_dict) == len(self.judgement_classes), 'Please use unique starting letters for judgement_classes'

        print dir(self)
        user_judgements = self.present_candidate_joins_single_foreign_unit(seedset)

        print user_judgements

        print seedset





        sys.exit('stop before autoencoder stuff')


        natural_junctures = self.get_natural_junctures()
        self.train_autoencoder(natural_junctures, max_epoch=3) ## creates self.autoencoder

        natural_scores = self.get_prediction_errors(natural_junctures)


        fake_scores = []
        for i in xrange(self.number_of_units):
            if i % 100 == 0:
                print i
            self.cursor.execute("SELECT to_index FROM transitions WHERE from_index=%s"%(i)) 
            to_index = self.cursor.fetchall()
            #print to_index
            to_index = np.array(to_index, dtype=int).flatten()
            #print to_index

            ## Remove natural transitions:- TODO: in SQL
            to_index = to_index[to_index!=i+1]

            from_ixx = np.ones(to_index.shape, dtype=int) * i
            features = np.hstack([self.end_join_feats[from_ixx,:], self.start_join_feats[to_index,:]])
            fake_scores.append(self.get_prediction_errors(features))

        fake_scores = np.concatenate(fake_scores)

        #pylab.subplot(211)
        pylab.hist(natural_scores, bins=30, alpha=0.6, color='g', normed=1)

        #pylab.subplot(212)
        pylab.hist(fake_scores, bins=30, alpha=0.6, color='b', normed=1)

        pylab.show()






        sys.exit('sdvsfbv')


        print natural_scores


        unit_seq = self.sample_uniform_a(nsteps=800)
        self.concatenate(unit_seq, '/tmp/out.wav')
        

    def present_candidate_joins_nonsense(self, index_pairs):
        CONTEXT = 20
        judgements = []
        for (fro, to) in index_pairs:
            print '======== testing join (nonsense method) ========'
            print (fro, to)            
            start = max(0, fro - CONTEXT)
            end = min(self.number_of_units, to + CONTEXT)
            fname = self.get_join_stimulus_filename(fro, to)
            self.concatenate(range(start, fro) + range(to, end), fname)
            judgements.append(self.present_audio_instance(fro, to))
        return judgements




    def get_join_stimulus_filename(self, fro, to, natural=False, prefix='join'):
        stimulus_dir = '/tmp/'
        if natural:
            assert prefix != 'paired'
            return '%s/%s_%s_%s_nat.wav'%(stimulus_dir, prefix, fro, to)
        else:
            return '%s/%s_%s_%s.wav'%(stimulus_dir, prefix, fro, to)

    def pair_fake_and_natural_joins(self, fro, to, random_order=False, silence_secs=0.75):
        fake_fname = self.get_join_stimulus_filename(fro, to, prefix='join')
        nat_fname = self.get_join_stimulus_filename(fro, to, prefix='join', natural=True)
        fnames = [fake_fname, nat_fname]
        if random_order:
            random.shuffle(fnames)


        wave_a, fs = read_wave(fnames[0])
        wave_b, fs = read_wave(fnames[1])
        
        sil = np.zeros(int(silence_secs * fs))

        
        out_fname = self.get_join_stimulus_filename(fro, to, prefix='paired')

        write_wave(np.concatenate([wave_a, sil, wave_b]), out_fname, fs)

        #print out_fname
        #sys.exit('rgswrbetb')



    def make_join_stimulus(self, fro, to, context=20, make_natural=True):
        start = max(0, fro - context)
        end = min(self.number_of_units, fro + context + 1)
        insert_point = context + 1
        if fro - context < 0:
            insert_point += fro - context
            print 'CORRETION!!!!!!!!!! ---------------'
        unit_sequence = range(start, end)
        if make_natural:
            self.concatenate(unit_sequence, self.get_join_stimulus_filename(fro, to, natural=True))
        unit_sequence[insert_point] = to
        print unit_sequence
        self.concatenate(unit_sequence, self.get_join_stimulus_filename(fro, to, natural=False))


    def present_candidate_joins_single_foreign_unit(self, index_pairs, CONTEXT=20, paired=False):
        judgements = []
        for (fro, to) in index_pairs:
            print '======== testing join ========'
            print (fro, to)
            print (self.train_unit_names[fro], self.train_unit_names[to])
            print 'try replacing %s with %s (%s with %s)'%(fro + 1, to, self.train_unit_names[fro + 1], self.train_unit_names[to])
            self.make_join_stimulus(fro, to, context=CONTEXT, make_natural=True)
            if paired:
                self.pair_fake_and_natural_joins(fro, to)
                judgements.append(self.present_audio_instance(fro, to, natural_reference=False, paired=True))
            else:
                judgements.append(self.present_audio_instance(fro, to, natural_reference=True))
        return judgements


    def make_prompt(self, audio_prompt=False, natural_reference=''):
        prompt = ["%s for %s, "%(code, category) for (code, category) in self.class_dict.items()] 
        prompt = " ".join(prompt)
        prompt = "  Type: " + prompt
        if audio_prompt:
            prompt += " (or r to replay)"
        if natural_reference:            
            prompt += " (or n to play natural reference)"
        prompt += " ... "
        return prompt


    def present_audio_instance(self, fro, to, natural_reference=False, paired=False):  

        fname = self.get_join_stimulus_filename(fro, to)
        assert os.path.isfile(fname) and fname.endswith('.wav') 

        if natural_reference:
            assert not paired
            natural_fname = self.get_join_stimulus_filename(fro, to, natural=True)
            assert os.path.isfile(natural_fname) and fname.endswith('.wav') 
        else:
            natural_fname = ''

        if paired:
            assert not natural_reference
            fname = self.get_join_stimulus_filename(fro, to, prefix='paired', natural=False)
            assert os.path.isfile(fname) and fname.endswith('.wav') 





        print "  Please listen to this example:"
        os.system("play " + fname)
        prompt = self.make_prompt(audio_prompt=True, natural_reference=natural_reference)
        valid_response_received = False
        while not valid_response_received:       
            raw_reponse = raw_input(prompt)
            print 
            print "  ------------------------------"
            print 
            if raw_reponse=="r":
                os.system("play " + fname)
            if natural_reference:
                if raw_reponse == 'n':
                    os.system("play " + natural_reference)
            if raw_reponse in self.class_dict:
                valid_response_received=raw_reponse
        return self.class_dict[valid_response_received]

    def get_natural_junctures(self):
        '''
        TODO: exclude joins between sentences
        '''
        return np.hstack([self.end_join_feats[:-1,:], self.start_join_feats[1:,:]])


    def sample_uniform_a(self, nsteps=100):
        '''
        draw from all transitions given by basic linguistic constraints only (allowing natural)
        '''
        ## assemble list of unit indices from which to synthesise:
        sampled = [random.randint(0, self.number_of_units)]
        while len(sampled) < nsteps:
            self.cursor.execute("SELECT to_index FROM transitions WHERE from_index=%s"%(sampled[-1])) 
            #print("\nfetch one:")
            res = self.cursor.fetchall()
            ## TODO: move random selection to sql call (https://stackoverflow.com/questions/2279706/select-random-row-from-an-sqlite-table)
            
            sel = random.choice(res)[0]
            sampled.append(sel)            

        return sampled

            


    def get_prediction_errors(self, data):
        predictions = self.autoencoder.predict(data)
        sq_errs = (predictions - data)**2
        scores = sq_errs.mean(axis=1)    
        return scores




    def train_autoencoder(self, data, max_epoch=30):

        
        m,n = data.shape

        model = Sequential()

        model.add(Dense(units=1000, input_dim=n))
        model.add(Activation('relu'))
        model.add(Dense(units=1000))
        model.add(Activation('relu'))

        model.add(Dense(units=60))
    #    model.add(Activation('relu'))

        model.add(Dense(units=1000))
        model.add(Activation('relu'))
        model.add(Dense(units=n))

        model.compile(loss='mean_squared_error', optimizer='adam')

        earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')

        model.fit(data, data, epochs=max_epoch, batch_size=64, callbacks=[earlyStopping], validation_split=0.10, shuffle=True)

        self.autoencoder = model

        ## TODO: save model to disk



    def index_units_by_left_phone(self):
        ### Build index of units by l phone only (c phones are already handled by synth.unit_index):
        print 'build l index...'
        l_index = {}
        seen = {}
        for ll_l_c_r_rr in self.train_unit_names:
            (ll,l,c,r,rr) = ll_l_c_r_rr.split('/')
            if c.endswith('_L'):
                #assert '%s/%s'%(l,c) not in l_index
                if '%s/%s'%(l,c) not in seen:
                    seen['%s/%s'%(l,c)] = 0
                    if l not in l_index:
                        l_index[l] = []
                    l_index[l].extend(self.unit_index['%s/%s'%(l,c)])
        print '...done'
        return l_index


    def setup_join_database(self, fname, connect_to_existing=True):

        if os.path.isfile(fname):
            if connect_to_existing:
                self.connection = sqlite3.connect(fname) # ("/tmp/transitions.db")
                self.cursor = self.connection.cursor()
                print '------'            
                print 'sql file already exists -- assume database has been made' ## TODO: aactually check table within file
                print fname
                print '------'
                return      
            else:
                sys.exit('SQL file already exists (%s) - remove and start again?'%(fname))

        ## else continue to create the data base      

        #transitions = [] ## list of lists
        #transitions = shelve.open('/tmp/transitions')


        self.connection = sqlite3.connect(fname) # ("/tmp/transitions.db")
        self.cursor = self.connection.cursor()

        ## TODO: number of votes? annotator? time? fatigue? recently challenged?
        sql_command = """
        CREATE TABLE transitions ( 
        from_index INTEGER, 
        to_index INTEGER, 
        natural BOOLEAN, 
        labelled BOOLEAN, 
        labelled_ok BOOLEAN,
        trained_on BOOLEAN,
        confidence FLOAT);"""

        self.cursor.execute(sql_command)   
   
        # ### try adding entries for all joins:
        # for i in xrange(2): # self.number_of_units):
        #     print i
        #     for j in xrange(self.number_of_units):

        #         natural = False
        #         if j == i+1:
        #             natural = True
        #         format_str = """INSERT INTO transitions (from_index, to_index, natural, labelled, labelled_ok, trained_on, confidence)
        #         VALUES ("{from_index}", "{to_index}", "{natural}", "{labelled}", "{labelled_ok}", "{trained_on}", "{confidence}");"""

        #         sql_command = format_str.format(from_index=i, to_index=j, natural=natural,\
        #                                          labelled=False, labelled_ok=False, trained_on=False, \
        #                                          confidence=0.5)
        #         self.cursor.execute(sql_command)

        self.connection.commit()

        self.have_sql_db = True

    def add_join_candidate_to_database(self, i, j, sql=True):
        if sql:
            self.add_join_candidate_to_database_sql(i, j)
        else:
            self.add_join_candidate_to_database_dict(i, j)
        

    def add_join_candidate_to_database_dict(self, i, j):
        natural = False
        if j == i+1:
            natural = True

        labelled=False
        labelled_ok=False
        trained_on=False
        confidence=0.5            

        self.join_db[(i,j)] = (natural, labelled, labelled_ok, trained_on, confidence)


    def add_join_candidate_to_database_sql(self, i, j):
        natural = False
        if j == i+1:
            natural = True
        format_str = """INSERT INTO transitions (from_index, to_index, natural, labelled, labelled_ok, trained_on, confidence)
        VALUES ("{from_index}", "{to_index}", "{natural}", "{labelled}", "{labelled_ok}", "{trained_on}", "{confidence}");"""

        sql_command = format_str.format(from_index=i, to_index=j, natural=natural,\
                                         labelled=False, labelled_ok=False, trained_on=False, \
                                         confidence=0.5)
        self.cursor.execute(sql_command)



    def junk():

        ## determine candidate joins:
        ## Work out the minimal linguistic criteria which should signal a possible join.
        l_index = self.index_units_by_left_phone()


        nlinks = 0 # count number of possibles
        for (i,name) in enumerate(self.train_unit_names):
            if i % 100 == 0:
                print '%s of %s'%(i, self.number_of_units)
            (ll,l,c,r,rr) = name.split('/')
            if c.endswith('_L'):
                next_unit = c.replace('_L','_R')
                #transitions.append(self.unit_index[next_unit])
                transitions_from_here = self.unit_index[next_unit]  
            else:
                ## find all following units compatible with it:
                this_current = c.split('_')[0]
                this_right = r
                transitions_from_here = l_index.get(this_current, [])  +   self.unit_index.get(this_right + '_L', []) 


                # for (l,c) in lc_inventory:
                #     if this_current == l or this_right == c:
                #         next_unit = '%s/%s_L'%(l,c)
                #         #print '%s %s   %s'%(this_current, this_right, next_unit)
                #         transitions_from_here.extend(self.unit_index[next_unit])


                #transitions.append(transitions_from_here)

            ### add to database:
            for j in transitions_from_here:

                natural = False
                if j == i+1:
                    natural = True
                format_str = """INSERT INTO transitions (from_index, to_index, natural, labelled, labelled_ok, trained_on, confidence)
                VALUES ("{from_index}", "{to_index}", "{natural}", "{labelled}", "{labelled_ok}", "{trained_on}", "{confidence}");"""

                sql_command = format_str.format(from_index=i, to_index=j, natural=natural,\
                                                 labelled=False, labelled_ok=False, trained_on=False, \
                                                 confidence=0.5)
                self.cursor.execute(sql_command)

            self.connection.commit()
            nlinks += len(transitions_from_here)




    def junk():

        #print transitions
        #print transitions['101']
        #nlinks = sum([len(sublist) for sublist in transitions])


        cursor.execute("SELECT * FROM transitions") 
        print("fetchall:")
        result = cursor.fetchall() 
        for r in result:
            print(r)
        cursor.execute("SELECT * FROM transitions") 
        print("\nfetch one:")
        res = cursor.fetchone() 
        print(res)
        print '----'

        print nlinks
        print self.number_of_units
        print 'done'

        connection.close()

        sys.exit('evaeveb1111')  

    def determine_candidate_joins_SHELVE(self):


        ## Work out the minimal linguistic criteria which should signal a possible join.
        l_index = self.index_units_by_left_phone()

        #transitions = [] ## list of lists
        transitions = shelve.open('/tmp/transitions')

        nlinks = 0 # count number of possibles
        for (i,name) in enumerate(self.train_unit_names):
            if i % 100 == 0:
                print '%s of %s'%(i, self.number_of_units)
            (ll,l,c,r,rr) = name.split('/')
            if c.endswith('_L'):
                next_unit = c.replace('_L','_R')
                #transitions.append(self.unit_index[next_unit])
                transitions_from_here = self.unit_index[next_unit]
                
            else:
                ## find all following units compatible with it:
                this_current = c.split('_')[0]
                this_right = r
                transitions_from_here = l_index.get(this_current, [])  +   self.unit_index.get(this_right + '_L', []) 


                # for (l,c) in lc_inventory:
                #     if this_current == l or this_right == c:
                #         next_unit = '%s/%s_L'%(l,c)
                #         #print '%s %s   %s'%(this_current, this_right, next_unit)
                #         transitions_from_here.extend(self.unit_index[next_unit])


                #transitions.append(transitions_from_here)
            



            transitions[str(i)] = numpy.array(transitions_from_here, dtype=int)
            nlinks += len(transitions_from_here)

        #print transitions
        print transitions['101']
        #nlinks = sum([len(sublist) for sublist in transitions])
        print nlinks
        print self.number_of_units
        print 'done'