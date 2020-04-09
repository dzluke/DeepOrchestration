import itertools
from functools import reduce
from generateDB import generateDBLabels, n_fft
import librosa.display
import random
import math
from torch.utils import data
import torch
import numpy as np
import os
import pickle

RATE = 44100
HOP_LENGTH = 44100

def showSample(s):
    print(s[1])
    librosa.display.specshow(s[0], x_axis='time',y_axis='mel',sr=44100)

class IndexIncrementor:
    def __init__(self, N, base, offset=None):
        self.N = N
        if type(base) == int:
            self.base = [base]*N
        elif type(base) == list and len(base) == N:
            self.base = base
        else:
            raise Exception("Wrong argument for IndexIncrementor (must be int or list of size N)")
        self.max_count = reduce((lambda x,y:x*y), self.base)
        self.restart(offset)
    
    def findStep(self):
        self.step = int(random.randint(int(0.05*self.max_count),int(0.95*self.max_count)))
        while math.gcd(self.step, self.max_count) > 1:
            self.step += 1
    
    def add(self, e):
        r = e
        i = 0
        while i < self.N and r != 0:
            r,self.index[i] = divmod(self.index[i]+r,self.base[i])
            i += 1
            
    def incr(self):
        self.add(self.step)
        self.count += 1
        
    def restart(self, offset = None):
        self.count = 0
        self.index = [0]*self.N
        self.findStep()
        if offset is None:
            self.add(random.randint(0,self.max_count-1))
        elif offset >= 0:
            self.add(offset)
        else:
            raise Exception("Wrong offset")
    
    def isOver(self):
        return self.count >= self.max_count
            

class RawDatabase:
    def __init__(self, path, random_granularity = 10, instr_filter = None):
        self.db,self.pr = generateDBLabels(path, random_granularity, instr_filter)
        self.random_granularity = random_granularity
        self.instr_filter = instr_filter
        
class OrchDataSet(data.Dataset):
    def __init__(self, raw_db, class_encoder):
        self.mel_basis = librosa.filters.mel(44100, n_fft)
        self.db = raw_db.db
        self.pr = raw_db.pr
        
        #Function that takes as an argument the list of samples combined, and returns the corresponding label vector
        self.class_encoder = class_encoder
    
    def generate(self, N, nb_samples=None, nb_pitch_range = 0):
        self.N = N
        self.nb_pitch_range = nb_pitch_range or len(self.pr)
        self.nb_samples = sum(reduce((lambda x,y : x*y), [sum(len(e) for e in self.db[i[x]]) for x in range(N)]) for i in itertools.combinations(self.db.keys(), N))
        print("Total number of possible samples : {}".format(self.nb_samples))
        self.out_classes = []
        for i in self.db:
            set_pitches = set()
            for j in self.db[i]:
                for x in j:
                    set_pitches.add(int(x['pitch_idx']*self.nb_pitch_range/len(self.pr)))
            self.out_classes.extend([(i,x) for x in set_pitches])
            
        print("Total number of classes : {}".format(len(self.out_classes)))
        if not nb_samples is None:
            self.nb_samples = min(self.nb_samples,nb_samples)
        
        self.instr_comb = list(itertools.combinations(self.db.keys(), self.N))
        random.shuffle(self.instr_comb)
        self.indexes = [IndexIncrementor(N,[len(self.db[i[x]]) for x in range(N)]) for i in self.instr_comb]
        
        
        self.index_comb = 0
        
        print("Generating the Dataset")
        self.data = []
        for i in range(self.nb_samples):
            self.data.append(self.getNextSample())
            if i % 1000 == 0:
                print("{}/{} samples done".format(i, self.nb_samples))
        print("Finished generating the Dataset")
    
    def getNextSample(self):
        ii = self.indexes[self.index_comb]
        pitch_indexes = ii.index
        sample_list_to_combine = [(self.instr_comb[self.index_comb][x],
                                   pitch_indexes[x],
                                   random.randint(0,len(self.db[self.instr_comb[self.index_comb][x]][pitch_indexes[x]])-1)) for x in range(self.N)]
           
        ii.incr()
        if ii.isOver():
            ii.restart()
            
        self.index_comb += 1
        if self.index_comb == len(self.instr_comb):
            self.index_comb = 0
            
        return sample_list_to_combine

    def save(self, path):
        f = open(path, 'wb')
        pickle.dump(self.data, f)
        f.close()

    def load(self, path):
        f = open(path, 'rb')
        self.data = pickle.load(f)
        self.nb_samples = len(self.data)
        self.N = len(self.data[0])
        f.close()
    
    def __len__(self):
        return self.nb_samples

    def __getitem__(self, idx):
        stft = None
        list_samp = []
        for i in self.data[idx]:
            samp = self.db[i[0]][i[1]][i[2]]
            list_samp.append(samp)
            if stft is None:
                stft = np.copy(samp['stft'])
            else:
                stft += samp['stft']
        stft = stft / self.N
        s = np.real(stft*np.conjugate(stft))
        mel_spec = librosa.feature.melspectrogram(S=s, sr=RATE)
        pow_db = librosa.power_to_db(mel_spec)
        mfcc = librosa.feature.mfcc(S=pow_db)
        return torch.tensor(np.array([mfcc])), self.class_encoder(list_samp)
    
pitch_instruments = ['Vn', 'Fl']
other_instruments = ['Bn', 'BTb', 'Cb', 'ClBb', 'Hn', 'Ob', 'Tbn', 'TpC', 'Va', 'Vc']
pitches = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
    
def class_encoder(list_samp):
    label = [0 for i in range(34)]
    for s in list_samp:
        if s['instrument'] in pitch_instruments:
            label[12*pitch_instruments.index(s['instrument']) + pitches.index(s['pitch_name'][:-1])] = 1
        else:
            label[24+other_instruments.index(s['instrument'])] = 1
    return np.array(label).astype(np.float32)

try:
    rdb
except NameError:
    rdb = RawDatabase('./TinySOL', 10, None)

#odb = OrchDataSet(rdb, class_encoder)
#odb.generate(2, 400000)
#odb.save('./db/400000.pkl')
#odb2 = OrchDataSet(rdb, class_encoder)
#odb2.load('./db/400000.pkl')