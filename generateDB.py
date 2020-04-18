import os
import itertools
import argparse
from functools import reduce
import librosa
import numpy as np
import random

from parameters import MEL_HOP_LENGTH, N_FFT, TIME_LENGTH

def getSampleMetaData(path,name):
    s = {}
    s['path'] = path
    y,sr=librosa.load(path,sr=None)
    nb_samples = int(TIME_LENGTH*sr)
    if len(y) < nb_samples:
        y = np.append(y, np.zeros((1,(nb_samples-len(y))), dtype=np.float32))
    else:
        y = y[:nb_samples]
    s['stft'] = librosa.stft(y=y,hop_length=MEL_HOP_LENGTH,n_fft=N_FFT)
    t = name.split('.')[0].split('-')
    s['instrument'] = t[0]
    s['style'] = t[1]
    s['pitch_name'] = t[2]
    s['nuance'] = t[3]
    if len(t) == 5:
        s['other'] = t[4]
    
    return s
    

def recursiveSearch(path):
    l = []
    for x in os.listdir(path):
        p = os.path.join(path,x)
        if os.path.isdir(p):
            l.extend(recursiveSearch(p))
        elif os.path.isfile(p) and p.split('.')[-1] == 'wav':
            l.append(getSampleMetaData(p,x))
    return l

def getPitchRange(d):
    pr = set()
    for i in d:
        pr.add(i['pitch_name'])
    pr = list(pr)
    pr.sort(key = lambda x : 10000*ord(x[-1]) + 100*((ord(x[0])-ord('C'))%7) + (len(x)==3))
    return pr


def generateDBLabels(path, nb_pitch_range = 1, instr_filter=None):
    '''
        Generates the short term fourier transforms for each sample of the data base.
        In order to ensure a good repartition of the pitches when generating the combinations,
        the set is divided into pitch_bins (the total number of bins amond the pitch range is defined
        by nb_pitch_range).
        If instr_filter is not None, the generated set will contain only samples of the instruments
        specified by instr_filter.
        
        Note that if the global paramaters MEL_HOP_LENGTH, N_FFT or TIME_LENGTH are modified,
        this function needs to be called again
    '''
    l = recursiveSearch(path)
    if not instr_filter is None:
        l = [x for x in l if x['instrument'] in instr_filter]
    instr_db = {}
    for i in l:
        if i["instrument"] not in instr_db:
            instr_db[i["instrument"]] = []
        instr_db[i["instrument"]].append(i)
    pr = getPitchRange(l)
    
    db = {}
    
    nb_pitch_classes = nb_pitch_range or len(pr)
    for i in instr_db:
        db[i] = [[] for i in range(nb_pitch_classes)]
    for x in l:
        x['pitch_idx'] = pr.index(x['pitch_name'])
        db[x['instrument']][int(x['pitch_idx']*nb_pitch_classes/len(pr))].append(x)
        
    for i in db:
        db[i] = [x for x in db[i] if len(x) > 0]
        for j in db[i]:
            random.shuffle(j)
    return db,pr