import os
import itertools
import argparse
from functools import reduce
import librosa
import numpy as np
import random

mel_hop_length = 512
n_fft = 2048
time_length = 4

def getSampleMetaData(path,name):
    s = {}
    s['path'] = path
    y,sr=librosa.load(path,sr=None)
    nb_samples = int(time_length*sr)
    if len(y) < nb_samples:
        y = np.append(y, np.zeros((1,(nb_samples-len(y))), dtype=np.float32))
    else:
        y = y[:nb_samples]
    s['stft'] = librosa.stft(y=y,hop_length=mel_hop_length,n_fft=n_fft)
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
        Generates the dataset labels and organizes all samples in those labels.
        Outputs the labels as a dictionary along with the list of pitch ranges used.
        Labels are generated using all possible combinations of N instruments (among instr_filter if set) using pitches within same pitch ranges and nuances.
        If nb_pitch_range is set, the whole pitch range will be split in nb_pitch_range subsets.
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

if __name__ == "__maihn__":
    
    parser = argparse.ArgumentParser(description='combination of orch')
    parser.add_argument('--path', default='.\\TinySOL')
    parser.add_argument('--is_resume', default='False',
                        choices=['True', 'False'])

    args = parser.parse_args()

    importDB(args.path)