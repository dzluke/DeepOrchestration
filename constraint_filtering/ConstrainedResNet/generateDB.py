import os
from functools import reduce
import librosa
import numpy as np
import random

from parameters import GLOBAL_PARAMS

def getSampleMetaData(path,name):
    s = {}
    s['path'] = path
    s['stft'] = {}
    X = np.linspace((1-GLOBAL_PARAMS.coeff_freq_shift_data_augment)*GLOBAL_PARAMS.RATE, (1+GLOBAL_PARAMS.coeff_freq_shift_data_augment)*GLOBAL_PARAMS.RATE, 5)
    for freq in X:
        y,sr=librosa.load(path,sr=int(freq))
        nb_samples = int(GLOBAL_PARAMS.TIME_LENGTH*GLOBAL_PARAMS.RATE)
        if len(y) < nb_samples:
            y = np.append(y, np.zeros((1,(nb_samples-len(y))), dtype=np.float32))
        else:
            y = y[:nb_samples]
        s['stft'][int(freq)] = librosa.stft(y=y,hop_length=GLOBAL_PARAMS.MEL_HOP_LENGTH,n_fft=GLOBAL_PARAMS.N_FFT)
    t = name.split('.')[0].split('-')
    s['instrument'] = t[0]
    s['style'] = t[1]
    s['pitch_name'] = t[2]
    s['nuance'] = t[3]
    if len(t) == 5:
        s['other'] = t[4]
    
    return s
    

def recursiveSearch(path, i=0):
    """
    create a list sample metadata dictionaries for each sample in path
    @param path: path to database
    @param i:
    @return: list of sample metadata dictionaries
    """
    samples_metadata = []
    s = i
    for x in os.listdir(path):
        p = os.path.join(path, x)
        if os.path.isdir(p):
            samples_metadata.extend(recursiveSearch(p, s))
        elif os.path.isfile(p) and p.split('.')[-1] == 'wav':
            print("Sample {}, Path: {}".format(s, p))
            samples_metadata.append(getSampleMetaData(p, x))
            s += 1
    return samples_metadata

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
    # TODO: replace recursiveSearch with librosa.util.find_files()
    samples = recursiveSearch(path)
    # we sort the sample list so that the ordering of the samples is the same on every machine.
    # otherwise, the indices in the labels could correspond to different samples
    # because of differences in file structure on different machines
    samples.sort(key=lambda s: s['path'])

    if instr_filter is not None:
        samples = [x for x in samples if x['instrument'] in instr_filter]
    instr_db = {}
    for i in samples:
        if i["instrument"] not in instr_db:
            instr_db[i["instrument"]] = []
        instr_db[i["instrument"]].append(i)
    pitch_range = getPitchRange(samples)
    
    database = {}
    
    nb_pitch_classes = nb_pitch_range or len(pitch_range)
    for i in instr_db:
        database[i] = [[] for i in range(nb_pitch_classes)]
    for x in samples:
        x['pitch_idx'] = pitch_range.index(x['pitch_name'])
        database[x['instrument']][int(x['pitch_idx']*nb_pitch_classes/len(pitch_range))].append(x)
        
    for i in database:
        database[i] = [x for x in database[i] if len(x) > 0]
        for j in database[i]:
            random.shuffle(j)
    return database, pitch_range
