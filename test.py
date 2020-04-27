import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import librosa
import argparse
import time
import json
import pickle
import os
import random

from model import OrchMatchNet
from OrchDataset import RawDatabase
from train import prediction, getPosNMax
from parameters import rdm_granularity, coeff_freq_shift_data_augment, coeff_freq_shift_data_augment
from parameters import N_FFT, RATE, N_MELS

import matplotlib.pyplot as plt
import soundfile as sf

# SET THE FOLLOWING VARS:

# path to TinySOL data
tinysol_path = './TinySOL'

# path to store solutions as .wav
solutions_path = './orchestrated_targets'

# path to a trained version of the model
state_path = 'cnn_test_state_n=10.pth'

# cnn or resnet
model_type = 'cnn'

# path to target samples
target_path = './target_samples'

# instruments to be used (all instruments will be used)
instr_filter = ['Hn', 'Ob', 'Vn', 'Va',
                'Vc', 'Fl', 'Tbn', 'Bn', 'TpC', 'ClBb']

# number of samples to be used in solution
n = 10


def test(model, state_path, data, targets):
    device = torch.device('cpu')    
    state = torch.load(state_path, map_location=device)
    model.load_state_dict(state['state_dict'])
    model.eval()

    outputs = model(data)
    outputs = outputs.detach().cpu().clone().numpy()

    # text file to write solutions to
    f = open(solutions_path + '/orchestration_results.txt', 'w+')

    for i in range(len(outputs)):
        output = outputs[i]
        # get indices of top n probabilities
        indices = getPosNMax(output, n)
        # turn indices into [instr, pitch]
        classes = get_classes(indices) # returns (instr, pitch)
        # get top n probabiltiies
        probs = [output[i] for i in indices]
        # turn probabilities into dynamic markings (pp, mf, ff)
        dynamics = prob_to_dynamic(probs)
        # combine into (instr, pitch, dynamic)
        for j in range(len(classes)):
            classes[j].append(dynamics[j])
        
        # turn (instr, pitch, dynamic) into list of actual TinySOL sample paths
        sample_paths = find_sample_paths(classes)
        targets[i]['classes'] = classes
        # combine samples and write to wav
        combine_and_write(sample_paths, targets[i], f)
    f.close()

        
def find_sample_paths(classes):
    samples = []
    for c in classes:
        instr, pitch, dynamic = c
        instr_samples = rdb.db[instr]
        for lst in instr_samples:
            for sample in lst:
                if sample['instrument'] == instr and sample['pitch_name'] == pitch and sample['nuance'] == dynamic:
                    samples.append(sample['path'])
                    break
    return samples

'''
converts a list of probabilities to a list of dynamics
0 - 0.33 -> pp
0.34 - 0.66 -> mf
0.67 - 1 -> ff
'''
def prob_to_dynamic(probs):
    dynamics = []
    pp = 0.33
    mf = 0.66
    for prob in probs:
        if prob > mf:
            dynamics.append('ff')
        elif prob > pp:
            dynamics.append('mf')
        else:
            dynamics.append('pp')
    return dynamics    

# given a list of indices, return the corresponding [instrument, pitch] in lab_class
def get_classes(indices):
    classes = [None for i in indices]
    for instrument, pitch_dict in lab_class.items():
        for pitch, class_number in pitch_dict.items():
            if class_number in indices:
                index = indices.index(class_number)
                classes[index] = [instrument, pitch]
    assert len(indices) == len(classes)
    return classes



# load a sample at the given path and return the melspectrogram  and duration of the sample
def load_sample(path):
    
    y, sr = librosa.load(path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    mel_hop_length = sr * duration / (87 - 1) # based on training data size
    mel_hop_length = int(mel_hop_length)
   
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=mel_hop_length)
    return torch.tensor(np.array([mel_spec])), duration

# load in a folder of data and return a list of melspectrograms
def load_data(folder_path):
    
    data = []
    targets = []
    for entry in os.listdir(folder_path):
        if (entry.endswith('.wav')):
            full_path = os.path.join(folder_path, entry)
            mel_spec, duration = load_sample(full_path)
            data.append(mel_spec)
            targets.append({'name': entry[:-4], 'duration': duration})
    print("Loaded {} target samples".format(len(data)))

    return torch.stack(data), targets


def combine_and_write(soundlist, target, f):
    mixed_file = np.zeros((1, 1))
    for sound in soundlist:
        sfile, sr = librosa.load(sound, sr=None)
        mixed_file = mix(mixed_file, sfile)
    mixed_file = mixed_file/len(soundlist)

    # trim to target length
    trim_index = int(target['duration']*sr)
    mixed_file = mixed_file[:trim_index]

    # write wav
    file_name = solutions_path + '/orchestrated_' + target['name'] + '.wav'
    sf.write(file_name, mixed_file, sr)

    # write to file
    f.write('Target: {}; Samples used: {}\n'.format(target['name'], target['classes']))



def mix(fa, fb):
    diff = len(fa) - len(fb)

    if diff >= 0:
        add = np.zeros((1, diff), dtype=np.float32)
        fb = np.append(fb, add)
    else:
        add = np.zeros((1, -diff), dtype=np.float32)
        fa = np.append(fa, add)

    return fa+fb



if __name__ == "__main__":
    rdb = RawDatabase(tinysol_path, rdm_granularity, instr_filter)

    # Create dictionary for label indexing
    lab_class = {}
    tot_size = 0
    for k in rdb.db:
        lab_class[k] = {}
        a = set()
        for l in rdb.db[k]:
            for e in l:
                a.add(e['pitch_name'])
        for p in rdb.pr:
            if p in a:
                lab_class[k][p] = tot_size
                tot_size += 1

    def class_encoder(list_samp):
        label = [0 for i in range(tot_size)]
        for s in list_samp:
            label[lab_class[s['instrument']][s['pitch_name']]] = 1
        return np.array(label).astype(np.float32)
    
    num_classes = len(class_encoder([]))
    features_shape = [128, 87] # the dim of the data used to train the network
    
    model = OrchMatchNet(num_classes, model_type, features_shape)
    
    data, targets = load_data(target_path)
    test(model, state_path, data, targets)
