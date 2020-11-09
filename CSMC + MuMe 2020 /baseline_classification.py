# Adapted from code written by Carmine E. Cella, 2020


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from collections import Counter
from matplotlib import pyplot as plt

import librosa
import librosa.display
import json
import os
import random
import sys
from time import process_time

# raw dataset
# file hierarchy: (note that folders Brass, Winds, Strings are not present)
# ----TinySOL
#   ----Bn
#   ----Cb
#   ----Va
#   etc...
path = './TinySOL_0.6/TinySOL'

# time duration
time = 4

# number of samples between successive frames in librosa.melspectrogram
mel_hop_length = 44100

pitch_classes = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']

# maps an instrument or instrument-pitch to a class #
label_mapping = {}

def mix(fa, fb):
    diff = len(fa) - len(fb)

    if diff > 0:
        add = np.zeros((1, diff), dtype=np.float32)
        fb = np.append(fb, add)
    elif diff < 0:
        add = np.zeros((1, -diff), dtype=np.float32)
        fa = np.append(fa, add)

    return fa+fb
    
def combine_sounds(soundlist):
    mixed_file = np.zeros((1, 1))
    sr = 0
    for sound in soundlist:
        sound_path = os
        sfile, sr = librosa.load(sound, sr=None)
        if len(sfile) > time*sr:
            # randomly select one part of the raw audio
            n = np.random.randint(0, len(sfile)-time*sr)
            sfile = sfile[n:n+time*sr]
        # add augment
        # sfile = wav_augment(sfile, sr)
        mixed_file = mix(mixed_file, sfile)
    mixed_file = mixed_file/len(soundlist)
    
    return [mixed_file, sr]

def calculate_features(sample, sr):
    feature = librosa.feature.mfcc(y=sample, sr=sr,
                                             hop_length=mel_hop_length)

    # zero padding
    expected_length = sr*time // mel_hop_length + 1
    diff = expected_length - feature.shape[1]
    if diff > 0:
        padding = np.zeros((feature.shape[0], diff), dtype=np.float32)
        feature = np.append(feature, padding)
    return feature

def extract_label(sample):
    '''
        return the instrument name and pitch class given a file path
        expects sample to be similar to "./TinySOL_0.6/TinySOL/BTb/BTb-ord-F2-ff.wav"
    '''
    sample = sample.split('/')[-1]
    sample = sample.split('.')[0]
    sample = sample.split('-')

    instrument = sample[0]
    playing_style = sample[1]
    pitch = sample[2]
    dynamic = sample[3]

    pitch_class = pitch[:-1]
    return [instrument, pitch_class]


def create_label_mapping(orchestra, pitch_instruments):
    '''
        creates a dictionary that maps an instrument and pitch class to its index in the label

        for instruments in 'pitch_instruments' a key looks like: Fl-C or Vn-G
        for all other instruments, a key is just the instrument name: Fl or Vn or Cb
        and each of these instruments maps to the same index, because they all fall under the same class
    '''
    assert len(label_mapping) == 0
    i = 0
    for instrument in pitch_instruments:
        for pitch in pitch_classes:
            key = instrument + '-' + pitch
            label_mapping[key] = i
            i += 1
    for instrument in orchestra:
        if instrument not in pitch_instruments:
            label_mapping[instrument] = i

def create_binary_label(samples, pitch_instruments, orchestra):
    '''
        given a list of samples, return a binary vector

        for N pitch_instruments, the first N * 12 indices correspond to each pitch_instrument and one of 12
        pitches associated with it.
        The N * 12 + 1 index corresponds to the class that represents "noise" i.e. the fact that some other
        instrument that is not a pitch_instrument is present
    '''
    if set(pitch_instruments) == set(orchestra):
        # if these lists are identical, then we have N*12 classes
        label_length = (len(pitch_instruments) * 12)
    else:
        # else we assume len(pitch_instruments) < len(orchestra), and we have N*12 + 1 classes
        label_length = (len(pitch_instruments) * 12) + 1
    assert len(set(label_mapping.values())) == label_length

    label = np.zeros(label_length, dtype=np.float32)
    for sample in samples:
        instrument, pitch_class = extract_label(sample)
        key = instrument
        if instrument in pitch_instruments:
            key = key + '-' + pitch_class
        index = label_mapping[key]
        label[index] = 1.0
    return label

def generate_data(orchestra, pitch_instruments, n, num_samples):
    '''
        create combinations of n instruments using only the instruments defined in 'orchestra'
        the instruments in 'pitch_instruments' will have their pitch class indentified as well

    '''
    create_label_mapping(orchestra, pitch_instruments)

    # dictionary where key is instrument name 
    # and value is a list of all the samples in the dataset for that instrument
    samples = {} 
    for instrument in orchestra:
        samples[instrument] = []
        instrument_path = os.path.join(path, instrument)
        for sample in os.listdir(instrument_path):
            sample_path = os.path.join(instrument_path, sample)
            samples[instrument].append(sample_path)

    X = [] # data
    y = [] # labels

    i = 0
    while i < num_samples:
        instruments = []
        if (len(pitch_instruments) > 0):
            # select one of 'pitch_instruments'
            instruments.append(random.choice(pitch_instruments))
        else:
            instruments.append(random.choice(orchestra))

        # select n - 1 instruments
        instruments.extend(random.sample(orchestra, n - 1))
    
        samples_to_combine = []
        # for each of n chosen instruments, randomly select one sample
        for instrument in instruments:
            sample = random.choice(samples[instrument])
            samples_to_combine.append(sample)

        # combine sounds, storing combined sound in `mixture`
        mixture, sr = combine_sounds(samples_to_combine)
        # calculate feature of combined sound
        features = calculate_features(mixture, sr)
        # create label from the samples that were chosen
        label = create_binary_label(samples_to_combine, pitch_instruments, orchestra)

        # since you cant know the dimensions until the features have been computed,
        # you can't make the ndarray until now
        if i == 0:
            num_features = features.flatten().shape[0]
            X = np.zeros((num_samples, num_features))
            y = np.zeros((num_samples, label.shape[0]))
        
        X[i] = features.flatten()
        y[i] = label

        if i % 100 == 0:
            print("{} / {} have finished".format(i, num_samples))

        i += 1

    return X, y


def train_and_test(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.4,
                                                    random_state = 42,
                                                    shuffle=True)

    clfs = []

    clf = SVC(kernel='rbf')
    clf = MultiOutputClassifier(clf)
    clfs.append(clf)

    # clf = RandomForestClassifier(max_depth=15)
    # clf = MultiOutputClassifier(clf)
    # clfs.append(clf)

    test_scores = []

    print("\nRunning classifications...")
    for classifier in clfs:
        start_time = process_time()
        pipeline = Pipeline([
            ('normalizer', StandardScaler()),
            ('clf', classifier)
        ])
        print('---------------------------------')
        print(str(classifier))
        print('---------------------------------')
        shuffle = KFold(n_splits=5, random_state=5, shuffle=True)
        scores = cross_val_score(pipeline, X, y, cv=shuffle)

        print("model scores: ", scores)
        print("average training score: ", scores.mean())

        pipeline.fit(X_train, y_train)
        ncvscore = pipeline.score(X_test, y_test)
        print("test accuracy: ", ncvscore)
        print("time: ", process_time() - start_time)
        test_scores.append(ncvscore)

    return test_scores


orchestra = ['Vc', 'Fl', 'Va', 'Vn', 'Ob', 'BTb',
               'Cb', 'ClBb', 'Hn', 'TpC', 'Bn', 'Tbn']


string_duo = ['Vn', 'Vc']
brass_trio = ['Hn', 'TpC', 'Tbn']
string_quartet = ['Vn', 'Va', 'Vc', 'Cb']

num_samples = 50000
scores = []
n = 3

orchestra = brass_trio
pitch_instruments = brass_trio
X, y = generate_data(orchestra, pitch_instruments, n, num_samples)

print("\nNumber of classes: ", y[0].shape)

score = train_and_test(X, y)
scores.append(score)

print('---------------------------------')
print("orchestra size: ", len(orchestra))
print("n: ", n)
print("number of samples: ", num_samples)
print("scores from this run: ", scores)
print('---------------------------------')

#eof
