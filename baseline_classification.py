# ------------------------------------
# OrchideaSOL classification baselines
# ------------------------------------
#
# Written by Carmine E. Cella, 2020
#
# This code is distributed with the OrchideaSOL dataset
# of extended instrumental techniques.
#
# For more information, please see:
# C. E. Cella, D. Ghisi, V. Lostanlen, F. Lévy, J. Fineberg and Y. Maresz,
#       OrchideaSOL: a dataset of extended instrumental techniques for computer-assisted orchestration,
#       ICMC 2020, Santiago, Chile.
#

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

        # mfcc = librosa.feature.mfcc(y=sfile, sr=sr, hop_length=mel_hop_length)
        # librosa.display.specshow(mfcc)
        # plt.title(sound)
        # plt.show()

        if len(sfile) > time*sr:
            # randomly select one part of the raw audio
            n = np.random.randint(0, len(sfile)-time*sr)
            sfile = sfile[n:n+time*sr]
        # add augment
        # sfile = wav_augment(sfile, sr)
        mixed_file = mix(mixed_file, sfile)
    mixed_file = mixed_file/len(soundlist)

    # mfcc = librosa.feature.mfcc(y=sfile, sr=sr, hop_length=mel_hop_length)
    # librosa.display.specshow(mfcc)
    # plt.title('mix')
    # plt.show()
    
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

def create_binary_label(instruments, orchestra):
    '''
        given a list of instruments, return a binary vector whose length is the number
        of instruments in the orchestra
    '''
    label = np.zeros(len(orchestra), dtype=np.float32)
    for instrument in instruments:
        index = orchestra.index(instrument)
        label[index] = 1.0
    return label

# def create_index_label(instruments):
#     '''
#         given a list of instruments, return a vector of length n where each value is an index
#         of an instrument used
#     '''
#     assert len(instruments) == n
#     label = np.zeros(n, dtype=np.float32)
#     for i in range(n):
#         instrument = instruments[i]
#         index = orchestra.index(instrument)
#         label[i] = float(index)
#     return label

def generate_data(orchestra, n, num_samples):
    '''
        create combinations of n instruments using only the instruments defined in 'orchestra'

    '''

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

    for i in range(num_samples):
        # select n instruments
        instruments = random.sample(orchestra, n)
        samples_to_combine = []
        # for each of n chosen instruments, randomly select one sample
        for instrument in instruments:
            sample = random.choice(samples[instrument])
            samples_to_combine.append(sample)

        # combine sounds, storing combined sound in `mixture`
        mixture, sr = combine_sounds(samples_to_combine)
        # calculate feature of combined sound
        features = calculate_features(mixture, sr)
        # create label from the instruments that were chosen
        label = create_binary_label(instruments, orchestra)

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

    return X, y


# orchestra = ['Vc', 'Fl', 'Va', 'Vn', 'Ob', 'BTb',
#                'Cb', 'ClBb', 'Hn', 'TpC', 'Bn', 'Tbn']


X, y = generate_data(orchestra=['Fl', 'Vn', 'Cb'], n=2, num_samples=1000)

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,
                                                   test_size = 0.2,
                                                   random_state = 42,
                                                   shuffle=True)

print('X train: ', X_train.shape)
print('X test: ', X_test.shape)
print('y train: ', y_train.shape)
print('y test: ', y_test.shape)


clfs = []


clf = SVC(kernel='rbf')
clf = MultiOutputClassifier(clf)
clfs.append(clf)

clf = KNeighborsClassifier()
clf = MultiOutputClassifier(clf)
clfs.append(clf)

clf = RandomForestClassifier()
clf = MultiOutputClassifier(clf)
clfs.append(clf)

print ("\nRunning classifications...")
for classifier in clfs:
    pipeline = Pipeline([
        ('normalizer', StandardScaler()),
        ('clf', classifier)
    ])
    print('---------------------------------')
    print(str(classifier))
    print('---------------------------------')
    shuffle = KFold (n_splits=5, random_state=5, shuffle=True)
    scores = cross_val_score (pipeline, X, y, cv=shuffle)

    print("model scores: ", scores)
    print("average score: ", scores.mean ())

    pipeline.fit (X_train, y_train)
    ncvscore = pipeline.score(X_test, y_test)
    print("non cross-validated score: ", ncvscore)

#eof
