import itertools
from functools import reduce
import librosa
import librosa.display
import math
from torch.utils import data
import torch
import pickle
import hashlib
import os
import numpy as np
import random

from parameters import GLOBAL_PARAMS

"""
OrchDataset.py

This file contains the RawDatabase and OrchDataSet classes, which are used to generate the dataset
"""



class RawDatabase:
    '''
        Class wrapping the result of the function generateDBLabels containing
        short term fourier transforms of all individual samples.

        self.db: dictionary that maps an instrument to a list of lists of sample metadata objects
            first level: dictionary mapping an instrument to a list
            second level: list of lists. each list corresponds to an octave
            third level: list of metadata objects
    '''
    def __init__(self, path, random_granularity=10, instr_filter=None):
        self.db, self.pr = RawDatabase.generateDBLabels(path, random_granularity, instr_filter)
        self.random_granularity = random_granularity
        self.instr_filter = instr_filter
        
    def save(self, path):
        print("Saving RDB...")
        f = open(path, 'wb')
        pickle.dump(self, f)
        f.close()
        print("Done")

    @staticmethod
    def generateDBLabels(path, nb_pitch_range=1, instr_filter=None):
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
        samples = RawDatabase.recursiveSearch(path)
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
        pitch_range = RawDatabase.getPitchRange(samples)

        database = {}

        nb_pitch_classes = nb_pitch_range or len(pitch_range)
        for i in instr_db:
            database[i] = [[] for i in range(nb_pitch_classes)]
        for x in samples:
            x['pitch_idx'] = pitch_range.index(x['pitch_name'])
            database[x['instrument']][int(x['pitch_idx'] * nb_pitch_classes / len(pitch_range))].append(x)

        for i in database:
            database[i] = [x for x in database[i] if len(x) > 0]
            for j in database[i]:
                random.shuffle(j)
        return database, pitch_range

    @staticmethod
    def getSampleMetaData(path, name):
        """
        @param path: path to sample
        @param name: filename of sample, expected to be 'Fl-ord-G4-pp.wav'
        @return: dictionary of metadata
        """
        sample = {}
        sample['path'] = path
        sample['stft'] = {}
        X = np.linspace((1 - GLOBAL_PARAMS.coeff_freq_shift_data_augment) * GLOBAL_PARAMS.RATE,
                        (1 + GLOBAL_PARAMS.coeff_freq_shift_data_augment) * GLOBAL_PARAMS.RATE, 5)
        for freq in X:
            y, sr = librosa.load(path, sr=int(freq))
            nb_samples = int(GLOBAL_PARAMS.TIME_LENGTH * GLOBAL_PARAMS.RATE)
            if len(y) < nb_samples:
                y = np.append(y, np.zeros((1, (nb_samples - len(y))), dtype=np.float32))
            else:
                y = y[:nb_samples]
            sample['stft'][int(freq)] = librosa.stft(y=y, hop_length=GLOBAL_PARAMS.MEL_HOP_LENGTH,
                                                     n_fft=GLOBAL_PARAMS.N_FFT)
        t = name.split('.')[0].split('-')
        sample['instrument'] = t[0]
        sample['style'] = t[1]
        sample['pitch_name'] = t[2]
        sample['nuance'] = t[3]
        if len(t) == 5:
            sample['other'] = t[4]

        return sample

    @staticmethod
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
                samples_metadata.extend(RawDatabase.recursiveSearch(p, s))
            elif os.path.isfile(p) and p.split('.')[-1] == 'wav':
                print("Sample {}, Path: {}".format(s, p))
                samples_metadata.append(RawDatabase.getSampleMetaData(p, x))
                s += 1
        return samples_metadata

    @staticmethod
    def getPitchRange(d):
        pr = set()
        for i in d:
            pr.add(i['pitch_name'])
        pr = list(pr)
        pr.sort(key=lambda x: 10000 * ord(x[-1]) + 100 * ((ord(x[0]) - ord('C')) % 7) + (len(x) == 3))
        return pr


class OrchDataSet(data.Dataset):
    '''
        Main class for generating the dataset.
    '''
    def __init__(self, raw_db, feature_type):
        '''
            To avoid unnecessary computations when trying new settings, the RawDatabase
            object is passed as an argument.
            
            The argument class_encoder is a function that takes as an input a list
            of combinations (e.g. instrument + pitch) and returns the label vector
            containing 1 in the positions corresponding to the class of each sample
            in the combination.
            
            feature_type defines whether the features generated are mel spectrograms or mfcc.
            It must be either 'mel' or 'mfcc'.
        '''
        
        # For data augmentation, a set of mel basis is generated, using slightly shifted
        # sample rates. This ensures frequency variation of the selected samples.
        self.mel_basis = librosa.filters.mel(GLOBAL_PARAMS.RATE, GLOBAL_PARAMS.N_FFT, n_mels=GLOBAL_PARAMS.N_MELS)
            
        # Create delay filters for data augmentation
        # Delay is a periodic geometric decaying sequence of impulses with an offset
        self.delay_filters = []
        for i in range(100):
            s = np.zeros((GLOBAL_PARAMS.N_FFT,), dtype = np.float32)
            s[0] = 1.0
            k = 1
            
            # Allow 10% of variability around given averages
            off_delay = int(GLOBAL_PARAMS.RATE*GLOBAL_PARAMS.delay_offset_avg*(1 + 0.1*(1-2*random.random())))
            per_delay = int(GLOBAL_PARAMS.RATE*GLOBAL_PARAMS.delay_period_avg*(1 + 0.1*(1-2*random.random())))
            fdb_delay = GLOBAL_PARAMS.delay_feedback_avg*(1 + 0.1*(1-2*random.random()))
            for t in range(1,len(s)):
                if t >= off_delay and (t-off_delay) % per_delay == 0:
                    s[t] = fdb_delay**k
                    k += 1
            self.delay_filters.append(np.fft.fft(s))
        
        self.db = raw_db.db
        self.pr = raw_db.pr
        
        if feature_type not in ['mfcc', 'mel']:
            raise Exception("Feature type must be mfcc or mel")
        self.feature_type = feature_type

        # maps a class (instr, pitch) to an index in the label
        # nested dict, usage is class_indices[instrument][pitch]
        self.class_indices = {}
        self.num_classes = None
        self.createClassIndices(raw_db)

        def class_encoder(list_samp):
            label = [0 for _ in range(self.num_classes)]
            for sample in list_samp:
                index = self.class_indices[sample['instrument']][sample['pitch_name']]
                label[index] = 1
            return np.array(label).astype(np.float32)
        
        self.class_encoder = class_encoder


    
    def generate(self, N, nb_samples=None):
        '''
            This function is called to generate the list of samples. The only parameters
            that will be saved are the combinations of samples. Data augmentation parameters
            such as frequency shift, row and column zeroing, etc
        '''
        self.N = N
        self.nb_samples = sum(reduce((lambda x,y : x*y), [sum(len(e) for e in self.db[i[x]]) for x in range(N)]) for i in itertools.combinations(self.db.keys(), N))
        print("Total number of possible samples : {}".format(self.nb_samples))
        print("Total number of classes : {}".format(len(self.class_encoder([]))))
        
        # The number of samples cannot exceed the number of possible combinations
        if not nb_samples is None:
            self.nb_samples = min(self.nb_samples,nb_samples)
        
        self.list_instr = list(self.db.keys())
        self.list_sr = np.linspace((1-GLOBAL_PARAMS.coeff_freq_shift_data_augment)*GLOBAL_PARAMS.RATE, (1+GLOBAL_PARAMS.coeff_freq_shift_data_augment)*GLOBAL_PARAMS.RATE, 5)
        self.list_sr = [int(x) for x in self.list_sr]
        
        # Store in self.data all the combinations
        print("Generating the Dataset")
        self.data = []
        hash_data = HashTable(4)
        i = 0
        while i < self.nb_samples:
            is_in_hash = True
            while is_in_hash:
                samp_comb = self.getNextSample()
                # Use hash of the combination to avoid same data twice
                hash_comb = ';'.join(['_'.join([str(y) for y in x]) for x in samp_comb])
                hash_comb = hashlib.sha256(bytes(hash_comb, 'utf')).hexdigest()
                
                is_in_hash = hash_data.isin(hash_comb)
            self.data.append(samp_comb)
            hash_data.add(hash_comb)
            if i % 1000 == 0:
                print("{}/{} samples done".format(i, self.nb_samples))
            i += 1
        print("Finished generating the Dataset")
    
    def getNextSample(self):
        '''
            Function that generates one combination. It is done by iterating over
            the list of instrument combinations, and for each one, choose a combination
            of pitches chosen in the pitch bins pointetd by the corresponding IndexIncrementor objects.
        '''
        
        # Combination of instruments chosen (note that the same instrument can appear multiple times,
        # but avoid more than three times)
        comb = []
        while len(comb) < self.N:
            selected_instr = self.list_instr[random.randint(0, len(self.list_instr)-1)]
            if comb.count(selected_instr) < 3:
                comb.append(selected_instr)
        
        # Will contain the formatted version of the combination, with instruments, pitches, 
        sample_list_to_combine = []
        
        for samp_instr in comb:
            acceptable = False
            while not acceptable:
                samp_pitch_class = random.randint(0,len(self.db[samp_instr])-1)
                if len(self.db[samp_instr][samp_pitch_class]) == 0:
                    acceptable = False
                else:
                    samp_samp = random.randint(0,len(self.db[samp_instr][samp_pitch_class])-1)
                    samp_rate = self.list_sr[random.randint(0,4)]
                    
                    # A sample is acceptable if it is not too close from the an already chohsen sample
                    # ie instrument + pitch range
                    acceptable = len([x for x in sample_list_to_combine if x[0] == samp_instr and x[1] == samp_pitch_class]) == 0
            sample_list_to_combine.append((samp_instr,
                                          samp_pitch_class,
                                          samp_samp,
                                          samp_rate))
        return sample_list_to_combine

    def createClassIndices(self, raw_database):
        """
        populates the class_indices dictionary, which maps an (instr, pitch) pair to an index in the label
        @param raw_database:
        @return: None
        """
        i = 0
        for instrument in raw_database.db:
            if instrument in GLOBAL_PARAMS.instr_filter:
                self.class_indices[instrument] = {}
                pitches = set()
                for octave in raw_database.db[instrument]:
                    for sample in octave:
                        pitches.add(sample['pitch_name'])
                for pitch in pitches:
                    self.class_indices[instrument][pitch] = i
                    i += 1
        print("createClassIndices: Calculated {} classes".format(i))
        self.num_classes = i

    def save(self, path):
        '''
            Save the list containing all the combinations, without the data augmentation features
        '''
        f = open(path, 'wb')
        pickle.dump(self.data, f)
        f.close()

    def load(self, path):
        '''
            Loads a list of combinations from a pickle file.
        '''
        print("Loading dataset from file {}".format(path))
        f = open(path, 'rb')
        self.data = pickle.load(f)
        self.nb_samples = len(self.data)
        self.N = len(self.data[0])
        f.close()
    
    def __len__(self):
        return self.nb_samples

    def __getitem__(self, idx):
        '''
            Generates the actual feature fed to the model. The computations are as follow
                - Sum all STFTs from selected samples to combine and normalize it
                - Select a mel basis among those generated in the constructor (Frequency augmentation)
                - Generate the Mel spectrogram of the sum
                - Randomly zero rows and columns in a proportion that is on avergae given by prop_zero_row and prop_zero_col
                - If 'mfcc' is selected, compute the mfcc and return it
        '''
        stft = None
        list_samp = []
        sample_list_to_combine = self.data[idx]
        
        for i in sample_list_to_combine:
            samp = self.db[i[0]][i[1]][i[2]]
            list_samp.append(samp)
            if stft is None:
                stft = np.copy(samp['stft'][i[3]])
            else:
                stft += samp['stft'][i[3]]
        stft = stft / self.N
        s = np.real(stft*np.conjugate(stft))
        mel_spec = np.dot(self.mel_basis, s)
        
        zero_row = np.random.rand(mel_spec.shape[0]) < GLOBAL_PARAMS.prop_zero_row
        zero_col = np.random.rand(mel_spec.shape[1]) < GLOBAL_PARAMS.prop_zero_col
        mel_spec[zero_row] = 0.
        mel_spec[:,zero_col] = 0.
        
        if self.feature_type == 'mfcc':
            pow_db = librosa.power_to_db(mel_spec)
            mfcc = librosa.feature.mfcc(S=pow_db)
            return torch.tensor(np.array([mfcc])), self.class_encoder(list_samp)
        elif self.feature_type == 'mel':
            return torch.tensor(np.array([mel_spec])), self.class_encoder(list_samp)


class IndexIncrementor:
    '''
        This class provides a way to iterate over the set of features contained in
        a RawDatabase object to ensure that all pitch bins are uniformly chosen.
        This helps generating a homogeneous database that doesn't miss some dynamics.

        A certain degree of randomness is added in the sense that the step and starting point
        are randomized.
    '''

    def __init__(self, N, base, offset=None):
        self.N = N
        if type(base) == int:
            self.base = [base] * N
        elif type(base) == list and len(base) == N:
            self.base = base
        else:
            raise Exception("Wrong argument for IndexIncrementor (must be int or list of size N)")
        self.max_count = reduce((lambda x, y: x * y), self.base)
        self.restart(offset)

    def findStep(self):
        self.step = int(random.randint(int(0.05 * self.max_count), int(0.95 * self.max_count)))
        while math.gcd(self.step, self.max_count) > 1:
            self.step += 1

    def add(self, e):
        r = e
        i = 0
        while i < self.N and r != 0:
            r, self.index[i] = divmod(self.index[i] + r, self.base[i])
            i += 1

    def incr(self):
        self.add(self.step)
        self.count += 1

    def restart(self, offset=None):
        self.count = 0
        self.index = [0] * self.N
        self.findStep()
        if offset is None:
            self.add(random.randint(0, self.max_count - 1))
        elif offset >= 0:
            self.add(offset)
        else:
            raise Exception("Wrong offset")

    def isOver(self):
        return self.count >= self.max_count


class HashTable:
    def __init__(self, max_depth, depth=0):
        self.depth = depth
        self.max_depth = max_depth
        if max_depth == depth + 1:
            self.data = []
        else:
            self.data = {}
            for i in '0123456789abcdef':
                self.data[i] = HashTable(max_depth, depth + 1)

    def isin(self, h):
        if self.max_depth == self.depth + 1:
            return h in self.data
        else:
            return self.data[h[self.depth]].isin(h)

    def add(self, h):
        if self.max_depth == self.depth + 1:
            return self.data.append(h)
        else:
            return self.data[h[self.depth]].add(h)


#############################################################
# Following code is just for testing, delete before release #
#############################################################


# print("Loading raw database...")
# with open("SAVED_RAW_DATABASE", 'rb') as pickle_file:
#     raw_db = pickle.load(pickle_file)
# print("Done.")
#
# print("Creating class_indices dict...")
# GLOBAL_PARAMS.createClassIndices(raw_db)
# print("Done.")
#
# num_classes = 0
# for instrument in GLOBAL_PARAMS.class_indices.keys():
#     num_classes += len(GLOBAL_PARAMS.class_indices[instrument])
#
# print("OrchDataset.py: Calculated {} classes".format(num_classes))



# orch_db = OrchDataSet(raw_db, 'mel')
# orch_db.generate(GLOBAL_PARAMS.N)
# orch_db.save("SAVED_ORCH_DATABASE")