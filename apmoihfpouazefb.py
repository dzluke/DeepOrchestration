from parameters import GLOBAL_PARAMS, SimParams
from OrchDataset import OrchDataSet, RawDatabase
import numpy as np
import matplotlib.pyplot as plt
from model import OrchMatchNet
import os
import librosa
import random

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import data

PATH = './model/ResNet/run1'
train_list = list(range(1))



GLOBAL_PARAMS.load_parameters(PATH)

tot_size = sum(len(GLOBAL_PARAMS.lab_class[k]) for k in GLOBAL_PARAMS.lab_class)
    
def class_encoder(list_samp):
    label = [0 for i in range(tot_size)]
    for s in list_samp:
        label[GLOBAL_PARAMS.lab_class[s['instrument']][s['pitch_name']]] = 1
    return np.array(label).astype(np.float32)

def evaluate(preds, labels):
    if preds.shape != labels.shape:
        print("[Error]: size difference")
    # compute the label-based accuracy
    result = {}

    result['acc'] = np.sum(preds*labels)/max(1.0, np.sum(labels))
    pitch_acc = {}
    for i in GLOBAL_PARAMS.lab_class:
        l = [GLOBAL_PARAMS.lab_class[i][x] for x in GLOBAL_PARAMS.lab_class[i]]
        f = np.zeros(preds.shape, dtype = np.float32)
        f[:,min(l):max(l)+1] = 1.0
        f = labels*f
        pitch_acc[i] = np.sum(preds*f)/max(1.0, np.sum(f))
    result['pitch_acc'] = pitch_acc

    return result


def testMix(mix):
    samps = []
    for i in mix:
        instr,pitch,nuance=i
        p = './TinySOL/{}/ordinario'.format(instr)
        l = [x for x in os.listdir(p) if pitch in x and nuance in x]
        if len(l) != 1:
            raise Exception('Wrong sample {}'.format(i))
        y,sr = librosa.load(p + '/' + l[0], sr=None)
        nb_samples = int(GLOBAL_PARAMS.TIME_LENGTH*sr)
        if len(y) < nb_samples:
            y = np.append(y, np.zeros((1,(nb_samples-len(y))), dtype=np.float32))
        else:
            y = y[:nb_samples]
        samps.append(librosa.feature.melspectrogram(y=y,sr=sr,hop_length=GLOBAL_PARAMS.MEL_HOP_LENGTH,n_fft=GLOBAL_PARAMS.N_FFT,n_mels=GLOBAL_PARAMS.N_MELS))
    return sum(samps)/(len(samps)**2.)

def getModelOutput(epoch, mix):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.BCELoss()
    state = torch.load(PATH + '/epoch_{}.pth'.format(epoch))
    
    samp = torch.tensor(np.array([[testMix(mix)]]))
    samp = samp.to(device)
    
    labels = []
    for x in mix:
        s = {}
        s['instrument'] = x[0]
        s['pitch_name'] = x[1]
        labels.append(s)
    
    out_num = len(class_encoder([]))
    features_shape = samp.shape[2:]
    model = OrchMatchNet(out_num, GLOBAL_PARAMS.model_type, features_shape)
    
    model.load_state_dict(state['state_dict'])
    model = model.to(device)
    
    model.eval()
    outputs = model(samp)
    
    plt.plot(outputs.cpu().detach().numpy().reshape(-1))
    plt.plot(class_encoder(labels))

def getRandMix(N):
    NN = random.randint(0,N-1)

getModelOutput(13, [('Ob','A#5','ff'),('Hn','A#2','ff')])

#rdb = RawDatabase('./TinySOL', GLOBAL_PARAMS.rdm_granularity, GLOBAL_PARAMS.instr_filter)
#
#train_dataset = OrchDataSet(rdb,class_encoder, GLOBAL_PARAMS.FEATURE_TYPE)
#train_dataset.load(PATH + '/trainset.pkl')
#test_dataset = OrchDataSet(rdb,class_encoder, GLOBAL_PARAMS.FEATURE_TYPE)
#test_dataset.load(PATH + '/testset.pkl')
#
## load data
#train_load = torch.utils.data.DataLoader(dataset=train_dataset,
#                                         batch_size=GLOBAL_PARAMS.batch_size,
#                                         shuffle = True)
#
#test_load = torch.utils.data.DataLoader(dataset=test_dataset,
#                                        batch_size=GLOBAL_PARAMS.batch_size,
#                                        shuffle = False)
#
#out_num = len(class_encoder([]))
#features_shape = train_dataset[0][0].shape[1:]
#model = OrchMatchNet(out_num, GLOBAL_PARAMS.model_type, features_shape)
#
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#criterion = nn.BCELoss()
#    
#for epoch in train_list:
#    state = torch.load(PATH + '/epoch_{}.pth'.format(epoch))
#    model.load_state_dict(state['state_dict'])
#    model = model.to(device)
#    print("Epoch {}".format(epoch))
#    
#    loss_hist = []
#    
#    model.eval()
#    
#    for i, (trains, labels) in enumerate(test_load):
#        print("Batch {}".format(i))
#        trains = trains.to(device)
#        labels = labels.to(device)
#
#        outputs = model(trains)
#        loss_hist.append(float(criterion(outputs, labels)))