import librosa
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
import pickle

from model import OrchMatchNet
from parameters import GLOBAL_PARAMS, SimParams
from OrchDataset_2 import RawDatabase, OrchDataSet

path = "D:/DeepOrchestration/New_tests/CNN/run0/epoch_39.pth"

print("Loading rdb")
try:
    rdb
except:
    with open("D:/DeepOrchestration/rdb.pkl", 'rb') as f:
        rdb = pickle.load(f)
    
GLOBAL_PARAMS.lab_class = {}
GLOBAL_PARAMS.N = 10
GLOBAL_PARAMS.nb_samples = 100

tot_size = 0
for i in rdb.db:
    GLOBAL_PARAMS.lab_class[i] = {}
    a = set()
    for k in rdb.db[i]:
        for j in k:
            a.add(j['pitch_name'])
    for x in a:
        GLOBAL_PARAMS.lab_class[i][x] = tot_size
        tot_size += 1
    
def class_encoder(list_samp):
    label = [0 for i in range(tot_size)]
    for s in list_samp:
        label[GLOBAL_PARAMS.lab_class[s['instrument']][s['pitch_name']]] = 1
    return np.array(label).astype(np.float32)
    
train_dataset = OrchDataSet(rdb,class_encoder, GLOBAL_PARAMS.FEATURE_TYPE)
train_dataset.generate(GLOBAL_PARAMS.N, GLOBAL_PARAMS.nb_samples)

print("Loading model")
print("Shape : {}".format(train_dataset[0][0].shape[1:]))
model = OrchMatchNet(424, 'cnn', train_dataset[0][0].shape[1:])
state = torch.load(path)
model.load_state_dict(state['state_dict'])
    
print("Generating sample")
samp = train_dataset[0][0].view([1,1,128,87])

res = [samp]
res.extend(model.getLatentSpace(samp))

def plot_latent_space(res, layer):
    nrows = 2**(int(np.log2(res[layer].shape[1]))//2)
    ncols = res[layer].shape[1]/nrows
    
    plt.figure()
    for i in range(res[layer].shape[1]):
        axi = plt.subplot(nrows, ncols, i+1)
        axi.matshow(res[layer][0][i])
        axi.get_xaxis().set_visible(False)
        axi.get_yaxis().set_visible(False)
    plt.savefig('./paper/latex/figs/latent_space_layer{}.png'.format(layer))
        
plt.close('all')
for i in range(len(res)-1):
    plot_latent_space(res, i)