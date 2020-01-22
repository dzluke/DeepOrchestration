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

from model import OrchMatchNet
# from process_TinySOL import show_all_class_num, show_all_instru_num, stat_test_db, decode, N
from dataset import OrchDataSet

out_num = 3674
time = 4
N = 5


def get_data():
    target_path = '/home/data/happipub/gradpro_l/target'
    features = []
    for x in os.listdir(target_path):
        y, sr = librosa.load(os.path.join(target_path, x), sr=None)

        feature = librosa.feature.melspectrogram(y, sr).T

        if feature.shape[0] <= 256:
            # add zero
            zero = np.zeros((256-feature.shape[0], 128), dtype=np.float32)
            feature = np.vstack((feature, zero))
        else:
            feature = feature[:256]
            # feature = feature[:-1*(feature.shape[0] % 128)]

        num_chunk = feature.shape[0]/128
        feature = np.split(feature, num_chunk)
        features.append([torch.tensor([feature]), x])

    return features


def test():
    server_model_path = '/home/data/happipub/gradpro_l/model/three'
    state = torch.load(server_model_path+"/epoch_best.pth")
    model = OrchMatchNet(out_num, 'cnn')
    model.load_state_dict(state['state_dict'])

    datas = get_data()
    model = model.float()
    model.eval()
    for data, x in datas:
        print('--------------------------')
        print('target: ', x)
        out = model(data.float())
        get_pred_file(out)


def synthesize():
    f = open('./exp/five/five.txt')
    path = './new_OrchDB_ord'
    lines = f.readlines()
    name = ''
    soundlist = []
    f_inx = {}
    for f in os.listdir(path):
        if f.split('.')[0].endswith('c'):
            f_inx[f.split('.')[0][:-3]] = os.path.join(path, f)
        else:
            f_inx[f.split('.')[0]] = os.path.join(path, f)

    for i in range(len(lines)):
        line = lines[i]
        if line.startswith('target:'):
            name = line.strip().split(':')[-1][2:]
            for j in range(N):
                soundlist.append(f_inx[lines[i+2+j][:-1]])
            print(name, soundlist)
            combine(soundlist, name)
            soundlist = []


def combine(soundlist, n):
    mixed_file = np.zeros((1, 1))
    sr = 0
    for sound in soundlist:
        sfile, sr = librosa.load(sound, sr=None)
        mixed_file = mix(mixed_file, sfile)
    mixed_file = mixed_file/len(soundlist)
    mixed_file = mixed_file[:time*sr]

    name = './exp/five/' + n
    librosa.output.write_wav(name, y=mixed_file, sr=sr)
    print('finish')
    # return [mixed_file, sr, mixed_label]


def mix(fa, fb):
    diff = len(fa) - len(fb)

    if diff >= 0:
        add = np.zeros((1, diff), dtype=np.float32)
        fb = np.append(fb, add)
    else:
        add = np.zeros((1, -diff), dtype=np.float32)
        fa = np.append(fa, add)

    return fa+fb


def get_pred_file(output):
    '''
        get Top N prediction
        or set a threshold
    '''

    pred = np.zeros(output.shape)
    inx = json.load(open('class.index', 'r'))

    preidx = []
    print("orch:")
    for i, p in enumerate(output[0]):
        if p > 0.01:
            preidx.append(i)
            orch_file = list(inx.keys())[list(inx.values()).index(i)]
            print(round(float(p), 3), orch_file)

        # for i in range(N):
        #     idx = output.max(1, keepdim=True)[1]
        #     preidx.append(idx)
        #     output[0][idx] = -1


if __name__ == "__main__":
    test()
