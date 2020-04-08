import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import data
import numpy as np
import argparse
import time
import json
import os
import itertools
import argparse
from functools import reduce
from generateDB import generateDBLabels
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from model import OrchMatchNet
from OrchDataset import OrchDataSet,RawDatabase,showSample

# Stick to the problem of Luke (classify 24 classes)
# Increase the number of samples
# Put pitches into bins

path = "./TinySOL"
N = 2
nb_samples = 40000
rdm_granularity = 10
random_pool_size = 100
nb_pitch_range = 8
instr_filter = None
batch_size = 16
model_type = 'cnn'
nb_epoch = 200
model_path = './model'
resume_model = False
train_proportion = 0.8

class Timer:
    def __init__(self, size_buffer):
        self.measuring = False
        self.begin = []
        self.end = []
        self.size_buffer = size_buffer
        
    def start(self):
        if not self.measuring:
            if len(self.begin) == self.size_buffer:
                self.begin.pop(0)
            self.begin.append(time.time())
            self.measuring = True
            
    def stop(self):
        if self.measuring:
            if len(self.end) == self.size_buffer:
                self.end.pop(0)
            self.end.append(time.time())
            self.measuring = False
    
    def estimate(self, remaining):
        if len(self.end) > 0 and not self.measuring:
            diff = np.array(self.end)-np.array(self.begin)
            diff = np.mean(diff)
            h,r=divmod(diff*remaining,3600)
            m,s=divmod(r,60)
            print(diff)
            print("Estimated remaining time : {}h{}m{}s".format(int(h), int(m), int(s)))
            

class DataSaver:
    def __init__(self, path_file):
        self.path = path_file
        self.count = 0
        
    def save(self, outs, labels, loss, loss_min, epoch, step):
        r = {}
        r['epoch'] = epoch
        r['step'] = step
        r['outputs'] = outs
        r['labels'] = labels
        r['loss'] = loss
        r['loss_min'] = loss_min
        np.save(self.path, r)
        self.count += 1
        
ds = DataSaver('./outputs.npy')

timer = Timer(1000)

def main(rdb = None):
    #print("Starting parsing -----")
    #arg = arg_parse()
    #print("End parsing")

    # dataset
    print("Start loading data -----")

    if rdb is None:
        raw_db = RawDatabase(path, rdm_granularity, instr_filter)
    else:
        raw_db = rdb
    train_dataset = OrchDataSet(raw_db,N,int(train_proportion*nb_samples),random_pool_size,nb_pitch_range)
    test_dataset = OrchDataSet(raw_db,N,int((1-train_proportion)*nb_samples),random_pool_size,nb_pitch_range)

    # load data
    train_load = torch.utils.data.DataLoader(dataset=train_dataset,
                                             batch_size=batch_size)

    test_load = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size)
    print("End loading data")

    # model construction
    model = OrchMatchNet(len(train_dataset.out_classes), model_type)
    
    start_epoch = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # model load
    if resume_model:
        ckpt = os.listdir(model_path)
        ckpts = []
        for x in ckpt:
            if x.endswith('.pth'):
                ckpts.append(int(x.split('.')[0].split('_')[-1]))

        if len(ckpt) != 0:
            model_resume_path = 'model_epoch_'+str(max(ckpts))+'.pth'
            state = torch.load(model_path+'/'+model_resume_path)
            start_epoch = state['epoch']
            model.load_state_dict(state['state_dict'])
            optimizer.load_state_dict(state['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

            print("Load %s successfully! " % model_resume_path)
        else:
            print("[Error] no checkpoint ")

    # train model
    train(model, optimizer, train_load, test_load, start_epoch, len(train_dataset.out_classes))
    
def train(model, optimizer, train_load, test_load, start_epoch, out_num):
    print("Starting training")
    # model = torch.nn.DataParallel(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.BCELoss()
    sig = F.sigmoid

    best_acc = 0
    total_acc = 0
    best_epoch = None
    total_loss = 0

    weight_decay = 0.01
    size_train_load = int(train_proportion*nb_samples/batch_size)
    size_test_load = int((1-train_proportion)*nb_samples/batch_size)
    
    loss_min = None
    
    for epoch in range(start_epoch, nb_epoch):
        train_load.dataset.reinitialize()
        test_load.dataset.reinitialize()
        
        print("Epoch {}".format(epoch))
        model.train()
        
        for i, (trains, labels) in enumerate(train_load):
            print("Step {}".format(i))
            timer.stop()
            timer.estimate(size_train_load*(nb_epoch-epoch-1) + (size_train_load-i-1))
            timer.start()
            trains = trains.to(device)
            labels = labels.to(device)

            outputs = model(trains)

            if loss_min is None:
                loss_min = float(criterion(labels, labels))

            loss = criterion(outputs, labels)
            ds.save(outputs.cpu().detach().numpy(), labels.cpu().detach().numpy(), float(loss), loss_min, epoch, i)

            total_loss += float(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 50 == 0:
                print('Epoch:[{}/{}], Step:[{}/{}], Loss:{:.6f} '.format(
                    epoch + 1, nb_epoch, i+1, size_train_load, out_num*total_loss/50))
                total_loss = 0

        if (epoch+1) % 5 == 0:
            model.eval()

            total_time = 0.0
            test_loss = 0.0
            pret_tmp = np.zeros((0, out_num))
            grod_tmp = np.zeros((0, out_num))
            s = 0

            for tests, labels in test_load:
                tests = tests.to(device)
                labels = labels.to(device)

                start = time.time()
                outputs = model(tests)
                end = time.time()
                loss = criterion(outputs, labels)

                predicts = get_pred(outputs)
                #predicts = outputs.detach().cpu().clone().numpy()
                pret_tmp = np.vstack([pret_tmp, predicts])

                # pret_tmp[s:s+batch_size,
                #          :] = outputs.cpu().detach().numpy().reshape((batch_size, -1))
                grod_tmp = np.vstack([grod_tmp, labels.cpu().detach().numpy().reshape(predicts.shape)])
                s += batch_size

                total_time += float(end-start)
                test_loss += float(loss)

                # result_ins = evaluate(predicts, labels, result_ins)

            # pret_tmp[pret_tmp >= 0.05] = 1
            # pret_tmp[pret_tmp < 0.05] = 0
            result = evaluate(pret_tmp, grod_tmp)

            
            print("Test Loss: {:.6f}".format(out_num*test_loss/size_test_load))
        
        state = {
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'state_dict': model.state_dict(),
        }
        torch.save(state, model_path+"/epoch_"+str(epoch)+".pth")
        print("Model saved")

    print("Best test accuracy: {} at epoch: {}".format(best_acc, best_epoch))
        
def test(path, raw_db, out_num):
    state = torch.load(path)
    epoch = state['epoch']
    criterion = F.binary_cross_entropy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OrchMatchNet(out_num, model_type)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()    
    odb = OrchDataSet(raw_db, N, int((1-train_proportion)*nb_samples), random_pool_size, nb_pitch_range)
    test_load = torch.utils.data.DataLoader(odb, batch_size = batch_size)
    
    model.eval()

    total_time = 0.0
    test_loss = 0.0
    pret_tmp = np.zeros((0, out_num))
    grod_tmp = np.zeros((0, out_num))
    s = 0
    
    for tests, labels in test_load:
        tests = tests.to(device)
        labels = labels.to(device)
    
        start = time.time()
        outputs = model(tests)
        end = time.time()
        loss = criterion(outputs, labels)
    
        predicts = get_pred(outputs)
        #predicts = outputs.detach().cpu().clone().numpy()
        pret_tmp = np.vstack([pret_tmp, predicts])
    
        # pret_tmp[s:s+batch_size,
        #          :] = outputs.cpu().detach().numpy().reshape((batch_size, -1))
        grod_tmp = np.vstack([grod_tmp, labels.cpu().detach().numpy().reshape(predicts.shape)])
        s += batch_size
    
        total_time += float(end-start)
        test_loss += float(loss)
    
    accurate = pret_tmp*grod_tmp
    miss = pret_tmp - grod_tmp
    accuracy = np.sum(accurate,axis=1)/N

    

def evaluate(pret, grot):
    if pret.shape != grot.shape:
        print("[Error]: size difference")
    total_num = pret.shape[0]
    # compute the label-based accuracy
    result = {}

    gt_pos = np.sum((grot == 1).astype(float), axis=0)
    gt_neg = np.sum((grot == 0).astype(float), axis=0)
    pt_pos = np.sum(pret *
                    (grot == 1).astype(float), axis=0)
    pt_neg = np.sum((grot == 0).astype(float) *
                    (pret == 0).astype(float), axis=0)
    label_pos_acc = 1.0*pt_pos/np.array([max(e,1) for e in gt_pos])

    return result


def get_pred(output):
    '''
        get Top N prediction
    '''


    pred = np.zeros(output.shape)
    for k, o in enumerate(output):
        preidx = []
        for i in range(N):
            idx = o.max(0)[1]
            preidx.append(idx)
            o[int(idx)] = -1

        for idx in preidx:
            pred[k][idx] = 1.0

    return pred

#rdb = RawDatabase(path, rdm_granularity, instr_filter)
#test('./model/nb_pitch_range_1/epoch_19.pth', rdb, 12)
main(rdb)