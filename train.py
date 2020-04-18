import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import data
import numpy as np
import argparse
import time
import pickle
import os
from functools import reduce
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from model import OrchMatchNet
from parameters import N, FEATURE_TYPE
from parameters import nb_samples, rdm_granularity, nb_pitch_range, instr_filter, batch_size, model_type, nb_epoch, train_proportion
from parameters import coeff_freq_shift_data_augment, prop_zero_col, prop_zero_row
from parameters import load_parameters, save_parameters
from parameters import model_path, model_run_resume, model_epoch_resume, resume_model
from OrchDataset import OrchDataSet,RawDatabase

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
    '''
        Class used to write data for visualization (see showOutputs.py)
    '''
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
    train_dataset = OrchDataSet(raw_db,class_encoder, FEATURE_TYPE)
    train_dataset.generate(N,int(train_proportion*nb_samples))
    test_dataset = OrchDataSet(raw_db,class_encoder, FEATURE_TYPE)
    test_dataset.generate(N,nb_samples-int(train_proportion*nb_samples))

    # load data
    train_load = torch.utils.data.DataLoader(dataset=train_dataset,
                                             batch_size=batch_size,
                                             shuffle = True)

    test_load = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle = False)
    print("End loading data")

    # model construction
    out_num = len(class_encoder([]))
    features_shape = train_dataset[0][0].shape[1:]
    model = OrchMatchNet(out_num, model_type, features_shape)
    
    start_epoch = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # model load
    if resume_model:
        mpath = model_path+'/run{}/epoch_{}.pth'.format(model_run_resume, model_epoch_resume)
        save_path = model_path+'/run{}'.format(model_run_resume)
        if os.path.exists(mpath):
            state = torch.load(mpath)
            start_epoch = state['epoch']
            model.load_state_dict(state['state_dict'])
            optimizer.load_state_dict(state['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

            print("Load %s successfully! " % mpath)
        else:
            print("[Error] no checkpoint ")
    else:
        i = 0
        for d in os.listdir(model_path):
            if 'run' in d and i <= int(d[3:]):
                i = int(d[3:]) + 1
        save_path = model_path + '/run' + str(i)
        os.mkdir(save_path)
        train_dataset.save(save_path+'/trainset.pkl')
        test_dataset.save(save_path+'/testset.pkl')
        save_parameters(save_path)

    # train model
    train(model, save_path, optimizer, train_load, test_load, start_epoch, out_num)
    
def train(model, save_path, optimizer, train_load, test_load, start_epoch, out_num):
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

            for k, (tests, labels) in enumerate(test_load):
                tests = tests.to(device)
                labels = labels.to(device)

                start = time.time()
                outputs = model(tests)
                end = time.time()
                loss = criterion(outputs, labels)

                predicts = prediction(outputs, N)
                #predicts = outputs.detach().cpu().clone().numpy()
                pret_tmp = np.vstack([pret_tmp, predicts])

                # pret_tmp[s:s+batch_size,
                #          :] = outputs.cpu().detach().numpy().reshape((batch_size, -1))
                grod_tmp = np.vstack([grod_tmp, labels.cpu().detach().numpy().reshape(predicts.shape)])
                s += batch_size
                
                if k%100 == 0:
                    print("Test set {}/{}".format(k, len(test_load)))

                total_time += float(end-start)
                test_loss += float(loss)

                # result_ins = evaluate(predicts, labels, result_ins)

            # pret_tmp[pret_tmp >= 0.05] = 1
            # pret_tmp[pret_tmp < 0.05] = 0
            result = evaluate(pret_tmp, grod_tmp)
            f = open(save_path + '/result{}.pkl'.format(epoch), 'wb')
            pickle.dump(result, f)
            f.close()
            
        state = {
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'state_dict': model.state_dict(),
        }
        torch.save(state, save_path+"/epoch_"+str(epoch)+".pth")
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
    
        predicts = prediction(outputs)
        #predicts = outputs.detach().cpu().clone().numpy()
        pret_tmp = np.vstack([pret_tmp, predicts])
    
        # pret_tmp[s:s+batch_size,
        #          :] = outputs.cpu().detach().numpy().reshape((batch_size, -1))
        grod_tmp = np.vstack([grod_tmp, labels.cpu().detach().numpy().reshape(predicts.shape)])
        s += batch_size
    
        total_time += float(end-start)
        test_loss += float(loss)
    
    result = evaluate(pret_tmp, grod_tmp)
    return result

def getPosNMax(l, N):
    ind = [0]
    sort_key=lambda x : l[x]
    for j in range(1,len(l)):
        if len(ind) == N:
            if l[j] > l[ind[0]]:
                ind[0] = j
                ind.sort(key=sort_key)
        else:
            ind.append(j)
            ind.sort(key=sort_key)
    return ind
            

def prediction(outputs, N):
    pred = np.zeros(outputs.shape)
    for i in range(pred.shape[0]):
        for j in getPosNMax(outputs[i],N):
            pred[i,j] = 1.0
    return pred
    

#rdb = RawDatabase(path, rdm_granularity, instr_filter)
#test('./model/nb_pitch_range_1/epoch_19.pth', rdb, 12)
if __name__=='__main__':

    try:
        rdb
    except NameError:
        rdb = RawDatabase('./TinySOL', rdm_granularity, instr_filter)
        
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
    
    def evaluate(preds, labels):
        if preds.shape != labels.shape:
            print("[Error]: size difference")
        total_num = preds.shape[0]
        # compute the label-based accuracy
        result = {}
    
        result['acc'] = np.sum(preds*labels)/max(1.0, np.sum(labels))
        pitch_acc = {}
        j=0
        for i in lab_class:
            l = [lab_class[i][x] for x in lab_class[i]]
            f = np.zeros(preds.shape, dtype = np.float32)
            f[:,min(l):max(l)+1] = 1.0
            f = labels*f
            pitch_acc[i] = np.sum(preds*f)/max(1.0, np.sum(f))
            j+=12
        result['pitch_acc'] = pitch_acc
    
        return result
        
    main(rdb)