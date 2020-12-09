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

from model import OrchMatchNet, CustomLoss
from parameters import GLOBAL_PARAMS
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


timer = Timer(1000)


def main(resume_model, rdb = None):
    # dataset
    print("Start loading data -----")

    if rdb is None:
        raw_db = RawDatabase(GLOBAL_PARAMS.path, GLOBAL_PARAMS.rdm_granularity, GLOBAL_PARAMS.instr_filter)
    else:
        raw_db = rdb

    train_dataset = OrchDataSet(raw_db, class_encoder, GLOBAL_PARAMS.FEATURE_TYPE)

    if resume_model['to_resume']:
        train_dataset.load(resume_model['model_path']+'/run{}/trainset.pkl'.format(resume_model['model_run_resume']))
    else:
        train_dataset.generate(GLOBAL_PARAMS.N, int(GLOBAL_PARAMS.train_proportion * GLOBAL_PARAMS.nb_samples))
    test_dataset = OrchDataSet(raw_db, class_encoder, GLOBAL_PARAMS.FEATURE_TYPE)
    
    if resume_model['to_resume']:
        test_dataset.load(resume_model['model_path']+'/run{}/testset.pkl'.format(resume_model['model_run_resume']))
    else:
        test_dataset.generate(GLOBAL_PARAMS.N, GLOBAL_PARAMS.nb_samples - int(GLOBAL_PARAMS.train_proportion * GLOBAL_PARAMS.nb_samples))

    # load data
    train_load = torch.utils.data.DataLoader(dataset=train_dataset,
                                             batch_size=GLOBAL_PARAMS.batch_size,
                                             shuffle = True)

    test_load = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=GLOBAL_PARAMS.batch_size,
                                            shuffle = False)
    print("End loading data")

    # model construction
    out_num = len(class_encoder([]))
    features_shape = train_dataset[0][0].shape[1:]
    model = OrchMatchNet(out_num, GLOBAL_PARAMS.model_type, features_shape)
    
    start_epoch = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # model load
    if resume_model['to_resume']:
        GLOBAL_PARAMS.load_parameters(resume_model['model_path']+'/run{}'.format(resume_model['model_run_resume']))

        mpath = resume_model['model_path']+'/run{}/epoch_{}.pth'.format(resume_model['model_run_resume'], resume_model['model_epoch_resume'])
        save_path = resume_model['model_path']+'/run{}'.format(resume_model['model_run_resume'])
        if os.path.exists(mpath):
            state = torch.load(mpath)
            start_epoch = state['epoch'] + 1
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
        for d in os.listdir(resume_model['model_path']):
            if 'run' in d and i <= int(d[3:]):
                i = int(d[3:]) + 1
        save_path = resume_model['model_path'] + '/run' + str(i)
        os.mkdir(save_path)
        train_dataset.save(save_path+'/trainset.pkl')
        test_dataset.save(save_path+'/testset.pkl')
        GLOBAL_PARAMS.save_parameters(save_path)

    # train model
    train(model, save_path, optimizer, train_load, test_load, start_epoch, out_num)
    
def train(model, save_path, optimizer, train_load, test_load, start_epoch, out_num):
    print("Starting training")
    # model = torch.nn.DataParallel(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = CustomLoss(0.001)
    sig = F.sigmoid

    best_acc = 0
    total_acc = 0
    best_epoch = None
    total_loss = 0

    weight_decay = 0.01
    size_train_load = int(GLOBAL_PARAMS.train_proportion*GLOBAL_PARAMS.nb_samples/GLOBAL_PARAMS.batch_size)
    size_test_load = int((1-GLOBAL_PARAMS.train_proportion)*GLOBAL_PARAMS.nb_samples/GLOBAL_PARAMS.batch_size)
    
    loss_min = None
        
    ds = DataSaver(save_path + '/outputs.npy')
    
    for epoch in range(start_epoch, GLOBAL_PARAMS.nb_epoch):
        print("Epoch {}".format(epoch))
        model.train()
        
        for i, (trains, labels) in enumerate(train_load):
            print("Step {}".format(i))
            timer.stop()
            timer.estimate(size_train_load*(GLOBAL_PARAMS.nb_epoch-epoch-1) + (size_train_load-i-1))
            timer.start()
            trains = trains.to(device)
            labels = labels.to(device)

            # outputs is a vector of probabilities
            outputs = model(trains)

            # TODO: Change constraints to be a result of analysis of the harmonic peaks
            constraints = calculateConstraint(labels)

            if loss_min is None:
                loss_min = float(criterion(labels, labels, constraints))

            loss = criterion(outputs, labels, constraints)

            ds.save(outputs.cpu().detach().numpy(), labels.cpu().detach().numpy(), float(loss), loss_min, epoch, i)

            total_loss += float(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 50 == 0:
                print('Epoch:[{}/{}], Step:[{}/{}], Loss:{:.6f} '.format(
                    epoch + 1, GLOBAL_PARAMS.nb_epoch, i+1, size_train_load, out_num*total_loss/50))
                total_loss = 0

        graph_loss(criterion.bce_loss_list, criterion.semantic_loss_list, epoch + 1, criterion.semantic_weight)

        if (epoch+1) % 5 == 0:
            model.eval()

            test_loss = 0.0
            pret_tmp = np.zeros((0, out_num))
            grod_tmp = np.zeros((0, out_num))
            s = 0

            for k, (tests, labels) in enumerate(test_load):
                tests = tests.to(device)
                labels = labels.to(device)

                outputs = model(tests)
                
                loss = criterion(outputs, labels)

                predicts = prediction(outputs.detach().cpu().clone().numpy(), GLOBAL_PARAMS.N)
                pret_tmp = np.vstack([pret_tmp, predicts])

                grod_tmp = np.vstack([grod_tmp, labels.cpu().detach().numpy().reshape(predicts.shape)])
                
                s += GLOBAL_PARAMS.batch_size
                
                if k%100 == 0:
                    print("Test set {}/{}".format(k, len(test_load)))

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
    

def getAccuracyLoadedModel(model_dir, epoch, raw_db = None, tst = True):
    if os.path.exists(model_dir):
        GLOBAL_PARAMS.load_parameters(model_dir)
        print('Parameters loaded successfully')
        
        if raw_db is None or GLOBAL_PARAMS.rdm_granularity != raw_db.random_granularity or not (all(x in GLOBAL_PARAMS.instr_filter for x in raw_db.instr_filter) and all(x in raw_db.instr_filter for x in GLOBAL_PARAMS.instr_filter)):
            rdb = RawDatabase(GLOBAL_PARAMS.path, GLOBAL_PARAMS.rdm_granularity, GLOBAL_PARAMS.instr_filter)
        else:
            rdb = raw_db
            
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
            # compute the label-based accuracy
            result = {}
        
            result['acc'] = np.sum(preds*labels)/max(1.0, np.sum(labels))
            pitch_acc = {}
            for i in lab_class:
                l = [lab_class[i][x] for x in lab_class[i]]
                f = np.zeros(preds.shape, dtype = np.float32)
                f[:,min(l):max(l)+1] = 1.0
                f = labels*f
                pitch_acc[i] = np.sum(preds*f)/max(1.0, np.sum(f))
            result['pitch_acc'] = pitch_acc
            
        dataset = OrchDataSet(rdb,class_encoder, GLOBAL_PARAMS.FEATURE_TYPE)
        if tst:
            dataset.load(model_dir+'/trainset.pkl')
        else:
            dataset.load(model_dir+'/testset.pkl')
        
        dataset_load = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=GLOBAL_PARAMS.batch_size,
                                            shuffle = False)
        
        out_num = len(class_encoder([]))
        features_shape = dataset[0][0].shape[1:]
        model = OrchMatchNet(out_num, GLOBAL_PARAMS.model_type, features_shape)
        
        state = torch.load(model_dir+'/epoch_{}.pth'.format(epoch))
        model.load_state_dict(state['state_dict'])
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
    
        criterion = nn.BCELoss()
        s = 0
        
        for k, (samps, labels) in enumerate(dataset_load):
            samps = samps.to(device)
            labels = labels.to(device)

            outputs = model(samps)
            
            loss = criterion(outputs, labels)

            predicts = prediction(outputs.detach().cpu().clone().numpy(), GLOBAL_PARAMS.N)
            
            s += 1
            
            if k%100 == 0:
                print("Test set {}/{}".format(k, len(dataset_load)))


def calculateConstraint(labels):
    """
    takes in a single label, not a batch
    harmonic constraints: include sample if exact note is present in training data
    :param labels: shape (num_labels,)
    :return: tensor shape (batch size, num features) with binary values
    """
    constraints = torch.zeros_like(labels)
    for i in range(labels.shape[0]):
        label = labels[i]
        pitches_to_include = set()  # list of pitches allowed in the orchestration
        indices = label.nonzero(as_tuple=True)[0]
        for index in indices.tolist():
            instr, pitch = GLOBAL_PARAMS.index2sample[index]
            pitches_to_include.add(pitch)
        # this part might be slow
        for instr, pitches in GLOBAL_PARAMS.sample2index.items():
            for pitch in pitches_to_include:
                if pitch in pitches:
                    index = GLOBAL_PARAMS.sample2index[instr][pitch]
                    constraints[i][index] = 1.0
    return constraints


def graph_loss(bce_loss_list, semantic_loss_list, epoch, semantic_weight):
    plt.title("Loss after {} epochs; {} samples per epoch; semantic weight = {}"
              .format(epoch, GLOBAL_PARAMS.nb_samples, semantic_weight))
    plt.plot(bce_loss_list[1:], label="BCE Loss")
    plt.plot(semantic_loss_list[1:], label="Semantic Loss")
    plt.legend()
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="Path to the directory containing all the data of the runs")
    parser.add_argument("--samples_path", help="Path to the pkl containing the raw samples. If ")
    parser.add_argument("--N", type=int, help="Number of instruments in combination")
    parser.add_argument("--nb_samples", type=int, help="Number of samples to use per epoch")
    parser.add_argument("--epochs", type=int, help="Total number of epochs")
    parser.add_argument("--resume", help="If need to resume model", action="store_true")
    parser.add_argument("--resume_run", type=int, help="Path to the directory containing checkpoint of the model")
    parser.add_argument("--resume_epoch", type=int, help="Path to the directory containing checkpoint of the model")
    args = parser.parse_args()
    if args.resume:
        resume_model = {"to_resume": True,
                        "model_path": args.model_path,
                        "model_run_resume": args.resume_run,
                        "model_epoch_resume": args.resume_epoch}
        print("Resuming model located in {} run {} at epoch {}".format(resume_model['model_path'], resume_model['model_run_resume'], resume_model['model_epoch_resume']))
    else:
        resume_model = {"to_resume": False,
                        "model_path": args.model_path}
        # GLOBAL_PARAMS.N = args.N
        # GLOBAL_PARAMS.nb_samples = args.nb_samples
        # GLOBAL_PARAMS.nb_epoch = args.epochs
        print("Number of instruments per combination : {}".format(GLOBAL_PARAMS.N))
        print("Number of samples used per epoch : {}".format(GLOBAL_PARAMS.nb_samples))
        print("Total number of epochs : {}".format(GLOBAL_PARAMS.nb_epoch))
    
    print("Loading database")
    with open('../SAVED_RAW_DATABASE', 'rb') as f:
        rdb = pickle.load(f)
    print("Database loaded")
    
    GLOBAL_PARAMS.sample2index = {}
    tot_size = 0
    for instr in rdb.db:
        if instr in GLOBAL_PARAMS.instr_filter:
            GLOBAL_PARAMS.sample2index[instr] = {}
            pitches = set()
            for k in rdb.db[instr]:
                for j in k:
                    pitches.add(j['pitch_name'])
            for pitch in pitches:
                GLOBAL_PARAMS.sample2index[instr][pitch] = tot_size
                GLOBAL_PARAMS.index2sample.append((instr, pitch))
                tot_size += 1

    def class_encoder(list_samp):
        label = [0 for i in range(tot_size)]
        for s in list_samp:
            label[GLOBAL_PARAMS.sample2index[s['instrument']][s['pitch_name']]] = 1
        return np.array(label).astype(np.float32)
    
    def evaluate(preds, labels):
        if preds.shape != labels.shape:
            print("[Error]: size difference")
        # compute the label-based accuracy
        result = {}
    
        result['acc'] = np.sum(preds*labels)/max(1.0, np.sum(labels))
        pitch_acc = {}
        for i in GLOBAL_PARAMS.sample2index:
            l = [GLOBAL_PARAMS.sample2index[i][x] for x in GLOBAL_PARAMS.sample2index[i]]
            f = np.zeros(preds.shape, dtype = np.float32)
            f[:,min(l):max(l)+1] = 1.0
            f = labels*f
            pitch_acc[i] = np.sum(preds*f)/max(1.0, np.sum(f))
        result['pitch_acc'] = pitch_acc
    
        return result
    
    main(resume_model, rdb)