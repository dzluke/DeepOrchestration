import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import argparse
import time
import json
import os

from model import OrchMatchNet
from process_OrchDB import instruments, decode, N, out_num, stat_test_db
from dataset import OrchDataSet


epoch_num = 200
batch_size = 16
my_model_path = './model'
server_model_path = './model'
my_data_path = './data/five'
featurized_data_path = './featurized_data/'

TEST_MODEL = True

# db = {'Vc': 3560.0, 'Fl': 2831.0, 'Va': 3469.0, 'Vn': 3328.0, 'Ob': 2668.0, 'BTb': 2565.0,
#       'Cb': 3305.0, 'ClBb': 2960.0, 'Hn': 3271.0, 'TpC': 2245.0, 'Bn': 2944.0, 'Tbn': 2854.0}

db = stat_test_db()

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
            

timer = Timer(10)

def arg_parse():
    parser = argparse.ArgumentParser(description='combination of orch')
    parser.add_argument('--model', default='resnet')
    parser.add_argument('--is_resume', default='False',
                        choices=['True', 'False'])

    return parser.parse_args()


def main():
    print("Starting parsing -----")
    arg = arg_parse()
    print("End parsing")

    # dataset
    print("Start loading data -----")

    trainset = OrchDataSet(
        featurized_data_path, 'training', transforms.ToTensor())
    testset = OrchDataSet(
        featurized_data_path, 'test', transforms.ToTensor())

    # load data
    train_load = torch.utils.data.DataLoader(dataset=trainset,
                                             batch_size=batch_size,
                                             shuffle=True)

    test_load = torch.utils.data.DataLoader(dataset=testset,
                                            batch_size=batch_size,
                                            shuffle=False)
    print("End loading data")

    # model construction
    model = OrchMatchNet(out_num, arg.model)
    
    if TEST_MODEL:
        model_resume_path = 'epoch_199.pth'
        state = torch.load(server_model_path+'/'+model_resume_path)
        start_epoch = state['epoch']
        model.load_state_dict(state['state_dict'])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.load_state_dict(state['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
                    model.eval()
                    
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        criterion = F.binary_cross_entropy
        pret_tmp = np.zeros((len(test_load)*batch_size, out_num))
        grod_tmp = np.zeros((len(test_load)*batch_size, out_num))
        s = 0

        for tests, labels in test_load:
            tests = tests.to(device)
            labels = labels.to(device)

            outputs = model(tests)
            loss = criterion(outputs, labels/N)

            #predicts = get_pred(outputs)
            predicts = outputs.detach().cpu().clone().numpy()
            pret_tmp[s:s+batch_size,
                     :] = predicts.reshape((batch_size, -1))

            # pret_tmp[s:s+batch_size,
            #          :] = outputs.cpu().detach().numpy().reshape((batch_size, -1))
            grod_tmp[s:s+batch_size,
                     :] = labels.cpu().detach().numpy().reshape((batch_size, -1))
            s += batch_size


            # result_ins = evaluate(predicts, labels, result_ins)

        # pret_tmp[pret_tmp >= 0.05] = 1
        # pret_tmp[pret_tmp < 0.05] = 0
        result = evaluate(pret_tmp, grod_tmp)
        a = 5

        #print("Test Loss: {:.6f}".format(out_num*test_loss/len(test_load)))
        #total_acc = 100*float(result['a'][-1])/len(test_load)
        
    else:
        start_epoch = 0
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
        # model load
        if arg.is_resume == 'True':
            ckpt = os.listdir(server_model_path)
            ckpts = []
            for x in ckpt:
                if x.endswith('.pth'):
                    ckpts.append(int(x.split('.')[0].split('_')[-1]))
    
            if len(ckpt) != 0:
                model_resume_path = 'model_epoch_'+str(max(ckpts))+'.pth'
                state = torch.load(server_model_path+'/'+model_resume_path)
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
        train(model, optimizer, train_load, test_load, start_epoch)


def train(model, optimizer, train_load, test_load, start_epoch):
    print("Starting training")
    # model = torch.nn.DataParallel(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    saved_model = False

    criterion = F.binary_cross_entropy
    sig = F.sigmoid

    best_acc = 0
    total_acc = 0
    best_epoch = None
    total_loss = 0

    weight_decay = 0.01
    size_train_load = len(train_load)

    for epoch in range(start_epoch, epoch_num):
        print("Epoch {}".format(epoch))
        model.train()

        size_train_load = len(train_load)
        for i, (trains, labels) in enumerate(train_load):
            print("Step {}".format(i))
            timer.stop()
            timer.estimate(size_train_load*(epoch_num-epoch-1) + (size_train_load-i-1))
            timer.start()
            trains = trains.to(device)
            labels = labels.to(device)

            outputs = model(trains)
            # L2 regularization
            # l2_reg = torch.tensor(0.)
            # for param in model.parameters():
            #     l2_reg += torch.norm(param, p=2)

            loss = criterion(outputs, labels/N)

            total_loss += float(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 50 == 0:
                print('Epoch:[{}/{}], Step:[{}/{}], Loss:{:.6f} '.format(
                    epoch + 1, epoch_num, i+1, len(train_load), out_num*total_loss/50))
                total_loss = 0

        # save cuurent model
        # if (epoch+1) % 100 == 0:
        #     model_name = server_model_path + \
        #         '/model_epoch_'+str(epoch+1)+'.pth'
        #     state = {
        #         'epoch': epoch,
        #         'optimizer': optimizer.state_dict(),
        #         'state_dict': model.state_dict(),
        #     }

        #     torch.save(state, model_name)
        #     print('model_epoch_'+str(epoch+1)+' saved')

        if (epoch+1) % 5 == 0:
            model.eval()

            total_time = 0.0
            test_loss = 0.0
            pret_tmp = np.zeros((len(test_load)*batch_size, out_num))
            grod_tmp = np.zeros((len(test_load)*batch_size, out_num))
            s = 0

            for tests, labels in test_load:
                tests = tests.to(device)
                labels = labels.to(device)

                start = time.time()
                outputs = model(tests)
                end = time.time()
                loss = criterion(outputs, labels/N)

                #predicts = get_pred(outputs)
                predicts = outputs.detach().cpu().clone().numpy()
                pret_tmp[s:s+batch_size,
                         :] = predicts.reshape((batch_size, -1))

                # pret_tmp[s:s+batch_size,
                #          :] = outputs.cpu().detach().numpy().reshape((batch_size, -1))
                grod_tmp[s:s+batch_size,
                         :] = labels.cpu().detach().numpy().reshape((batch_size, -1))
                s += batch_size

                total_time += float(end-start)
                test_loss += float(loss)

                # result_ins = evaluate(predicts, labels, result_ins)

            # pret_tmp[pret_tmp >= 0.05] = 1
            # pret_tmp[pret_tmp < 0.05] = 0
            result = evaluate(pret_tmp, grod_tmp)

            avg_time = total_time/float(len(test_load))
            print("Average Time: {:2.3f} ms".format(1000*avg_time))
            print("Test Loss: {:.6f}".format(out_num*test_loss/len(test_load)))
            total_acc = 100*float(result['a'][-1])/len(test_load)
            
            state = {
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
            }
            torch.save(state, server_model_path+"/epoch_"+str(epoch)+".pth")
            saved_model = True
            print("Model saved")

        if total_acc > best_acc:
            best_acc = total_acc
            best_epoch = epoch+1
            state = {
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
            }
            torch.save(state, server_model_path+"/epoch_best.pth")
            saved_model = True
            print("Model saved")

    print("Best test accuracy: {} at epoch: {}".format(best_acc, best_epoch))
    if not saved_model:
        state = {
            'epoch': epoch+1,
            'optimizer': optimizer.state_dict(),
            'state_dict': model.state_dict(),
        }
        torch.save(state, server_model_path+"/epoch_best.pth")
        print("Model saved")

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
    # label_neg_acc = 1.0*pt_neg/gt_neg
    # label_acc = (label_pos_acc + label_neg_acc)/2

    result['label_pos_acc'] = label_pos_acc
    # result['label_neg_acc'] = label_neg_acc
    # result['label_acc'] = label_acc
    print("label_pos_acc: ", label_pos_acc)
    # print("label_neg_acc: ", label_neg_acc)
    # print("label_acc: ", label_acc)

    inx = json.load(open('class.index', 'r'))
    acc_num = {}
    stat_result = {}
    for i in instruments:
        acc_num[i] = 0

    for i in inx.keys():
        ins_type = i.split('-')[0]
        id = inx[i]
        acc_num[ins_type] += pt_pos[id]

    for i in instruments:
        print("{}: {}/{} = {:.3f}% ".format(i,
                                            acc_num[i], db[i], 100.0*acc_num[i]/db[i]))
    # compute the instance-based accuracy
    # precision
    gt_pos = np.sum((grot == 1).astype(float), axis=1)
    pt_pos = np.sum((pret == 1).astype(float), axis=1)
    floatersect_pos = np.sum((grot == 1).astype(
        float)*(pret == 1).astype(float), axis=1)

    union_pos = np.sum(((grot == 1)+(pret == 1)).astype(float), axis=1)
    cnt_eff = float(grot.shape[0])
    for iter, key in enumerate(gt_pos):
        if key == 0:
            union_pos[iter] = 1
            pt_pos[iter] = 1
            gt_pos[iter] = 1
            cnt_eff = cnt_eff - 1
            continue
        if pt_pos[iter] == 0:
            pt_pos[iter] = 1

    instance_acc = np.sum(floatersect_pos/union_pos)/cnt_eff
    instance_precision = np.sum(floatersect_pos/pt_pos)/cnt_eff
    instance_recall = np.sum(floatersect_pos/gt_pos)/cnt_eff
    floatance_F1 = 2*instance_precision*instance_recall / \
        max(1,instance_precision+instance_recall)
    result['instance_acc'] = instance_acc
    result['instance_precision'] = instance_precision
    result['instance_recall'] = instance_recall
    result['instance_F1'] = floatance_F1
    print("instance_acc: ", instance_acc)
    print("instance_precision: ", instance_precision)
    print("instance_recall: ", instance_recall)
    print("instance_F1: ", floatance_F1)

    result['a'] = N*[0]
    for num in floatersect_pos:
        if num > 0:
            result['a'][int(num)-1] += 1

    for i in range(N):
        print("Correct: {}/{}, Accuracy: {:.3f}% ".format(i +
                                                          1, N, 100*float(result['a'][i])/total_num))

    # inx = json.load(open('class.index', 'r'))
    # print("+++++++++++++++")
    # d_g = decode(grot)
    # for g in d_g:
    #     if g.split('-')[0] in ['Va', 'Vn', 'Vc']:
    #         d_p = decode(pret)
    #         print("g: ", d_g)
    #         print("p: ", d_p)
    #         break

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


if __name__ == "__main__":
    main()
