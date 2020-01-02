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
from process_OrchDB import ins, decode, N, out_num
from dataset import OrchDataSet


epoch_num = 200
batch_size = 32
my_model_path = './model'
server_model_path = '/home/data/happipub/gradpro_l/model/five'
my_data_path = './data/five'
server_data_path = '/home/data/happipub/gradpro_l/five'
result = N*[0]
db = {'Va': 25908.0, 'Cb': 25285.0, 'Cbs': 5657.0, 'Vns': 9404.0, 'Vc': 27583.0, 'Fl': 16569.0, 'Vn': 23219.0, 'BTbn': 2942.0, 'BClBb': 4499.0, 'ClBb': 8368.0,
      'BFl': 1767.0, 'CbTb': 3679.0, 'Picc': 2866.0, 'ClEb': 1598.0, 'Ob': 8507.0, 'EH': 4953.0, 'CbFl': 1788.0, 'Bn': 9064.0, 'Vas': 7778.0, 'Vcs': 7272.0, 'CbClBb': 1294.0}


def arg_parse():
    parser = argparse.ArgumentParser(description='combination of orch')
    parser.add_argument('--model', default='cnn')
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
        server_data_path, 'train', transforms.ToTensor())
    testset = OrchDataSet(
        server_data_path, 'test', transforms.ToTensor())

    # load data
    train_load = torch.utils.data.DataLoader(dataset=trainset,
                                             batch_size=batch_size,
                                             shuffle=True)

    test_load = torch.utils.data.DataLoader(dataset=testset,
                                            batch_size=1,
                                            shuffle=False)
    print("End loading data")

    # model construction
    model = OrchMatchNet(out_num, arg.model)

    start_epoch = 0
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, betas=(0.9, 0.999))
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

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

    criterion = F.binary_cross_entropy

    best_acc = 0
    total_acc = 0
    best_epoch = None
    total_loss = 0
    result_ins = np.array(out_num*[0], dtype=np.float32)

    for epoch in range(start_epoch, epoch_num):
        model.train()

        for i, (trains, labels) in enumerate(train_load):
            trains = trains.to(device)
            labels = labels.to(device)

            outputs = model(trains)
            weights = np.ones(labels.shape, dtype=np.float32)
            weights = Variable(torch.tensor(weights)).to(device)

            loss = criterion(outputs, labels/N,
                             weight=weights)*out_num

            total_loss += float(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 50 == 0:
                print('Epoch:[{}/{}], Step:[{}/{}], Loss:{:.4f} '.format(
                    epoch + 1, epoch_num, i+1, len(train_load), total_loss/50))
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

        if (epoch+1) % 10 == 0:
            model.eval()

            total_num = 0
            total_time = 0.0

            for tests, labels in test_load:
                tests = tests.to(device)
                labels = labels.to(device)

                start = time.time()
                outputs = model(tests)
                end = time.time()

                predicts = get_pred(outputs)

                total_num += labels.size(0)
                total_time += float(end-start)

                result_ins = evaluate(predicts, labels, result_ins)

            avg_time = total_time/float(len(test_load))
            print("Average Time: {:2.3f} ms".format(1000*avg_time))
            total_acc = 100*float(result[-1])/total_num

            for i in range(N):
                print("Correct: {}/{}, Accuracy: {:.3f}% ".format(i +
                                                                  1, N, 100*float(result[i])/total_num))
                result[i] = 0

            inx = json.load(open('class.index', 'r'))
            stat_result = {}
            for i in ins:
                stat_result[i] = 0

            for i in inx.keys():
                ins_type = i.split('-')[0]
                id = inx[i]
                stat_result[ins_type] += result_ins[id]
                result_ins[id] = 0

            for i in ins:
                print("{}: {}/{} = {:.3f}% ".format(i,
                                                    stat_result[i], db[i], 100*float(stat_result[i]/db[i])))

        if total_acc > best_acc:
            best_acc = total_acc
            best_epoch = epoch+1
            state = {
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
            }
            torch.save(state, server_model_path+"/epoch_best.pth")

    print("Best test accuracy: {} at epoch: {}".format(best_acc, best_epoch))


def evaluate(pret, grot, result_ins):
    grot = grot.cpu().numpy()
    if pret.shape != grot.shape:
        print("[Error]: size difference")

    acc_num = np.sum((pret == 1).astype(float) *
                     (grot == 1).astype(float), axis=1)

    for num in acc_num:
        if num > 0:
            result[int(num)-1] += 1

    pos_acc = np.sum((pret == 1).astype(float) *
                     (grot == 1).astype(float), axis=0)
    result_ins += pos_acc

    # inx = json.load(open('class.index', 'r'))
    # print("+++++++++++++++")
    # d_g = decode(grot)
    # for g in d_g:
    #     if g.split('-')[0] in ['Va', 'Va+S', 'Va+SP', 'Vn', 'Vn+S', 'Vn+SP', 'Vc', 'Vc+S', 'Vc+SP', 'Cb']:
    #         d_p = decode(pret)
    #         print("g: ", d_g)
    #         print("p: ", d_p)
    #         break

    return result_ins


def get_pred(output):
    '''
        get Top N prediction
    '''

    pred = np.zeros(output.shape)

    preidx = []
    for i in range(N):
        idx = output.max(1, keepdim=True)[1]
        preidx.append(idx)
        output[0][idx] = -1

    for idx in preidx:
        pred[0][idx] = 1.0

    return pred


if __name__ == "__main__":
    main()
