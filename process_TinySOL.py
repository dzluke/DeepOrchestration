##################################################
# Process the data of orchestration combinations
# and make preparation for training, the steps include:
#   1. combine N instruments with various pitches
#   2. extract the melspectrugram features of these combinations as input while training
#   3. save the corresponding label(index)
#   4. make the train and test dataset

import librosa
import numpy as np
import torch
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import random
import json
import copy
import pickle
import os

from dataset import OrchDataSet


data_path = './data/'
brass = '/Brass'
strings = '/Strings'
winds = '/Winds'
instru_type = [brass, strings, winds]
instruments = ['BTb', 'TpC', 'Hn', 'Tbn', 'Va', 'Vn',
       'Vc', 'Cb', 'Ob', 'Fl', 'Bn', 'ClBb']
N = 2
time = 4
MAX_NUM = 200
featurized_data_path = './featurized_data/'

"""
Creates .wav files of combinations of instruments
"""
def random_combine():
    all_instrument = []
    all_mixture = []

    for instruments in instru_type:
        for instrument in os.listdir(data_path + instruments):
            if instrument.startswith('.'):
                continue
            newpath = os.path.join(data_path + instruments, instrument)
            all_instrument.append(newpath)

    # combine
    all_selects = []
    all_mix = []
    init = 0
    while init < MAX_NUM:
        # select N instruments randomly
        selects = random.sample(range(len(all_instrument)), N)

        soundlist = []
        labellist = []

        for instrument in selects:
            instr = all_instrument[instrument]
            instr_file_list = os.listdir(instr)

            # select a file from the instrument
            num = random.randint(0, len(instr_file_list)-1)
            soundlist.append(os.path.join(instr, instr_file_list[num]))
            labellist.append(instr.split('/')[-1]+'!'+str(num))

        if set(labellist) in all_selects:
            continue

        combine(soundlist, labellist)
        # mix = deal_mix(mix)

        all_selects.append(set(labellist))
        # all_mix.append(mix)

        init += 1
        if init % 100 == 0:
            print(
                "{} / {} have finished".format(init, MAX_NUM))

    # save in disk
    # division = int(0.8*len(all_mix))
    # pickle.dump(all_mix[:division], open(
    #     '/home/data/happipub/gradpro_l/trainset.pkl', 'wb'))
    # pickle.dump(all_mix[division:], open(
    #     '/home/data/happipub/gradpro_l/testset.pkl', 'wb'))


# combine instruments(N)
def combine(soundlist, labellist):
    brass_file, sr1 = librosa.load(soundlist[0], sr=None)
    brass_file = np.array(brass_file)
    strings_file, sr2 = librosa.load(soundlist[1], sr=None)
    strings_file = np.array(strings_file)

    diff = len(brass_file) - len(strings_file)

    if diff >= 0:
        add = np.zeros((1, diff), dtype=np.float32)
        strings_file = np.append(strings_file, add)
    else:
        add = np.zeros((1, -diff), dtype=np.float32)
        brass_file = np.append(brass_file, add)

    output = (brass_file+strings_file)/2
    label = labellist[0]+'-'+labellist[1] + '.wav'

    name = data_path + label
    # print("about to create {} in {}".format(label, server_data_path))
    librosa.output.write_wav(name, y=output, sr=sr1)
    # return [output, sr1, label]


def deal_mix(mix):
    y = mix[0]
    sr = mix[1]
    label = mix[2]
    librosa.output.write_wav('tmp.wav', y=y, sr=sr)
    y1, sr1 = librosa.load('tmp.wav', duration=time, sr=None)
    feature = librosa.feature.melspectrogram(y=y1, sr=sr1).T

    if feature.shape[0] <= 256:
        # add zero
        zero = np.zeros((256-feature.shape[0], 128), dtype=np.float32)
        feature = np.vstack((feature, zero))
    else:
        feature = feature[:-1*(feature.shape[0] % 128)]

    num_chunk = feature.shape[0]/128
    feature = np.split(feature, num_chunk)

    # (2, 128, 128)
    feature = torch.tensor(feature)

    label = label.split('.')[0].split('-')
    label = encode(label)

    return [feature, label]


def show_all_class_num():
    division = {}
    start = 0
    end = 0
    for instruments in instru_type:
        for instrument in os.listdir(data_path + instruments):
            if instrument.startswith('.'):
                continue
            newpath = os.path.join(data_path + instruments, instrument)
            length = len(os.listdir(newpath))
            # print(newpath, len(os.listdir(newpath)))
            end = start + length - 1
            division[instrument] = [start, end]
            start = end + 1

    return end+1, division


def show_all_instru_num():
    ins = []
    for instruments in instru_type:
        for instrument in os.listdir(data_path + instruments):
            if instrument.startswith('.'):
                continue
            ins.append(instrument)

    return ins


def stat_all_db():
    root = './TinySOL/Combine'
    _, class_div = show_all_class_num()
    for key in class_div.keys():
        class_div[key] = 0

    for files in os.listdir(root):
        if files.endswith('.DS_Store'):
            continue
        lists = files.split('.')[0].split('-')
        for l in lists:
            key = l.split('!')[0]
            class_div[key] += 1
    print(class_div)
    return class_div


def stat_test_db():
    _, class_div = show_all_class_num()
    stat_result = {}
    for key in class_div.keys():
        stat_result[key] = 0

    testset = OrchDataSet(
        featurized_data_path + 'testset_mini.pkl', transforms.ToTensor())
    test_load = torch.utils.data.DataLoader(dataset=testset,
                                            batch_size=1,
                                            shuffle=False)

    for _, labels in test_load:
        labels = decode(labels)
        for label in labels:
            stat_result[instruments[label[0]]] += 1
            stat_result[instruments[label[1]]] += 1
            # for key in class_div.keys():
            #     if label[0] >= class_div[key][0] and label[0] <= class_div[key][1]:
            #         stat_result[key] += 1
            #     if label[1] >= class_div[key][0] and label[1] <= class_div[key][1]:
            #         stat_result[key] += 1

    print(stat_result)
    return stat_result


def encode(labels):
    class_num, class_div = show_all_class_num()
    # class_num = len(ins)

    encode_label = np.array(class_num*[0], dtype=np.float32)

    for label in labels:
        type, index = label.split('!')
        # encode_label[ins.index(type)] = float(1)
        start = class_div[type][0]
        encode_label[start+int(index)] = float(1)

    encode_label = torch.tensor(encode_label)
    return encode_label


def decode(labels):
    decode_label = []
    labels_copy = copy.deepcopy(labels)
    for i in range(len(labels_copy)):
        one = list(labels_copy[i]).index(1)
        labels_copy[i][one] = 0
        two = list(labels_copy[i]).index(1)
        decode_label.append([one, two])

    return torch.tensor(decode_label)

"""
Featurizes .wav files and stores as picklized data
"""
def make_dataset():
    # (feature, label)

    print("Start to make dataset -- ")
    # root = my_data_path+'Combine'
    # audio_dirs = [os.path.join(root, x) for x in os.listdir(root)]
    # audio_dirs = []
    # for instruments in instru_type:
    #     for instrument in os.listdir(path+instruments):
    #         if instrument.startswith('.'):
    #             continue
    #         for i in os.listdir(path+instruments+'/'+instrument):
    #             newpath = os.path.join(path+instruments+'/'+instrument, i)
    #             audio_dirs.append(newpath)
    # random.shuffle(audio_dirs)
    # audio_feature = []
    audio_dirs = [data_path + "training", data_path + "test"]
    sets = []

    for data_type in ["training", "test"]:
        directory = data_path + data_type
        for file in os.listdir(directory):
            if file.endswith('.DS_Store'):
                continue

            path_to_file = os.path.join(directory, file)
            y, sr = librosa.load(path_to_file, duration=time, sr=None)
            # (128, len)
            feature = librosa.feature.melspectrogram(y, sr).T
            # (len, 128)

            if feature.shape[0] <= 256:
                # add zero
                zero = np.zeros((256-feature.shape[0], 128), dtype=np.float32)
                feature = np.vstack((feature, zero))
            else:
                feature = feature[:-1*(feature.shape[0] % 128)]

            num_chunk = feature.shape[0]/128
            feature = np.split(feature, num_chunk)

            # if feature.shape[0] <= 128:
            #     # add zero
            #     zero = np.zeros((128-feature.shape[0], 128), dtype=np.float32)
            #     feature = np.vstack((feature, zero))
            # else:
            #     feature = feature[:128]

            # feature = np.split(feature, 1)

            # (2, 128, 128)
            feature = torch.tensor(feature)

            if len(sets) % 100 == 0:
                print(
                    "{} have finished".format(len(sets)))
            label = torch.zeros(len(instruments), dtype=torch.float32)
            instruments_used = file.split('-')
            for instrument in instruments_used:
                instrument = instrument.split('!')[0]
                index = instruments.index(instrument)
                label[index] = 1.0
            # label = encode(label)

            sets.append([feature, label])

        # save in disk
        division = int(0.8*len(sets))
        pickle.dump(sets[:division],
                    open(featurized_data_path + '{}_featurized_data.pkl'.format(data_type), 'wb'))


def draw_total():
    f = open('./all_cnn_500/acc.csv', 'r')
    epoch_num = []
    total_acc_cnn = []
    total_acc_res = []
    # single_acc = []

    for line in f.readlines()[:4]:
        acc = line.split(',')
        epoch_num.append(acc[0])
        total_acc_cnn.append(round(float(acc[1]), 2))
        # single_acc.append(round(float(acc[2]), 2))
    f.close()

    f = open('./acc.csv', 'r')
    for line in f.readlines():
        acc = line.split(',')
        # epoch_num.append(acc[0])
        total_acc_res.append(round(float(acc[1]), 2))
    f.close()

    plt.figure()
    plt.plot(epoch_num, total_acc_cnn, color='b')
    # plt.plot(epoch_num, single_acc, color='r')
    plt.plot(epoch_num, total_acc_res, color='r')

    plt.xlabel('epoch')
    plt.ylabel('b:total_acc_cnn r:total_acc_res')
    plt.savefig('acc_compare.png')
    plt.show()


def draw_specific():
    f = open('./all_cnn_500/specific_acc.json', 'r')
    specific_acc = json.load(f)

    x = []
    y = []

    for key in specific_acc.keys():
        x.append(key)
        y.append(specific_acc[key])

    plt.figure()
    plt.plot(x, y, color='r')

    plt.xlabel("type")
    plt.ylabel("acc")
    plt.savefig("specific_acc.png")
    plt.show()


def proc_out():
    root = './lstm-300/slstm/'
    f = open(root+'myout.txt', 'r')

    total_acc = []
    loss_log = []
    lines = f.readlines()
    loss = 0
    best = 0
    cnt = 0
    for line in lines:
        if line.startswith('Epoch:'):
            l = line.split(':')[-1]
            loss += float(l)
            cnt += 1
            if cnt % 500 == 0:
                loss_log.append(loss/500)
                loss = 0
        elif line.startswith('Total'):
            a = line.split('%')[0].split(' ')[-1]
            total_acc.append(float(a))
            if float(a) > best:
                best = float(a)

    epoch_num1 = range(0, 10*len(loss_log), 10)
    epoch_num2 = range(10, 10*len(total_acc)+10, 10)

    plt.figure()
    plt.plot(epoch_num1, loss_log, color='b')
    plt.ylabel('loss')
    plt.savefig(root+"lstm_loss.png")
    plt.show()

    plt.figure()
    plt.plot(epoch_num2, total_acc, color='r')
    plt.ylabel('total_accuracy')
    plt.savefig(root+"lstm_acc.png")
    plt.show()
    print("best: ", best)


if __name__ == "__main__":
    make_dataset()