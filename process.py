##################################################
# Process the data of orchestration combinations
# and make preparation for training, the steps include:
#   1. combine N instruments with various pitches
#   2. extract the melspectrugram features of these combinations as input while training
#   3. save the corresponding label(index)
#   4. make the train and test dataset

import librosa
from pydub import AudioSegment
import numpy as np
import torch
from torchvision.transforms import transforms
import os
import pickle
import matplotlib.pyplot as plt
import random
import json

from dataset import OrchDataSet


time = 4
path = './TinySOL'
brass = '/Brass'
strings = '/Strings'
winds = '/Winds'
instru_type = [brass, strings, winds]
N = 2
MAX_NUM = 20000


def random_combine():
    all_instrument = []
    all_mixture = []

    for instruments in instru_type:
        for instrument in os.listdir(path+instruments):
            if instrument.startswith('.'):
                continue
            newpath = os.path.join(path+instruments, instrument)
            all_instrument.append(newpath)

    # combine
    all_selects = []
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

        if labellist in all_selects:
            continue

        combine(soundlist, labellist)
        all_selects.append(labellist)
        init += 1


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
    name = path+'/Combine/' + label

    librosa.output.write_wav(name, y=output, sr=sr1)


def extract_feature(file):
    y, sr = librosa.load(file, duration=time, sr=None)
    mel_feature = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=1024, hop_length=512)

    return mel_feature


# def get_class_num():
#     return len(os.listdir(path+brass+'/BTb'))+len(os.listdir(path+strings+'/Vn'))


def show_all_class_num():
    division = {}
    start = 0
    end = 0
    for instruments in instru_type:
        for instrument in os.listdir(path+instruments):
            if instrument.startswith('.'):
                continue
            newpath = os.path.join(path+instruments, instrument)
            length = len(os.listdir(newpath))
            # print(newpath, len(os.listdir(newpath)))
            end = start + length - 1
            division[instrument] = [start, end]
            start = end + 1

    return end+1, division


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

    return class_div


def stat_test_db():
    _, class_div = show_all_class_num()
    stat_result = {}
    for key in class_div.keys():
        stat_result[key] = 0

    testset = OrchDataSet('./data/testset.pkl', transforms.ToTensor())
    test_load = torch.utils.data.DataLoader(dataset=testset,
                                            batch_size=1,
                                            shuffle=False)

    for _, labels in test_load:
        labels = decode(labels)
        for label in labels:
            for key in class_div.keys():
                if label[0] >= class_div[key][0] and label[0] <= class_div[key][1]:
                    stat_result[key] += 1
                if label[1] >= class_div[key][0] and label[1] <= class_div[key][1]:
                    stat_result[key] += 1

    print(stat_result)
    return stat_result


def encode(labels):
    class_num, class_div = show_all_class_num()

    encode_label = np.array(class_num*[-1], dtype=np.float32)

    for label in labels:
        type, index = label.split('!')
        start = class_div[type][0]
        encode_label[start+int(index)] = float(1)

    encode_label = torch.tensor(encode_label)
    return encode_label


def decode(labels):
    decode_label = []

    for i in range(len(labels)):
        one = list(labels[i]).index(1)
        labels[i][one] = -1
        two = list(labels[i]).index(1)
        decode_label.append([one, two])

    return torch.tensor(decode_label)


def make_dataset():
    # (feature, label)
    root = './TinySOL/Combine'
    audio_dirs = [os.path.join(root, x) for x in os.listdir(root)]
    random.shuffle(audio_dirs)
    audio_feature = []
    sets = []

    for x in audio_dirs:
        if x.endswith('.DS_Store'):
            continue
        y, sr = librosa.load(x, duration=time, sr=None)
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

        # (2, 128, 128)
        feature = torch.tensor(feature)

        audio_feature.append(feature)

        if len(audio_feature) % 100 == 0:
            print(
                "{} / {} have finished".format(len(audio_feature), len(audio_dirs)))

        label = x.split('.')[1].split('/')[-1].split('-')
        label = encode(label)
        label = torch.Tensor(label)

        sets.append([feature, label])

    # save in disk
    division = int(0.8*len(sets))
    pickle.dump(sets[:division],
                open('./data/trainset.pkl', 'wb'))
    pickle.dump(sets[division:],
                open('./data/testset.pkl', 'wb'))


def draw_total():
    f = open('acc.csv', 'r')
    epoch_num = []
    total_acc = []
    single_acc = []

    for line in f.readlines():
        acc = line.split(',')
        epoch_num.append(acc[0])
        total_acc.append(round(float(acc[1]), 2))
        single_acc.append(round(float(acc[2]), 2))

    plt.figure()
    plt.plot(epoch_num, total_acc, color='b')
    plt.plot(epoch_num, single_acc, color='r')

    plt.xlabel('epoch')
    plt.ylabel('b:total_acc r:single_acc')
    plt.savefig('acc.png')
    plt.show()


def draw_specific():
    f = open('specific_acc.json', 'r')
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


if __name__ == "__main__":
    draw_specific()
