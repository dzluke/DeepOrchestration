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


path = './new_OrchDB_ord'
ins = ['Va', 'Cb', 'Vns', 'Vc', 'BTb', 'Fl', 'Vn', 'Hn', 'BTbn', 'BClBb', 'ClBb', 'TpC', 'TTbn',
       'BFl', 'CbTb', 'Picc', 'ClEb', 'Acc', 'Ob', 'EH', 'CbFl', 'Bn', 'Vas', 'Vcs', 'ASax', 'CbClBb']
N = 5
time = 4
MAX_NUM = 250
out_num = 3674
my_data_path = './data/five/'


def random_combine():
    all_class = []
    all_mixture = []
    inx = json.load(open('class.index', 'r'))

    for inst in os.listdir(path):
        if inst.startswith('.'):
            continue
        newpath = os.path.join(path, inst)
        all_class.append(newpath)

    # combine
    all_selects = []
    all_mix = []
    init = 0
    while init < MAX_NUM:
        # select N files randomly
        selects = random.sample(range(len(all_class)), N)
        flag = False
        soundlist = []
        labellist = []

        for num in selects:
            f = all_class[num].split('.')[1].split('/')[-1]
            if f.endswith('c'):
                idx = inx[f[:-3]]
            else:
                idx = inx[f]

            if idx in labellist:
                flag = True
                break

            labellist.append(str(idx))
            soundlist.append(all_class[num])

        if flag or set(labellist) in all_selects:
            continue

        mixed = combine(soundlist, labellist)
        mix = deal_mix(mixed)

        all_selects.append(set(labellist))
        all_mix.append(mix)

        init += 1
        if init % 100 == 0:
            print(
                "{} / {} have finished".format(init, MAX_NUM))

        # distributed data
        # if init % 50000 == 0:
        #     # save in disk
        #     num = int(init/50000)
        #     division = int(0.8*len(all_mix))
        #     pickle.dump(all_mix[:division], open(
        #         './data/five/trainset-'+str(num)+'.pkl', 'wb'))
        #     pickle.dump(all_mix[division:], open(
        #         './data/five/testset-'+str(num)+'.pkl', 'wb'))
        #     all_mix = []
        #     print("store "+str(num))
    pickle.dump(all_mix, open(
        './data/five/set.pkl', 'wb'))

# combine(N)


def combine(soundlist, labellist):
    mixed_file = np.zeros((1, 1))
    sr = 0
    for sound in soundlist:
        sfile, sr = librosa.load(sound, sr=None)
        if len(sfile) > time*sr:
            n = np.random.randint(0, len(sfile)-time*sr)
            sfile = sfile[n:n+time*sr]
        mixed_file = mix(mixed_file, sfile)
    mixed_file = mixed_file/len(soundlist)
    # mixed_file = mixed_file[:time*sr]

    mixed_label = ''
    for label in labellist:
        mixed_label = mixed_label+label+'-'

    # name = '/home/data/happipub/gradpro_l/Combine/' + mixed_label
    # librosa.output.write_wav(name, y=mixed_file, sr=sr)
    return [mixed_file, sr, mixed_label]


def mix(fa, fb):
    diff = len(fa) - len(fb)

    if diff > 0:
        add = np.zeros((1, diff), dtype=np.float32)
        fb = np.append(fb, add)
    elif diff < 0:
        add = np.zeros((1, -diff), dtype=np.float32)
        fa = np.append(fa, add)

    return fa+fb


def deal_mix(mix):
    y = mix[0]
    sr = mix[1]
    label = mix[2]

    feature = librosa.feature.melspectrogram(y=y, sr=sr).T

    # if feature.shape[0] <= 256:
    #     # add zero
    #     zero = np.zeros((256-feature.shape[0], 128), dtype=np.float32)
    #     feature = np.vstack((feature, zero))
    # else:
    #     feature = feature[:-1*(feature.shape[0] % 128)]

    # num_chunk = feature.shape[0]/128
    # feature = np.split(feature, num_chunk)

    # (1, 128, 128)
    feature = np.split(feature, 1)
    feature = torch.tensor(feature)

    label = label.split('-')[:-1]

    label = encode(label)

    return [feature, label]


def show_all_class_num():
    cmt = 0
    m = []
    inx = {}
    for f in os.listdir(path):
        if f.startswith("."):
            continue

        if f.split('.')[0].endswith('c'):
            n = f.split('.')[0].split('-')[:-1]
            if n not in m:
                m.append(n)
                inx[f.split('.')[0][:-3]] = cmt
                cmt += 1

        else:
            m.append(f.split('.')[0].split('-'))
            inx[f.split('.')[0]] = cmt
            cmt += 1

    f = open('class.index', 'w')

    json.dump(inx, f)
    print(cmt)

    return cmt


def remove():
    c = 0
    for f in os.listdir(path):
        if f.startswith("."):
            continue
        y, sr = librosa.load(path+'/'+f, sr=None)
        if len(y) < sr*time:
            print("!")


def crop_data(mode):
    root = '/home/data/happipub/gradpro_l/five'

    mix = []
    for data in os.listdir(root):
        if data.startswith(mode):
            new_path = os.path.join(root, data)
            print(new_path)
            inp = pickle.load(open(new_path, 'rb'))
            for i, x in enumerate(inp):
                x[0] = torch.tensor(
                    [np.vstack((x[0][0].numpy(), x[0][1].numpy()))])
                mix.append(x)

            pickle.dump(mix, open(new_path+'new.pkl', 'wb'))
            mix = []


def show_all_instru_num():
    ins = []
    ins_dic = {}
    for i in ins:
        ins_dic[i] = 0

    for f in os.listdir(path):
        if f.startswith('.'):
            continue
        inst = f.split('-')[0]
        # if '+' in inst:
        #     continue
        if inst not in ins:
            ins.append(inst)
        # ins_dic[inst] += 1
    print(ins)
    print(len(ins))
    # print(ins_dic)
    # print(len(os.listdir(path)))

    return ins


def stat_test_db():
    inx = json.load(open('class.index', 'r'))
    t = np.array(out_num*[0], dtype=np.float32)

    stat_result = {}
    for key in ins:
        stat_result[key] = 0

    testset = OrchDataSet(
        my_data_path, 'test', transforms.ToTensor())
    test_load = torch.utils.data.DataLoader(dataset=testset,
                                            batch_size=1,
                                            shuffle=False)

    for _, labels in test_load:
        labels = labels.cpu().numpy()
        p = (labels == 1).astype(float)
        t += p[0]

    for i in inx.keys():
        ins_type = i.split('-')[0]
        id = inx[i]
        stat_result[ins_type] += t[id]

    print(stat_result)
    return stat_result


def encode(labels):
    encode_label = np.array(out_num*[0], dtype=np.float32)

    for index in labels:
        encode_label[int(index)] = float(1)

    encode_label = torch.tensor(encode_label)
    return encode_label


def decode(labels, f=0):
    decode_label = []
    labels_copy = copy.deepcopy(labels)
    inx = json.load(open('class.index', 'r'))

    for i in range(len(labels_copy)):
        one = list(labels_copy[i]).index(1)
        labels_copy[i][one] = 0
        one = list(inx.keys())[list(inx.values()).index(one)]

        two = list(labels_copy[i]).index(1)
        labels_copy[i][two] = 0
        two = list(inx.keys())[list(inx.values()).index(two)]

        # three = list(labels_copy[i]).index(1)
        # labels_copy[i][three] = 0
        # three = list(inx.keys())[list(inx.values()).index(three)]

        if f == 1:
            four = list(labels_copy[i]).index(1)
            four = list(inx.keys())[list(inx.values()).index(four)]
            decode_label.append([one, two, three, four])
        else:
            decode_label.append([one, two])

    return decode_label[0]


def loss(s):
    root = './exp/after1-13'
    f = open(root+'/myout-resnet-aug-sig.txt', 'r')
    loss_log = []
    test_log = []
    lines = f.readlines()
    loss = 0
    cnt = 0
    for line in lines:
        if line.startswith('Epoch:'):
            l = line.split(':')[-1]
            loss += float(l)
            cnt += 1
            if cnt % 625 == 0:
                loss_log.append(loss/625)
                loss = 0
        if line.startswith('Test'):
            l = line.split(':')[-1]
            test_log.append(float(l))

    return loss_log, test_log


def draw_loss_figure():
    # loss_two = loss('two')
    # loss_three = loss('three')
    loss_five, test_five = loss('five')
    # loss_ten = loss('ten')

    epoch_num = range(0, 5*len(loss_five[:10]), 5)

    plt.figure()
    # plt.plot(epoch_num, loss_two[:30], color='r', label='two')
    # plt.plot(epoch_num, loss_three[:30], color='g', label='three')
    plt.plot(epoch_num, loss_five[:10], color='b', label='train')
    plt.plot(epoch_num, test_five[:10], color='r', label='test')
    # plt.plot(epoch_num, loss_ten[:30], color='y', label='ten')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig("./exp/after1-13/loss_resnet.png")
    plt.show()


if __name__ == "__main__":
    draw_loss_figure()
