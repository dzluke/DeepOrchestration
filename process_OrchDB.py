import numpy as np
import torch
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import random
import json
import copy
import librosa
import pickle
import os

from dataset import OrchDataSet
from augment import wav_augment

# original dataset
path = './TinySOL_0.6/TinySOL'
# generated dataset
my_data_path = './data/'

instruments = ['Vc', 'Fl', 'Va', 'Vn', 'Ob', 'BTb',
       'Cb', 'ClBb', 'Hn', 'TpC', 'Bn', 'Tbn']

# number of mixture
N = 2
# time duration
time = 4
# max sample number
MAX_NUM = 400
# class number
out_num = 505


def random_combine():
    '''
        call this function to generate dataset and devide it into training set and test set
    '''
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
            f = all_class[num].split('.')[1].split('/')[-1].split('-')[:3]
            f = f[0]+'-'+f[1]+'-'+f[2]
            idx = inx[f]

            if str(idx) in labellist:
                # ignore the difference of dynamics
                flag = True

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
            print("{} / {} have finished".format(init, MAX_NUM))

    if os.path.exists(my_data_path) == False:
        os.makedirs(my_data_path)

    division = int(0.8*len(all_mix))
    pickle.dump(all_mix[:division], open(
        my_data_path+'trainset.pkl', 'wb'))
    pickle.dump(all_mix[division:], open(
        my_data_path+'testset.pkl', 'wb'))

    # divide the data into sevaral parts when its size is large
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


def combine(soundlist, labellist):
    mixed_file = np.zeros((1, 1))
    sr = 0
    for sound in soundlist:
        sfile, sr = librosa.load(sound, sr=None)
        if len(sfile) > time*sr:
            # randomly select one part of the raw audio
            n = np.random.randint(0, len(sfile)-time*sr)
            sfile = sfile[n:n+time*sr]
        # add augment
        # sfile = wav_augment(sfile, sr)
        mixed_file = mix(mixed_file, sfile)
    mixed_file = mixed_file/len(soundlist)

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

    feature = librosa.feature.melspectrogram(y=y, sr=sr)

    # (1, 128, 345)
    feature = np.split(feature, 1)
    feature = torch.tensor(feature)

    label = label.split('-')[:-1]
    label = encode(label)

    return [feature, label]


def show_all_class_num():
    '''
        get all index and class num from the dataset
    '''
    cmt = 0
    m = []
    inx = {}
    for f in os.listdir(path):
        if f.startswith("."):
            continue

        n = f.split('.')[0].split('-')[:3]
        print(n)
        if n not in m:
            m.append(n)
            i = n[0]+'-'+n[1]+'-'+n[2]
            inx[i] = cmt
            cmt += 1

        y, sr = librosa.load(path+f, sr=None)
        if len(y) < time*sr:
            add = np.zeros((1, time*sr-len(y)), dtype=np.float32)
            y = np.append(y, add)
            librosa.output.write_wav(path+f, y, sr)

    f = open('class.index', 'w')

    json.dump(inx, f)
    print('class num: ', cmt)

    return cmt


# def remove():
#     c = 0
#     for f in os.listdir(path):
#         if f.startswith("."):
#             continue
#         y, sr = librosa.load(path+f, sr=None)
#         if len(y) < sr*time:
#             add = np.zeros((1, sr*time-len(y)))
#             y = np.append(y, add)


# def crop_data(mode):
#     root = '/home/data/happipub/gradpro_l/five'

#     mix = []
#     for data in os.listdir(root):
#         if data.startswith(mode):
#             new_path = os.path.join(root, data)
#             print(new_path)
#             inp = pickle.load(open(new_path, 'rb'))
#             for i, x in enumerate(inp):
#                 x[0] = torch.tensor(
#                     [np.vstack((x[0][0].numpy(), x[0][1].numpy()))])
#                 mix.append(x)

#             pickle.dump(mix, open(new_path+'new.pkl', 'wb'))
#             mix = []


def show_all_instru_num():
    ins = []
    ins_dic = {}
    for i in ins:
        ins_dic[i] = 0

    for f in os.listdir(path):
        if f.startswith('.'):
            continue
        inst = f.split('-')[0]
        if inst not in ins:
            ins.append(inst)

    print(ins)
    print(len(ins))

    return ins


def stat_test_db():
    '''
        get the numbers of files for each instrument
        return a dictionary
    '''
    inx = json.load(open('class.index', 'r'))
    t = np.array(out_num*[0], dtype=np.float32)

    stat_result = {}
    for key in instruments:
        stat_result[key] = 0

    testset = OrchDataSet(
        my_data_path, 'testset', transforms.ToTensor())
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
    root = './exp/'
    f = open(root+'/myout-'+s+'.txt', 'r')
    loss_log = []
    test_log = []
    lines = f.readlines()
    loss = 0
    cnt = 0
    k = {"two": 200, "three": 300, "five": 500}
    for line in lines:
        if line.startswith('Epoch:'):
            l = line.split(':')[-1]
            loss += float(l)
            cnt += 1
            if cnt % k[s] == 0:
                loss_log.append(loss/k[s])
                loss = 0
        if line.startswith('Test'):
            l = line.split(':')[-1]
            test_log.append(float(l))

    return loss_log, test_log


def draw_loss_figure():
    loss_two, _ = loss('two')
    loss_three, _ = loss('three')
    loss_five, _ = loss('five')
    # loss_ten = loss('ten')

    epoch_num = range(5, 5*len(loss_five[:20])+5, 5)

    plt.figure()
    plt.plot(epoch_num, loss_two[:20], color='r', label='two')
    plt.plot(epoch_num, loss_three[:20], color='g', label='three')
    plt.plot(epoch_num, loss_five[:20], color='b', label='five')
    # plt.plot(epoch_num, test_five[:20], color='r', label='test')
    # plt.plot(epoch_num, loss_ten[:30], color='y', label='ten')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig("./exp/loss_resnet_100.png")
    plt.show()


def draw_acc_figure():
    root = './exp/'
    f = open(root+'/myout-three.txt', 'r')
    lines = f.readlines()
    acc = []
    precision = []
    recall = []

    for line in lines:
        if line.startswith('instance_acc'):
            l = line.split(':')[-1]
            acc.append(float(l))
        elif line.startswith('instance_precision'):
            l = line.split(':')[-1]
            precision.append(float(l))
        elif line.startswith('instance_recall'):
            l = line.split(':')[-1]
            recall.append(float(l))

    epoch_num = range(5, 5*len(acc[:20])+5, 5)

    plt.figure()

    plt.plot(epoch_num, acc[:20], color='r', label='acc')
    plt.plot(epoch_num, precision[:20], color='b', label='precision')
    # plt.plot(epoch_num, recall[:20], color='g', label='recall')

    plt.ylabel('acc')
    plt.legend()
    plt.savefig("./exp/acc_three_resnet_100.png")
    plt.show()


def draw_acc_comp():
    x = [2, 3, 5]
    y = [88.99, 84.21, 65.98]
    plt.bar(x=range(len(x)), height=y, width=0.4, label='accuracy',
            color='steelblue', tick_label=x, alpha=0.8)

    plt.savefig('./exp/acc_comp.png')
    plt.show()


if __name__ == "__main__":
    show_all_class_num()
