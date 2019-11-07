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


time = 4
path = './TinySOL'
brass = '/Brass/BTb'
strings = '/Strings/Vn'


def random_combine(N=2):
    brass_file_list = os.listdir(path+brass)
    strings_file_list = os.listdir(path+strings)

    # combine
    for brass_num, brass_file in enumerate(brass_file_list):
        for strings_num, strings_file in enumerate(strings_file_list):
            combine([brass_file, strings_file], [
                    str(brass_num), str(strings_num)])


# combine instruments(2)
def combine(soundlist, labellist):
    brass_file = AudioSegment.from_wav(path+brass+'/'+soundlist[0])
    strings_file = AudioSegment.from_wav(path+strings+'/'+soundlist[1])
    output = brass_file.overlay(strings_file)

    label = labellist[0]+'-'+labellist[1] + '.wav'
    name = path+'/Combine/' + label
    output.export(name, format='wav')


def extract_feature(file):
    y, sr = librosa.load(file, duration=time)
    mel_feature = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=1024, hop_length=512)

    return mel_feature


def get_class_num():
    return len(os.listdir(path+brass))+len(os.listdir(path+strings))


# def lable_stat():
#     all_pitch = []

#     if os.path.exists(path+brass):
#         filenames = os.listdir(path+brass)
#         for filename in filenames:
#             name = filename.split('/')[-1].split('.')[0]
#             pitch = name.split('-')[2]
#             if pitch not in all_pitch:
#                 all_pitch.append(pitch)

#     if os.path.exists(path+strings):
#         filenames = os.listdir(path+strings)
#         for filename in filenames:
#             name = filename.split('/')[-1].split('.')[0]
#             pitch = name.split('-')[2]
#             if pitch not in all_pitch:
#                 all_pitch.append(pitch)

#     print(len(all_pitch), all_pitch)


def make_dataset():
    # (feature, label)
    root = './TinySOL/Combine'
    audio_dirs = [os.path.join(root, x) for x in os.listdir(root)]
    audio_path = []
    audio_feature = []
    labels = []
    sets = []

    for x in audio_dirs:
        audio_path.append(x)
        y, sr = librosa.load(x, duration=time)
        # (128, len)
        feature = librosa.feature.melspectrogram(y, sr).T
        # (len, 128)

        if feature.shape[0] <= 128:
            # add zero
            zero = np.zeros((128-feature.shape[0], 128), dtype=np.float32)
            feature = np.vstack((feature, zero))
            feature = np.split(feature, 1)
        else:
            feature = feature[:-1*(feature.shape[0] % 128)]
            num_chunk = feature.shape[0]/128
            feature = np.split(feature, num_chunk)

        # (128, 128)
        feature = torch.tensor(feature)
        audio_feature.append(feature)

        if len(audio_feature) % 100 == 0:
            print(
                "{} / {} have finished".format(len(audio_feature), len(audio_dirs)))

        label = x.split('.')[1].split('/')[-1].split('-')
        label = np.array(label, dtype=int)
        label = torch.Tensor(label)
        labels.append(label)

        sets.append([feature, label])

    # save in disk
    division = int(0.8*len(sets))
    pickle.dump(sets[:division],
                open('./data/trainset.pkl', 'wb'))
    pickle.dump(sets[division:],
                open('./data/testset.pkl', 'wb'))


def crop_data():
    train = pickle.load(open('./data/trainset.pkl', 'rb'))
    train = crop(train)
    pickle.dump(train, open('./data/trainset1.pkl', 'wb'))

    test = pickle.load(open('./data/testset.pkl', 'rb'))
    test = crop(test)
    pickle.dump(test, open('./data/testset1.pkl', 'wb'))


def crop(data):
    min = 10000
    for sets in data:
        feature = sets[0]
        if feature.shape[1] < min:
            min = feature.shape[1]

    print(min)

    for sets in data:
        sets[0] = sets[0][:, :min]
        sets[0] = np.split(sets[0].numpy(), 1)
        sets[0] = torch.Tensor(sets[0])

    return data


def draw():
    f = open('acc.csv', 'r')
    epoch_num = []
    total_acc = []
    first_acc = []
    second_acc = []

    for line in f.readlines():
        acc = line.split(',')
        epoch_num.append(acc[0])
        total_acc.append(round(float(acc[1]), 2))
        first_acc.append(round(float(acc[2]), 2))
        second_acc.append(round(float(acc[3]), 2))

    plt.figure()
    plt.plot(epoch_num, total_acc, color='b')
    plt.plot(epoch_num, first_acc, color='g')
    plt.plot(epoch_num, second_acc, color='r')
    plt.xlabel('epoch')
    plt.ylabel('b:total_acc g:first_acc r:second_acc')
    plt.savefig('acc.png')
    plt.show()


if __name__ == "__main__":
    draw()
