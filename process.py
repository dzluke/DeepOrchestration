##################################################
# Process the data of orchestration combinations
# and make preparation for training, the steps include:
#   1. combine N instruments with various pitches
#   2. extract the melspectrugram features of these combinations as input while training
#   3. save the corresponding label(index)
#   4. make the train and test dataset

import librosa
from pydub import AudioSegment
import os
import pickle
from dataset import OrchDataSet

time = 3
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


def make_dataset(type):
    # (feature, label)
    list_combination = os.listdir(path+'/Combine')
    print(len(list_combination))
    features = []
    labels = []

    for combination in list_combination:
        feature = extract_feature(path+'/Combine/'+combination)
        label = combination.split('.')[0].split('-')

        features.append(feature)
        labels.append(label)

    division = int(0.8*len(features))
    if type == 'train':
        return features[:division], labels[:division]
    elif type == 'test':
        return features[division:-1], labels[:division:-1]


if __name__ == "__main__":
    a, b = make_dataset('test')
    print(len(a), len(b))
