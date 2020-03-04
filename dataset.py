from torch.utils import data
import torchvision.transforms as transforms
import torch
import numpy as np
import pickle
import librosa
import os
from augment import spec_augment


class OrchDataSet(data.Dataset):
    def __init__(self, root, mode, transform):
        if not os.path.exists(root):
            print("[Error] root does not exist")
            return
        self.mode = mode
        self.audio_feature = []
        self.labels = []
        self.mix = []

        for dir in os.listdir(root):
            if dir.startswith(mode):
                new_path = os.path.join(root, dir)
                f = open(new_path, 'rb')
                inp = pickle.load(f)
                for i, x in enumerate(inp):
                    self.audio_feature.append(x[0])
                    self.labels.append(x[1])

        if transform is not None:
            self.transform = transform

    def __len__(self):
        return len(self.audio_feature)

    def __getitem__(self, idx):
        audio = self.audio_feature[idx]

        if self.mode.startswith('train'):
            a = np.array(audio[0])
            a = spec_augment(a)
            audio = np.array([a])

            audio = torch.tensor(audio)

        return audio, self.labels[idx]


if __name__ == '__main__':
    root = './featurized_data'
    training = OrchDataSet(root, 'training', transforms.ToTensor())
    testing = OrchDataSet(root, 'test', transforms.ToTensor())
    print("training set size: {}".format(len(training)))
    print("testing set size: {}".format(len(testing)))

    print("training set sample feature shape: {}".format(training[0][0].shape))
    # print("testing set  feature shape: {}".format(testing[0].shape))

