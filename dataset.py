from torch.utils import data
import torchvision.transforms as transforms
import torch
import numpy as np
import pickle
import librosa
import os


class OrchDataSet(data.Dataset):
    def __init__(self, root, transform):
        if not os.path.exists(root):
            print("[Error] root not exits")
            return

        self.audio_feature = []
        self.labels = []

        if root.split('.')[-1] == 'pkl':
            inp = pickle.load(open(root, 'rb'))
            for x in inp:
                self.audio_feature.append(x[0])
                self.labels.append(x[1])

                # print(x[0].shape)

        if transform is not None:
            self.transform = transform

    def __len__(self):
        return len(self.audio_feature)

    def __getitem__(self, idx):
        audio = self.audio_feature[idx]

        audio = np.array(audio)
        audio = torch.tensor(audio)
        # print(audio.shape)
        # if self.transform is not None:
        #     audio = self.transform(audio)

        return audio, self.labels[idx]


if __name__ == '__main__':
    # root = './TinySOL/Combine'
    root = './data/trainset.pkl'
    a = OrchDataSet(root, transforms.ToTensor())
    aa = torch.utils.data.DataLoader(dataset=a, batch_size=1, shuffle=True)

    for (train, labels) in aa:
        if train.shape[1] != 2:
            print(train.shape)
            print(train)
