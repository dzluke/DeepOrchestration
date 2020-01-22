from torch.utils import data
import torchvision.transforms as transforms
import torch
import numpy as np
import pickle
import librosa
import os
import json
from augment import spec_augment


class OrchDataSet(data.Dataset):
    def __init__(self, root, mode, transform):
        if not os.path.exists(root):
            print("[Error] root not exits")
            return

        self.audio_feature = []
        self.labels = []
        self.mix = []

        num = 1

        # cb = np.zeros((1, 3674))
        # inx = json.load(open('class.index', 'r'))
        # for i in inx.keys():
        #     ins_type = i.split('-')[0]
        #     if ins_type == 'Cb':
        #         cb[inx[i]] = 1

        for data in os.listdir(root):
            if data.startswith(mode):
                new_path = os.path.join(root, data)
                print(new_path)
                inp = pickle.load(open(new_path, 'rb'))
                for i, x in enumerate(inp):
                    # if (x[1].numpy()*cb).any():
                    #     continue
                    self.audio_feature.append(x[0])
                    self.labels.append(x[1])
                    # self.mix.append(x)
                    # if (i+1) % 20000 == 0:
                    #     pickle.dump(self.mix, open(
                    #         root+'/'+mode+str(num)+'.pkl', 'wb'))
                    #     num += 1
                    #     self.mix = []
                # break

        if transform is not None:
            self.transform = transform

    def __len__(self):
        return len(self.audio_feature)

    def __getitem__(self, idx):
        audio = self.audio_feature[idx]

        a = np.array(audio[0]).T
        a = spec_augment(a)
        a = a.T

        audio = np.array([a])
        audio = torch.tensor(audio)

        return audio, self.labels[idx]


if __name__ == '__main__':
    root = './data/five'
    a = OrchDataSet(root, 'set', transforms.ToTensor())

    aa = torch.utils.data.DataLoader(dataset=a, batch_size=1, shuffle=True)

    for (train, labels) in aa:
        print("!")
        break
