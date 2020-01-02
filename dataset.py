from torch.utils import data
import torchvision.transforms as transforms
import torch
import numpy as np
import pickle
import librosa
import os


class OrchDataSet(data.Dataset):
    def __init__(self, root, mode, transform):
        if not os.path.exists(root):
            print("[Error] root not exits")
            return

        self.audio_feature = []
        self.labels = []
        self.mix = []
        num = 1

        for data in os.listdir(root):
            if data.startswith(mode):
                new_path = os.path.join(root, data)
                print(new_path)
                inp = pickle.load(open(new_path, 'rb'))
                for i, x in enumerate(inp):
                    # self.audio_feature.append(x[0])
                    # self.labels.append(x[1])
                    self.mix.append(x)
                    if (i+1) % 10000 == 0:
                        pickle.dump(self.mix, open(
                            root+'/'+mode+str(num)+'.pkl', 'wb'))
                        num += 1
                        self.mix = []
                # break

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
    root = './data/ten'
    a = OrchDataSet(root, 'trainset-1', transforms.ToTensor())
    a = OrchDataSet(root, 'trainset-2', transforms.ToTensor())
    a = OrchDataSet(root, 'trainset-3', transforms.ToTensor())
    a = OrchDataSet(root, 'trainset-4', transforms.ToTensor())
    # aa = torch.utils.data.DataLoader(dataset=a, batch_size=1, shuffle=True)

    # for (train, labels) in aa:
    #     for i, ind in enumerate(labels[0]):
    #         if ind == 1:
    #             print(i)
    #     break
