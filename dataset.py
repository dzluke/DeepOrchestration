import torch.utils.data as data


class OrchDataSet(data.Dataset):
    def __init__(self, features, labels, transform):
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        feature = self.transform(feature)
        return feature, self.labels[idx]
