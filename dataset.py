from torch.utils.data import Dataset
import torch


def load_dataset(path):
    dataset = torch.load(path)
    return dataset


class OneSentenceDataset(Dataset):
    def __init__(self, dataset):
        self.features = dataset[0]
        self.labels = dataset[1]
        self.num_samples = len(dataset[0])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    
    def get_n_classes(self):
        return len(set(self.labels))