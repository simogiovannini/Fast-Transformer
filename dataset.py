from torch.utils.data import Dataset
import torch


def load_dataset(path):
    dataset = torch.load(path)
    return dataset


class OneSentenceDataset(Dataset):
    def __init__(self, dataset, n_classes = 0):
        self.features = dataset[0]
        self.labels = dataset[1]
        self.num_samples = len(dataset[0])
        self.n_classes = n_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


class TwoSentencesDataset(Dataset):
    def __init__(self, dataset, n_classes = 0):
        self.first_sentences = dataset[0]
        self.second_sentences = dataset[1]
        self.labels = dataset[2]
        self.num_samples = len(dataset[0])
        self.n_classes = n_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return (self.first_sentences[index], self.second_sentences[index]), self.labels[index]
