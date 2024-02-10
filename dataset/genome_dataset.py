import torch
from torch.utils.data import Dataset

class GenomeDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val, dtype=torch.long) for key, val in self.data[idx].items()}
        return item