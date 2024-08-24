import torch
from torch.utils.data import Dataset

class GenomeDataset(Dataset):
    """A custom Dataset class for genomic data"""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert each item in the dataset to a tensor with long data type
        item = {key: torch.tensor(val, dtype=torch.long) for key, val in self.data[idx].items()} 
        return item