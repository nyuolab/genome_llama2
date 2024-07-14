import torch
from torch.utils.data import Dataset

# Define a custom Dataset class for genomic data
class GenomeDataset(Dataset):

    # Initialize the dataset with data
    def __init__(self, data):
        self.data = data

    # Return the length of the dataset
    def __len__(self):
        return len(self.data)

    # Retrieve an item from the dataset at a given index
    def __getitem__(self, idx):
        # Convert each item in the dataset to a tensor with long data type
        item = {key: torch.tensor(val, dtype=torch.long) for key, val in self.data[idx].items()} 
        return item