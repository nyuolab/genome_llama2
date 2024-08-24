import os
from datasets import load_dataset

class GenomeSequenceProcessor:
    """A class to process genome sequence datasets"""

    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.train_dataset = None
        self.valid_dataset = None

    def load_genome_dataset(self):
        """Load genome sequence datasets from specified files"""

        # Define the paths for the train and validation datasets
        train_file_path = os.path.join(self.dir_path, 'train.txt')
        valid_file_path = os.path.join(self.dir_path, 'dev.txt')

        # Load the training dataset from the train file
        self.train_dataset = load_dataset('text', data_files={'sequence': train_file_path}, split='sequence')

        # Load the validation dataset from the validation file
        self.valid_dataset = load_dataset('text', data_files={'sequence': valid_file_path}, split='sequence')

        # Return the loaded datasets
        return self.train_dataset, self.valid_dataset