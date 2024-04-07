import os
from datasets import Dataset

class GenomeSequenceProcessor:

    def __init__(self, directory, train_config):
        self.directory = directory
        self.train_config = train_config

    def split_sequence_generator(self, sequence, chunk_size):
        """
        Split a sequence into chunks of the given size and yield them one by one.
        """
        for i in range(0, len(sequence), chunk_size):
            yield sequence[i:i+chunk_size]

    def get_nucleotide_sequences(self, chunk_size):
        for filename in os.listdir(self.directory):
            if filename.endswith(".fna"):
                file_path = os.path.join(self.directory, filename)
                with open(file_path, 'r') as f:
                    sequence = ''
                    for line in f:
                        if not line.startswith('>'):
                            clean_line = line.upper().strip().replace('N', '')
                            sequence += clean_line
                    # Yield sequence chunks instead of extending a list
                    for chunk in self.split_sequence_generator(sequence, chunk_size):
                        yield chunk
    
    def create_and_split_dataset(self, train_size, chunk_size=10000):
        # Initialize lists to hold training and validation sequences
        train_sequences = []
        valid_sequences = []

        # Generate sequences and split them into training and validation sets
        for sequence in self.get_nucleotide_sequences(chunk_size):
            # Randomly decide if the sequence goes into the training or validation set
            # based on the specified training size
            if len(train_sequences) / (len(train_sequences) + len(valid_sequences) + 1) < train_size:
                train_sequences.append(sequence)
            else:
                valid_sequences.append(sequence)

        # Convert lists to datasets
        train_dataset = Dataset.from_dict({"sequence": train_sequences})
        valid_dataset = Dataset.from_dict({"sequence": valid_sequences})

        return train_dataset, valid_dataset