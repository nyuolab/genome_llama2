import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

class GenomeSequenceProcessor():

    def __init__(self, file_path, train_config):
        self.file_path = file_path
        self.train_config = train_config
        self.DNA_VALID_NUCLEOTIDES = {'A', 'C', 'G', 'T', 'N'}

    def is_valid_genome_sequence(self, sequence):
        return all(nucleotide in self.DNA_VALID_NUCLEOTIDES for nucleotide in sequence)

    def parse_fasta_file(self):
        sequences = []
        current_sequence = ''

        with open(self.file_path, 'r') as file:
            for line in file:
                line = line.strip().upper()
                if line.startswith('>'):
                    if current_sequence:
                        sequences.append(current_sequence)
                        current_sequence = ''
                elif self.is_valid_genome_sequence(line):
                    current_sequence += line

        if current_sequence:
            sequences.append(current_sequence)

        return sequences
    
    def process_fasta_file(self):
        genome_sequence_chunks = []
        parsed_fasta = self.parse_fasta_file()

        for sequence in parsed_fasta:
            # Adjusted to generate chunks and labels in parallel
            seq_length = len(sequence)
            chunk_size = self.train_config.MIN_SEQUENCE_LENGTH

            for i in range(0, seq_length, chunk_size):
                chunk = sequence[i:i + chunk_size]
                genome_sequence_chunks.append(chunk)

        return genome_sequence_chunks
    
    def create_dataset(self, sequences):
        df = pd.DataFrame(sequences, columns=['sequence'])
        return Dataset.from_pandas(df)

    def split_dataset(self, dataset, train_size):
        train_dataset, temp_dataset = train_test_split(dataset, test_size=1-train_size, random_state=self.train_config.SEED)
        valid_dataset, test_dataset = train_test_split(temp_dataset, test_size=0.5, random_state=self.train_config.SEED)
        return train_dataset, valid_dataset, test_dataset