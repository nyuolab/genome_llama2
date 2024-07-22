from datasets import DatasetDict
from transformers import AutoTokenizer
from config import ModelConfig
from processor import GenomeSequenceProcessor
import argparse

# Define a class to handle tokenization of genome sequences
class GenomeSequenceTokenizer():

    def __init__(self, tokenizer, model_config):
        self.tokenizer = tokenizer
        self.model_config = model_config

    # Tokenize the dataset
    def shared_transform(self, processed_dataset):
        # Map the tokenize method to the dataset, removing original columns and not using cached results
        tokenized_ds = processed_dataset.map(
            self.tokenize,
            remove_columns=processed_dataset["train"].column_names,
            batched=True,
            load_from_cache_file=False,
        )

        return tokenized_ds

    # Tokenize individual elements of the dataset
    def tokenize(self, element):
        # Tokenize the text element with specified options
        outputs = self.tokenizer(
            element['text'],
            truncation=True,
            padding="max_length",
            max_length=self.model_config.CONTEXT_LENGTH,
            return_overflowing_tokens=True,
            return_length=True,
            return_tensors="pt"
        )

        return {"input_ids": outputs["input_ids"]}
    
 # Tokenize the dataset
def tokenize_dataset(model_config):

    # Load the tokenizer from the pretrained model specified in the configuration
    tokenizer = AutoTokenizer.from_pretrained(model_config.TOKENIZER_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize the genome sequence processor with the directory path for raw data
    genome_sequence_processor = GenomeSequenceProcessor(model_config.RAW_TRAIN_DATA_DIR_PATH)
    ds_train, ds_valid  = genome_sequence_processor.load_genome_dataset()
    
    # Create a DatasetDict containing the train and validation datasets
    raw_datasets = DatasetDict({
        "train": ds_train,
        "valid": ds_valid
    })

    # Initialize the genome sequence tokenizer
    tokenizer = GenomeSequenceTokenizer(tokenizer, model_config)

    # Perform tokenization on the raw datasets
    print("Genome dataset tokenization started...")
    tokenized_dataset = tokenizer.shared_transform(raw_datasets)
    print("Genome dataset tokenized successfully...")

    # Save the tokenized dataset to disk
    print(f"Saving Tokenized Genome dataset to {model_config.TOKENIZED_DATASET_PATH}...")
    tokenized_dataset.save_to_disk(model_config.TOKENIZED_DATASET_PATH)
    print(f"Tokenized Genome dataset saved successfully to {model_config.TOKENIZED_DATASET_PATH}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, default="", help="Path to the training data.", required=True)
    parser.add_argument("--tokenized_dataset_path", type=str, default="", help=" Path to the directory where tokenized dataset will be saved.", required=True)
    parser.add_argument("--context_length", type=int, default=128, help="Context length (number of tokens) in each sequence (default: 128)")
    args = parser.parse_args()

    model_config = ModelConfig()
    model_config.RAW_TRAIN_DATA_DIR_PATH = args.train_data_path
    model_config.TOKENIZED_DATASET_PATH = args.tokenized_dataset_path
    model_config.CONTEXT_LENGTH = args.context_length

    tokenize_dataset(model_config)