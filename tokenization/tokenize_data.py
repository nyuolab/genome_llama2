from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer
import sys
import os
import pandas as pd
import argparse
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util import ConfigDict

class GenomeSequenceTokenizer():
    """Define a class to handle tokenization of genome sequences"""
    def __init__(self, tokenizer, model_config, pretrain=True):
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.pretrain = pretrain

    def shared_transform(self, processed_dataset):
        """Apply tokenization to the entire dataset."""
        tokenized_ds = processed_dataset.map(
            self.tokenize,
            remove_columns=processed_dataset["train"].column_names,
            batched=True,
            load_from_cache_file=False,
        )

        return tokenized_ds
    
    def tokenize(self, element):
        """Tokenize a single element of the dataset."""

        # Use 'text' key for pretraining, 'sequence' for fine-tuning
        text_key = 'text' if self.pretrain else 'sequence'
        
        outputs = self.tokenizer(
            element[text_key],
            truncation=True,
            padding="max_length",
            max_length=self.model_config.CONTEXT_LENGTH,
            return_overflowing_tokens=True if self.pretrain else False,
            return_length=True,
            return_tensors="pt"
        )

        # Return different outputs based on pretraining or fine-tuning mode
        if(self.pretrain):
            return {"input_ids": outputs["input_ids"]}
        
        return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"], "labels": element['label']}
    
def tokenize_dataset(model_config, tokenizer, pretrain=True):
    """Tokenize the dataset for either pretraining or fine-tuning."""
    if pretrain:
        # Pretraining data loading
        from processor import GenomeSequenceProcessor
        genome_sequence_processor = GenomeSequenceProcessor(model_config.RAW_TRAIN_DATA_DIR_PATH)
        ds_train, ds_valid = genome_sequence_processor.load_genome_dataset()
        raw_datasets = DatasetDict({
            "train": ds_train,
            "valid": ds_valid
        })
    else:
        # Fine-tuning data loading
        identifier = model_config.FINETUNE_DATA_DIR_PATH
        train_path = f"{identifier}train.csv"
        valid_path = f"{identifier}dev.csv"
        test_path = f"{identifier}test.csv"
        
        df_train = pd.read_csv(train_path)
        df_valid = pd.read_csv(valid_path)
        df_test = pd.read_csv(test_path)
        num_labels = df_train['label'].max() + 1
        
        raw_datasets = DatasetDict({
            "train": Dataset.from_pandas(df_train),
            "valid": Dataset.from_pandas(df_valid),
            "test": Dataset.from_pandas(df_test)
        })

    # Initialize tokenizer
    tokenizer = GenomeSequenceTokenizer(tokenizer, model_config, pretrain)
    
    print("Genome dataset tokenization started...")
    tokenized_dataset = tokenizer.shared_transform(raw_datasets)
    print("Genome dataset tokenized successfully...")

    # Save tokenized dataset only for pretraining
    if pretrain:
        print(f"Saving Tokenized Genome dataset to {model_config.TOKENIZED_DATASET_PATH}...")
        tokenized_dataset.save_to_disk(model_config.TOKENIZED_DATASET_PATH)
        print(f"Tokenized Genome dataset saved successfully to {model_config.TOKENIZED_DATASET_PATH}.")

    return tokenized_dataset, num_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenize_config", type=str, default="", help="Path to file that contains all the args for finetuning.", required=True)
    args = parser.parse_args()

    # Load the YAML file
    with open(args.tokenize_config, 'r') as file:
        tokenize_config_dict = yaml.safe_load(file)

    # Convert the dictionary into a ConfigDict object
    tokenize_config = ConfigDict(tokenize_config_dict)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenize_config.TOKENIZER_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    tokenize_dataset(tokenize_config, tokenizer)