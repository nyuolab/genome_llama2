# Genome-Llama-2: An Autoregressive Foundation Model for Multi-Species Genomes

## Table of Contents
- [Introduction](#introduction)
- [Pre-training](#pre-training)
- [Example Usage](#example-usage)
- [Todos](#todos)

## Introduction
Genome-Llama-2 is an autoregressive large language model with 119 million parameters, specifically trained on extensive multi-species genomic data. This model integrates cutting-edge advancements in natural language processing and deep learning, making it a powerful tool for genomic research. Building on the Llama-2 architecture, Genome-Llama-2 is optimized to approach the performance of [DNABERT-2](https://github.com/MAGICS-LAB/DNABERT_2).

By treating genome sequences as text, Genome-Llama-2 can learn intricate patterns and relationships within the data. This capability makes it suitable for a range of tasks, including Epigenetic Marks Prediction, Covid Variants Classification, Splice Site Prediction, and other tasks specified by the [GUE benchmark](https://github.com/MAGICS-LAB/DNABERT_2).

This repository contains the complete training pipeline for Genome-Llama-2, leveraging PyTorch Lightning to enable efficient training in a distributed environment. Whether you are a researcher or developer, Genome-Llama-2 offers a robust framework for advancing your genomic studies.

## Pre-training
We used the same dataset as DNABERT-2 to pretrain Genome-Llama-2. The training data can be accessed [here](https://drive.google.com/file/d/1dSXJfwGpDSJ59ry9KAp8SugQLK35V83f/view).

## Example Usage
To tokenize the pre-training data, use the following command:
```bash
python tokenize_data.py --train_data_path /path/to/pre_training_data --tokenized_dataset_path /path/to/store/tokenized_data
```

To pre-train the model, use this command:
```bash
python train_model.py --tokenized_dataset_path /path/to/tokenized_dataset --checkpoint_dir_path /path/to/store/checkpoints --log_dir_path /path/to/store/logs
```

## Todos
- Add code for fine-tuning Genome-Llama-2 on specific dataset.
