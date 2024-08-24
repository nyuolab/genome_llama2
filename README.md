# Genome-Llama-2: An Autoregressive Foundation Model for Multi-Species Genomes

## Table of Contents
- [Introduction](#introduction)
- [Pre-training](#pre-training)
- [Example Usage](#example-usage)

## Introduction
Genome-Llama-2 is a family of autoregressive large language models with sizes ranging from 119 million to 744 million parameters, specifically trained on extensive multi-species genomic data. This model integrates cutting-edge advancements in natural language processing and deep learning, making it a powerful tool for genomic research. Building on the Llama-2 architecture, Genome-Llama-2 is optimized to approach the performance of [DNABERT-2](https://github.com/MAGICS-LAB/DNABERT_2). The family includes models of varying sizes to accommodate different computational needs:
-  Base model: 119 million parameters.
-  Medium model: 411 million parameters.
-  Large model: 744 million parameters.

By treating genome sequences as text, Genome-Llama-2 can learn intricate patterns and relationships within the data. This capability makes it suitable for a range of tasks, including Epigenetic Marks Prediction, Covid Variants Classification, Splice Site Prediction, and other tasks specified by the [GUE benchmark](https://github.com/MAGICS-LAB/DNABERT_2).

This repository contains the complete training pipeline for Genome-Llama-2, leveraging PyTorch Lightning to enable efficient training in a distributed environment. Whether you are a researcher or developer, Genome-Llama-2 offers a robust framework for advancing your genomic studies.

## Pre-training
We used the same dataset as DNABERT-2 to pretrain Genome-Llama-2. The training data can be accessed [here](https://drive.google.com/file/d/1dSXJfwGpDSJ59ry9KAp8SugQLK35V83f/view).

## Example Usage
To tokenize the pre-training dataset, use the following command:
```bash
python genome_llama2/tokenization/tokenize_data.py --tokenize_config genome_llama2/config/pretrain_config.yaml
```

To pre-train the model, use the following command:
```bash
python genome_llama2/pretrain_model.py --pretrain_config genome_llama2/config/pretrain_config.yaml
```

To fine-tune the model on a specific data, use the following command:
```bash
python genome_llama2/finetune_model.py --finetune_config genome_llama2/config/finetune_config.yaml
```

The user can change the configuration in the pretrain_config.yaml and finetune_config.yaml files to customize tokenization, pre-training, and fine-tuning according to their requirements.
