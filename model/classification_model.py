from transformers import AutoConfig, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from lightning.pytorch import LightningModule
from dataset.genome_dataset import GenomeDataset
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, precision_score, recall_score
from tokenization.tokenize_data import tokenize_dataset
import torch
import os

class GenomeLlama2ForClassification(LightningModule):
    """A LightningModule for finetuning a Llama-2 LM for sequence classification with genome data"""

    def __init__(self, tokenizer, model_config):
        super().__init__()
        self.model = None
        self.tokenized_genome_dataset = None
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.true_labels = []
        self.predictions = []
        self.tokenized_genome_dataset, self.num_labels = tokenize_dataset(model_config, tokenizer, pretrain=False)
        self.genome_train = GenomeDataset(self.tokenized_genome_dataset["train"])
        self.genome_valid = GenomeDataset(self.tokenized_genome_dataset["valid"])
        self.genome_test = GenomeDataset(self.tokenized_genome_dataset["test"])

    def forward(self, input_ids, attention_mask, labels):
        """Forward pass of the model"""
        output = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        return output.loss, output.logits

    def _step(self, batch):
        """Helper function to compute the loss for a batch"""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, _ = self(input_ids, attention_mask, labels)
        return loss

    def training_step(self, batch, batch_idx):
        """Training step"""
        loss = self._step(batch)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        loss = self._step(batch)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        """Test step"""
        input_ids = batch['input_ids']
        labels = batch['labels']
        _, generated_ids = self(input_ids, None, None)
        preds = torch.argmax(generated_ids, dim=-1)

        self.true_labels.append(labels)
        self.predictions.append(preds)

        return {'test_preds': preds, 'test_labels': labels}
    
    def on_test_epoch_end(self):
        """Compute and print key evaluation metrics at the end of a test epoch."""
        pred_flattened_list = [item for sublist in self.predictions for item in sublist.tolist()]
        true_flattened_list = [item for sublist in self.true_labels for item in sublist.tolist()]
    
        # Calculate metrics
        accuracy = accuracy_score(pred_flattened_list, true_flattened_list)
        f1 = f1_score(pred_flattened_list, true_flattened_list, average="macro", zero_division=0)
        matthews_correlation = matthews_corrcoef(pred_flattened_list, true_flattened_list)
        precision = precision_score(pred_flattened_list, true_flattened_list, average="macro", zero_division=0)
        recall = recall_score(pred_flattened_list, true_flattened_list, average="macro", zero_division=0)

        # Print metrics
        print("Accuracy:", accuracy)
        print("F1 Score:", f1)
        print("Matthews Correlation Coefficient:", matthews_correlation)
        print("Precision:", precision)
        print("Recall:", recall)
    
    def configure_model(self):
        """Configure the model architecture and loads a pretrained checkpoint for sequence classification based on the specified model size."""
        if self.model is not None:
            return
        config = AutoConfig.from_pretrained(
            self.model_config.LLM_MODEL,
            vocab_size=len(self.tokenizer),
            n_ctx=self.model_config.CONTEXT_LENGTH,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            num_labels=self.num_labels,
            finetuning_task="classification"
        )

        # Set the model parameters
        # Total number of parameters: 119 M
        if(self.model_config.MODEL_SIZE == "base"):
            # The following model parameters align with DNABERT-2 architecture
            config.num_hidden_layers = 12 # Number of transformer layers
            config.num_attention_heads = 12 # Number of attention heads
            config.num_key_value_heads = 12 # Number of key-value heads in attention mechanism
            config.hidden_size = 768 # Size of hidden layers
            config.intermediate_size = 3072 # Size of intermediate layers in the feed-forward network

        # Total number of parameters: 411 M
        elif(self.model_config.MODEL_SIZE == "medium"):
            config.num_hidden_layers = 24
            config.num_attention_heads = 16
            config.num_key_value_heads = 16
            config.hidden_size = 1024
            config.intermediate_size = 4096

        # Total number of parameters: 744 M
        elif(self.model_config.MODEL_SIZE == "large"):
            config.num_hidden_layers = 28
            config.num_attention_heads = 20
            config.num_key_value_heads = 20
            config.hidden_size = 1280
            config.intermediate_size = 5120

        # Load checkpoint
        checkpoint_files = [f for f in os.listdir(self.model_config.CHECKPOINT_DIR) if f.endswith(".ckpt")]
        checkpoint_file = os.path.join(self.model_config.CHECKPOINT_DIR, checkpoint_files[0])
        checkpoint = torch.load(checkpoint_file)
        adjusted_state_dict = {key.replace("model.lm_head.weight", "lm_head.weight"): value for key, value in checkpoint['state_dict'].items()}
        adjusted_state_dict = {key.replace("model.model.", "model."): value for key, value in adjusted_state_dict.items()}

        pretrained_model = AutoModelForSequenceClassification.from_pretrained(None, config=config, state_dict=adjusted_state_dict)
        self.model = pretrained_model

    def configure_optimizers(self):
        """Configure the optimizer and learning rate scheduler"""

        # Explicitly cast config values to float
        lr_max = float(self.model_config.LR_MAX)
        betas = tuple(map(float, self.model_config.BETAS))
        eps = float(self.model_config.EPS)
        weight_decay = float(self.model_config.WEIGHT_DECAY)
        warm_up = float(self.model_config.WARM_UP)
        steps = int(self.model_config.STEPS)  # Ensure steps is an integer

        optimizer = torch.optim.AdamW(self.parameters(), 
                                     lr=lr_max, 
                                     betas=betas, 
                                     eps=eps, 
                                     weight_decay=weight_decay)

        scheduler = OneCycleLR(
            optimizer,
            max_lr = lr_max, # Upper learning rate boundaries in the cycle for each parameter group
            total_steps = steps, # Total number of steps to train for.
            anneal_strategy = 'linear',
            pct_start = float(warm_up / steps))

        lr_scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1,
        }

        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        """Create a DataLoader for the training set"""
        return DataLoader(self.genome_train, batch_size=self.model_config.BATCH_SIZE, shuffle=True, num_workers=self.model_config.NUM_WORKERS)

    def val_dataloader(self):
        """Create a DataLoader for the validation set"""
        return DataLoader(self.genome_valid, batch_size=self.model_config.BATCH_SIZE, shuffle=False, num_workers=self.model_config.NUM_WORKERS)

    def test_dataloader(self):
        """Create a DataLoader for the test set"""
        return DataLoader(self.genome_test, batch_size=self.model_config.BATCH_SIZE, shuffle=False, num_workers=self.model_config.NUM_WORKERS)
    