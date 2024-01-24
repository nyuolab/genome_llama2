import torch
from transformers import AutoConfig, LlamaForCausalLM
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule

class Llama2Genome(LightningModule):
    def __init__(self, tokenizer, tokenized_dataset, learning_rate, train_config):
        super().__init__()
        self.model = None
        self.tokenizer = tokenizer
        self.tokenized_dataset = tokenized_dataset
        self.learning_rate = learning_rate
        self.train_config = train_config

    def forward(self, input_ids, attention_mask, label_ids):
        output = self.model(input_ids, attention_mask=attention_mask, labels=label_ids)
        return output.loss, output.logits

    def _transform_tensors(self, input_tensors):
        # Assuming all tensors have the same length
        num_tensors = len(input_tensors)
        tensor_length = len(input_tensors[0])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Transposing the tensors
        output_tensors = [torch.tensor([input_tensors[j][i] for j in range(num_tensors)]) for i in range(tensor_length)]

        return torch.stack(output_tensors).to(device)

    def _step(self, batch):
        input_ids = self._transform_tensors(batch['input_ids'])
        attention_mask = self._transform_tensors(batch['attention_mask'])
        labels = self._transform_tensors(batch['label_ids'])
        loss, _ = self(input_ids, attention_mask, labels)
        # Write the loss function
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('test_loss', loss, prog_bar=True)
        return loss
    
    def configure_model(self):
        if self.model is not None:
            return
        config = AutoConfig.from_pretrained(
            self.train_config.LLM_MODEL,
            vocab_size=len(self.tokenizer),
            n_ctx=self.train_config.CONTEXT_LENGTH,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        self.model = LlamaForCausalLM(config)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10),
            'monitor': 'val_loss',
            'frequency': 1,
            'interval': 'epoch'
        }
        return [optimizer], [lr_scheduler]
        
    def _get_dataloader(self, split_name):
        sampler = torch.utils.data.DistributedSampler(self.tokenized_dataset[split_name], shuffle=True)
        return DataLoader(self.tokenized_dataset[split_name], batch_size=self.train_config.BATCH_SIZE, num_workers=self.train_config.NUM_WORKERS, sampler=sampler)

    def train_dataloader(self):
        return self._get_dataloader('train')

    def val_dataloader(self):
        return self._get_dataloader('valid')

    def test_dataloader(self):
        return self._get_dataloader('test')