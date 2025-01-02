from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, default_data_collator
from torch.utils.data import DataLoader
import torch

class Trainer:
    def __init__(self, model, tokenizer, num_train_epochs, train_dataset, eval_dataset, output_dir):
        self.output_dir = output_dir
        self.args = Seq2SeqTrainingArguments(
            predict_with_generate=True,
            eval_strategy='epoch',
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=24,
            per_device_eval_batch_size=24,
            output_dir=output_dir,
            # Enable multi-GPU training
            fp16=True,  # Use mixed precision for faster training
            dataloader_num_workers=4,  # Number of CPU threads to use for data loading
            # Automatically distribute across all available GPUs
            deepspeed=None,  # Optional: Enable deepspeed for large scale distributed training
            load_best_model_at_end=True,  # Always load the best model at the end of training
        )

        self.trainer = Seq2SeqTrainer(
            model=model,
            tokenizer=tokenizer,
            args=self.args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=default_data_collator
        )

    def train(self):
        # Ensure multi-GPU support is enabled with the Trainer API
        if torch.cuda.device_count() > 1:
            print(f"Training on {torch.cuda.device_count()} GPUs.")
            self.trainer.args.device = 'cuda'

        self.trainer.train()
        self.trainer.save_model()
        self.trainer.tokenizer.save_pretrained(self.output_dir)