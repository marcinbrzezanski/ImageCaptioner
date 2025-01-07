from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, default_data_collator
from torch.utils.data import DataLoader
import torch

class Trainer:
    def __init__(self, model, tokenizer, feature_extractor, num_train_epochs, train_dataset, eval_dataset, output_dir):
        self.output_dir = output_dir
        self.args = Seq2SeqTrainingArguments(
            predict_with_generate=True,
            eval_strategy='epoch',
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            output_dir=output_dir,
            dataloader_num_workers=4,
            deepspeed=None
        )

        self.trainer = Seq2SeqTrainer(
            model=model,
            tokenizer=feature_extractor,
            args=self.args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=default_data_collator,
            #processing_class=[tokenizer, feature_extractor]
        )

    def train(self):
        # Ensure multi-GPU support is enabled with the Trainer API
        if torch.cuda.device_count() > 1:
            print(f"Training on {torch.cuda.device_count()} GPUs.")

        self.trainer.train()
        self.trainer.save_model()
        self.trainer.tokenizer.save_pretrained(self.output_dir)