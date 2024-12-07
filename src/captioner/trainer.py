from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator

class Trainer:
    def __init__(self, model, tokenizer, train_dataset, eval_dataset, output_dir):
        self.args = Seq2SeqTrainingArguments(
            predict_with_generate=True,
            evaluation_strategy='epoch',
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            output_dir=output_dir
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
        self.trainer.train()
        self.trainer.save_model()
        self.self.trainer.tokenizer.save_pretrained(self.args.output_dir)
