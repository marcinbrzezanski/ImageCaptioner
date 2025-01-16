import torch
from accelerate import Accelerator
from tqdm import tqdm

class Trainer:
    def __init__(self, model, optimizer, scheduler, accelerator = None, num_epochs=3):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accelerator = accelerator or Accelerator(mixed_precision="fp16")
        self.num_epochs = num_epochs

    def train(self, train_dataloader, num_epochs):
        self.model, self.optimizer, train_dataloader= self.accelerator.prepare(
            self.model, self.optimizer, train_dataloader
        )

        for epoch in range(num_epochs):
            self.model.train()
            total_steps = 25000
            progress_bar = tqdm(
                total=total_steps,
                desc=f"Epoch {epoch+1}",
                disable=not self.accelerator.is_local_main_process
            )
            for batch in train_dataloader:
                outputs = self.model(**batch)
                loss = outputs.loss
                progress_bar.update(1)
                self.accelerator.backward(loss)

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                progress_bar.set_postfix({"loss": loss.item()})
            
            self.accelerator.wait_for_everyone()
            self.accelerator.save(
                self.model.state_dict(), f"checkpoint-{epoch+1}.pt"
            )