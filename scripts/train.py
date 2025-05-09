from transformers import AdamW, get_scheduler
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from captioner.model import ImageCaptionerModel
from captioner.preprocessor import DataPreprocessor
from captioner.trainer import Trainer
from utils.logger import logger
from captioner.dataset import DatasetManager
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import torch
BATCH_SIZE = 10
NUM_EPOCHS = 5

def main(args=None):
    logger.info("Initializing Image Captioning Training Pipeline")
    #ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision="fp16")
    
    # Step 1: Initialize model
    encoder_model = "google/vit-base-patch16-224-in21k"
    decoder_model = "sdadas/polish-gpt2-small"
    model_obj = ImageCaptionerModel(encoder_model, decoder_model)
    model, tokenizer, feature_extractor = model_obj.get_model_components()

    # Step 2: Load and preprocess dataset
    data_preprocessor = DataPreprocessor(tokenizer, feature_extractor)
    #preprocess_fn = lambda example: {
    #    "labels": data_preprocessor.tokenize(example["translated_text"], max_len=512),
     #   "pixel_values": data_preprocessor.extract_features(example["processed_image"]),
    #}
    dataset_manager = DatasetManager(batch_size=BATCH_SIZE)
    train_dataloader = dataset_manager.load_dataset(
        dataset_name = "marcinbrzezanski/captioning-final-100k-v3-features",
        split = "train",
    )
    #eval_dataloader = dataset_manager.load_dataset(
    #    "marcinbrzezanski/captioning", 
    #    "test",
    #   preprocess_fn)
    
    # Step 3: Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = 95000 * NUM_EPOCHS // BATCH_SIZE
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    model, train_dataloader, optimizer, scheduler = accelerator.prepare(
        model, train_dataloader, optimizer, scheduler
    )

    # Step 4: Initialize Trainer
    trainer = Trainer(model, optimizer, scheduler, accelerator, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
    trainer.train(train_dataloader, num_epochs=NUM_EPOCHS)
  

if __name__ == "__main__":
    from accelerate import notebook_launcher
    notebook_launcher(main, num_processes=1)
