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
import torch

def main():
    logger.info("Initializing Image Captioning Training Pipeline")
    accelerator = Accelerator(mixed_precision="fp16")
    
    # Step 1: Initialize model
    encoder_model = "marcinbrzezanski/vit-gpt2-polish-captionier-v1"
    decoder_model = "marcinbrzezanski/vit-gpt2-polish-captionier-v1"
    model_obj = ImageCaptionerModel(encoder_model, decoder_model)
    model = model_obj.get_model_components()

    # Step 2: Load and preprocess dataset
    data_preprocessor = DataPreprocessor(tokenizer, feature_extractor)
    preprocess_fn = lambda example: {
        "labels": data_preprocessor.tokenize(example["text"], max_len=1024),
        "pixel_values": data_preprocessor.extract_features(example["image"]),
    }
    dataset_manager = DatasetManager(batch_size=5)
    train_dataloader = dataset_manager.stream_dataset(
        dataset_name = "marcinbrzezanski/captioning-v6",
        split = "train",
        num_samples=95000
    )
    #eval_dataloader = dataset_manager.load_dataset(
    #    "marcinbrzezanski/captioning", 
    #    "test",
    #   preprocess_fn)
    
    # Step 3: Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = 95000 * 1 // 5
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    # Step 4: Initialize Trainer
    trainer = Trainer(model, optimizer, scheduler, accelerator)
    trainer.train(train_dataloader, num_epochs=1)
  

if __name__ == "__main__":
    main()
