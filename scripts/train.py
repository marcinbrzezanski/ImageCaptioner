from transformers import AdamW, get_scheduler
from captioner.model import ImageCaptionerModel
from captioner.preprocessor import DataPreprocessor
from captioner.trainer import Trainer
from utils.logger import logger

def main():
    logger.info("Initializing Image Captioning Training Pipeline")
    accelerator = Accelerator(fp16=torch.cuda.is_available())
    
    # Step 1: Initialize model
    encoder_model = "google/vit-base-patch16-224-in21k"
    decoder_model = "sdadas/polish-gpt2-small"
    model_obj = ImageCaptionerModel(encoder_model, decoder_model)
    model, tokenizer, feature_extractor = model_obj.get_model_components()

    # Step 2: Load and preprocess dataset
    data_preprocessor = DataPreprocessor(tokenizer, feature_extractor)
    preprocess_fn = lambda example: {
        "labels": data_preprocessor.tokenize(example["text"], max_len=1024),
        "pixel_values": data_preprocessor.extract_features(example["image"]),
    }
    dataset_manager = DatasetManager(batch_size=8)
    train_dataloader = dataset_manager.stream_dataset(
        "marcinbrzezanski/captioning-v6",
        "train",
        num_samples=25000
    )
    #eval_dataloader = dataset_manager.load_dataset(
    #    "marcinbrzezanski/captioning", 
    #    "test",
    #   preprocess_fn)
    
    # Step 3: Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = len(train_dataloader) * Trainer.num_epochs
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    # Step 4: Initialize Trainer
    trainer = Trainer(model, optimizer, scheduler, accelerator)
    trainer.train(train_dataloader, num_epochs=3)
  

if __name__ == "__main__":
    main()
