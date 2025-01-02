import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from captioner.model import ImageCaptionerModel
from captioner.preprocessor import DataPreprocessor
from captioner.dataset import DatasetManager
from captioner.trainer import Trainer
from utils.logger import logger

def main():
    logger.info("Initializing Image Captioning Training Pipeline")

    # Step 1: Initialize model
    encoder_model = "google/vit-base-patch16-224-in21k"
    decoder_model = "sdadas/polish-gpt2-small"
    model_obj = ImageCaptionerModel(encoder_model, decoder_model)
    model, tokenizer, feature_extractor = model_obj.get_model_components()

    # Step 2: Load and preprocess dataset
    data_preprocessor = DataPreprocessor(tokenizer, feature_extractor)
    def preprocess_func(example):
        model_inputs = {}
        model_inputs['labels'] = data_preprocessor.tokenize(example["text"], max_len=512)
        model_inputs['pixel_values'] = data_preprocessor.extract_features(example['image'])
        return model_inputs
    dataset_manager = DatasetManager()
    train_dataset = dataset_manager.load_dataset("marcinbrzezanski/captioning-v3", "train",preprocess_func)
    eval_dataset = dataset_manager.load_dataset("marcinbrzezanski/captioning", "test",preprocess_func)
    num_epochs = 3
    # Step 3: Initialize trainer
    trainer = Trainer(model, feature_extractor, num_epochs, train_dataset, eval_dataset, output_dir="./output")
    trainer.train()
    logger.info("Training Complete. Model saved to './output'")

if __name__ == "__main__":
    main()
