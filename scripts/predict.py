import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from captioner.model import ImageCaptionerModel
from captioner.inference import ImageCaptionerService
from utils.logger import logger

def main():
    logger.info("Starting prediction pipeline")

    # Step 1: Load model
    encoder_model = "google/vit-base-patch16-224-in21k"
    decoder_model = "gpt2"
    model_obj = ImageCaptionerModel(encoder_model, decoder_model)
    model, tokenizer, feature_extractor = model_obj.get_model_components()

    # Step 2: Initialize inference service
    captioner_service = ImageCaptionerService(model, tokenizer, feature_extractor)

    # Step 3: Predict on new images
    image_path = "69152837.jpg"  # Change to your image path
    caption = captioner_service.predict(image_path)
    logger.info(f"Generated Caption: {caption}")

if __name__ == "__main__":
    main()
