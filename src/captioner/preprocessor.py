from PIL import Image
from transformers import AutoFeatureExtractor, AutoTokenizer
from utils.logger import logger

class DataPreprocessor:
    def __init__(self, tokenizer, feature_extractor):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor

    def tokenize(self, captions, max_len):
        return self.tokenizer(captions, padding="max_length", max_length=max_len).input_ids

    def extract_features(self, image_paths):
        images = []
        to_keep = []
        for image in image_paths:
            try:
                if isinstance(image, str):  # Assuming image is a path or URL
                    img = Image.open(image)
                else:  # If image is already a PIL image
                    img = image
                images.append(img)
                to_keep.append(True)
            except Exception as e:
                logger.error(f"Error loading image {image}: {e}")
                to_keep.append(False)
        pixel_values = self.feature_extractor(images=images, return_tensors="pt").pixel_values
        return pixel_values
