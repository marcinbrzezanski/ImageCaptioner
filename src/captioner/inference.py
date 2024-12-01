from transformers import pipeline

class ImageCaptionerService:
    def __init__(self, model, tokenizer, feature_extractor):
        self.pipeline = pipeline("image-to-text", model=model, tokenizer=tokenizer, feature_extractor=feature_extractor)

    def predict(self, image):
        return self.pipeline(image)
