from captioner.model import ImageCaptionerModel
from captioner.preprocessor import DataPreprocessor
from captioner.dataset import DatasetManager
from captioner.metrics import MetricsCalculator
from utils.logger import logger

def main():
    logger.info("Starting evaluation pipeline")

    # Step 1: Load model
    encoder_model = "google/vit-base-patch16-224-in21k"
    decoder_model = "gpt2"
    model_obj = ImageCaptionerModel(encoder_model, decoder_model)
    model, tokenizer, feature_extractor = model_obj.get_model_components()

    # Step 2: Load and preprocess evaluation dataset
    data_preprocessor = DataPreprocessor(tokenizer, feature_extractor)
    def preprocess_func(example):
        return {
            "pixel_values": data_preprocessor.extract_features([example["image_path"]])[0],
            "labels": data_preprocessor.tokenize(example["caption"], max_len=128),
        }
    dataset_manager = DatasetManager()
    eval_dataset = dataset_manager.load_dataset("ydshieh/coco_dataset_script", "validation", preprocess_func)

    # Step 3: Perform evaluation
    predictions, references = [], []
    for sample in eval_dataset:
        pixel_values = sample["pixel_values"]
        labels = sample["labels"]
        predictions.append(model.generate(pixel_values))
        references.append(labels)

    predictions = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
    references = [tokenizer.decode(ref, skip_special_tokens=True) for ref in references]

    bleu_score = MetricsCalculator.compute_bleu(predictions, references)
    logger.info(f"BLEU Score: {bleu_score}")

if __name__ == "__main__":
    main()
