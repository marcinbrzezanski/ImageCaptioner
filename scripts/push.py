import os
import sys
from huggingface_hub import HfApi
from datasets import Dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.captioner.dataset import DatasetManager
from src.captioner.model import ImageCaptionerModel
from src.captioner.preprocessor import DataPreprocessor

# Function to load the dataset from disk
def load_dataset_from_disk(dataset_path):
    if os.path.exists(dataset_path):
        dataset = Dataset.load_from_disk(dataset_path)
        print(f"Dataset loaded from {dataset_path}")
        return dataset
    else:
        raise FileNotFoundError(f"The dataset path '{dataset_path}' does not exist.")
def load_dataset_from_hub(repo_id, data_preprocessor):
    dataset_manager = DatasetManager()
    def preprocess_func(example):
        model_inputs = {}
        model_inputs['labels'] = data_preprocessor.tokenize(example["text"], max_len=1024)
        model_inputs['pixel_values'] = data_preprocessor.extract_features(example['image'])
        return model_inputs
    dataset = dataset_manager.load_dataset(repo_id, "train", preprocess_func)
    return dataset


# Function to push the dataset to the Hugging Face Hub
def push_dataset_to_hub(dataset, repo_id, token):
    api = HfApi()
    try:
        # Ensure the repository exists, create it if necessary
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, token=token)
        dataset.push_to_hub(repo_id, token=token)
        print(f"Dataset successfully pushed to Hugging Face Hub under '{repo_id}'")
    except Exception as e:
        print(f"Failed to push dataset to Hugging Face Hub: {e}")

def main():
    # Step 1: Define the path where the dataset is saved locally
    #local_save_path = "./translated_flickr8k"  # Update this path if necessary
    encoder_model = "google/vit-base-patch16-224-in21k"
    decoder_model = "sdadas/polish-gpt2-small"
    model_obj = ImageCaptionerModel(encoder_model, decoder_model)
    model, tokenizer, feature_extractor = model_obj.get_model_components()
    data_preprocessor = DataPreprocessor(tokenizer, feature_extractor)
    # Step 2: Load the dataset from disk
    try:
        #dataset = load_dataset_from_disk(local_save_path)
        dataset = load_dataset_from_hub("marcinbrzezanski/captioning-v5", data_preprocessor)
    except FileNotFoundError as e:
        print(e)
        return

    local_save_path = "./translated_captions"
    os.makedirs(local_save_path, exist_ok=True)
    dataset.save_to_disk(local_save_path)
    # Step 3: Define the repository details for Hugging Face Hub
    repo_id = "marcinbrzezanski/captioning-v6"  # Update the repo ID
    token = "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"  # Your Hugging Face token

    # Step 4: Push the dataset to Hugging Face Hub
    push_dataset_to_hub(dataset, repo_id, token)

if __name__ == "__main__":
    main()
