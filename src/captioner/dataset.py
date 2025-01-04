import datasets
from src.utils.logger import logger
import tqdm

class DatasetManager:
    @staticmethod
    def load_dataset(dataset_name, split,preprocess):
        ds = datasets.load_dataset(dataset_name,split=split)
        return ds.map(
            function=preprocess,
            batched=True)
    @staticmethod
    def stream_dataset(dataset_name, split,preprocess,num_samples=1):
        ds = datasets.load_dataset(dataset_name,split=split,streaming=True)
        ds  = ds.take(num_samples)
        processed_samples = []

        for sample in tqdm.tqdm(ds, total=num_samples, desc="Processing dataset"):
            processed_sample = preprocess(sample)
            processed_samples.append(processed_sample)
        return processed_samples