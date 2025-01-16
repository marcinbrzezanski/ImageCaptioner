import datasets
from utils.logger import logger

class DatasetManager:
    @staticmethod
    def load_dataset(dataset_name, split,preprocess):
        ds = datasets.load_dataset(dataset_name,split=split)
        return ds.map(
            function=preprocess,
            batched=True)
    @staticmethod
    def stream_dataset(dataset_name, split,num_samples=1):
        ds = datasets.load_dataset(dataset_name,split=split,streaming=True)
        ds.select_columns(["labels","pixel_values"])
        ds  = ds.take(num_samples)
        return ds