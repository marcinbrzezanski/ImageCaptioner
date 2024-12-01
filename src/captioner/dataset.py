import datasets
from utils.logger import logger

class DatasetManager:
    @staticmethod
    def load_dataset(dataset_name, split,preprocess):
        ds = datasets.load_dataset(dataset_name,split=split)
        return ds.map(
            function=preprocess,
            batched=True)