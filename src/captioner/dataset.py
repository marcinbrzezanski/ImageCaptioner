import datasets
from utils.logger import logger
from torch.utils.data import DataLoader
from transformers import default_data_collator

class DatasetManager:
    def __init__(self, batch_size=4, num_workers=4):
        self.batch_size = batch_size
        self.num_workers = num_workers
    def load_dataset(self, dataset_name, split, preprocess_func):
        ds = datasets.load_dataset(dataset_name,split=split)
        ds = ds.map(
            function=preprocess_func,
            batched=True)
        return DataLoader(ds, batch_size=self.batch_size, collate_fn=default_data_collator, num_workers=self.num_workers)
    def stream_dataset(self, dataset_name, split,num_samples=1):
        ds = datasets.load_dataset(dataset_name,split=split,streaming=True)
        ds = ds.remove_columns([col for col in ds.column_names if col not in ["labels", "pixel_values"]])
        ds  = ds.take(num_samples)
        return DataLoader(ds, batch_size=self.batch_size, collate_fn=default_data_collator, num_workers=self.num_workers)