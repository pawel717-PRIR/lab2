import pandas as pd
from prir.files.dataset_file_manager import DatasetFileManager


class MnistFileManager(DatasetFileManager):
    def __init__(self,
                 train_dataset_file_path="../dataset/mnist_train.csv",
                 test_dataset_file_path="../dataset/mnist_test.csv"):
        self.train_dataset_file_path = train_dataset_file_path
        self.test_dataset_file_path = test_dataset_file_path

    def read_dataset(self):
        train_dataset = pd.read_csv(filepath_or_buffer=self.train_dataset_file_path, header=None)
        test_dataset = train_dataset[4500:6000]
        train_dataset = train_dataset[:4500]

        return {
            "train_data": self._drop_target(train_dataset),
            "train_target": self._get_target(train_dataset),
            "test_data": self._drop_target(test_dataset),
            "test_target": self._get_target(test_dataset)
        }

    def _get_target(self, data):
        return data[0].to_numpy()

    def _drop_target(self, data):
        return data.drop(0, axis=1)
