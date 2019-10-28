from prir.files.mnist_file_manager import MnistFileManager
import pandas as pd


class MnistTruncFileManager(MnistFileManager):
    def __init__(self,
                 dataset_file_path="../dataset/mnist_train.csv"):
        self.train_dataset_file_path = dataset_file_path
        self.test_dataset_file_path = dataset_file_path

    def read_dataset(self, train_size, test_size):
        self.__train_size = train_size
        self.__test_size = test_size

        train_dataset = pd.read_csv(filepath_or_buffer=self.train_dataset_file_path, header=None)
        test_dataset = train_dataset[self.__train_size:self.__train_size+self.__test_size]
        train_dataset = train_dataset[:self.__train_size-self.__test_size]

        return {
            "train_data": self._drop_target(train_dataset),
            "train_target": self._get_target(train_dataset),
            "test_data": self._drop_target(test_dataset),
            "test_target": self._get_target(test_dataset)
        }
