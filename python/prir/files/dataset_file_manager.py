from abc import ABC, abstractmethod


class DatasetFileManager(ABC):
    @abstractmethod
    def read_dataset(self):
        pass
