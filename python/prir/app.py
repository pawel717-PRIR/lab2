from prir.config import app_logging
from prir.files.mnist_trunc_file_manager import MnistTruncFileManager
from prir.metrics import timer
import prir.machinelearning.standarization.standarizator as standarizator
import prir.machinelearning.classifier.knn as knn
from prir.config.app_config import app_conf


logger = app_logging.get_logger(__name__)
logger.debug("APPLICATION START")

dataset = MnistTruncFileManager().read_dataset(train_size=4500, test_size=1500)

train_data_standarized_min_max = timer.call_and_measure(
        """standarizator.min_max_scaler(dataset.get('train_data'))""",
        number=app_conf['measures_count'])
test_data_standarized_min_max = timer.call_and_measure(
        """standarizator.min_max_scaler(dataset.get('test_data'))""",
        number=app_conf['measures_count'])
timer.call_and_measure(
        """knn.train_predict(train_data_standarized_min_max, dataset.get('train_target'), 
        test_data_standarized_min_max, dataset.get('test_target'))""",
        number=app_conf['measures_count'])
del train_data_standarized_min_max


train_data_standarized_standard_scaler = timer.call_and_measure(
    """standarizator.standard_scaler(dataset.get('train_data'))""",
    number=app_conf['measures_count'])
test_data_standarized_standard_scaler = timer.call_and_measure(
    """standarizator.standard_scaler(dataset.get('test_data'))""",
    number=app_conf['measures_count'])
timer.call_and_measure(
        """knn.train_predict(train_data_standarized_standard_scaler, dataset.get('train_target'), 
        test_data_standarized_standard_scaler, dataset.get('test_target'))""",
        number=app_conf['measures_count'])
del train_data_standarized_standard_scaler

logger.debug("APPLICATION END")
