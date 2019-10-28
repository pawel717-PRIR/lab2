import logging
import os
import json
import logging.config


def setup_logging(
    default_path='config/logger-config.json',
    env_key='LOG_CFG'
):
    """Setup logging configuration

    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.DEBUG, filename='logs/app.log', filemode='a',
                            format='%(asctime)s | %(process)d | %(levelname)s | %(module)s %(funcName)s | %(message)s')


def get_logger(name='default'):
    return logging.getLogger(name)


setup_logging()


