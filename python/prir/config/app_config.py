import os
import json


def load_app_config(
    default_path='config/app-config.json',
    env_key='APP_CFG'
):
    """Load app configuration

    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
        return config
    else:
        return {
            "jobs_number": 1,
            "measures_count": 1
        }


app_conf = load_app_config()
