import yaml
import json


def read_config(config_name):
    if '/' not in config_name and '\\' not in config_name:
        config_name = f'Configs/{config_name}'

    with open(config_name, 'r') as f:
        if config_name.endswith('.json'):
            config = json.load(f)
        else:
            config = yaml.safe_load(f)

    return config
