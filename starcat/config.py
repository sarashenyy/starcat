import os

import toml

# load config file
module_dir = os.path.dirname(__file__)
config_file = os.path.join(module_dir, 'data', 'config.toml')
config = toml.load(config_file)
# load data_dir
path = config['data_dir']
data_dir = config[path]
