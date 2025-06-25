import json
import re

# read config
def read_config(config_path):
    with open(config_path, 'r') as f:
        json_str = f.read()
        uncomment = re.sub(r'//.*', '', json_str)
        config = json.loads(uncomment)
        # config = jsonc.loads(f.read())
        f.close()
    return config
