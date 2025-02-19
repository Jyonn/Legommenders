import base64
import hashlib
import json
import os
import random
import string
import sys
from typing import Optional

import numpy as np
import torch
from oba import Obj


def combine_config(config: dict, **kwargs):
    for k, v in kwargs.items():
        if k not in config:
            config[k] = v
    return config


def seeding(seed=2023):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # tensorflow.random.set_seed(seed)


def argparse(arguments=None):
    arguments = arguments or sys.argv[1:]
    kwargs = {}

    key: Optional[str] = None
    for arg in arguments:
        if key is not None:
            kwargs[key] = arg                                          
            key = None
        else:
            assert arg.startswith('--')
            key = arg[2:]

    for key, value in kwargs.items():  # type: str, str
        if value == 'null':
            kwargs[key] = None
        elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
            kwargs[key] = int(value)
        elif value.lower() == 'true':
            kwargs[key] = True
        elif value.lower() == 'false':
            kwargs[key] = False
        else:
            try:
                kwargs[key] = float(value)
            except ValueError:
                pass
    return kwargs


def get_signature(data, embed, model, exp):
    configuration = {
        'data': Obj.raw(data),
        'embed': Obj.raw(embed),
        'model': Obj.raw(model),
        'exp': Obj.raw(exp),
    }
    canonical_str = json.dumps(configuration, sort_keys=True, ensure_ascii=False)
    md5_digest = hashlib.md5(canonical_str.encode('utf-8')).digest()
    b64_str = base64.urlsafe_b64encode(md5_digest).decode('utf-8').rstrip('=')
    return b64_str[:8]


def get_random_string(length=6):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def package_require(name, version=None):
    import pkg_resources

    try:
        installed_version = pkg_resources.get_distribution(name).version
    except pkg_resources.DistributionNotFound:
        raise ImportError(f"Package {name} not found. Please use 'pip install {name}' to install.")

    if version is not None:
        if installed_version < version:
            raise ImportError(f"Package {name} version is {installed_version}, but require version {version}. Please use 'pip install {name} -U' to upgrade.")
