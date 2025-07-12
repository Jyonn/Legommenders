"""
utils.py

A collection of small helper utilities that are reused across the project.
The tools provided here include:

1. Merging dictionaries with default values (`combine_config`)
2. Setting global random seeds for reproducibility (`seeding`)
3. A very light-weight command line argument parser (`argparse`)
4. Building a short, reproducible “experiment signature” (`get_signature`)
5. Generating a random alphanumeric string (`get_random_string`)
6. Checking Python package requirements at runtime (`package_require`)
"""

import base64
import hashlib
import json
import os
import random
import string
import sys
from typing import Optional, List, Dict, Any

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------
def combine_config(config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Add key–value pairs into an existing configuration dictionary
    *only* if they do not already exist.

    Parameters
    ----------
    config : Dict[str, Any]
        The user-supplied configuration dictionary that may already contain
        some keys.
    **kwargs
        Additional default parameters and their default values.

    Returns
    -------
    Dict[str, Any]
        The same dictionary object with missing keys filled in.
    """
    for k, v in kwargs.items():
        if k not in config:
            config[k] = v
    return config


# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------
def seeding(seed: int = 2023) -> None:
    """
    Set random seeds for Python, NumPy and PyTorch in a best-effort manner
    so that experiments become (more) deterministic.

    NOTE: Full determinism is not guaranteed on all GPU kernels.

    Parameters
    ----------
    seed : int, default=2023
        The global random seed value to use.
    """
    random.seed(seed)                           # Python’s built-in RNG
    os.environ["PYTHONHASHSEED"] = str(seed)     # Hash-based functions
    np.random.seed(seed)                         # NumPy RNG
    torch.manual_seed(seed)                      # PyTorch CPU RNG
    torch.cuda.manual_seed(seed)                 # PyTorch current GPU RNG
    torch.backends.cudnn.deterministic = True    # Make certain CuDNN ops deterministic
    # TensorFlow users could call: tensorflow.random.set_seed(seed)


# ---------------------------------------------------------------------------
# Minimal command line argument parser
# ---------------------------------------------------------------------------
def argparse(arguments: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Parse a list of command-line arguments of the form
    `--key value --another_key value2 ...` into a dictionary.

    Type inference rules (in order of precedence):
        null      -> None
        int       -> int(...)
        true/false-> bool(...)
        float     -> float(...)   (fallback)
        str       -> leave as str

    This tiny parser is *not* as powerful as the standard `argparse` module
    but is convenient for quick scripts.

    Parameters
    ----------
    arguments : List[str] | None
        The list of argument tokens. If `None` (default) we use `sys.argv[1:]`.

    Returns
    -------
    Dict[str, Any]
        Parsed key–value pairs with best-effort type casting.
    """
    arguments = arguments or sys.argv[1:]
    kwargs: Dict[str, Any] = {}

    key: Optional[str] = None
    for arg in arguments:
        if key is not None:
            # Current token is the *value* for the previous `--key`
            kwargs[key] = arg
            key = None
        else:
            # Current token should start with '--' indicating a *key*
            assert arg.startswith('--'), f"Unexpected token {arg}, expecting a key starting with '--'."
            key = arg[2:]

    # ---------------------------------------------------------------------
    # Best-effort type conversion
    # ---------------------------------------------------------------------
    for key, value in kwargs.items():
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
                # Attempt to parse as float
                kwargs[key] = float(value)
            except ValueError:
                # Keep as string
                pass
    return kwargs


# ---------------------------------------------------------------------------
# Experiment signature helpers
# ---------------------------------------------------------------------------
def get_signature(data, embed, model, exp) -> str:
    """
    Create a short (8-char) deterministic signature string that uniquely
    identifies an experimental configuration.

    The signature is built by:
       1. Calling the provided callables to obtain *configurations*.
       2. Serialising them as JSON with sorted keys.
       3. Computing MD5 digest of the JSON string.
       4. Base64-url-safe encoding the digest.
       5. Returning the first 8 characters.

    Parameters
    ----------
    data, embed, model, exp : Callable[[], Any]
        Zero-argument callables that return JSON-serializable objects
        representing the configuration of each component.

    Returns
    -------
    str
        An 8-character base64URL string that serves as the experiment ID.
    """
    configuration = {
        'data': data(),
        'embed': embed(),
        'model': model(),
        'exp': exp(),
    }

    # Convert to canonical JSON (sorted keys, no ASCII escaping)
    canonical_str = json.dumps(configuration, sort_keys=True, ensure_ascii=False)

    # MD5 -> 128-bit digest
    md5_digest = hashlib.md5(canonical_str.encode('utf-8')).digest()

    # Encode using URL-safe base64 and strip '=' padding
    b64_str = base64.urlsafe_b64encode(md5_digest).decode('utf-8').rstrip('=')

    # Use only the first 8 chars for brevity
    return b64_str[:8]


# ---------------------------------------------------------------------------
# Miscellaneous helpers
# ---------------------------------------------------------------------------
def get_random_string(length: int = 6) -> str:
    """
    Generate a random alphanumeric string (uppercase+lowercase letters
    and digits) of a given length.

    Parameters
    ----------
    length : int, default=6
        Desired length of the resulting string.

    Returns
    -------
    str
        Randomly generated string.
    """
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def package_require(name: str, version: Optional[str] = None) -> None:
    """
    Runtime dependency checker. Ensures that a package is installed and,
    optionally, that it meets the minimum version requirement.

    Parameters
    ----------
    name : str
        The package name as known by `pip` / `pkg_resources`.
    version : str | None
        A minimum required version string (e.g. "1.2.3"). If `None`, any
        version is accepted.

    Raises
    ------
    ImportError
        If the package is missing, or if its version is older than required.
    """
    import pkg_resources

    try:
        installed_version = pkg_resources.get_distribution(name).version
    except pkg_resources.DistributionNotFound:
        raise ImportError(f"Package {name} not found. Please use 'pip install {name}' to install.")

    if version is not None:
        # NOTE: `pkg_resources.parse_version` could be used for more robust
        # version comparison, but string comparison is often sufficient for
        # x.y.z style semantic versions.
        if installed_version < version:
            raise ImportError(
                f"Package {name} version is {installed_version}, but at least {version} is required. "
                f"Please use 'pip install {name} -U' to upgrade."
            )
