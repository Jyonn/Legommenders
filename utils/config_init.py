import abc
import os.path
import warnings
from typing import cast

import refconfig
from oba import Obj
from refconfig import RefConfig

from utils import function, io

# Ensure required package versions are installed
function.package_require('refconfig', '0.1.2')
function.package_require('smartdict', '0.2.1')
function.package_require('unitok', '4.4.3')
function.package_require('oba', '0.1.1')
function.package_require('pigmento', '0.2.3')


class CommandInit:
    """
    A utility class for initializing and validating configurations from command-line arguments or other sources.
    """

    def __init__(self, required_args, default_args=None):
        """
        Initializes the CommandInit object with required and default arguments.

        Args:
            required_args (list): A list of argument names that are required.
            default_args (dict, optional): A dictionary of arguments with default values.
        """
        self.required_args = required_args
        self.default_args = default_args or {}

    def parse(self, kwargs=None):
        """
        Parses the arguments, validates required arguments, and applies default values.

        Args:
            kwargs (dict, optional): A dictionary of input arguments. If None, arguments are parsed from the command line.

        Returns:
            Obj: A parsed configuration object.
        """
        kwargs = kwargs or function.argparse()

        # Check for missing required arguments
        for arg in self.required_args:
            if arg not in kwargs:
                raise ValueError(f'miss argument {arg}')

        # Set default values for missing arguments
        for arg in self.default_args:
            if arg not in kwargs:
                kwargs[arg] = self.default_args[arg]

        # Create and parse a RefConfig object
        config = RefConfig().add(refconfig.CType.SMART, **kwargs).parse()
        config = Obj(config)  # Convert configuration to an object for easy attribute access

        return config


class ConfigInit(abc.ABC):
    """
    An abstract base class for managing configuration files.
    Subclasses must implement the `examples` method to provide example configurations.
    """
    _d: dict = None  # A class-level dictionary to store parsed configurations

    @classmethod
    def parse(cls):
        """
        Parses the configuration file associated with the subclass.

        Returns:
            dict: A dictionary containing the parsed configuration key-value pairs.

        Raises:
            Warning: If the configuration file does not exist, a warning is issued with example configurations.
        """
        if cls._d is not None:
            return cls._d  # Return cached configuration if already parsed

        cls._d = dict()  # Initialize an empty dictionary

        # Check if the configuration file exists
        if not os.path.exists(f'.{cls.classname()}'):
            warnings.warn(f'config file .{cls.classname()} not found')
            print(f'example .{cls.classname()} config:')
            print(cls.examples())  # Print example configurations
            return cls._d

        # Read and parse the configuration file
        # with open(f'.{cls.classname()}') as f:
        #     config = f.read()
        config = io.file_load(f'.{cls.classname()}')  # Load the configuration file content

        for line in config.strip().split('\n'):  # Process each line in the configuration file
            key, value = line.split('=')
            cls._d[key.strip().lower()] = cast(str, value).strip()  # Store key-value pairs in lowercase

        return cls._d

    @classmethod
    def get(cls, key, **kwargs):
        """
        Retrieves a value from the configuration dictionary by key.

        Args:
            key (str): The key to look up in the configuration.
            **kwargs: Optional arguments, including a `default` value if the key is not found.

        Returns:
            str: The value associated with the key.

        Raises:
            ValueError: If the key is not found and no default value is provided.
        """
        d = cls.parse()

        key = key.lower()  # Ensure the key is lowercase for case-insensitive matching
        if key not in d:
            # Return default value if provided, otherwise raise an error
            if 'default' in kwargs:
                return kwargs['default']
            raise ValueError(f'key {key} not found in config')

        return d[key]

    @classmethod
    def classname(cls):
        """
        Derives the class name for the configuration file by removing 'Init' and converting to lowercase.

        Returns:
            str: The derived class name.
        """
        return cls.__name__.lower().replace('init', '')

    @classmethod
    def examples(cls):
        """
        Provides example configurations for the subclass.

        Returns:
            str: Example configuration entries.

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        raise NotImplementedError


class DataInit(ConfigInit):
    """
    A subclass of ConfigInit for managing data-related configurations.
    """

    @classmethod
    def examples(cls):
        """
        Provides example paths for datasets.

        Returns:
            str: Example configuration entries for datasets.
        """
        return '\n'.join(['mind = /path/to/mind', 'goodreads = /path/to/goodreads'])


class ModelInit(ConfigInit):
    """
    A subclass of ConfigInit for managing model-related configurations.
    """

    @classmethod
    def examples(cls):
        """
        Provides example configurations for pre-trained models.

        Returns:
            str: Example configuration entries for models.
        """
        return '\n'.join(['bertbase = bert-base-uncased', 'bertlarge = bert-large-uncased'])


class AuthInit(ConfigInit):
    """
    A subclass of ConfigInit for managing authentication-related configurations.
    """

    @classmethod
    def examples(cls):
        """
        Provides example configurations for authentication.

        Returns:
            str: Example configuration entries for authentication credentials.
        """
        return '\n'.join(['user = user', 'password = password'])
