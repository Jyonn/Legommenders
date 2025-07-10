import json
import torch
from utils.iterating import Iterating


class TensorShape:
    """
    A class to represent the shape and data type of a PyTorch tensor.
    """

    def __init__(self, shape, dtype):
        """
        Initializes a TensorShape instance.

        Args:
            shape (torch.Size): The shape of the tensor.
            dtype (torch.dtype): The data type of the tensor.
        """
        self.shape = list(shape)  # Convert shape to a list for easier manipulation
        self.dtype = dtype  # Store the tensor's data type

    def __str__(self):
        """
        Returns a string representation of the TensorShape.

        Returns:
            str: A human-readable representation of the tensor shape and data type.
        """
        return f'tensor({self.shape}, dtype={self.dtype})'

    def __repr__(self):
        """
        Returns the string representation for debugging.

        Returns:
            str: The string representation of the TensorShape.
        """
        return self.__str__()


class ListShape:
    """
    A class to represent the shape of a nested list.
    """

    def __init__(self, data):
        """
        Initializes a ListShape instance by analyzing the shape of a nested list.

        Args:
            data (list): The input nested list to analyze.
        """
        self.shape = []  # Store the dimensions of the list
        while isinstance(data, list):
            self.shape.append(len(data))  # Append the size of each dimension
            data = data[0]  # Move to the next nested level

    def __str__(self):
        """
        Returns a string representation of the list shape.

        Returns:
            str: A human-readable representation of the list shape.
        """
        return f'list({self.shape})'

    def __repr__(self):
        """
        Returns the string representation for debugging.

        Returns:
            str: The string representation of the ListShape.
        """
        return self.__str__()


class Structure(Iterating):
    """
    A class to analyze and represent the structure of nested data, such as dictionaries,
    lists, and tensors.
    """

    def __init__(self, use_shape=False):
        """
        Initializes a Structure instance.

        Args:
            use_shape (bool): Whether to include detailed shape information for tensors and lists.
        """
        super().__init__()
        self.use_shape = use_shape  # Flag to control whether to use detailed shapes

    def custom_worker(self, x):
        """
        Processes non-iterable elements in the data structure.

        Args:
            x (any): The input element to process.

        Returns:
            any: The processed representation of the element.
        """
        if isinstance(x, torch.Tensor):
            # Handle PyTorch tensors
            if self.use_shape:
                return TensorShape(x.shape, x.dtype)  # Return detailed shape object
            return f'tensor({list(x.shape)}, dtype={x.dtype})'  # Return summarized shape
        elif isinstance(x, list):
            # Handle nested lists
            shape = ListShape(x)  # Analyze the list shape
            if self.use_shape:
                return shape  # Return detailed shape object
            return str(shape)  # Return summarized shape as a string
        else:
            # For other types, return their type name
            return type(x).__name__

    def worker(self, x):
        """
        Recursively processes the input data structure.

        Args:
            x (any): The input data to process.

        Returns:
            any: The processed data structure.
        """
        if isinstance(x, dict):
            # Process dictionaries recursively
            return self.worker_dict(x)
        return self.custom_worker(x)  # Process non-iterable elements

    def analyse(self, x):
        """
        Analyzes the structure of the input data.

        Args:
            x (any): The input data to analyze.

        Returns:
            any: The analyzed structure.
        """
        return self.worker(x)

    def analyse_and_stringify(self, x):
        """
        Analyzes the structure and returns it as a JSON-formatted string.

        Args:
            x (any): The input data to analyze.

        Returns:
            str: The JSON-formatted string representation of the analyzed structure.

        Raises:
            AssertionError: If `use_shape` is True, as stringification is not supported in that mode.
        """
        assert not self.use_shape, 'Cannot stringify shape'  # Ensure `use_shape` is False
        structure = self.analyse(x)  # Analyze the structure
        return json.dumps(structure, indent=4)  # Convert to JSON string


if __name__ == '__main__':
    # Example nested data structure
    a = dict(
        x=torch.rand(3, 5, 6).tolist(),  # Convert tensor to nested list
        y=dict(
            z=torch.rand(3, 6).tolist(),  # Convert tensor to nested list
            k=[torch.rand(3, 2, 6).tolist()]  # List containing nested tensor as a list
        )
    )

    # Analyze the structure of the data with detailed shapes
    print(Structure(use_shape=True).analyse(a))
