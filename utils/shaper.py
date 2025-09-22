import torch

from utils.iterating import Iterating
from utils.structure import Structure


class Reshaper(Iterating):
    """
    Reshaper is a subclass of Iterating designed to reshape tensors in a nested data structure.
    It flattens the middle dimensions of tensors while preserving the batch size.
    """

    def __init__(self, batch_size):
        """
        Initializes the Reshaper with a given batch size.

        Args:
            batch_size (int or None): The size of the batch. If None, it will be inferred from the first tensor processed.
        """
        super().__init__()
        self.batch_size = batch_size

    def custom_worker(self, x):
        """
        Defines the custom transformation to be applied to non-iterable elements (e.g., tensors).
        For tensors, it reshapes them to have shape [-1, last_dimension], where -1 infers the size dynamically.

        Args:
            x (any): The input element to process.

        Returns:
            torch.Tensor: The reshaped tensor.

        Raises:
            AssertionError: If the batch size of the tensor does not match the expected batch size.
        """
        if not isinstance(x, torch.Tensor):
            return None

        if self.batch_size is None:
            # Infer batch size from the first tensor processed
            self.batch_size = x.shape[0]
        else:
            # Ensure the batch size matches the expected size
            assert self.batch_size == x.shape[0], 'Batch size mismatch'
        return x.view(-1, x.shape[-1])  # Flatten all but the last dimension


class Recover(Iterating):
    """
    Recover is a subclass of Iterating designed to reshape flattened tensors back to their original batch size.
    """

    def __init__(self, batch_size):
        """
        Initializes the Recover with a given batch size.

        Args:
            batch_size (int): The size of the batch to restore.
        """
        super().__init__()
        self.batch_size = batch_size

    def custom_worker(self, x):
        """
        Defines the custom transformation to be applied to non-iterable elements (e.g., tensors).
        For tensors, it reshapes them to have shape [batch_size, -1, last_dimension].

        Args:
            x (any): The input element to process.

        Returns:
            torch.Tensor: The reshaped tensor.
        """
        if isinstance(x, torch.Tensor):
            return x.view(self.batch_size, -1, x.shape[-1])  # Restore the batch size


class Shaper:
    """
    Shaper is a utility class that handles reshaping and recovering nested data structures containing tensors.
    It uses the `Reshaper` to flatten tensors and the `Recover` class to restore the original structure.
    """

    def __init__(self):
        """
        Initializes the Shaper with no predefined structure and a Reshaper with a dynamic batch size.
        """
        self.structure = None
        self.reshaper = Reshaper(batch_size=None)

    def transform(self, data: any):
        """
        Transforms the data by reshaping tensors in the nested structure.
        It also analyzes and stores the structure of the input data for later recovery.

        Args:
            data (any): The input data to be transformed.

        Returns:
            any: The transformed data with reshaped tensors.
        """
        self.reshaper.batch_size = None  # Reset batch size
        # Analyze the data structure and store it
        self.structure = Structure(use_shape=True).analyse(data)
        return self.reshaper.worker(data)

    def recover(self, data: any):
        """
        Recovers the original structure of the data by reshaping tensors back to their original dimensions.

        Args:
            data (any): The flattened data to be recovered.

        Returns:
            any: The recovered data with tensors restored to their original shapes.
        """
        recover = Recover(batch_size=self.reshaper.batch_size)  # Use the same batch size as the Reshaper
        return recover.worker(data)


if __name__ == '__main__':
    # Example usage of Shaper and related classes

    shaper = Shaper()

    # Example nested data structure containing tensors
    d = {
        "input_ids": {
            "title": torch.rand([64, 5, 23]),
            "cat": torch.rand([64, 5, 23]),
            "__cat_inputer_special_ids": torch.rand([64, 5, 23])
        },
        "attention_mask": torch.rand([64, 5, 23])
    }

    # Analyze and print the structure of the original data
    print(Structure().analyse_and_stringify(d))

    # Transform the data (flatten tensors)
    d = shaper.transform(d)

    # Analyze and print the structure of the transformed data
    print(Structure().analyse_and_stringify(d))

    # Simulate some transformation on the flattened data
    d = torch.rand(64 * 5, 768)

    # Analyze and print the structure of the modified flattened data
    print(Structure().analyse_and_stringify(d))

    # Recover the original structure of the data
    d = shaper.recover(d)

    # Analyze and print the structure of the recovered data
    print(Structure().analyse_and_stringify(d))
