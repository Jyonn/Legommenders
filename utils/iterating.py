class Iterating:
    """
    A base class designed for recursively processing elements in nested data structures like dictionaries, lists, tuples, or sets.
    This class provides a framework for applying custom operations on data through the `custom_worker` method,
    which can be overridden in subclasses for specific use cases.
    """

    def __init__(self, list_types=None):
        """
        Initializes the Iterating object.

        Args:
            list_types (list, optional): A list of types to be treated as iterable (like list, tuple, set).
                                         Defaults to [list, tuple, set] if not provided.
        """
        self.list_types = list_types or [list, tuple, set]

    def worker_dict(self, d: dict):
        """
        Recursively processes a dictionary by applying the `worker` method to all its values.

        Args:
            d (dict): The input dictionary to process.

        Returns:
            dict: A new dictionary with the same keys but transformed values.
        """
        return {k: self.worker(d[k]) for k in d}

    def worker_list(self, l: list):
        """
        Recursively processes a list (or similar iterable) by applying the `worker` method to all its elements.

        Args:
            l (list): The input list to process.

        Returns:
            list: A new list with transformed elements.
        """
        return [self.worker(x) for x in l]

    def is_list(self, x):
        """
        Checks if the given object `x` is of a type that is considered iterable (e.g., list, tuple, set).

        Args:
            x: The object to check.

        Returns:
            bool: True if `x` is an instance of any type in `self.list_types`, otherwise False.
        """
        for t in self.list_types:
            if isinstance(x, t):
                return True
        return False

    def custom_worker(self, x):
        """
        A placeholder method meant to be overridden in subclasses.
        This method defines the custom operation to be performed on non-iterable elements.

        Args:
            x: The input element to process.

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        raise NotImplementedError

    def worker(self, x):
        """
        A recursive method that processes the input `x` based on its type:
        - If it's a dictionary, it processes its values using `worker_dict`.
        - If it's an iterable (list, tuple, set), it processes its elements using `worker_list`.
        - Otherwise, it applies the `custom_worker` method to the element.

        Args:
            x: The input element to process (can be of any type).

        Returns:
            The processed element, with transformations applied recursively.
        """
        if isinstance(x, dict):
            return self.worker_dict(x)
        elif self.is_list(x):
            return self.worker_list(x)
        return self.custom_worker(x)
