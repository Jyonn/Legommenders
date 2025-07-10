class Meaner:
    """
    A utility class to calculate the running mean of a sequence of values.
    The `Meaner` class allows you to add new values incrementally and updates the mean in constant time.
    """

    def __init__(self):
        """
        Initializes the `Meaner` object with an initial mean of 0 and a count of 0.
        """
        self.mean = 0  # The current mean of the values added so far
        self.count = 0  # The number of values added so far

    def add(self, value):
        """
        Adds a new value to the mean calculation and updates the running mean.

        Args:
            value (float or int): The new value to incorporate into the mean.
        """
        # Update the mean using the formula for running mean:
        # new_mean = (current_mean * current_count + new_value) / (current_count + 1)
        self.mean = (self.mean * self.count + value) / (self.count + 1)
        # Increment the count of values
        self.count += 1

    def __call__(self, value):
        """
        Makes the `Meaner` object callable, allowing new values to be added directly, and returns the updated mean.

        Args:
            value (float or int): The new value to incorporate into the mean.

        Returns:
            float: The updated running mean.
        """
        self.add(value)  # Add the new value to the mean calculation
        return self.mean  # Return the updated mean
