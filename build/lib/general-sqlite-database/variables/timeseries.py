from ..variable import Variable

class Timeseries:

    @classmethod
    def load(cls, name: str, data_object, hash: str = None) -> 'Timeseries':
        """
        Load the timeseries variable from the database.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def save(self, name: str, data_object) -> None:
        """
        Save the timeseries variable to the database.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")