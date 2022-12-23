import numpy as np


class AntColony:
    def __init__(
        self,
        num_locations: int,
        num_ant_paths: int,
        num_trials: int,
        evaporation_rate: float,
    ):
        self.num_locations = num_locations
        self.num_ant_paths = num_ant_paths
        self.num_trials = num_trials
        self.evaporation_rate = evaporation_rate

        # Each element of the distance matrix represents the distance between
        # two locations, and each element of the flow matrix represents the
        # number of students that flow between two locations.
        self.distance_matrix, self.flow_matrix = self.process_dataset_file(
            "dataset.txt"
        )

    def process_dataset_file(self, file_name) -> tuple:
        """
        Processes a .txt file that contains the number of locations, the
        distance matrix, and the flow matrix.

        Arguments:
            file_name: The name of the .txt file to get the data from.
        Returns:
            A tuple of numpy arrays containing the distance matrix and the
            flow matrix.
        """
        return ()


def main():
    pass


if __name__ == "__main__":
    main()
