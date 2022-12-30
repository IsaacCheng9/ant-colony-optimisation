import numpy as np


class AntColony:
    def __init__(
        self,
        num_trials: int,
        num_ant_paths: int,
        evaporation_rate: float,
    ):
        self.num_trials = num_trials
        self.num_ant_paths = num_ant_paths
        self.evaporation_rate = evaporation_rate

        # Each element of the distance matrix represents the distance between
        # two locations, and each element of the flow matrix represents the
        # number of students that flow between two locations.
        (
            self.num_locations,
            self.distance_matrix,
            self.flow_matrix,
        ) = self.load_data_file("data/Uni50a.dat")

    def load_data_file(self, file_name: str) -> tuple:
        """
        Load a data file that contains the number of locations, the distance
        matrix, and the flow matrix.

        Arguments:
            file_name: The name of the file to get the data from.

        Returns:
            A tuple of the number of locations, and two 2D numpy arrays
            containing the distance matrix and the flow matrix.
        """
        first_line = np.loadtxt(file_name, ndmin=2, usecols=(0))
        num_locations = int(first_line[0, 0])
        matrix_data = np.loadtxt(file_name, skiprows=1, unpack=True, ndmin=2)
        distance_matrix = matrix_data[0:, 0:num_locations]
        flow_matrix = matrix_data[0:, num_locations:]

        return (num_locations, distance_matrix, flow_matrix)


if __name__ == "__main__":
    ant_colony = AntColony(5, 100, 0.90)
    print(ant_colony.num_locations)
    print(ant_colony.distance_matrix)
    print(ant_colony.flow_matrix)
