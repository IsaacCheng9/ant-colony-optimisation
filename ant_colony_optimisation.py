import numpy as np
import random


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

        # Randomly distribute small amounts of pheromone between 0 and 1 on the
        # construction graph.
        self.pheromone = np.random.uniform(
            0, 1, (self.num_locations, self.num_locations)
        )

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

    def generate_ant_paths(self, num_ant_paths: int) -> list:
        """
        Simulate each ant randomly selecting a path between the locations.

        Args:
            num_ant_paths: The number of ant paths to generate.

        Returns:
            A list of ant paths.
        """
        paths = []

        for _ in range(num_ant_paths):
            path = list(range(self.num_locations))
            random.shuffle(path)
            paths.append(path)

        return paths

    def evaporate_pheromone(self):
        """
        Evaporate the pheromone values according to the evaporation rate.
        """
        self.pheromone *= self.evaporation_rate


if __name__ == "__main__":
    ant_colony = AntColony(num_trials=5, num_ant_paths=100, evaporation_rate=0.90)
    print(ant_colony.num_locations)
    print(ant_colony.distance_matrix)
    print(ant_colony.flow_matrix)
    print(ant_colony.pheromone)
