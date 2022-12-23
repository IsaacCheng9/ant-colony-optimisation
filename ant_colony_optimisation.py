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
        ) = self.process_dataset_file("data/dataset.txt")

    def process_dataset_file(self, file_name: str) -> tuple:
        """
        Process a .txt file that contains the number of locations, the distance
        matrix, and the flow matrix.

        Arguments:
            file_name: The name of the .txt file to get the data from.

        Returns:
            A tuple of the number of locations, and two 2D numpy arrays
            containing the distance matrix and the flow matrix.
        """
        with open(file_name, "r") as file:
            file_lines = file.readlines()

        # The first line of the file contains the number of locations, followed
        # by a blank line.
        num_locations = int(file_lines[0])
        # The next num_locations lines contains the distance matrix, followed
        # by a blank line, then the flow matrix over num_locations lines.
        distance_matrix_start = 2
        flow_matrix_start = distance_matrix_start + num_locations + 1

        # Generate the distance matrix and flow matrix.
        distance_matrix = np.empty((num_locations, num_locations), dtype=int)
        flow_matrix = np.empty((num_locations, num_locations), dtype=int)
        for row in range(num_locations):
            distance_matrix[row] = np.fromstring(
                file_lines[row + distance_matrix_start], dtype=int, sep=" "
            )
        for row in range(num_locations):
            flow_matrix[row] = np.fromstring(
                file_lines[row + flow_matrix_start], dtype=int, sep=" "
            )

        return (num_locations, distance_matrix, flow_matrix)


def main():
    ant_colony = AntColony(5, 100, 0.90)
    print(ant_colony.num_locations)
    print(ant_colony.distance_matrix)
    print(ant_colony.flow_matrix)


if __name__ == "__main__":
    main()
