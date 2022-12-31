import math
import random
import time

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

        Args:
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

    def calculate_fitness(self, path: list) -> float:
        """
        Calculate the fitness of a path.

        Args:
            The path to calculate the fitness of.

        Returns:
            The fitness of the path.
        """
        fitness = 0
        path.append(path[0])
        for i in range(len(path) - 1):
            fitness += (
                self.distance_matrix[path[i], path[i + 1]]
                * self.flow_matrix[path[i], path[i + 1]]
            )
        return fitness

    def update_pheromone(self, pheromone: np.ndarray, paths: list) -> np.ndarray:
        for path in paths:
            fitness = self.calculate_fitness(path)
            for i in range(len(path)):
                pheromone[path[i], path[(i + 1) % len(path)]] += 1 / fitness
        return pheromone

    def evaporate_pheromone(self, pheromone: np.ndarray) -> np.ndarray:
        """
        Evaporate the pheromone values according to the evaporation rate.

        Args:
            pheromone: The pheromone matrix for the paths.

        Returns:
            The updated pheromone matrix after evaporation has occurred.
        """
        pheromone *= self.evaporation_rate
        return pheromone

    def run_aco_evaluation(self) -> float:
        """
        Run the ant colony optimisation algorithm.

        Returns:
            The best fitness found after finishing the evaluation.
        """
        # Randomly distribute small amounts of pheromone between 0 and 1 on the
        # construction graph.
        pheromone = np.random.uniform(0, 1, (self.num_locations, self.num_locations))
        paths = self.generate_ant_paths(self.num_ant_paths)
        self.update_pheromone(pheromone, paths)
        self.evaporate_pheromone(pheromone)
        best_fitness = math.inf
        for path in paths:
            fitness = self.calculate_fitness(path)
            best_fitness = min(best_fitness, fitness)

        return best_fitness

    def run_fitness_evaluations(self, num_evaluations: int) -> float:
        """
        Run the ant colony optimisation algorithm for a number of fitness
        evaluations.

        Args:
            num_evaluations: The number of fitness evaluations to run the
                             algorithm for.

        Returns:
            The best fitness found after finishing all evaluations.
        """
        best_fitness = math.inf
        for _ in range(num_evaluations):
            best_fitness = min(best_fitness, self.run_aco_evaluation())

        return best_fitness

    def run_trials(self, num_evaluations_per_trial: int) -> list:
        """
        Run the ant colony optimisation algorithm for a number of trials.

        Args:
            num_evaluations_per_trial: The number of fitness evaluations to run
                                       the algorithm for each trial.
        Returns:
            A list of the best fitnesses found after finishing all trials.
        """
        best_fitnesses = []
        for _ in range(self.num_trials):
            best_fitnesses.append(
                self.run_fitness_evaluations(num_evaluations_per_trial)
            )

        return best_fitnesses


if __name__ == "__main__":
    start = time.perf_counter()
    num_evaluations_per_trial = 10000

    ant_colony1 = AntColony(num_trials=5, num_ant_paths=100, evaporation_rate=0.90)
    print(ant_colony1.num_locations)
    print(ant_colony1.distance_matrix)
    print(ant_colony1.flow_matrix)
    print(f"Experiment 1: {ant_colony1.run_trials(num_evaluations_per_trial)}")

    ant_colony2 = AntColony(num_trials=5, num_ant_paths=100, evaporation_rate=0.50)
    print(ant_colony2.num_locations)
    print(ant_colony2.distance_matrix)
    print(ant_colony2.flow_matrix)
    print(f"Experiment 2: {ant_colony2.run_trials(num_evaluations_per_trial)}")

    ant_colony3 = AntColony(num_trials=5, num_ant_paths=10, evaporation_rate=0.90)
    print(ant_colony3.num_locations)
    print(ant_colony3.distance_matrix)
    print(ant_colony3.flow_matrix)
    print(f"Experiment 3: {ant_colony3.run_trials(num_evaluations_per_trial)}")

    ant_colony4 = AntColony(num_trials=5, num_ant_paths=10, evaporation_rate=0.50)
    print(ant_colony4.num_locations)
    print(ant_colony4.distance_matrix)
    print(ant_colony4.flow_matrix)
    print(f"Experiment 4: {ant_colony4.run_trials(num_evaluations_per_trial)}")

    end = time.perf_counter()
    print(f"Time taken: {end - start} seconds")
