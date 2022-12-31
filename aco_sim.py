"""
Ant colony optimisation to solve the quadratic assignment problem.
"""
import random
import time
from typing import List, Tuple

import numpy as np


class AntColonySimulation:
    """
    Simulation of an ant colony.
    """

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
        ) = load_data_file("data/Uni50a.dat")

    def generate_ant_paths(self, num_ant_paths: int) -> List[List[int]]:
        """
        Simulate each ant randomly selecting a path between the locations.

        Args:
            num_ant_paths: The number of ant paths to generate.

        Returns:
            A list of ant paths that contain numerical locations.
        """
        # TODO: Implement this with numpy.
        paths = []

        # TODO: Generate paths based on probabilities that can be calculated from the pheromone matrix.
        for _ in range(num_ant_paths):
            path = list(range(self.num_locations))
            random.shuffle(path)
            paths.append(path)

        return paths

    def calculate_fitness(self, path: list) -> float:
        """
        Calculate the fitness of a path.

        Args:
            path: The path to calculate the fitness of.

        Returns:
            The fitness of the path.
        """
        fitness = 0
        path.append(path[0])
        for i in range(len(path) - 1):
            for j in range(len(path) - 1):
                fitness += (
                    self.distance_matrix[path[i], path[j]]
                    * self.flow_matrix[path[i], path[j]]
                )
            # fitness += (
            #     self.distance_matrix[path[i], path[i + 1]]
            #     * self.flow_matrix[path[i], path[i + 1]]
            # )

        return fitness

    def update_pheromone(self, pheromone: np.ndarray, paths: list) -> np.ndarray:
        """
        Update the pheromone values for the paths.

        Args:
            pheromone: The pheromone matrix for the paths.
            paths: The paths to update the pheromone values for.

        Returns:
            The pheromone matrix after updating the pheromone values.
        """
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
        best_fitness = float("inf")
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
        best_fitness = float("inf")
        for i in range(num_evaluations):
            print(f"Iteration {i} - Current best fitness: {best_fitness}")
            best_fitness = min(best_fitness, self.run_aco_evaluation())

        return best_fitness

    def run_trials(self, num_evaluations_per_trial: int) -> List[float]:
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


def load_data_file(file_name: str) -> Tuple[int, np.ndarray, np.ndarray]:
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


if __name__ == "__main__":
    start = time.perf_counter()
    NUM_TRIALS = 5
    NUM_EVALUATIONS_PER_TRIAL = 10000
    # (num_ant_paths, evaporation_rate)
    configs = [
        (100, 0.90),
        (100, 0.50),
        (10, 0.90),
        (10, 0.50),
    ]

    for index, config in enumerate(configs):
        m, e = config
        print(f"Running experiment with m = {m}, e = {e}...")
        aco_sim = AntColonySimulation(NUM_TRIALS, num_ant_paths=m, evaporation_rate=e)
        fitness_results = aco_sim.run_trials(NUM_EVALUATIONS_PER_TRIAL)
        print(f"Experiment {index + 1} (m = {m}, e = {e}):" f"{fitness_results}\n")

    end = time.perf_counter()
    print(f"\nTime taken: {end - start} seconds")
