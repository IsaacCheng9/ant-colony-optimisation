"""
Ant colony optimisation to solve the quadratic assignment problem.
"""
import time
from typing import List, Tuple

import numpy as np


class AntColonyQAPSimulation:
    """
    Simulation of an ant colony for the quadratic assignment problem (QAP).
    """

    def __init__(
        self,
        num_trials: int,
        num_evaluations_per_trial: int,
        num_locations: int,
        distance_matrix: np.ndarray,
        flow_matrix: np.ndarray,
        num_ant_paths: int,
        evaporation_rate: float,
    ):
        self.num_trials = num_trials
        self.num_evaluations_per_trial = num_evaluations_per_trial
        self.num_locations = num_locations
        # Each element of the distance matrix represents the distance between
        # two locations, and each element of the flow matrix represents the
        # number of students that flow between two locations.
        self.distance_matrix = distance_matrix
        self.flow_matrix = flow_matrix
        self.num_ant_paths = num_ant_paths
        self.evaporation_rate = evaporation_rate
        # Randomly distribute small amounts of pheromone between 0 and 1 on
        # the construction graph.
        self.pheromone_matrix = np.random.uniform(
            0, 1, (self.num_locations, self.num_locations)
        )

    def choose_next_facility(self, ant_path: np.ndarray, row: int) -> int:
        """
        Choose the next facility to assign to a location based on the pheromone
        levels.

        Args:
            ant_path: The path that the ant has already taken.
            row: The row of the pheromone matrix to use.

        Returns:
            The index of the next facility to assign to a location.
        """
        total_pheromone_in_next_paths = 0
        for i in range(self.num_locations):
            if i in ant_path:
                continue
            total_pheromone_in_next_paths += self.pheromone_matrix[row][i]

        # Assign probabilities for each location based on the pheromone levels.
        probabilities = np.empty(self.num_locations, dtype="float64")
        for i in range(self.num_locations):
            # If the facility has already been assigned a location, ensure that
            # it can't be selected in the future.
            if i in ant_path:
                probabilities[i] = 0
            # Calculate the probability of selecting this facility based on
            # pheromone levels.
            else:
                next_facility_pheromone = self.pheromone_matrix[row][i]
                probabilities[i] = (
                    next_facility_pheromone / total_pheromone_in_next_paths
                )

        # Choose the next facility based on the weighted probability.
        probabilities /= np.sum(probabilities)
        next_facility = np.random.choice(
            list(range(self.num_locations)), p=probabilities
        )

        return next_facility

    def generate_ant_path(self) -> Tuple[np.ndarray, float]:
        """
        Generate a path for an ant to follow using probabilities based on
        pheromone levels.

        Returns:
            The path that the ant took and the fitness of the path.
        """
        ant_path = np.array([self.num_locations + 1] * self.num_locations)
        for i in range(self.num_locations):
            ant_path[i] = self.choose_next_facility(ant_path, i)
        ant_fitness = self.calculate_fitness(ant_path)
        return ant_path, ant_fitness

    def calculate_fitness(self, ant_path: np.ndarray) -> float:
        """
        Calculate the fitness of a path.

        Args:
            ant_path: The path to calculate the fitness of.

        Returns:
            The fitness of the path.
        """
        fitness = 0.0
        for i in range(self.num_locations):
            for j in range(self.num_locations):
                fitness += (
                    self.distance_matrix[ant_path[i]][ant_path[j]]
                    * self.flow_matrix[i][j]
                )
        return fitness

    def update_pheromone_matrix(
        self, ant_paths: np.ndarray, ant_fitnesses: np.ndarray
    ) -> None:
        """
        Update the pheromone matrix values for the paths.

        Args:
            ant_paths: The paths to update the pheromone values for.
            ant_fitnesses: The fitnesses of the paths that we use to update the
                       pheromone values.
        """
        for i in range(self.num_ant_paths):
            for j in range(self.num_locations):
                self.pheromone_matrix[j][ant_paths[i][j]] += 1 / ant_fitnesses[i]

    def evaporate_pheromone(self) -> None:
        """
        Evaporate the pheromone values according to the evaporation rate.
        """
        self.pheromone_matrix *= self.evaporation_rate

    def run_trial(self) -> Tuple[float, np.ndarray]:
        """
        Run the ant colony optimisation algorithm for a number of fitness
        evaluations.

        Returns:
            The best fitness and the best ant path found after finishing all
            evaluations.
        """
        best_fitness = float("inf")
        best_ant_path = np.array([self.num_locations + 1] * self.num_locations)

        for i in range(self.num_evaluations_per_trial):
            ant_paths = np.empty((self.num_ant_paths, self.num_locations), dtype=int)
            ant_fitnesses = np.empty(self.num_ant_paths, dtype=float)

            # Generate the path for each ant to search for the best path and
            # fitness.
            for j in range(self.num_ant_paths):
                ant_path, ant_fitness = self.generate_ant_path()
                ant_paths[j], ant_fitnesses[j] = ant_path, ant_fitness
                if ant_fitness < best_fitness:
                    best_fitness = ant_fitness
                    best_ant_path = ant_path

            self.update_pheromone_matrix(ant_paths, ant_fitnesses)
            self.evaporate_pheromone()
            print(f"Iteration {i} - current best fitness: {best_fitness}")

        return best_fitness, best_ant_path

    def run_experiment(self) -> Tuple[List[float], List[np.ndarray]]:
        """
        Run the ant colony optimisation algorithm for a number of trials.

        Returns:
            A tuple containing the best fitnesses and the best ant paths found
            after finishing all trials.
        """
        print(
            "Running experiment with "
            f"m = {self.num_ant_paths}, e = {self.evaporation_rate}..."
        )
        best_fitnesses = []
        best_ant_paths = []

        for _ in range(self.num_trials):
            # Reset the pheromone matrix to ensure each trial is independent.
            self.pheromone_matrix = np.random.uniform(
                0, 1, (self.num_locations, self.num_locations)
            )
            best_fitness, best_ant_path = self.run_trial()
            best_fitnesses.append(best_fitness)
            best_ant_paths.append(best_ant_path)

        print(
            f"Experiment {index + 1} (m = {self.num_ant_paths}, "
            f"e = {self.evaporation_rate}):"
        )
        print(f"Best Fitnesses: {best_fitnesses}")
        print(f"Best Ant Paths: {best_ant_paths}")
        return best_fitnesses, best_ant_paths


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
    first_line = np.loadtxt(file_name, ndmin=2, usecols=0)
    num_locations = int(first_line[0, 0])
    matrix_data = np.loadtxt(file_name, skiprows=1, unpack=True, ndmin=2)
    distance_matrix = matrix_data[0:, 0:num_locations]
    flow_matrix = matrix_data[0:, num_locations:]
    return num_locations, distance_matrix, flow_matrix


if __name__ == "__main__":
    # Set up the experiment configurations.
    start = time.perf_counter()
    locations, distances, flows = load_data_file("data/Uni50a.dat")
    NUM_TRIALS = 5
    NUM_EVALUATIONS_PER_TRIAL = 10_000
    # (num_ant_paths, evaporation_rate)
    configs = [
        (100, 0.90),
        (100, 0.50),
        (10, 0.90),
        (10, 0.50),
    ]

    # Run the experiments.
    best_fitness_results = []
    best_ant_path_results = []
    for index, (m, e) in enumerate(configs):
        aco_sim = AntColonyQAPSimulation(
            num_trials=NUM_TRIALS,
            num_evaluations_per_trial=NUM_EVALUATIONS_PER_TRIAL,
            num_locations=locations,
            distance_matrix=distances,
            flow_matrix=flows,
            num_ant_paths=m,
            evaporation_rate=e,
        )
        best_fitnesses_res, best_ant_paths_res = aco_sim.run_experiment()
        best_fitness_results.append(best_fitnesses_res)
        best_ant_path_results.append(best_ant_paths_res)

    end = time.perf_counter()
    print(f"\nTime taken: {end - start} seconds\n")
    # Display the results of the experiments.
    print(
        f"\n\nResults of experiments with {NUM_TRIALS} trials and "
        f"{NUM_EVALUATIONS_PER_TRIAL} evaluations per trial:\n"
    )
    for index, (m, e) in enumerate(configs):
        print(
            f"\nExperiment {index + 1} (m = {m}, e = {e}):\n"
            f"    Best Fitness Results: {best_fitness_results[index]}\n"
            f"    Best Ant Path Results: {best_ant_path_results[index]}"
        )
