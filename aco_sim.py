"""
Ant colony optimisation to solve the quadratic assignment problem.
"""
import time
from typing import List, Tuple

import numpy as np


class AntColonyOptimisationForQAPSimulation:
    """
    Simulation of an ant colony optimisation for the quadratic assignment
    problem (QAP). Each experiment consists of a number of trials, and each
    trial consists of a number of evalutions according to the parameters.
    """

    def __init__(
        self,
        num_trials: int,
        num_evals_per_trial: int,
        num_locations: int,
        distance_matrix: np.ndarray,
        flow_matrix: np.ndarray,
        num_ant_paths: int,
        evaporation_rate: float,
    ):
        self.num_trials = num_trials
        self.num_evals_per_trial = num_evals_per_trial
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

    def calculate_total_pheromone_for_unassigned_facilities(
        self, location_index: int, ant_path_set: set
    ) -> float:
        """
        Calculate the total pheromone levels for facilities that haven't been
        assigned to a location yet according to the ant path set.

        Args:
            location_index: The row of the pheromone matrix that represents
                            the location we're choosing for.
            ant_path_set: A set of facilities that have already been assigned
                          to a location, used for fast look-ups.

        Returns:
            The total pheromone level for facilities that haven't been assigned
            to a location.
        """
        total_remaining_pheromone = 0.0
        for facility_index in range(self.num_locations):
            if facility_index in ant_path_set:
                continue
            total_remaining_pheromone += self.pheromone_matrix[location_index][
                facility_index
            ]
        return total_remaining_pheromone

    def calculate_probabilities_of_facilities(
        self, location_index: int, ant_path_set: set, total_remaining_pheromone: float
    ) -> np.ndarray:
        """
        Calculate probabilities for each facility based on the pheromone
        levels.

        Args:
            location_index: The row of the pheromone matrix that represents
                            the location we're choosing for.
            ant_path_set: A set of facilities that have already been assigned
                          to a location, used for fast look-ups.
            total_remaining_pheromone: The total pheromone level for facilities
                                       that haven't been assigned to a
                                       location.

        Returns:
            The probability of going to each facility from the current
            location.
        """
        probabilities = np.empty(self.num_locations, dtype=float)
        for facility_index in range(self.num_locations):
            # If the facility has already been assigned a location, ensure that
            # it can't be selected in the future.
            if facility_index in ant_path_set:
                probabilities[facility_index] = 0
            # Calculate the probability of selecting this facility based on
            # pheromone levels.
            else:
                facility_pheromone = self.pheromone_matrix[location_index][
                    facility_index
                ]
                probabilities[facility_index] = (
                    facility_pheromone / total_remaining_pheromone
                )
        return probabilities

    def randomly_choose_next_facility_for_location(
        self, location_index: int, ant_path_set: set
    ) -> int:
        """
        Randomly choose the next facility to assign to a location using
        weighted probabilities based on the pheromone levels.

        Args:
            location_index: The row of the pheromone matrix that represents
                            the location we're choosing for.
            ant_path_set: A set of facilities that have already been assigned
                          to a location, used for fast look-ups.

        Returns:
            The index of the next facility to assign to a location.
        """
        # Only count the total remaining pheromone for facilities that haven't
        # been assigned.
        total_remaining_pheromone = (
            self.calculate_total_pheromone_for_unassigned_facilities(
                location_index, ant_path_set
            )
        )
        # The probabilities of going to each facility is based on the pheromone
        # levels.
        probabilities = self.calculate_probabilities_of_facilities(
            location_index, ant_path_set, total_remaining_pheromone
        )
        # Choose the next facility based on the weighted probability.
        next_facility_index = np.random.choice(
            list(range(self.num_locations)), p=probabilities
        )
        return next_facility_index

    def generate_ant_path(self) -> np.ndarray:
        """
        Generate a path for an ant to follow using probabilities based on
        pheromone levels.

        Returns:
            The path that the ant took.
        """
        ant_path = np.zeros(self.num_locations, dtype=int)
        # Keep a separate set for the facilities that have been assigned for
        # O(1) rather than O(n) lookups.
        ant_path_set = set()
        for location_index in range(self.num_locations):
            facility_index = self.randomly_choose_next_facility_for_location(
                location_index, ant_path_set
            )
            ant_path[location_index] = facility_index
            ant_path_set.add(facility_index)
        return ant_path

    def calculate_ant_path_fitness(self, ant_path: np.ndarray) -> np.ndarray:
        """
        Calculate the fitness of a path.

        Args:
            ant_path: The path to calculate the fitness of.

        Returns:
            The fitness of the path.
        """
        return np.sum(self.distance_matrix[ant_path][:, ant_path] * self.flow_matrix)

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
            self.pheromone_matrix[np.arange(self.num_locations), ant_paths[i]] += (
                1 / ant_fitnesses[i]
            )

    def evaporate_pheromone(self) -> None:
        """
        Evaporate the pheromone values according to the evaporation rate.
        """
        self.pheromone_matrix *= self.evaporation_rate

    def run_trial(self) -> Tuple[float, int, int]:
        """
        Run the ant colony optimisation algorithm for a number of fitness
        evaluations.

        Returns:
            The best fitness found after finishing all evaluations, the
            number of times a better fitness was found, and the last evaluation
            when a better fitness was found.
        """
        best_fitness = float("inf")
        better_fitness_count = 0
        last_eval_improved = 1

        for eval_num in range(self.num_evals_per_trial):
            # Keep track of the best fitness found in this evaluation.
            best_fitness_in_eval = float("inf")
            ant_paths = np.empty((self.num_ant_paths, self.num_locations), dtype=int)
            ant_fitnesses = np.empty(self.num_ant_paths, dtype=float)

            # Generate the path for each ant to search for the best path and
            # fitness.
            for ant in range(self.num_ant_paths):
                ant_paths[ant] = self.generate_ant_path()
                ant_fitnesses[ant] = self.calculate_ant_path_fitness(ant_paths[ant])
                best_fitness_in_eval = min(best_fitness_in_eval, ant_fitnesses[ant])

            self.update_pheromone_matrix(ant_paths, ant_fitnesses)
            self.evaporate_pheromone()

            # Keep track of the best fitness found and how many times a better
            # fitness was found in the trial.
            if best_fitness_in_eval < best_fitness:
                best_fitness = best_fitness_in_eval
                better_fitness_count += 1
                last_eval_improved = eval_num + 1
            print(
                f"Evaluation {eval_num + 1} - "
                f"best fitness in evaluation: {best_fitness_in_eval}, "
                f"best fitness in trial: {best_fitness}, "
                f"better fitness found count: {better_fitness_count}, "
                f"last evaluation fitness improved: {last_eval_improved}"
            )

        return best_fitness, better_fitness_count, last_eval_improved

    def run_experiment(self) -> Tuple[List[float], List[int], List[int]]:
        """
        Run the ant colony optimisation algorithm for a number of trials.

        Returns:
            A list of the best fitnesses found, a list of the number of times
            a better fitness was found per trial in the experiment, and a list
            of the last evaluation when a better fitness was found.
        """
        print(
            "Running experiment with "
            f"m = {self.num_ant_paths}, e = {self.evaporation_rate}..."
        )
        best_fitness_results = []
        better_fitness_count_results = []
        last_eval_improved_results = []

        for _ in range(self.num_trials):
            # Reset the pheromone matrix to ensure each trial is independent.
            self.pheromone_matrix = np.random.uniform(
                0, 1, (self.num_locations, self.num_locations)
            )
            (
                best_fitness,
                better_fitness_count,
                last_evaluation_improved,
            ) = self.run_trial()
            best_fitness_results.append(best_fitness)
            better_fitness_count_results.append(better_fitness_count)
            last_eval_improved_results.append(last_evaluation_improved)

        return (
            best_fitness_results,
            better_fitness_count_results,
            last_eval_improved_results,
        )


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


# !IMPORTANT: Redirect stdout to a log file for optimal performance:
# !IMPORTANT: python -u aco_sim.py >> aco_sim.log
if __name__ == "__main__":
    # Set up the design problem provided by the file.
    locations, distances, flows = load_data_file("data/Uni50a.dat")
    # !IMPORTANT: Change these experiment parameters and the configs to run.
    NUM_TRIALS = 5
    NUM_EVALS_PER_TRIAL = 10_000
    # (num_ant_paths (m), evaporation_rate (e))
    experiment_configs = [
        # Experiments 1-4 contain the ones asked for in the assignment.
        (100, 0.90),
        (100, 0.50),
        (10, 0.90),
        (10, 0.50),
        # Experiments 5-7 contain extra experiments to test the effect of e.
        (10, 0.8),
        (10, 0.7),
        (10, 0.6),
    ]

    # Run the experiments.
    best_fitness_per_exp = []
    better_fitness_count_per_exp = []
    last_eval_improved_per_exp = []
    for exp_num, (m, e) in enumerate(experiment_configs):
        # Start the timer to track the runtime of the experiment.
        start = time.perf_counter()
        aco_sim = AntColonyOptimisationForQAPSimulation(
            num_trials=NUM_TRIALS,
            num_evals_per_trial=NUM_EVALS_PER_TRIAL,
            num_locations=locations,
            distance_matrix=distances,
            flow_matrix=flows,
            num_ant_paths=m,
            evaporation_rate=e,
        )
        (
            best_fitness_exp_results,
            better_fitness_count_exp_results,
            last_eval_improved_exp_results,
        ) = aco_sim.run_experiment()
        # Calculate the runtime of the experiment
        end = time.perf_counter()
        print(f"Experiment {exp_num + 1} completed in {end - start} seconds")
        best_fitness_per_exp.append(best_fitness_exp_results)
        better_fitness_count_per_exp.append(better_fitness_count_exp_results)
        last_eval_improved_per_exp.append(last_eval_improved_exp_results)
        print(f"\nExperiment {exp_num + 1} (m = {m}, e = {e}):")
        print(
            f"    Best Fitness Results: {best_fitness_exp_results}\n"
            "    Better Fitness Found Count Results: "
            f"{better_fitness_count_exp_results}\n"
            f"    Last Evaluation Improved Results: {last_eval_improved_exp_results}\n"
        )

    # Display a summary of the results of the experiments.
    print(
        f"Results of experiments with {NUM_TRIALS} trials and "
        f"{NUM_EVALS_PER_TRIAL} evaluations per trial:"
    )
    for exp_num, (m, e) in enumerate(experiment_configs):
        print(
            f"Experiment {exp_num + 1} (m = {m}, e = {e}):\n"
            f"    Best Fitness Results: {best_fitness_per_exp[exp_num]}\n"
            "    Better Fitness Found Count Results: "
            f"{better_fitness_count_per_exp[exp_num]}\n"
            "    Last Evaluation Improved Results: "
            f"{last_eval_improved_per_exp[exp_num]}"
        )
