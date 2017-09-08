from numpy import identity, sqrt, power, exp, floor, log, divide, sum, multiply, square, subtract
from numpy.random import multivariate_normal
from numpy.ma import sum, dot, transpose
from random import random
from numpy.linalg import inv


class CMAEvolutionStrategy:
    def __init__(self, problem_dimension, fitness_metric, learning_rate=1,
                 population_size=4, default_step_size=0.3):
        self.fitness_metric = fitness_metric
        self.problem_dimension = problem_dimension
        self.learning_rate = learning_rate

        self.identity_matrix = identity(self.problem_dimension)  # todo get value

        self.population_size = int(population_size + floor(3 * log(self.problem_dimension)))
        self.number_of_parents = self.population_size / 2

        self.weights = [log(self.number_of_parents + 1 / 2) - log(sample_index + 1) for sample_index in
                        range(int(self.number_of_parents))]
        self.number_of_parents = int(self.number_of_parents)
        self.weights = divide(self.weights, sum(self.weights))
        self.number_of_parents = int(floor(self.number_of_parents))

        self.variance_effective_selection_mass = power(sum([power(weight, 2) for weight in self.weights]), -1)

        self.variance_effectiveness_of_sum_of_weights = (self.variance_effective_selection_mass + 2) \
                                                        / (
                                                            self.problem_dimension + self.variance_effective_selection_mass + 5)

        self.time_constant_for_covariance_matrix = ((4 + self.variance_effectiveness_of_sum_of_weights
                                                     / self.problem_dimension)
                                                    / (self.problem_dimension + 4
                                                       + 2 * self.variance_effectiveness_of_sum_of_weights / 2))
        self.learning_rate_for_rank_one_update_of_covariance_matrix = 2 / square(problem_dimension)
        self.learning_rate_for_parent_rank_of_covariance_matrix = min(
            1 - self.learning_rate_for_rank_one_update_of_covariance_matrix,
            2 * self.variance_effectiveness_of_sum_of_weights - 1 / self.variance_effectiveness_of_sum_of_weights
            / (square(self.problem_dimension + 2) + self.variance_effectiveness_of_sum_of_weights))

        self.time_constant_for_step_size_control = ((self.variance_effectiveness_of_sum_of_weights + 5)
                                                    / (self.problem_dimension
                                                       + self.variance_effectiveness_of_sum_of_weights + 5))
        self.step_size_dampening = 1 + 2 * max(0, sqrt((self.variance_effectiveness_of_sum_of_weights - 1)
                                                       / (self.population_size + 1)) - 1) \
                                   + self.time_constant_for_covariance_matrix

        # Can also be 1 to save any bother
        self.expected_value_from_identity_normal = (sqrt(2) *
                                                    ((self.problem_dimension + 1) / 2) / (self.problem_dimension / 2))

        self.current_distribution_mean_of_normal = [random() for value in range(self.problem_dimension)]
        self.step_size = default_step_size  # should always be > 0
        self.covariance_matrix = self.identity_matrix
        self.isotropic_evolution_path = [0 for value in range(self.problem_dimension)]
        self.anisotropic_evolution_path = [0 for value in range(self.problem_dimension)]

        self.discount_factor_for_isotropic = 1 - self.time_constant_for_step_size_control
        self.discount_factor_for_anisotropic = (1
                                                - ((4 + self.variance_effective_selection_mass / self.population_size)
                                                   / (self.population_size + 4 + (
            2 * self.variance_effective_selection_mass)
                                                      / self.population_size)))  # todo DO!

        self.complements_of_discount_variance_for_isotropic = sqrt(1 - square(self.discount_factor_for_isotropic))
        self.complements_of_discount_variance_for_anisotropic = sqrt(1 - square(self.discount_factor_for_anisotropic))
        self.learning_rate_of_variance_effective_selection_mass = self.variance_effective_selection_mass / square(
            problem_dimension)
        self.division_thingy = 1 + 2 * max(
            [0, sqrt(((self.variance_effective_selection_mass - 1) / self.population_size + 1) + 1)]) \
                               + self.discount_factor_for_isotropic

    def train(self, fitness_requirement):
        generation = 0
        while self.fitness_metric.get_fitness(self.current_distribution_mean_of_normal) <= fitness_requirement:
            print("Generation", generation)
            self.update_for_next_generation()
            generation += 1
        return self.current_distribution_mean_of_normal

    def update_for_next_generation(self):
        sample_candidates = self.get_new_sample_candidates()
        sample_fitnesses = [self.fitness_metric.get_fitness(sample) for sample in sample_candidates]
        sorted_samples = self.get_current_population_sorted(sample_candidates, sample_fitnesses)

        next_generation_mean = self.get_updated_distribution_mean(sorted_samples)

        self.isotropic_evolution_path = self.get_updated_isotropic_evolution_path(next_generation_mean)
        self.anisotropic_evolution_path = self.get_updated_anisotropic_evolution_path(next_generation_mean)

        self.covariance_matrix = self.get_updated_covariance_matrix(sorted_samples)
        self.step_size = self.get_updated_step_size()

        self.current_distribution_mean_of_normal = next_generation_mean

        print("Step size", self.step_size)
        print("Current mean", self.current_distribution_mean_of_normal)
        print()

    def get_new_sample_candidates(self):
        return [self.get_sample_from_multivariate_normal_distribution() for candidate_sample in
                range(self.population_size)]

    def get_sample_from_multivariate_normal_distribution(self):
        sample_candidate = (self.current_distribution_mean_of_normal
                            + (self.step_size * multivariate_normal([0 for value in range(self.problem_dimension)],
                                                                    self.covariance_matrix)))
        # print("Candidate:", sample_candidate)
        return sample_candidate
        # return multivariate_normal(self.current_distribution_mean_of_normal,
        #                           (self.covariance_matrix * square(self.step_size)))

    def get_step_of_distribution_mean(self, sorted_sample_population):
        return sum([weight * self.get_adjusted_sample(sorted_sample)
                    for weight, sorted_sample in zip(self.weights, sorted_sample_population)])
        #  return dot(self.weights,
        #            [self.get_adjusted_sample(sorted_sample)
        #            for sorted_sample in sorted_sample_population[:int(self.number_of_parents)]])

    def get_adjusted_sample(self, sorted_sample):
        return (sorted_sample - self.current_distribution_mean_of_normal) / self.step_size

    def get_current_population_sorted(self, sample_population, fitness):
        sorted_population = [sample for (fitness, sample) in
                             sorted(zip(fitness, sample_population), key=lambda pair: pair[0])]
        return sorted_population[:self.number_of_parents]

    def get_updated_distribution_mean(self, sorted_sample_population):
        return self.current_distribution_mean_of_normal \
               + self.learning_rate \
                 * self.get_step_of_distribution_mean(sorted_sample_population)
        # def get_updated_distribution_mean(self, next_distribution_mean_of_normal ,step_of_distribution_mean):
        #   return next_distribution_mean_of_normal + (self.learning_rate * step_of_distribution_mean)

    def get_updated_isotropic_evolution_path(self, next_distribution_mean_of_normal):
        return multiply(self.discount_factor_for_isotropic, self.isotropic_evolution_path) \
               + self.complements_of_discount_variance_for_isotropic \
                 * sqrt(self.variance_effective_selection_mass) \
                 * self.get_square_root_inverse_of_covariance_matrix() \
                 * self.get_displacement_of_distribution_mean_of_normal(next_distribution_mean_of_normal)

    def distribute_identity_matrix_normal_under_neutral_selection(self, next_distribution_mean_of_normal):
        return sqrt(self.variance_effective_selection_mass) \
               * self.get_displacement_of_distribution_mean_of_normal(next_distribution_mean_of_normal) \
               * self.get_square_root_inverse_of_covariance_matrix()

    def get_square_root_inverse_of_covariance_matrix(self):
        inverse_of_covariance_matrix = self.get_inverse_of_covariance_matrix()
        return sqrt(inverse_of_covariance_matrix)

    def get_displacement_of_distribution_mean_of_normal(self, next_distribution_mean_of_normal):
        displacement_of_mean = divide((next_distribution_mean_of_normal - self.current_distribution_mean_of_normal),
                                      self.step_size)
        return displacement_of_mean

    def get_updated_step_size(self):
        return self.step_size * exp((self.time_constant_for_step_size_control / self.step_size_dampening) * (
            len(self.isotropic_evolution_path) / self.expected_value_from_identity_normal) - 1)
        # todo CURRENTLY WORKING HERE

    def get_updated_anisotropic_evolution_path(self, next_distribution_mean_of_normal):
        return multiply(self.discount_factor_for_anisotropic, self.anisotropic_evolution_path) \
               + self.get_indicator_result() * self.complements_of_discount_variance_for_anisotropic \
                 * sqrt(self.variance_effective_selection_mass) \
                 * self.get_displacement_of_distribution_mean_of_normal(next_distribution_mean_of_normal)

    def get_indicator_result(self):
        return 1 if (len(self.isotropic_evolution_path) / sqrt(1 - square(1 - self.time_constant_for_step_size_control))
                     < (1.4 + (2 / (self.problem_dimension + 1))) * self.expected_value_from_identity_normal) else 0

    def get_updated_covariance_matrix(self, sample_population):
        covariance_discount_factor = self.get_covariance_matrix_discount_factor()
        rank_one_matrix = self.get_rank_one_matrix()
        rank_minimum_matrix = self.get_rank_minimum_matrix(sample_population)

        return multiply(covariance_discount_factor, self.covariance_matrix) \
               + (multiply(self.learning_rate_for_rank_one_update_of_covariance_matrix, rank_one_matrix)) \
               + (multiply(self.learning_rate_for_parent_rank_of_covariance_matrix, rank_minimum_matrix))

    def get_covariance_matrix_discount_factor(self):
        return (1
                + self.learning_rate_for_rank_one_update_of_covariance_matrix
                * self.get_preventer_of_axes_increase_decider()
                - self.learning_rate_for_rank_one_update_of_covariance_matrix
                - self.learning_rate_for_parent_rank_of_covariance_matrix * sum(self.weights)
                )

    def get_preventer_of_axes_increase_decider(self):
        return (1 - power(self.get_indicator_result(),
                          2)) * self.learning_rate_for_rank_one_update_of_covariance_matrix * self.learning_rate \
               * (2 - self.learning_rate)

    def get_rank_one_matrix(self):
        return multiply(self.anisotropic_evolution_path, transpose(self.anisotropic_evolution_path))

    def get_rank_minimum_matrix(self, sorted_sample_population):
        return sum([multiply(
            (self.get_steped_difference(sorted_sample) * transpose(self.get_steped_difference(sorted_sample))), weight)
            for weight, sorted_sample in
            zip(self.get_adjusted_weights(sorted_sample_population), sorted_sample_population)])

    def get_adjusted_weights(self, sorted_sample_population):
        return [weight * self.decide_how_weight_is_adjusted(weight, sorted_sample)
                for weight, sorted_sample in zip(self.weights, sorted_sample_population)]

    def decide_how_weight_is_adjusted(self, weight, sorted_sample):
        return 1 if weight >= 0 else self.problem_dimension / square(len(self.get_inverse_of_covariance_matrix()
                                                                         * (
                                                                             sorted_sample - self.current_distribution_mean_of_normal
                                                                             / self.step_size)))

    def get_inverse_of_covariance_matrix(self):
        return inv(self.covariance_matrix)

    def get_steped_difference(self, sorted_sample):
        return divide(subtract(sorted_sample, self.current_distribution_mean_of_normal),
                      self.step_size)
