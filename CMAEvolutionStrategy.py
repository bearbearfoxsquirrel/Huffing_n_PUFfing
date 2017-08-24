from numpy import identity, sqrt, power, exp, floor, log, conjugate, divide, mean, hamming, sum, dot, count_nonzero, multiply
from numpy.random import multivariate_normal
from numpy.ma import sum, dot, transpose, argsort
from random import random
from numpy.linalg import inv
from Simplified_Arbiter_PUF import SimplifiedArbiterPUF

class ArbiterPUFFitnessMetric:
    def __init__(self, training_set):
        self.training_set = training_set

    def get_fitness(self, candidate_vector):
        candidate_puf = SimplifiedArbiterPUF(candidate_vector)
        return sum([count_nonzero(training_example.response - candidate_puf.get_response(training_example.challenge))
                    for training_example in self.training_set])

class CMAEvolutionStrategy:
    def __init__(self, fitness_metric, problem_dimension, learning_rate = 1,
                 population_size = 4, default_step_size = 0.3):
        self.fitness_metric = fitness_metric
        self.problem_dimension = problem_dimension
        self.learning_rate = learning_rate

        self.identity_matrix = identity(self.problem_dimension) #todo get value

        self.population_size = int(population_size + floor(3 * log(self.problem_dimension)))
        self.number_of_parents = self.population_size / 2
        self.weights = log(int(self.number_of_parents) + 1 / 2) - conjugate(log(range(1, int(self.number_of_parents))))
        self.weights = divide(self.weights, sum(self.weights))
        self.number_of_parents = floor(self.number_of_parents)

        self.variance_effectiveness_of_sum_of_weights = sum(power(self.weights, 2)) / sum(dot(self.weights, self.weights))
        self.time_constant_for_step_size_control = (self.variance_effectiveness_of_sum_of_weights + 2) \
                                                   / (self.population_size + self.variance_effectiveness_of_sum_of_weights + 5)
        self.step_size_dampening = 1 + 2 * max(0, sqrt((self.variance_effectiveness_of_sum_of_weights - 1)
                                                       / (self.population_size + 1)) - 1) \
                                   + self.time_constant_for_step_size_control
        #Can also be 1 to save any bother
        self.expected_value_from_identity_normal = mean(self.identity_matrix)

        self.learning_rate_for_rank_one_update = 2 / power(problem_dimension, 2)


        self.current_distribution_mean_of_normal = [random() for value in range(self.problem_dimension)]
        self.step_size = default_step_size #should always be > 0
        self.covariance_matrix = self.identity_matrix
        self.isotropic_evolution_path = [0]
        self.anisotropic_evolution_path = [0]
        #self.multivariate_normal_distribution = multivariate_normal(self.current_distribution_mean_of_normal, self.identity_matrix) #TODO find size #TODO set up

        self.discount_factor = 1 - learning_rate
        self.complements_of_discount_variance = sqrt(1 - power(learning_rate, 2))

    def get_best_candidate_solution(self):
        generation = 0
        current_fitness = 0
        while current_fitness != 1:
            print("Generation", generation)
            self.update_for_next_generation()
            current_fitness = self.fitness_metric.get_fitness(self.current_distribution_mean_of_normal)
            generation =+ 1

        return self.current_distribution_mean_of_normal

    def update_for_next_generation(self):
        sample_candidates = self.get_new_sample_candidates()
        sample_fitnesses = [self.fitness_metric.get_fitness(sample) for sample in sample_candidates]
        sorted_sample_indexes = self.get_current_population_sorted(sample_candidates, sample_fitnesses)

        current_generation_mean = self.current_distribution_mean_of_normal
        next_generation_mean = self.get_updated_distribution_mean(sorted_sample_indexes)

        self.isotropic_evolution_path = self.get_updated_isotropic_evolution_path(next_generation_mean)
        self.anisotropic_evolution_path = self.get_updated_isotropic_evolution_path(next_generation_mean)

        self.covariance_matrix = self.get_updated_covariance_matrix(sample_candidates)

        self.step_size = self.get_updated_step_size()

    def get_new_sample_candidates(self):
        return [self.get_sample_from_multivariate_normal_distribution() for candidate_sample in range(self.population_size)]

    def get_sample_from_multivariate_normal_distribution(self):
        return multivariate_normal(self.current_distribution_mean_of_normal,
                                       dot(self.covariance_matrix, power(self.step_size, 2)))

    def get_step_of_distribution_mean(self, sorted_sample_population):
        return sum([dot(self.weights, sorted_sample) for sorted_sample in sorted_sample_population])

    def get_current_population_sorted(self, sample_population, fitness):
        #return [index for (fitness,index) in sorted(zip(fitness, list(range(len(fitness)))))]
        return [sample for (fitness,sample) in sorted(zip(fitness, sample_population), key=lambda pair : pair[0])]
        #return argsort(sample_population, fitness)

    def get_updated_distribution_mean(self, sorted_sample_population):
        return self.current_distribution_mean_of_normal \
               + self.learning_rate \
               * self.step_size \
               * self.get_step_of_distribution_mean(sorted_sample_population)
   # def get_updated_distribution_mean(self, next_distribution_mean_of_normal ,step_of_distribution_mean):
     #   return next_distribution_mean_of_normal + (self.learning_rate * step_of_distribution_mean)

    def get_updated_isotropic_evolution_path(self, next_distribution_mean_of_normal):
        return multiply(self.discount_factor,  self.isotropic_evolution_path) \
               + multiply(self.complements_of_discount_variance,
                self.get_displacement_of_distribution_mean_of_normal(next_distribution_mean_of_normal))

    def distribute_identity_matrix_normal_under_neutral_selection(self, next_distribution_mean_of_normal):
        return self.distribute_as_normal_under_neutral_selection(next_distribution_mean_of_normal) \
               * self.get_inverse_of_covariance_matrix()

    def distribute_as_normal_under_neutral_selection(self, next_distribution_mean_of_normal):
        return sqrt(self.get_variance_selection_mass())\
               * self.get_displacement_of_distribution_mean_of_normal(next_distribution_mean_of_normal)

    def get_variance_selection_mass(self):
        return sum([power(weight, 2) for weight in self.weights])

    def get_inverse_of_covariance_matrix(self):
        return inv(self.covariance_matrix)

    def get_displacement_of_distribution_mean_of_normal(self, next_distribution_mean_of_normal):
        displacement_of_mean = divide((next_distribution_mean_of_normal - self.current_distribution_mean_of_normal), self.step_size)
        return displacement_of_mean

    def get_updated_step_size(self):
        return self.step_size * exp(self.learning_rate / self.step_size_dampening * (len(self.isotropic_evolution_path) / self.expected_value_from_identity_normal) * -1)
        #todo CURRENTLY WORKING HERE

    def get_updated_anisotropic_evolution_path(self):
        return self.discount_factor * self.anisotropic_evolution_path + self.get_indicator_result() * self.complements_of_discount_variance

    def get_indicator_result(self):
        return 1 if (len(self.anisotropic_evolution_path) <= int((1.5 * sqrt(self.problem_dimension)))) else 0

    def get_updated_covariance_matrix(self, sample_population):
        return self.get_covariance_matrix_discount_factor() \
               * self.covariance_matrix \
               + (self.learning_rate_for_rank_one_update * self.get_rank_one_matrix()) \
               + self.learning_rate_for_rank_one_update \
               + self.get_rank_minimum_matrix(sample_population)

    def get_covariance_matrix_discount_factor(self):
        return 1 - self.learning_rate_for_rank_one_update - self.get_variance_selection_mass() \
                + self.get_making_up_thing_for_incase_indicator_function_is_zero()

    def get_making_up_thing_for_incase_indicator_function_is_zero(self):
        return (1 - power(self.get_indicator_result(), 2)) * self.learning_rate_for_rank_one_update * self.learning_rate \
                * (2 - self.learning_rate)

    def get_rank_one_matrix(self):
        return dot(self.anisotropic_evolution_path, transpose(self.anisotropic_evolution_path))

    def get_rank_minimum_matrix(self, sample_population):
        minimum_matrix = self.get_minimum_matrix(sample_population)
        return [self.weights[sample_index] * dot(minimum_matrix, transpose(minimum_matrix))
                                              for sample_index in range(self.population_size)]

    def get_minimum_matrix(self, sorted_sample_population):
        return divide([sorted_sample - self.current_distribution_mean_of_normal for sorted_sample in sorted_sample_population],
                      self.step_size)
