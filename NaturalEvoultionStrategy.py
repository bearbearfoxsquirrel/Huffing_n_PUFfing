from numpy import log, zeros, exp, std, mean, square, transpose, floor_divide, sqrt, divide, abs, min, tanh
from numpy.random import randn
from numpy.ma import dot

class MyNaturalEvolutionStrategy:
    def __init__(self, problem_dimension, fitness_metric, sample_population_size = 50):
        self.problem_dimension = problem_dimension
        self.fitness_metric = fitness_metric
        self.sample_population_size = sample_population_size
        self.mean_solution = randn(problem_dimension) #initial guess
        self.mean_solutions_fitness = self.fitness_metric.get_fitness(self.mean_solution)
        self.noise_factor = 1

    def train(self, fitness_requirement):
        generation_index = 0
        print("Original guesses fitness", self.mean_solutions_fitness)
        print("\n")
        while self.mean_solutions_fitness < fitness_requirement:
            print("Generation", generation_index)

            self.noise_factor = self.get_noise_factor(fitness_requirement)
            noises = self.get_noises()
            samples = noises + self.mean_solution

            sample_rewards = self.get_fitness_of_samples(samples)
            print("sample rewards", sample_rewards)
            weighted_rewards = self.get_weighted_rewards(sample_rewards)

            direction_to_head = samples - self.mean_solution
            direction_to_head = dot(direction_to_head.transpose(), weighted_rewards)

            self.mean_solution += direction_to_head / self.sample_population_size#* self.get_smoothing_factor()
            self.mean_solutions_fitness = self.fitness_metric.get_fitness(self.mean_solution)

            print("population size", self.sample_population_size)
            print("noise factor", self.noise_factor)
            print("mean solution's fitness: %s" % (str(self.mean_solutions_fitness)))
            print('\n\n===================================\n')

            generation_index += 1
        return self.mean_solution

    def get_noise_factor(self, fitness_requirement):
        return  1 - ((self.mean_solutions_fitness) / (fitness_requirement)) + 1

    def get_fitness_of_samples(self, samples):
        return [self.fitness_metric.get_fitness(sample) for sample in samples]

    def get_weighted_rewards(self, samples_rewards):
        weighted_rewards =  (samples_rewards - mean(samples_rewards)) / std(samples_rewards)
        return weighted_rewards

    def get_noises(self):
        random_noises = randn(self.sample_population_size, self.problem_dimension) * self.noise_factor
        return random_noises

    def get_smoothing_factor(self):
        smoothing_degree = exp(self.noise_factor) / 10
        print("smoothing factor noise factor squared,", smoothing_degree)
        return smoothing_degree


class NaturalEvolutionStrategy:
    def __init__(self, problem_dimension, fitness_metric,
                 sample_population_size = 100, noise_factor = 0.1, learning_rate = 0.001):
        self.problem_dimension = problem_dimension
        self.fitness_metric = fitness_metric
        self.sample_population_size = sample_population_size
        self.noise_factor = noise_factor
        self.learning_rate = learning_rate

    def train(self):
        mean_solution =  randn(self.problem_dimension)
        i = 0
        while self.get_accuracy(mean_solution) < 1:
            print("Generation", i)
            print("mean solution's accuracy: %s" %
                    (str(self.get_accuracy(mean_solution))))
            print('\n\n===================================\n')
            sample_candidates = randn(self.sample_population_size, self.problem_dimension)
            jittered_samples_rewards = zeros(self.sample_population_size)

            for sample_index in range(self.sample_population_size):
                jittered_sample_candidate = mean_solution + (self.noise_factor * sample_candidates[sample_index])
                jittered_samples_rewards[sample_index] = self.fitness_metric.get_fitness(jittered_sample_candidate)

            standardised_rewards = ((jittered_samples_rewards - mean(jittered_samples_rewards))) \
                                   / std(jittered_samples_rewards)

            mean_solution = (mean_solution
                                      + self.learning_rate / (self.sample_population_size * self.noise_factor)
                                      * dot(sample_candidates.T, standardised_rewards))
            i += 1
        print(mean_solution)
        return mean_solution

    def get_accuracy(self, mean_solution):
        return self.fitness_metric.get_fitness(mean_solution) / len(self.fitness_metric.training_set)
