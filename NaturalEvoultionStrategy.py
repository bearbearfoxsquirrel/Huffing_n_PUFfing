from numpy import log, zeros, exp, std, mean, square, transpose, array, floor_divide, delete
from numpy.random import randn
from numpy.ma import dot

class MyNaturalEvolutionStrategy:
    def __init__(self, problem_dimension, fitness_metric, sample_population_size = 40):
        self.problem_dimension = problem_dimension
        self.fitness_metric = fitness_metric
        self.sample_population_size = sample_population_size
        self.noise_factor = sample_population_size / square(problem_dimension)
        self.mean_solution = randn(problem_dimension) #initial guess
        self.mean_solutions_fitness = self.fitness_metric.get_fitness(self.mean_solution)

    def train(self, fitness_requirement):
        generation_index = 0
        while self.mean_solutions_fitness < fitness_requirement:
            self.noise_factor = (fitness_requirement / self.mean_solutions_fitness)
            print("Generation", generation_index)
            print("population size", self.sample_population_size)
            print("noise factor", self.noise_factor)
            print("mean solution's accuracy: %s" %
                  (str(self.mean_solutions_fitness)))
            print('\n\n===================================\n')
            generation_samples = (randn(self.sample_population_size, self.problem_dimension)) - self.mean_solution
            generation_samples *= (self.noise_factor)

            generation_samples_rewards = [self.fitness_metric.get_fitness(sample) for sample in generation_samples]
            weighted_rewards = generation_samples_rewards - mean(generation_samples_rewards)
            for index in range(self.sample_population_size):
                if weighted_rewards[index] < 0:
                    delete(weighted_rewards, index)
                    delete(generation_samples, index)
            self.mean_solution = (self.mean_solution + dot(transpose(generation_samples), weighted_rewards)
                                  / (len(weighted_rewards) + 1))
            self.mean_solutions_fitness = self.fitness_metric.get_fitness(self.mean_solution)
            generation_index += 1
        return self.mean_solution

    '''
    def train(self, fitness_requirement):
        generation_index = 0
        while self.mean_solutions_fitness < fitness_requirement:
            self.noise_factor = 1 - (self.mean_solutions_fitness / fitness_requirement)
            print("Generation", generation_index)
            print("noise factor", self.noise_factor)
            print("mean solution's accuracy: %s" %
                  (str(self.mean_solutions_fitness)))
            print('\n\n===================================\n')
            generation_samples = randn(self.sample_population_size, self.problem_dimension) + self.mean_solution
            generation_samples *= self.noise_factor

            generation_samples_rewards = [self.fitness_metric.get_fitness(sample) for sample in generation_samples]
            weighted_rewards = generation_samples_rewards - mean(generation_samples_rewards)

            self.mean_solution += dot(transpose(generation_samples), weighted_rewards) / (
            self.noise_factor * self.sample_population_size)
            self.mean_solutions_fitness = self.fitness_metric.get_fitness(self.mean_solution)
            generation_index += 1
        return self.mean_solution
    '''

class NaturalEvolutionStrategy:
    def __init__(self, problem_dimension, fitness_metric,
                 sample_population_size = 100, noise_factor = 0.1, learning_rate = 0.001):
        self.problem_dimension = problem_dimension
        self.fitness_metric = fitness_metric
        self.sample_population_size = sample_population_size
        self.noise_factor = noise_factor
        self.learning_rate = learning_rate
        #remove learning rate
        #make noise factor vary
        #high population size
        #keep track of best solution

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
                                      * dot(sample_candidates.T, (standardised_rewards)))
            i += 1
        print(mean_solution)
        return mean_solution

    def get_accuracy(self, mean_solution):
        return self.fitness_metric.get_fitness(mean_solution) / len(self.fitness_metric.training_set)
