from numpy import log, zeros, exp, std, mean, square, transpose, array
from numpy.random import randn
from numpy.ma import dot

class MyNaturalEvolutionStrategy:
    def __init__(self, problem_dimension, fitness_metric, sample_population_size = 100):
        self.problem_dimension = problem_dimension
        self.fitness_metric = fitness_metric
        self.sample_population_size = sample_population_size
        self.noise_factor = sample_population_size / square(problem_dimension)
        self.mean_solution = randn(problem_dimension) #initial guess
        self.mean_solutions_fitness = self.fitness_metric.get_fitness(self.mean_solution)
        self.best_solution_thus_far = self.mean_solution
        self.best_solution_thus_far_fitness = self.mean_solutions_fitness

    def train(self, fitness_requirement):
        generation_index = 0
        while self.mean_solutions_fitness < fitness_requirement:
            self.noise_factor =  1 - (self.best_solution_thus_far_fitness / fitness_requirement)
            print("Generation", generation_index)
            print("noise factor", self.noise_factor)
            print("best solution's accuracy: %s" %
                  (str(self.best_solution_thus_far_fitness)))
            print("mean solution's accuracy: %s" %
                  (str(self.mean_solutions_fitness)))
            print('\n\n===================================\n')
            generation_samples = randn(self.sample_population_size, self.problem_dimension) + self.best_solution_thus_far
            generation_samples *= (self.noise_factor)

            generation_samples_rewards = [self.fitness_metric.get_fitness(sample) for sample in generation_samples]
            weighted_rewards = generation_samples_rewards - mean(generation_samples_rewards)
            self.mean_solution = (self.best_solution_thus_far + dot(transpose(generation_samples), weighted_rewards)
                                  / (self.noise_factor * self.sample_population_size))
            self.mean_solutions_fitness = self.fitness_metric.get_fitness(self.mean_solution)

            if self.mean_solutions_fitness >= self.best_solution_thus_far_fitness:
                self.best_solution_thus_far = self.mean_solution
                self.best_solution_thus_far_fitness = self.mean_solutions_fitness
            generation_index += 1
        return self.best_solution_thus_far

class NaturalEvolutionStrategy:
    def __init__(self, problem_dimension, fitness_metric,
                 sample_population_size = 100, noise_factor = 0.1, learning_rate = 0.001):
        self.problem_dimension = problem_dimension
        self.fitness_metric = fitness_metric
        self.sample_population_size = sample_population_size
        self.noise_factor = noise_factor
        self.min_learning_rate = log(0.001)
        self.learning_rate = learning_rate
        self.noise_factor_scaling = 1.2
        #remove learning rate
        #make noise factor vary
        #high population size
        #keep track of best solution

    def train(self):
        mean_solution =  randn(self.problem_dimension)
        proposed_mean_solution = mean_solution

        current_fitness = self.fitness_metric.get_fitness(proposed_mean_solution)

        i = 0
        while self.get_accuracy(mean_solution) < 1:
            print("Generation", i)
            print("mean solution's accuracy: %s" %
                    (str(self.get_accuracy(mean_solution))))
            print('\n\n===================================\n')

           # self.noise_factor =  (1 - self.get_accuracy(mean_solution)) * self.noise_factor_scaling

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
            #proposed_fitness = self.fitness_metric.get_fitness(proposed_mean_solution)

          #  self.learning_rate = self.learning_rate * 2  if (proposed_fitness > current_fitness) \
           #    else self.learning_rate * 0.80

#            if proposed_fitness >= current_fitness:
 #               mean_solution = proposed_mean_solution
  #              current_fitness = proposed_fitness
            i += 1
        print(mean_solution)
        return mean_solution

    def get_accuracy(self, mean_solution):
        return self.fitness_metric.get_fitness(mean_solution) / len(self.fitness_metric.training_set)
