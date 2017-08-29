from numpy import log, zeros, exp, std, mean, float128
from numpy.random import randn
from numpy.ma import dot

class NaturalEvolutionStrategy:
    def __init__(self, problem_dimension, fitness_metric,
                 sample_population_size = 10, noise_factor = 0.1, learning_rate = 0.1):
        self.problem_dimension = problem_dimension
        self.fitness_metric = fitness_metric
        self.sample_population_size = sample_population_size
        self.noise_factor = noise_factor
        self.min_log_learning_rate = log(0.001)
        self.log_learning_rate = log(learning_rate)
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
            print('mean_solution: %s, reward: %f' %
                    (str(self.get_accuracy(mean_solution)), self.fitness_metric.get_fitness(mean_solution)))

            sample_candidates = randn(self.sample_population_size, self.problem_dimension)
            jittered_samples_rewards = zeros(self.sample_population_size, dtype=float128)

            for sample_index in range(self.sample_population_size):

                jittered_sample_candidate = mean_solution + (self.noise_factor * sample_candidates[sample_index])


                jittered_samples_rewards[sample_index] = self.fitness_metric.get_fitness(jittered_sample_candidate)

            standardised_rewards = ((jittered_samples_rewards - mean(jittered_samples_rewards))
                                   / std(jittered_samples_rewards))

            proposed_mean_solution = (mean_solution
                                      + exp(self.log_learning_rate) / (self.sample_population_size * self.noise_factor)
                                      * dot(sample_candidates.T, standardised_rewards))

            print('standardised rewards', standardised_rewards)
            proposed_fitness = self.fitness_metric.get_fitness(proposed_mean_solution)


            self.log_learning_rate = log(exp(self.log_learning_rate) * 1.5)  if (proposed_fitness > current_fitness) \
               else (log(exp(self.log_learning_rate) * 0.5))

            if proposed_fitness >= current_fitness:
                mean_solution = proposed_mean_solution
                current_fitness = proposed_fitness

            print("learning rate", self.log_learning_rate)
            print("noise factor", self.noise_factor, '\n\n===============================\n')

            #self.noise_factor =  1 / current_fitness
            i += 1
        print(mean_solution)
        return mean_solution

    def get_accuracy(self, mean_solution):
        return self.fitness_metric.get_fitness(mean_solution) / len(self.fitness_metric.training_set)
