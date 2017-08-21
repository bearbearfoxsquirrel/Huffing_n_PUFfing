from CRP import CRP
from multiprocessing import Pool, Process
from numpy.ma import dot
from numpy import  sign, float_power
from math import e, exp
from functools import partial

class LogisticRegressionModel:
    def __init__(self, probability_vector, constant_bias = 0):
        self.probability_vector = probability_vector
        self.constant_bias = constant_bias

    def get_output_probability(self, input_vector):
        #assert len(input_vector) == len(self.probability_vector)
        sigmoid = lambda input: 1 / (1 + float_power(e, input))
        dot_product_of_input_and_probability = dot(input_vector, self.probability_vector)
        probability = sigmoid(dot_product_of_input_and_probability)
        return probability

class LogisticRegressionCostFunction:
    def __init__(self, logistic_regression_model):
        self.logistic_regression_model = logistic_regression_model

    def get_derivative_of_cost_function(self, training_examples, weight_index):
        return -(1 / len(training_examples)) * self.get_sum_of_squared_errors(training_examples, weight_index)

    def get_sum_of_squared_errors(self, training_examples, weight_index):
        return sum([self.get_squared_error(training_example.response,
                                           self.logistic_regression_model.get_output_probability(training_example.challenge),
                                           training_example.challenge[weight_index])
                    for training_example in training_examples])

    def get_squared_error(self, training_response, model_response, input):
        return (model_response - training_response) * input


class RPROP:
    def __init__(self, epoch = 300, default_step_size = 0.1, error_tolerance_threshold = 5.0):
        self.min_step_size = 1 * exp(-6)
        self.max_step_size = 50
        self.default_step_size = default_step_size
        self.step_size_increase_factor = 1.2
        self.step_size_decrease_factor = 0.5
        self.epoch = epoch
        self.error_tolerance_threshold = error_tolerance_threshold

    def train_model_irprop_minus_without_multiprocessing(self, model_to_train, cost_function, network_weights, training_set):
        step_size = [self.default_step_size for weight_step_size in range(len(network_weights))]
        weight_gradients_on_previous_iteration = [0.0 for value in range(len(network_weights))]

        for iteration in range(self.epoch):
            print("Starting epoch", iteration)
            for weight_index in range(len(network_weights)):
                gradient_on_current_iteration = cost_function.get_derivative_of_cost_function(training_set, weight_index)

                gradient_product = self.get_gradient_product(gradient_on_current_iteration,
                                                             weight_gradients_on_previous_iteration[weight_index])

                step_size[weight_index] = self.get_new_step_size(gradient_product, step_size[weight_index])

                gradient_on_current_iteration = self.get_new_gradient_with_gradient_product(gradient_on_current_iteration,
                                                                                                   gradient_product)

                network_weights[weight_index] = self.update_weight_with_step_size(network_weights[weight_index],
                                                                    gradient_on_current_iteration,
                                                                    step_size[weight_index])

                weight_gradients_on_previous_iteration[weight_index] = gradient_on_current_iteration
            print(network_weights, "\n")
        return network_weights

    def train_model_irprop_minus_with_multiprocessing(self, model_to_train, cost_function, network_weights, training_set):
        current_step_size = [self.default_step_size for weight_step_size in range(len(network_weights))]
        weight_gradients_on_previous_iteration = [0.0 for value in range(len(network_weights))]
        weight_indexes = list(range(len(network_weights)))

        for epoch in range(self.epoch):
            pool = Pool()
            print("Starting epoch", epoch)
            weight_gradients_on_current_iteration = pool.starmap(cost_function.get_derivative_of_cost_function,
                                                                     [(training_set, weight_index)
                                                                      for weight_index in weight_indexes])

            gradient_products = pool.starmap(self.get_gradient_product,
                                                 [(weight_gradients_on_current_iteration[weight_index],
                                                   weight_gradients_on_previous_iteration[weight_index])
                                                  for weight_index in weight_indexes])


            current_step_size = pool.starmap(self.get_new_step_size,
                                                 [(gradient_products[weight_index],
                                                   current_step_size[weight_index])
                                                  for weight_index in weight_indexes])


            weight_gradients_on_current_iteration = pool.starmap(self.get_new_gradient_with_gradient_product,
                                                                     [(weight_gradients_on_current_iteration[weight_index],
                                                                       gradient_products[weight_index])
                                                                      for weight_index in weight_indexes])

            network_weights = pool.starmap(self.update_weight_with_step_size,
                                               [(network_weights[weight_index],
                                                weight_gradients_on_current_iteration[weight_index],
                                                current_step_size[weight_index])
                                                for weight_index in weight_indexes])

            pool.close()
            pool.join()

            weight_gradients_on_previous_iteration = [gradient for gradient in weight_gradients_on_current_iteration]


            print(network_weights, "\n")
        return network_weights

    def get_new_gradient_with_gradient_product(self, current_weight_gradient, gradient_product):
        return 0 if gradient_product < 0 else current_weight_gradient

    def get_new_step_size(self, gradient_product, current_step_size):
            if gradient_product > 0:
                return self.get_increased_step_size(current_step_size)
            elif gradient_product < 0:
                return self.get_decreased_step_size(current_step_size)
            else:
                return current_step_size

    def get_increased_step_size(self, current_step_size):
        return min(current_step_size * self.step_size_increase_factor, self.max_step_size)

    def get_decreased_step_size(self, current_step_size):
        return max(current_step_size * self.step_size_decrease_factor, self.min_step_size)

    def get_gradient_product(self, weight_gradient_on_current_iteration, weight_gradients_on_previous_iteration):
        return weight_gradient_on_current_iteration * weight_gradients_on_previous_iteration

    def update_weight_with_step_size(self, weight, weight_gradient, update_step_size):
        return weight - sign(weight_gradient) * update_step_size

    def is_model_okie(self, testing_set):
        pass
        #TODO test to see if weights keep to a certain accuracy threshold