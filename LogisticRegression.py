from CRP import CRP
from numpy.ma import dot
from numpy import sign, log, float64, float_power
from math import e, exp


class LogisticRegressionModel:
    def __init__(self, classifier, probability_vector, constant_bias = 0):
        self.probability_vector = probability_vector
        self.constant_bias = constant_bias
        self.classifier = classifier

    def get_output_probability(self, input_vector):
        assert len(input_vector) == len(self.probability_vector)
        sigmoid = lambda input: float64(1 / (1 + float_power(e, input)))
        dot_product = dot(input_vector, self.probability_vector)
        probability = sigmoid(dot_product)
        return probability

    def train_probability_vector(self, training_set, iterations_for_training, model_trainer):
        return model_trainer.train_model_irprop_minus(self.probability_vector, training_set)

class LogisticRegressionCostFunction:
    def __init__(self, logisitic_regression_model):
        self.logistic_regression_model = logisitic_regression_model

    def get_derivative_of_cost_function(self, training_examples, weight_index):
        return -(1 / len(training_examples))  * self.get_sum_of_squared_errors(training_examples, weight_index)

    def get_sum_of_squared_errors(self, training_examples, weight_index):
        return sum([self.get_squared_error(training_example.response,
                                           self.logistic_regression_model.get_output_probability(training_example.challenge),
                                           training_example.challenge[weight_index])
                    for training_example in training_examples])

    def get_squared_error(self, training_response, model_response, input):
        return (model_response - training_response) * input

class RPROP:
    def __init__(self, model_to_train, cost_function, epoch = 300, default_step_size = 0.1):
        self.min_step_size = 1 * exp(-6)
        self.max_step_size = 50
        self.default_step_size = default_step_size
        self.step_size_increase_factor = 1.2
        self.step_size_decrease_factor = 0.5
        self.model_to_train = model_to_train
        self.cost_function = cost_function
        self.epoch = epoch

    def train_model_irprop_minus(self, network_weights, training_set):
        current_step_size = [self.default_step_size for weight_step_size in range(len(network_weights))]
        weight_gradients_on_current_iteration = [0.0 for value in range(len(network_weights))]
        weight_gradients_on_previous_iteration = [0.0 for value in range(len(network_weights))]

        for iteration in range(self.epoch):
            for weight_index in range(len(network_weights)):
                weight_gradients_on_current_iteration[weight_index] = self.cost_function.get_derivative_of_cost_function(training_set, weight_index)

                gradient_product = weight_gradients_on_current_iteration[weight_index] * weight_gradients_on_previous_iteration[weight_index]
                if gradient_product > 0:
                    current_step_size[weight_index] = min(current_step_size[weight_index] * self.step_size_increase_factor, self.max_step_size)
                elif gradient_product < 0:
                    current_step_size[weight_index] = max(current_step_size[weight_index] * self.step_size_decrease_factor, self.min_step_size)
                    weight_gradients_on_current_iteration[weight_index] = 0

                network_weights[weight_index] = self.update_weight_with_step_size(network_weights[weight_index],
                                                                                  weight_gradients_on_current_iteration[weight_index],
                                                                                  current_step_size[weight_index])
                weight_gradients_on_previous_iteration[weight_index] = weight_gradients_on_current_iteration[weight_index]
        print(network_weights)
        return network_weights

    def get_model_response(self, inputs):
        return self.model_to_train.get_output_probability(inputs)

    def get_weight_cost_function(self, network_response, training_response):
        return training_response * log(network_response) + ((1 - training_response) * log(1 - network_response))

    def update_weight_with_step_size(self, weight, weight_gradient, update_step_size):
        return weight - sign(weight_gradient) * update_step_size

    def is_model_okie(self, testing_set):
        pass

'''
class GradientDescent:
    def __init__(self):

        def get_cost(machine_answers, training_answers):
            return sum((get_error_of_machine_answer(machine_answer, training_answer) for machine_answer, training_answer in zip (machine_answers, training_answers)))

        def get_error_of_machine_answer(self, network_response, training_response):
            return pow(training_response - training_response, 2) / 2
'''