from CRP import CRP
from numpy.ma import dot
from numpy import sign, power, log, float64
from math import e, exp

class LogisticRegressionModel:
    def __init__(self, classifier, probability_vector, constant_bias = 0): #TODO: Need to check about initial probability vector
        self.probability_vector = probability_vector
        self.constant_bias = constant_bias
        self.classifier = classifier

    def get_output_probability(self, input_vector):
        assert len(input_vector) == len(self.probability_vector)
        sigmoid = lambda input: 1 / (1 + power(e, input))
        dot_product = dot(input_vector, self.probability_vector)
        probability = sigmoid(dot_product)
        return probability

    def train_probability_vector(self, training_set, iterations_for_training, model_trainer):
        return model_trainer.train_model(self.probability_vector, training_set, iterations_for_training)


class RPROP:
    def __init__(self, model_to_train):
        self.min_step_size = 1 * exp(-6)
        self.max_step_size = 50
        self.default_step_size = 0.1
        self.step_size_increase_factor = 1.2
        self.step_size_decrease_factor = 0.5
        self.model_to_train = model_to_train

    #TODO get back and do this thing
    def train_model(self, network_weights, training_set, max_epoch):
        current_step_size = [self.default_step_size for weight_step_size in range(len(network_weights))]
        weight_gradients_on_current_iteration = [0.0 for value in range(len(network_weights))]
        weight_gradients_on_previous_iteration = [0.0 for value in range(len(network_weights))]

        for example_index in range(len(training_set[:max_epoch])):
          #  error_value = self.get_error_value(training_set[:example_index])
            for weight_index in range(len(network_weights)):
                weight_gradients_on_current_iteration[weight_index] = self.get_derivative_of_weight_over_cost_function(training_set[:example_index], network_weights[weight_index])
                #TODO work out stuff to do with calculating the thing

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
        return network_weights

    def get_model_response(self, inputs):
        return self.model_to_train.get_output_probability(inputs)
        #return self.model_to_train.classifier.get_classification_from_probability(self.model_to_train.get_output_probability(inputs))

    def get_derivative_of_weight_over_cost_function(self, training_examples, weight):
        return -(1 / (len(training_examples) + 1))  * sum([self.get_derivied_output_on_error_function(training_example.response, self.get_model_response(training_example.challenge), weight) for training_example in training_examples])

    def get_derivied_output_on_error_function(self, training_response, model_response, weight):
        output =  training_response * log(weight) + (1 - training_response) * log(1 - weight)
        return output

    #todo update derivitive to use logisic regression cost function

    def get_error_value(self, set_of_training_examples):
        sum_of_training_examples_error = sum([self.get_weight_cost_function(self.get_model_response(training_example.challenge), training_example.response) for training_example in set_of_training_examples])
        return -(1 / (len(set_of_training_examples) + 1)) * sum_of_training_examples_error

    def get_weight_cost_function(self, network_response, training_response):
        return training_response * log(network_response) + ((1 - training_response) * log(1 - network_response))
        #return pow(training_response - network_response, 2) / 2
    #todo get logistic regression cost function

    def update_weight_with_step_size(self, weight, weight_gradient, update_step_size):
            return weight - sign(weight_gradient) * update_step_size





'''

class GradientDescent:
    def __init__(self):

        def get_cost(machine_answers, training_answers):
            return sum((get_error_of_machine_answer(machine_answer, training_answer) for machine_answer, training_answer in zip (machine_answers, training_answers)))

        def get_error_of_machine_answer(self, network_response, training_response):
            return pow(training_response - training_response, 2) / 2
'''