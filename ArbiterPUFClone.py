from LogisticRegression import LogisticRegressionModel, RPROP, LogisticRegressionCostFunction

class ArbiterPUFClone:
    def __init__(self, machine_learning_model, training_set, training_iterations, arbiter_challenge_bit_length):
        self.challenge_bit_length = arbiter_challenge_bit_length
        self.machine_learning_model = machine_learning_model
        self.model_trainer = RPROP(self.machine_learning_model, LogisticRegressionCostFunction(self.machine_learning_model))
        self.machine_learning_model.probability_vector = self.machine_learning_model.train_probability_vector(training_set, training_iterations, self.model_trainer)

    def get_response(self, challenge):
        probability_of_response_being_one = self.machine_learning_model.get_output_probability(challenge)
        return self.machine_learning_model.classifier.get_classification_from_probability(probability_of_response_being_one)


class PUFClassifier:
    def __init__(self, decision_boundary = 0.5):
        self.decision_boundary = decision_boundary

    def get_classification_from_probability(self, probability_of_output):
        return 1 if probability_of_output >= self.decision_boundary else -1