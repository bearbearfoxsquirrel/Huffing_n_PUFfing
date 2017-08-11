from LogisticRegression import LogisticRegressionModel, RPROP, LogisticRegressionCostFunction

class ArbiterPUFClone:
    def __init__(self, machine_learning_model, puf_classifier, training_set, training_iterations):
        self.machine_learning_model = machine_learning_model
        self.probability_classifier = puf_classifier
        self.model_trainer = RPROP(self.machine_learning_model, LogisticRegressionCostFunction(self.machine_learning_model))

        training_set = self.prepare_training_set_for_training(training_set)
        self.machine_learning_model.probability_vector = self.machine_learning_model.train_probability_vector(training_set, training_iterations, self.model_trainer)

    def get_response(self, challenge):
        probability_of_response_being_one = self.machine_learning_model.get_output_probability(challenge)
        return self.probability_classifier.get_classification_from_probability(probability_of_response_being_one)

    def prepare_training_set_for_training(self, training_set):
        for crp in training_set:
            if crp.challenge == -1:
                crp.challenge = 0
            if crp.response == -1:
                crp.response = 0
        return training_set


class PUFClassifier:
    def __init__(self, decision_boundary = 0.5):
        self.decision_boundary = decision_boundary

    def get_classification_from_probability(self, probability_of_output):
        return 1 if probability_of_output >= self.decision_boundary else -1