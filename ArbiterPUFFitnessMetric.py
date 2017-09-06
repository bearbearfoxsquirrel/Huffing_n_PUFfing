from numpy import count_nonzero
from Simplified_Arbiter_PUF import SimplifiedArbiterPUF
from XORArbiterPUF import XORArbiterPUF

class XORArbiterPUFFitnessMetric:
    def __init__(self, training_set):
        self.training_set = training_set

    def get_fitness(self, candidate_vectors):
        internal_pufs = [SimplifiedArbiterPUF(candidate_vector) for candidate_vector in candidate_vectors]
        candidate_puf = XORArbiterPUF(internal_pufs)
        hamming_distance = sum([count_nonzero(training_example.response - candidate_puf.get_response(training_example.challenge))
             for training_example in self.training_set])
        fitness = len(self.training_set) - hamming_distance
        return fitness

class ArbiterPUFFitnessMetric:
    def __init__(self, training_set):
        self.training_set = training_set

    def get_fitness(self, candidate_vector):
        candidate_puf = SimplifiedArbiterPUF(candidate_vector)
        hamming_distance = sum([count_nonzero(training_example.response - candidate_puf.get_response(training_example.challenge))
                                for training_example in self.training_set])
        fitness =  len(self.training_set) - hamming_distance
        return fitness