from numpy import count_nonzero
from Simplified_Arbiter_PUF import SimplifiedArbiterPUF

class ArbiterPUFFitnessMetric:
    def __init__(self, training_set):
        self.training_set = training_set

    def get_fitness(self, candidate_vector):
        candidate_puf = SimplifiedArbiterPUF(candidate_vector)
        hamming_distance = sum([count_nonzero(training_example.response - candidate_puf.get_response(training_example.challenge))
                                for training_example in self.training_set])
        fitness =  len(self.training_set) - hamming_distance
        return fitness