from numpy import sign, dot
import pandas

class SimplifiedArbiterPUF:
    def __init__(self, delay_vector):
        self.delay_vector = delay_vector
        self.challenge_bits = len(self.delay_vector)  # Number of stages that can be configured for a given challenge in the circuit


    def get_response(self, challenge_configuration):
        # Challenge_configuration refers to the vector representing a binary input of chosen path for the electrical signal
        # Return 0 if total delta is >= 0 else return 1
        return int(sign(dot(self.delay_vector, challenge_configuration)))