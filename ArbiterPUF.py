from numpy.ma import dot
from numpy import sign


class ArbiterPUF:
    '''
    This model of an Arbiter PUF is based off the Arbiter Model presented in
    Extracting Secret Keys from Integrated Circuits by Daihyun Lim
    '''

    def __init__(self, input_vector):
        self.puf_delay_parameters = input_vector  # 2D Vector to represent variances in circuit, defined with: p, r, s, q
        self.challenge_bits = len(
            self.puf_delay_parameters)  # Number of stages that can be configured for a given challenge in the circuit
        self.delay_vector = self.calculate_delay_vector()

    def get_response(self, challenge_configuration):
        # Challenge_configuration refers to the vector representing a binary input of chosen path for the electrical signal
        # Return 0 if total delta is >= 0 else return 1
        return int(sign(self.get_total_delay_vector_from_challenge(challenge_configuration)))

    def get_total_delay_vector_from_challenge(self, challenge_configuration):
        # Delta between top and bottom can be represented as the dot multiplication of the input vector and challenge configuration
        return dot(self.delay_vector, challenge_configuration)

    def calculate_delay_vector(self):
        delay_vector = [self.get_alpha(0)]
        # For all challenge bits except for the first and last bits
        for stage_number, stage in enumerate(self.puf_delay_parameters[1: self.challenge_bits - 1]):
            delay_vector.append(self.get_alpha(stage_number) + self.get_beta(stage_number - 1))
        delay_vector.append(self.get_beta(self.challenge_bits - 1))
        return delay_vector

    def get_alpha(self, stage_number):
        return (self.get_challenge_stage_delay(stage_number, 0) - self.get_challenge_stage_delay(stage_number, 3)
                + self.get_challenge_stage_delay(stage_number, 1) - self.get_challenge_stage_delay(stage_number, 2)) / 2

    def get_beta(self, stage_number):
        return (self.get_challenge_stage_delay(stage_number, 0) - self.get_challenge_stage_delay(stage_number, 3)
                - self.get_challenge_stage_delay(stage_number, 1) + self.get_challenge_stage_delay(stage_number, 2)) / 2

    def get_challenge_stage_delay(self, stage_number, delay_type):
        return self.puf_delay_parameters[stage_number][delay_type]
