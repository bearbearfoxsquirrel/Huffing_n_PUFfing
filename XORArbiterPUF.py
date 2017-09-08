from Simplified_Arbiter_PUF import SimplifiedArbiterPUF
from numpy import bitwise_xor


class XORArbiterPUF:
    def __init__(self, arbiter_pufs):
        assert len(arbiter_pufs) >= 2
        self.arbiter_pufs = arbiter_pufs
        self.challenge_bits = arbiter_pufs[0].challenge_bits

    def get_response(self, challenge_vector):
        responses = [arbiter_puf.get_response(challenge_vector) for arbiter_puf in self.arbiter_pufs]
        xor_result = responses[0]
        for next_puf_response in responses[1:]:
            xor_result = bitwise_xor(xor_result, next_puf_response)
        return xor_result
