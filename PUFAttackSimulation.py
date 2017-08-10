from ArbiterPUF import ArbiterPUF
from ArbiterPUFClone import ArbiterPUFClone, PUFClassifier
from CRP import CRP
import json
from pandas import DataFrame
from LogisticRegression import LogisticRegressionModel
import random

def generate_random_physical_characteristics_for_arbiter_puf(number_of_challenges):
    # 4 delays for each stage to represent p, q, r & s del
    return [[random.randint(10, 1000) for delay in range(4)] for challenge_stage in range(number_of_challenges)]

def generate_random_puf_challenge(puf_challenge_bit_length):
    return [random.choice([-1, 1]) for challenge_bit in range(puf_challenge_bit_length)]

def create_puf_clone_training_set(puf_to_generate_crps_from, training_set_size):
    training_set = []
    for challenge in range(training_set_size):
        random_challenge = generate_random_puf_challenge(puf_to_generate_crps_from.challenge_bits)
        training_set.append(CRP(random_challenge, puf_to_generate_crps_from.get_response(random_challenge)))
    return training_set

def is_clone_response_match_original(original_response, clone_response):
    return original_response == clone_response

def save_training_set_to_json(training_set, output_file):
    with open(output_file, 'w') as output_file:
        json.dump([training_example.__dict__ for training_example in training_set], output_file, indent=4)

def get_test_results_of_puf_clone_against_original(clone_puf, original_puf, tests):
    tests_passed = 0
    for test in tests:
        original_puf_response = original_puf.get_response(test)
        clone_puf_response = clone_puf.get_response(test)
        if clone_puf_response == original_puf_response:
            tests_passed += 1
    return tests_passed

def print_ml_accuracy(number_of_tests, tests_passed):
    print((tests_passed / number_of_tests) * 100, '% accuracy on tests')

def puf_attack_sim():
    #Original PUF to be cloned, has a randomly generated vector for input (physical characteristics) and a given challenge bit length (number of stages)
    puf_challenge_bit_length = 64
    random_physical_characteristics = generate_random_physical_characteristics_for_arbiter_puf(puf_challenge_bit_length)

    original_puf = ArbiterPUF(random_physical_characteristics, puf_challenge_bit_length)
    print(DataFrame(original_puf.puf_delay_parameters))

    #create a training set of CRPs for the clone to train on
    puf_clone_training_set = create_puf_clone_training_set(original_puf, 1500)
    save_training_set_to_json(puf_clone_training_set, 'ArbiterPUF_Training_Set.json')

    #create clone PUF
    clone_puf = ArbiterPUFClone(LogisticRegressionModel(PUFClassifier(), [random.random() for weight in range(puf_challenge_bit_length)]), puf_clone_training_set, 20000
                                , original_puf.challenge_bits)

    #testing the clone to ensure it has the same output as the original puf
    number_of_tests = 10000
    tests_for_puf = [generate_random_puf_challenge(original_puf.challenge_bits) for test in range(number_of_tests)]
    print_ml_accuracy(number_of_tests, get_test_results_of_puf_clone_against_original(clone_puf, original_puf, tests_for_puf))

if __name__ == '__main__':
    puf_attack_sim()