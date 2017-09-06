from ArbiterPUF import ArbiterPUF
from ArbiterPUFClone import ArbiterPUFClone, PUFClassifier
from numpy import shape
from CRP import CRP
import json
from pandas import DataFrame
from LogisticRegression import LogisticRegressionModel, LogisticRegressionCostFunction, RPROP
import random
from multiprocessing import Pool
from time import time
from Simplified_Arbiter_PUF import SimplifiedArbiterPUF
from CMAEvolutionStrategy import CMAEvolutionStrategy
from ArbiterPUFFitnessMetric import ArbiterPUFFitnessMetric, XORArbiterPUFFitnessMetric
from NaturalEvoultionStrategy import NaturalEvolutionStrategy, MyNaturalEvolutionStrategy
from XORArbiterPUF import XORArbiterPUF

def generate_random_physical_characteristics_for_arbiter_puf(number_of_challenges):
    # 4 delays for each stage to represent p, q, r & s delay
    return [[random.random() for delay in range(4)] for challenge_stage in range(number_of_challenges)]

def generate_random_puf_challenge(puf_challenge_bit_length):
    return [random.choice([-1, 1]) for challenge_bit in range(puf_challenge_bit_length)]

def create_puf_clone_training_set(puf_to_generate_crps_from, training_set_size):
    training_set = []
    for challenge in range(training_set_size):
        random_challenge = generate_random_puf_challenge(puf_to_generate_crps_from.challenge_bits)
        training_set.append(CRP(random_challenge, puf_to_generate_crps_from.get_response(random_challenge)))
    return training_set

def does_clone_response_match_original(original_response, clone_response):
    return original_response == clone_response

def save_training_set_to_json(training_set, output_file):
    with open(output_file, 'w') as output_file:
        json.dump([training_example.__dict__ for training_example in training_set], output_file, indent=4)

def get_test_results_of_puf_clone_against_original(clone_puf, original_puf, tests, pool):
    results = pool.starmap(does_clone_response_match_original,
                                      [(original_puf.get_response(test), clone_puf.get_response(test)) for test in tests])
    return sum(results)

def print_ml_accuracy(number_of_tests, tests_passed):
    print((tests_passed / number_of_tests) * 100, '% accuracy on tests')

def generate_arbiter_clone_with_my_nes(bit_length, training_set):
    puf_clone = SimplifiedArbiterPUF(get_random_vector(bit_length))
    puf_clone.delay_vector = MyNaturalEvolutionStrategy(puf_clone.challenge_bits,
                                                        ArbiterPUFFitnessMetric(training_set)).train(len(training_set))
    return puf_clone

def generate_xor_arbiter_clone_with_my_nes(bit_length, number_of_xors, training_set):
    puf_clone = generate_xor_arbiter_puf(bit_length, number_of_xors)
    print("Attack on", puf_clone.__str__())
    puf_vectors = MyNaturalEvolutionStrategy((len(puf_clone.arbiter_pufs), bit_length),
                                                        XORArbiterPUFFitnessMetric(training_set)).train(len(training_set))
    internal_pufs = [SimplifiedArbiterPUF(candidate_vector) for candidate_vector in puf_vectors]
    puf_clone.arbiter_pufs = internal_pufs
    return puf_clone

def generate_arbiter_clone_with_open_ai_nes(bit_length, training_set):
    puf_clone = SimplifiedArbiterPUF(get_random_vector(bit_length))
    puf_clone.delay_vector = NaturalEvolutionStrategy(puf_clone.challenge_bits,
                                                        ArbiterPUFFitnessMetric(training_set)).train(len(training_set))
    return puf_clone

def generate_arbiter_clone_with_cmaes(bit_length, training_set):
    puf_clone = SimplifiedArbiterPUF(get_random_vector(bit_length))
    puf_clone.delay_vector = CMAEvolutionStrategy(bit_length, ArbiterPUFFitnessMetric(training_set),
                                                  puf_clone.challenge_bits).train(len(training_set))
    return puf_clone

def generate_arbiter_clone_with_lr_rprop(bit_length, training_set):
    logistic_regression_model = LogisticRegressionModel(get_random_vector(bit_length))
    puf_clone = ArbiterPUFClone(logistic_regression_model, PUFClassifier())
    puf_clone.train_machine_learning_model_with_multiprocessing(RPROP(),
                                                               training_set,
                                                              LogisticRegressionCostFunction(puf_clone.machine_learning_model))
    return puf_clone


def get_random_vector(length):
    return [random.random() for weight in range(length)]

def generate_arbiter_puf(bit_length):
    return SimplifiedArbiterPUF(get_random_vector(bit_length))

def generate_xor_arbiter_puf(bit_length, number_of_xors):
    return XORArbiterPUF([generate_arbiter_puf(bit_length) for puf in range(number_of_xors + 1)])

def puf_attack_sim():
    #Original PUF to be cloned, has a randomly generated vector for input (physical characteristics) and a given challenge bit length (number of stages)
    puf_challenge_bit_length = 128
    number_of_xors = 1
    original_puf = generate_arbiter_puf(puf_challenge_bit_length)
    #original_puf = generate_xor_arbiter_puf(puf_challenge_bit_length, number_of_xors)

    #create a training set of CRPs for the clone to train on
    training_set_length = 4000
    puf_clone_training_set = create_puf_clone_training_set(original_puf, training_set_length)
    #save_training_set_to_json(puf_clone_training_set, 'ArbiterPUF_Training_Set.json')

    #create clone PUF
    start_time = time()
    puf_clone = generate_arbiter_clone_with_my_nes(puf_challenge_bit_length, puf_clone_training_set)
    #puf_clone = generate_xor_arbiter_clone_with_my_nes(puf_challenge_bit_length, number_of_xors, puf_clone_training_set)
    training_time = time() - start_time
    print("Time to train is", training_time)

    #testing the clone to ensure it has the same output as the original puf
    number_of_tests = 100000
    pool = Pool()
    tests_for_puf = pool.map(generate_random_puf_challenge, [(original_puf.challenge_bits) for length in range(number_of_tests)])

    print_ml_accuracy(number_of_tests, get_test_results_of_puf_clone_against_original(puf_clone, original_puf, tests_for_puf, pool))
    pool.close()
    pool.join()

if __name__ == '__main__':
    puf_attack_sim()