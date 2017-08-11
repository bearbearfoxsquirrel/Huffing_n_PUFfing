from numpy import identity
from numpy.random import multivariate_normal

class CMAEvolutionStrategy:
    def __init__(self, population_size = 4, distribution_mean_of_normal = 0, default_step_size = 0.1):
        self.population_size = population_size
        self.distribution_mean_of_normal = distribution_mean_of_normal
        self.step_size = default_step_size #should always be > 0
        self.covariance_matrix = identity() #TODO find size
        self.isotropic_evolution_path = [0]
        self.anisotropic_evolution_path = [0]

    