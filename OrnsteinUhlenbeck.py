import numpy as np

class OrnsteinUhlenbeck:
    def __init__(self, mu, sigma, omega):
        self.__rng = np.random.default_rng()
        self.__mu = mu
        self.__sigma = sigma
        self.__omega = omega
    
    def produce_noise(self, current_state, dt):
        return ((self.__omega**2 / (2 * self.__sigma**2)) * (self.__mu - current_state) * dt
                + np.sqrt(dt) * self.__omega * self.__rng.normal(size=current_state.shape))