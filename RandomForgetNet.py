from Net import Net
from OrnsteinUhlenbeck import OrnsteinUhlenbeck

class RandomForgetNet(Net):
    def __init__(self, N, eigenval_norm, noise_std, mu, sigma, omega):
        self.__ou_process = OrnsteinUhlenbeck(mu, sigma, omega)
        super().__init__(N, eigenval_norm, noise_std)
        
    def step(self, stimulus, dt):
        self._M += self.__ou_process.produce_noise(self._M, dt)
        self._normalise()
        return super().step(stimulus, dt)