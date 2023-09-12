from Net import Net

class OverwriteForgetNet(Net):
    def __init__(self, N, eigenval_norm, noise_std, gamma, alpha=1.0):
        self.__gamma = gamma
        self.__alpha = alpha
        super().__init__(N, eigenval_norm, noise_std)

    def step(self, stimulus, dt):
        self._M = (1 - self.__gamma) * self._M + self.__gamma * self.__alpha * (stimulus @ stimulus.T)
        self._normalise()
        return super().step(stimulus, dt)