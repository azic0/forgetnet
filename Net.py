import numpy as np

class Net:
    def __init__(self, N, eigenval_norm, noise_std):
        self.__N = N
        self.__eigenval_norm = eigenval_norm
        self._M = None
        self.__current_state = None
        self.__rng = np.random.default_rng()
        self.__noise_std = noise_std
        
    @property
    def N(self):
        return self.__N
        
    @property
    def current_state(self):
        return np.copy(self.__current_state)
        
    def set_init_state(self, init_state):
        self.__current_state = np.copy(init_state)
        
    def train(self, memories):
        self._M = memories.T @ memories
        np.fill_diagonal(self._M, 0)
        self._normalise()
    
    def train_random(self):
        self._M = np.random.normal(size=(self.__N, self.__N))
    
    def _normalise(self):
        self._M = (self._M - np.mean(self._M)) / np.std(self._M)
        eigenvals, _ = np.linalg.eig(self._M)
        self._M /= np.max(np.abs(eigenvals))
        self._M *= self.__eigenval_norm
        
    def step(self, stimulus, dt):
        noise = self.__rng.normal(self.__noise_std)
        for i in range(self.__N):
            self.__current_state[i] += (- self.__current_state[i]
                                        + np.dot(self._M[i,:], self.__current_state)
                                        + stimulus[i]
                                        + noise
                                       ) * dt
        self.__current_state /= np.linalg.norm(self.__current_state)
        return self.current_state
            
    def steady_state(self, x):
        eigenvals, eigenvecs = np.linalg.eig(self._M)
        return np.sum([(np.dot(eigenvecs[:,i], x) / (1. - eigenvals[i])) * eigenvecs[:,i] for i in range(self.__N)],
                     axis=0)