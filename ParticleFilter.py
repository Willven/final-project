import numpy as np
import scipy.stats
from numpy.random import randn, random
import cv2


class ParticleFilter():
    def __init__(self, N, pos, p_std=(0.005, 0.005), v_std=(5, 5)):
        self.N = N
        # Store x, y
        self.particles = np.empty((N, 2))
        self.particles[:, 0] = pos[0] + (randn(N) * p_std[0])
        self.particles[:, 1] = pos[1] + (randn(N) * p_std[1])

        # Update noisily
        self.velocities = np.empty((N, 2))
        self.velocities[:, 0] = randn(N) * v_std[0]
        self.velocities[:, 1] = randn(N) * v_std[1]
        self.time_since_update = 0
        self.bayes_draw = False
        self.weights = np.ones(self.N)

    def predict(self):
        self.particles += self.velocities

    def update(self, z, R=1):
        self.bayes_draw = False
        self.weights = np.ones(self.N)

        dists = np.linalg.norm(self.particles - z, axis=1)
        self.weights *= scipy.stats.norm(dists).pdf(0)


        self.weights += 1e-300
        self.weights /= sum(self.weights)
        self.time_since_update = 0

    def update_none(self):
        self.bayes_draw = True
        self.time_since_update += 1
        # Remove after 1 second if not detected again
        return self.time_since_update > 30


    def resample(self):
        if 1/ np.sum(np.square(self.weights)) < self.N/1.5:
            cum_sum = np.cumsum(self.weights)
            cum_sum[-1] = 1
            indexes = np.searchsorted(cum_sum, random(self.N))
            self.particles[:] = self.particles[indexes]
            self.weights.fill(1/self.N)


    def estimate(self):
        return np.average(self.particles, weights=self.weights, axis=0)

    def draw(self, map):
        x, y = self.estimate().astype(int)
        cv2.circle(map, (x, y), 3, (0, 0, 255) if self.bayes_draw else (255, 0, 0), -1)
        return map
