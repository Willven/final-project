import numpy as np
import scipy.stats
from numpy.random import randn, random
import cv2


class ParticleFilter():
    def __init__(self, N, pos, p_std=(0.005, 0.005), v_std=(5, 5)):
        """
        Method used to intialise a particle filter
        :param N: Number of particles
        :param pos: Starting coordinates
        :param p_std: Process noise
        :param v_std: Movement noise
        """
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
        """
        Predict the new position of the particles
        """
        self.particles += self.velocities

    def update(self, z):
        """
        Update the particles locations
        :param z: New observation
        """
        self.bayes_draw = False
        self.weights = np.ones(self.N)

        dists = np.linalg.norm(self.particles - z, axis=1)
        self.weights *= scipy.stats.norm(dists).pdf(0)


        self.weights += 1e-300
        self.weights /= sum(self.weights)
        self.time_since_update = 0

    def update_none(self):
        """
        Method used to update the particle filter if no observation is provided
        :return: A boolean value indicating if the filter has decayed, True if so.
        """
        self.bayes_draw = True
        self.time_since_update += 1
        # Remove after 1 second if not detected again
        return self.time_since_update > 30


    def resample(self):
        """
        Method used to resample the particles if required.
        """
        if 1/ np.sum(np.square(self.weights)) < self.N/1.5:
            cum_sum = np.cumsum(self.weights)
            cum_sum[-1] = 1
            indexes = np.searchsorted(cum_sum, random(self.N))
            self.particles[:] = self.particles[indexes]
            self.weights.fill(1/self.N)


    def estimate(self):
        """
        Method used to estimate the position of the tracked object
        :return: The estimated position of the object
        """
        return np.average(self.particles, weights=self.weights, axis=0)

    def draw(self, image):
        """
        Method used to draw the particle's mean position on a diagram
        :param image: The image to draw the estimate on
        :return: An updated version of image
        """
        x, y = self.estimate().astype(int)
        cv2.circle(image, (x, y), 3, (0, 0, 255) if self.bayes_draw else (255, 0, 0), -1)
        return image
