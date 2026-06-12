import math

import numba as nb
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import sequence
import sequence.rf
import simulate.kernels


class PSO_SAVL:
    def __init__(self, num_particles: int, num_iterations: int, bounds: np.ndarray, cost_function: callable):
        # Optimization parameters
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.bounds = bounds
        self.n_dims = bounds.shape[0]

        # Hyperparameters
        self.c1 = 2.05
        self.c2 = 2.05
        self.max_inertia = 0.7
        self.min_inertia = 0.4
        self.inertia_weight = 0.9

        self.f = 1.0
        self.velocity_limit = self.max_inertia * (self.bounds[:, 1] - self.bounds[:, 0])

        # Particles
        self.positions = np.random.rand(num_particles, self.n_dims) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        self.velocities = np.random.rand(num_particles, self.n_dims) * self.velocity_limit

        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_costs = np.apply_along_axis(cost_function, 1, self.positions)

        self.cost_function = cost_function

    def embed_initial_state(self, particle_index: int, initial_state: np.ndarray) -> None:
        if not np.all((self.bounds[:, 0] <= initial_state) & (initial_state <= self.bounds[:, 1])):
            raise ValueError("Initial state is out of bounds!")

        self.positions[particle_index] = np.copy(initial_state)
        self.personal_best_positions[particle_index] = np.copy(initial_state)
        self.personal_best_costs[particle_index] = self.cost_function(initial_state)

    @staticmethod
    @nb.njit(nb.float64(nb.float64[:, :], nb.int64), cache=True, fastmath=True)
    def calculate_evolutionary_state(positions: np.ndarray, best_particle_index: int) -> float:
        # Compute squared distances between all particles (broadcasted pairwise subtraction)
        squared_distances = np.sum((positions[:, np.newaxis, :] - positions[np.newaxis, :, :]) ** 2, axis=2)

        # Set diagonal to zero (self-distance)
        np.fill_diagonal(squared_distances, 0)

        # Compute mean distance for each particle (excluding itself)
        mean_distances = np.sum(np.sqrt(squared_distances), axis=1) / (squared_distances.shape[0] - 1)

        min_d = np.min(mean_distances)
        max_d = np.max(mean_distances)

        if min_d == max_d:
            return 0.0

        global_best = mean_distances[best_particle_index]

        return (global_best - min_d) / (max_d - min_d)

    @staticmethod
    @nb.njit(nb.float64[:](nb.float64[:], nb.float64[:, :]), cache=True, fastmath=True)
    def handle_position_bounds(positions: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        out_of_bounds = np.logical_or(positions > bounds[:, 1],
                                      positions < bounds[:, 0])
        if not out_of_bounds.any():
            return positions

        for index in range(out_of_bounds.shape[0]):
            if out_of_bounds[index]:
                positions[index] = (np.random.rand() * (bounds[index, 1] - bounds[index, 0]) + bounds[index, 0])

        return positions

    @staticmethod
    @nb.njit(nb.float64[:](nb.float64, nb.float64[:], nb.float64[:]), cache=True, fastmath=True)
    def handle_velocity_bounds(f: float, velocities: np.ndarray, velocity_limit: np.ndarray) -> np.ndarray:
        out_of_bounds = np.abs(velocities) > velocity_limit

        if not out_of_bounds.any():
            return velocities

        if f >= 0.5:
            for index in range(out_of_bounds.shape[0]):
                if out_of_bounds[index]:
                    velocities[index] = np.minimum(velocity_limit[index],
                                                   np.maximum(-velocity_limit[index], velocities[index]))
        else:
            for index in range(out_of_bounds.shape[0]):
                if out_of_bounds[index]:
                    velocities[index] = np.random.rand() * (2 * velocity_limit[index] - velocity_limit[index])

        return velocities

    def run(self) -> np.ndarray:
        best_cost = np.min(self.personal_best_costs)
        best_particle_index = np.argmin(self.personal_best_costs)
        global_best = np.copy(self.positions[best_particle_index])

        alpha = (1 / self.min_inertia) - 1
        beta = -math.log((1 / self.max_inertia - 1) / alpha)

        pbar = tqdm(range(1, self.num_iterations + 1), desc="PSO-SAVL Optimizing")
        for iteration in pbar:
            f = self.calculate_evolutionary_state(self.positions, best_particle_index)
            current_inertia = self.min_inertia + (self.max_inertia - self.min_inertia) * iteration / self.num_iterations
            self.velocity_limit = (1 / (1 + alpha * math.exp(-beta * f))) * (self.bounds[:, 1] - self.bounds[:, 0])

            self.velocities = (current_inertia * self.velocities + self.c1 *
                               np.random.rand(self.num_particles, self.n_dims) *
                               (self.personal_best_positions - self.positions))

            for index in range(self.num_particles):
                self.velocities[index] += self.c2 * np.random.rand() * (global_best - self.positions[index])
                self.velocities[index] = self.handle_velocity_bounds(f, self.velocities[index],
                                                                     self.velocity_limit)

                self.positions[index] = self.positions[index] + self.velocities[index]
                self.positions[index] = self.handle_position_bounds(self.positions[index], self.bounds)

                cost = self.cost_function(self.positions[index])
                if cost < best_cost:
                    best_cost = cost
                    best_particle_index = index
                    global_best = np.copy(self.positions[index])

                if cost < self.personal_best_costs[index]:
                    self.personal_best_positions[index] = self.positions[index].copy()
                    self.personal_best_costs[index] = cost

            pbar.set_postfix({'Best Cost': best_cost, 'Global Best': global_best.flatten()})
            pbar.set_description(f'PSO Optimizing {self.cost_function.__name__}')

        return global_best


def hyperbolic_secant(x):
    x[0] = round(x[0] / 10) * 10
    rf_pulse = sequence.rf.hyperbolic_secant(duration=x[0], mu=x[1], bandwidth=x[2], dt=10)
    rf_pulse.amplitude = x[3] * 1e-6

    magnetisation = simulate.kernels.cpu_non_selective(t1=np.inf, t2=15e-3, dt=10,
                                                       df=np.linspace(-1000, 1000, 100),
                                                       rf=rf_pulse.waveform.numpy().astype(np.complex128),
                                                       initial_magnetisation=np.array([0., 0., 1.]))

    mz = magnetisation[..., -1, 2]
    mxy = np.abs(magnetisation[..., -1, 0] + 1j * magnetisation[..., -1, 1])

    target_mxy, target_mz = 0, -1
    cost = np.sum(np.abs(mz - target_mz) ** 2) + np.sum((mxy - target_mxy) ** 2) + 1e-3 * x[3]

    return cost


test = PSO_SAVL(num_particles=70, num_iterations=100, bounds=np.array([[5000., 5000.],
                                                                       [1., 25.],
                                                                       [1000., 7000.],
                                                                       [0., 25.]]),
                cost_function=hyperbolic_secant)
test.run()
