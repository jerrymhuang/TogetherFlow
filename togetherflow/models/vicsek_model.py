
import numpy as np


class VicsekModel:
    def __init__(
            self,
            num_agents: int = 12,
            num_timesteps: int = 100,
            room_size: float | tuple = 10.0,
            rng: np.random.Generator = None
    ):
        self.num_agents = num_agents
        self.num_timesteps = num_timesteps
        self.room_size = room_size
        self.rng = rng

        if self.rng is None:
            self.rng = np.random.default_rng()

    def place(self):
        positions = self.rng.uniform(size=(self.num_agents, 2)) * self.room_size
        directions = self.rng.uniform(size=self.num_agents) * np.pi
        return positions, directions

    def hyperprior(
            self,
            rng: np.random.Generator = None
    ):

        if self.rng is None:
            self.rng = np.random.default_rng()

        alpha_j = rng.gamma(2, 2)
        beta_j = rng.gamma(2, 2)
        rho_j = rng.gamma(2, 1)
        mu_j = rng.uniform(0, np.pi)

        return np.array([alpha_j, beta_j, rho_j, mu_j]).astype(np.float32)

    def prior(self, hyperprior=None, rng=None, ):

        if rng is None:
            rng = np.random.default_rng()

        if hyperprior is not None:
            alpha_j, beta_j, rho_j, mu_j = hyperprior
        else:
            alpha_j, beta_j, rho_j, mu_j = self.hyperprior(rng=self.rng)

        # Conditional priors
        r = rng.gamma(rho_j, 1)
        v = rng.gamma(alpha_j, beta_j)
        eta = rng.uniform(0, mu_j)

        return np.array([r, v, eta]).astype(np.float32)

    def observation_model(self, params: np.ndarray = None):

        if params is not None:
            radius, speed, eta = params[0], params[1], params[2]
        else:
            radius, speed, eta = 2.0, 1.0, 0.1

        trajectories = np.zeros((self.num_timesteps + 1, self.num_agents, 2))
        headings = np.zeros((self.num_timesteps + 1, self.num_agents, 1))

        positions, directions = self.place()
        trajectories[0] = positions
        headings[0] = directions[:, np.newaxis]

        for t in range(self.num_timesteps):
            new_directions = np.zeros(self.num_agents)
            for i in range(self.num_agents):
                neighbors = []
                for j in range(self.num_agents):
                    if i != j and np.linalg.norm(positions[i] - positions[j]) < radius:
                        neighbors.append(directions[j])
                if neighbors:
                    avg_direction = np.arctan2(np.mean(np.sin(neighbors)), np.mean(np.cos(neighbors)))
                    new_directions[i] = avg_direction + self.rng.uniform(-eta * 0.5, eta * 0.5)
                else:
                    new_directions[i] = directions[i]

            directions = new_directions
            positions[:, 0] += speed * np.cos(directions)
            positions[:, 1] += speed * np.sin(directions)
            positions = np.mod(positions, self.room_size)

            trajectories[t + 1] = positions
            headings[t + 1] = directions[:, np.newaxis]

        return np.concatenate((trajectories, headings), axis=-1)

