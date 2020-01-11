import math
import random

import Box2D
import gym
import numpy as np
from gym import wrappers
from gym.envs import box2d, classic_control

from ray.tune.registry import register_env


class CustomEnv(gym.Env):
    _DEFAULT_CONFIG = {
        "time_limit": 500,
        "action_map": 'lambda x: x',
    }

    def __init__(self, env, default_config, config):
        self._config = {**CustomEnv._DEFAULT_CONFIG, **default_config, **config}
        self.env = env
        self._action_map = None

        self._reward = 0

        self.configure()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        print(self._action_map(0))
        print(self._reward)

        self.configure()
        self._reward = 0
        return self._modify(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self._modify(self.env.step(self._action_map(action)))
        self._reward += reward
        return obs, reward, done, info

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

    def config(self, key):
        vals = self._config[key]
        if isinstance(vals, list):
            return random.choice(vals)
        return vals

    def configure(self):
        self._configure()
        if self.config("time_limit"):
            self.env = wrappers.TimeLimit(self.env.unwrapped, self.config("time_limit"))
        self._action_map = eval(self.config("action_map"))


    def _modify(self, data):
        return data

    def _configure(self):
        raise NotImplementedError



class CartPole(CustomEnv):
    DEFAULT_CONFIG = {
        "gravity": 9.8,
        "masscart": 1.0,
        "masspole": 0.1,
        "length": 0.5,
        "force_mag": 10.0,
        "tau": 0.02,
        "kinematics_integrator": "euler",
        "theta_threshold": 12,
        "x_threshold": 2.4,
        "with_confounder": False,
    }

    def __init__(self, config):
        super().__init__(classic_control.CartPoleEnv(), CartPole.DEFAULT_CONFIG, config)

    def _configure(self):
        self.env.gravity = self.config("gravity")
        self.env.masscart = self.config("masscart")
        self.env.masspole = self.config("masspole")
        self.env.total_mass = (self.env.masspole + self.env.masscart)
        self.env.length = self.config("length")
        self.env.polemass_length = (self.env.masspole * self.env.length)
        self.env.force_mag = self.config("force_mag")
        self.env.tau = self.config("tau")
        self.env.kinematics_integrator = self.config("kinematics_integrator")
        self.env.theta_threshold_radians = self.config("theta_threshold") * 2 * math.pi / 360
        self.env.x_threshold = self.config("x_threshold")

        high = np.array([
            self.env.x_threshold * 2,
            np.finfo(np.float32).max,
            self.env.theta_threshold_radians * 2,
            np.finfo(np.float32).max,
        ])
        if self.config("with_confounder"):
            high = np.append(high, [1])

        self.env.action_space = gym.spaces.Discrete(2)
        self.env.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

    def _modify(self, data):
        if not self.config("with_confounder"):
            return data

        if isinstance(data, tuple):
            obs, reward, done, info = data
        else:
            obs = data

        if self._action_map(0) == 0:
            obs = np.append(obs, [1])
        elif self._action_map(0) == 1:
            obs = np.append(obs, [-1])
        else:
            obs = np.append(obs, [0])

        if isinstance(data, tuple):
            data = obs, reward, done, info
        else:
            data = obs

        return data


class LunarLanderContinuous(CustomEnv):
    DEFAULT_CONFIG = {
        "gravity_x": 0,
        "gravity_y": -10,
    }

    def __init__(self, config):
        super().__init__(box2d.LunarLanderContinuous(), LunarLanderContinuous.DEFAULT_CONFIG, config)

    def _configure(self):
        self.env.world = Box2D.b2World(gravity=(self.config("gravity_x"), self.config("gravity_y")))

        self.env.moon = None
        self.env.lander = None
        self.env.particles = []
        self.env.prev_reward = None

        self.reset()

    def _step(self, action):
        return self.env.step(action)
