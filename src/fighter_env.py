from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SimpleFighterEnergyEnv(gym.Env):
    """
    Simple 3D fighter-like environment for SAC.

    Observation:
        pos(3), vel(3), rpy(3), target_rel(3), speed(1), energy(1)

    Action:
        roll_rate, pitch_rate, yaw_rate, thrust
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        self.dt = 0.05
        self.max_steps = 600
        self.g = 9.81

        self.target = np.array([80.0, 40.0, 25.0], dtype=np.float32)

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([ 1.0,  1.0,  1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(14,),
            dtype=np.float32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.pos = np.array([0.0, 0.0, 20.0], dtype=np.float32)
        self.vel = np.array([20.0, 0.0, 0.0], dtype=np.float32)

        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        self.step_count = 0
        self.prev_dist = np.linalg.norm(self.target - self.pos)

        return self._get_obs(), {}

    def _rotation_matrix(self):
        cr, sr = np.cos(self.roll), np.sin(self.roll)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)

        Rz = np.array([
            [cy, -sy, 0.0],
            [sy,  cy, 0.0],
            [0.0, 0.0, 1.0],
        ])

        Ry = np.array([
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ])

        Rx = np.array([
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr,  cr],
        ])

        return Rz @ Ry @ Rx

    def _energy(self):
        speed = np.linalg.norm(self.vel)
        height = max(float(self.pos[2]), 0.0)

        # mass is omitted because it is constant
        potential = self.g * height
        kinetic = 0.5 * speed ** 2

        return potential + kinetic

    def _get_obs(self):
        speed = np.linalg.norm(self.vel)
        energy = self._energy()
        target_rel = self.target - self.pos

        obs = np.concatenate([
            self.pos,
            self.vel,
            np.array([self.roll, self.pitch, self.yaw], dtype=np.float32),
            target_rel,
            np.array([speed, energy], dtype=np.float32),
        ])

        return obs.astype(np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)

        roll_rate, pitch_rate, yaw_rate, thrust = action

        prev_dist = np.linalg.norm(self.target - self.pos)

        # attitude update
        self.roll += float(roll_rate) * self.dt
        self.pitch += float(pitch_rate) * self.dt
        self.yaw += float(yaw_rate) * self.dt

        self.roll = np.clip(self.roll, -1.2, 1.2)
        self.pitch = np.clip(self.pitch, -0.8, 0.8)

        R = self._rotation_matrix()
        forward = R @ np.array([1.0, 0.0, 0.0])

        speed = np.linalg.norm(self.vel)

        # simple dynamics
        thrust_accel = forward * float(thrust) * 35.0
        gravity = np.array([0.0, 0.0, -self.g])
        drag = -0.015 * self.vel * speed

        accel = thrust_accel + gravity + drag

        self.vel = self.vel + accel * self.dt
        self.pos = self.pos + self.vel * self.dt

        self.step_count += 1

        dist = np.linalg.norm(self.target - self.pos)
        progress = prev_dist - dist

        speed = np.linalg.norm(self.vel)
        height = float(self.pos[2])
        energy = self._energy()

        action_cost = np.sum(np.square(action))

        reward = 0.0

        # main objective
        reward += 3.0 * progress
        reward -= 0.015 * dist

        # energy advantage
        reward += 0.0015 * energy

        # penalties
        reward -= 0.002 * action_cost
        reward -= 0.002 * float(thrust) ** 2

        terminated = False
        truncated = False

        if dist < 5.0:
            reward += 200.0
            terminated = True

        if height < 0.0:
            reward -= 200.0
            terminated = True

        if speed < 5.0:
            reward -= 10.0

        if speed > 120.0:
            reward -= 50.0
            terminated = True

        if self.step_count >= self.max_steps:
            truncated = True

        info = {
            "dist": dist,
            "speed": speed,
            "height": height,
            "energy": energy,
            "progress": progress,
        }

        return self._get_obs(), float(reward), terminated, truncated, info
