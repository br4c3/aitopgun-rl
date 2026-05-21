import numpy as np


class MultiAgentFighterEnv:
    def __init__(self):
        self.n_agents = 2
        self.dt = 0.05
        self.max_steps = 600
        self.g = 9.81

        self.obs_dim = 17
        self.act_dim = 4

        self.step_count = 0
        self.reset()

    def reset(self):
        self.pos = np.array([
            [0.0, 0.0, 25.0],
            [80.0, 40.0, 25.0],
        ], dtype=np.float32)

        self.vel = np.array([
            [25.0, 0.0, 0.0],
            [-25.0, 0.0, 0.0],
        ], dtype=np.float32)

        self.rpy = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, np.pi],
        ], dtype=np.float32)

        self.step_count = 0
        return self._get_obs()

    def _rotation_matrix(self, roll, pitch, yaw):
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

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

    def _energy(self, i):
        speed = np.linalg.norm(self.vel[i])
        height = max(float(self.pos[i, 2]), 0.0)
        return self.g * height + 0.5 * speed ** 2

    def _get_obs(self):
        obs = []

        for i in range(self.n_agents):
            j = 1 - i

            rel_pos = self.pos[j] - self.pos[i]
            rel_vel = self.vel[j] - self.vel[i]

            speed = np.linalg.norm(self.vel[i])
            energy = self._energy(i)

            o = np.concatenate([
                self.pos[i],
                self.vel[i],
                self.rpy[i],
                rel_pos,
                rel_vel,
                np.array([speed, energy], dtype=np.float32),
            ])

            obs.append(o.astype(np.float32))

        return obs

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float32)

        prev_dist = np.linalg.norm(self.pos[0] - self.pos[1])

        for i in range(self.n_agents):
            roll_rate, pitch_rate, yaw_rate, thrust = actions[i]

            self.rpy[i, 0] += roll_rate * self.dt
            self.rpy[i, 1] += pitch_rate * self.dt
            self.rpy[i, 2] += yaw_rate * self.dt

            self.rpy[i, 0] = np.clip(self.rpy[i, 0], -1.2, 1.2)
            self.rpy[i, 1] = np.clip(self.rpy[i, 1], -0.8, 0.8)

            R = self._rotation_matrix(*self.rpy[i])
            forward = R @ np.array([1.0, 0.0, 0.0])

            speed = np.linalg.norm(self.vel[i])

            thrust_accel = forward * thrust * 35.0
            gravity = np.array([0.0, 0.0, -self.g])
            drag = -0.015 * self.vel[i] * speed

            accel = thrust_accel + gravity + drag

            self.vel[i] += accel * self.dt
            self.pos[i] += self.vel[i] * self.dt

        self.step_count += 1

        dist = np.linalg.norm(self.pos[0] - self.pos[1])
        closing = prev_dist - dist

        rewards = []

        for i in range(self.n_agents):
            j = 1 - i

            my_energy = self._energy(i)
            enemy_energy = self._energy(j)
            speed = np.linalg.norm(self.vel[i])
            height = self.pos[i, 2]

            energy_adv = my_energy - enemy_energy
            action_cost = np.sum(actions[i] ** 2)

            reward = 0.0

            # 상대에게 접근
            reward += 1.5 * closing

            # 에너지 우위
            reward += 0.001 * energy_adv

            # 너무 멀면 패널티
            reward -= 0.005 * dist

            # 조작 비용
            reward -= 0.002 * action_cost

            # 생존 조건
            if height < 0.0:
                reward -= 200.0

            if speed < 5.0:
                reward -= 10.0

            if speed > 130.0:
                reward -= 30.0

            rewards.append(float(reward))

        done = False

        if self.step_count >= self.max_steps:
            done = True

        if np.any(self.pos[:, 2] < 0.0):
            done = True

        obs = self._get_obs()

        info = {
            "dist": dist,
            "energy_0": self._energy(0),
            "energy_1": self._energy(1),
            "pos": self.pos.copy(),
        }

        return obs, rewards, done, info
