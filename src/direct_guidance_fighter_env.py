from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DirectGuidanceFighterEnv(gym.Env):
    metadata = {"render_modes": []}

    MODE_RECOVER = 0
    MODE_EVADE = 1
    MODE_ATTACK = 2
    MODE_PURSUIT = 3
    NUM_MODES = 4

    def __init__(self):
        super().__init__()

        self.dt = 0.05
        self.max_steps = 1500
        self.g = 9.81

        self.min_combat_altitude = 25.0
        self.safe_altitude = 42.0

        self.base_obs_dim = 21
        self.obs_dim = self.base_obs_dim + self.NUM_MODES
        self.act_dim = 4

        self.damage_scale = 8.0

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.pos = np.array([
            [0.0, 0.0, 50.0],
            [80.0, 25.0, 50.0],
        ], dtype=np.float32)

        self.vel = np.array([
            [30.0, 0.0, 0.0],
            [-22.0, 0.0, 0.0],
        ], dtype=np.float32)

        self.forward_vec = np.array([
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ], dtype=np.float32)

        self.pos += self.np_random.uniform(-5.0, 5.0, size=self.pos.shape).astype(np.float32)
        self.pos[:, 2] = np.clip(self.pos[:, 2], 42.0, 60.0)

        self.enemy_turn_dir = float(self.np_random.choice([-1.0, 1.0]))

        self.hp = np.array([100.0, 100.0], dtype=np.float32)
        self.step_count = 0
        self.current_mode = self.MODE_PURSUIT

        return self._get_obs(0), {}

    def _normalize(self, v):
        n = np.linalg.norm(v)
        if n < 1e-6:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return (v / n).astype(np.float32)

    def _forward(self, i):
        return self.forward_vec[i]

    def _right(self, i):
        f = self._forward(i)
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        r = np.cross(f, world_up)

        if np.linalg.norm(r) < 1e-6:
            r = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        return self._normalize(r)

    def _up(self, i):
        r = self._right(i)
        f = self._forward(i)
        return self._normalize(np.cross(r, f))

    def _speed(self, i):
        return float(np.linalg.norm(self.vel[i]))

    def _energy(self, i):
        height = max(float(self.pos[i, 2]), 0.0)
        speed = self._speed(i)
        return self.g * height + 0.5 * speed ** 2

    def _los_angle_deg(self, shooter, target):
        rel = self.pos[target] - self.pos[shooter]
        dist = float(np.linalg.norm(rel))
        to_target = rel / (dist + 1e-6)

        forward = self._forward(shooter)
        cos_theta = np.clip(np.dot(forward, to_target), -1.0, 1.0)
        theta = float(np.degrees(np.arccos(cos_theta)))

        return theta, dist

    def _compute_damage_rate(self, dist, theta_deg):
        if 5.0 <= dist <= 50.0 and theta_deg < 10.0:
            los_factor = 1.0 - theta_deg / 10.0
            range_factor = 1.0 - (dist - 5.0) / 45.0
            return 1.5 * max(los_factor, 0.0) * max(range_factor, 0.0)

        if 5.0 <= dist <= 80.0 and theta_deg < 25.0:
            los_factor = 1.0 - theta_deg / 25.0
            range_factor = 1.0 - (dist - 5.0) / 75.0
            return 0.6 * max(los_factor, 0.0) * max(range_factor, 0.0)

        if 5.0 <= dist <= 120.0 and theta_deg < 45.0:
            los_factor = 1.0 - theta_deg / 45.0
            range_factor = 1.0 - (dist - 5.0) / 115.0
            return 0.25 * max(los_factor, 0.0) * max(range_factor, 0.0)

        return 0.0

    def _is_enemy_behind(self, i, enemy):
        rel = self.pos[enemy] - self.pos[i]
        dist = np.linalg.norm(rel)
        to_enemy = rel / (dist + 1e-6)

        return float(np.dot(self._forward(i), to_enemy)) < -0.15

    def _enemy_on_my_six_score(self, i, enemy):
        rel = self.pos[enemy] - self.pos[i]
        dist = np.linalg.norm(rel)

        to_enemy = rel / (dist + 1e-6)
        to_me = -to_enemy

        enemy_behind_me = -np.dot(self._forward(i), to_enemy)
        enemy_aiming_me = np.dot(self._forward(enemy), to_me)

        return float(max(0.0, enemy_behind_me) * max(0.0, enemy_aiming_me))

    def _pseudo_rpy(self, i):
        f = self._forward(i)
        yaw = np.arctan2(f[1], f[0])
        pitch = np.arctan2(f[2], np.linalg.norm(f[:2]) + 1e-6)
        roll = 0.0
        return np.array([roll, pitch, yaw], dtype=np.float32)

    def _select_mode(self):
        height = float(self.pos[0, 2])
        speed = self._speed(0)

        dist = float(np.linalg.norm(self.pos[1] - self.pos[0]))
        agent_los_deg, _ = self._los_angle_deg(0, 1)

        enemy_behind = self._is_enemy_behind(0, 1)
        danger_score = self._enemy_on_my_six_score(0, 1)

        if height < self.safe_altitude + 5.0 or speed < 16.0:
            return self.MODE_RECOVER

        if enemy_behind and danger_score > 0.65:
            return self.MODE_EVADE

        if 5.0 < dist < 120.0 and agent_los_deg < 70.0:
            return self.MODE_ATTACK

        return self.MODE_PURSUIT

    def _mode_one_hot(self, mode):
        x = np.zeros(self.NUM_MODES, dtype=np.float32)
        x[mode] = 1.0
        return x

    def _get_base_obs(self, i):
        j = 1 - i

        rel_pos = self.pos[j] - self.pos[i]
        rel_vel = self.vel[j] - self.vel[i]

        my_speed = self._speed(i)
        enemy_speed = self._speed(j)
        my_energy = self._energy(i)
        enemy_energy = self._energy(j)

        obs = np.concatenate([
            self.pos[i] / 100.0,
            self.vel[i] / 100.0,
            self._pseudo_rpy(i),
            rel_pos / 100.0,
            rel_vel / 100.0,
            np.array([
                my_speed / 100.0,
                my_energy / 5000.0,
                enemy_speed / 100.0,
                enemy_energy / 5000.0,
                self.hp[i] / 100.0,
                self.hp[j] / 100.0,
            ], dtype=np.float32),
        ])

        return obs.astype(np.float32)

    def _get_obs(self, i):
        if i == 0:
            self.current_mode = self._select_mode()
            mode_one_hot = self._mode_one_hot(self.current_mode)
        else:
            mode_one_hot = self._mode_one_hot(self.MODE_PURSUIT)

        return np.concatenate([self._get_base_obs(i), mode_one_hot]).astype(np.float32)

    def _apply_guidance_action(self, i, action):
        dir_cmd = np.asarray(action[:3], dtype=np.float32)
        throttle = float(action[3])

        height = float(self.pos[i, 2])
        vertical_speed = float(self.vel[i, 2])

        desired_dir = self._normalize(dir_cmd)

        if height < self.safe_altitude:
            desired_dir[2] = max(desired_dir[2], 0.65)
            throttle = 1.0

        if height < self.safe_altitude + 8.0:
            desired_dir[2] = max(desired_dir[2], 0.35)
            throttle = max(throttle, 0.9)

        if vertical_speed < -3.0:
            desired_dir[2] = max(desired_dir[2], 0.45)
            throttle = max(throttle, 0.95)

        desired_dir[2] = np.clip(desired_dir[2], -0.15, 0.85)
        desired_dir = self._normalize(desired_dir)

        turn_alpha = 0.12
        new_forward = self._normalize(
            (1.0 - turn_alpha) * self.forward_vec[i] + turn_alpha * desired_dir
        )
        self.forward_vec[i] = new_forward

        throttle = 0.35 + 0.65 * throttle
        throttle = np.clip(throttle, 0.35, 1.0)

        speed = self._speed(i)

        thrust_accel = self.forward_vec[i] * throttle * 42.0
        gravity = np.array([0.0, 0.0, -self.g], dtype=np.float32)
        drag = -0.008 * self.vel[i] * speed

        accel = thrust_accel + gravity + drag

        self.vel[i] += accel * self.dt
        self.pos[i] += self.vel[i] * self.dt

    def _enemy_constant_turn_action(self):
        f = self._forward(1)
        yaw = np.arctan2(f[1], f[0])
        yaw += self.enemy_turn_dir * 0.35 * self.dt

        height = float(self.pos[1, 2])
        desired_z = 0.0
        throttle = 0.68

        if height < self.safe_altitude:
            desired_z = 0.5
            throttle = 1.0
        elif height > 70.0:
            desired_z = -0.25
            throttle = 0.55

        desired_dir = np.array([
            np.cos(yaw),
            np.sin(yaw),
            desired_z,
        ], dtype=np.float32)

        return np.array([
            desired_dir[0],
            desired_dir[1],
            desired_dir[2],
            throttle,
        ], dtype=np.float32)

    def _mode_reward(
        self,
        mode,
        dist,
        progress,
        heading_align,
        closing_speed,
        lateral_error,
        vertical_error,
        dir_cmd,
        agent_los_deg,
        enemy_damage_rate,
        agent_damage_rate,
        height,
        speed,
        danger_score,
        enemy_behind_agent,
    ):
        reward = 0.0
        los_reward = max(0.0, 1.0 - agent_los_deg / 60.0)

        desired_dir = self._normalize(dir_cmd)
        to_enemy = self._normalize(self.pos[1] - self.pos[0])

        reward += 4.0 * float(np.dot(desired_dir, to_enemy))

        if mode == self.MODE_RECOVER:
            reward -= 1.2 * abs(height - self.safe_altitude)

            if height >= self.safe_altitude:
                reward += 20.0

            if height >= self.safe_altitude + 8.0:
                reward += 10.0

            if 24.0 < speed < 90.0:
                reward += 3.0

            reward -= 80.0 * enemy_damage_rate

        elif mode == self.MODE_EVADE:
            reward += 4.0 * danger_score

            if enemy_behind_agent and progress < 0.0:
                reward += 5.0 * (-progress)

            reward -= 100.0 * enemy_damage_rate

            if height >= self.safe_altitude:
                reward += 10.0

            if 30.0 < speed < 100.0:
                reward += 1.0

        elif mode == self.MODE_ATTACK:
            reward += 220.0 * agent_damage_rate
            reward -= 120.0 * enemy_damage_rate

            reward += 12.0 * heading_align
            reward += 10.0 * los_reward
            reward += 2.0 * progress

            # Good firing geometry reward.
            if agent_los_deg < 45.0 and 5.0 < dist < 120.0:
                reward += 5.0

            if agent_los_deg < 25.0 and 5.0 < dist < 80.0:
                reward += 10.0

            if agent_los_deg < 10.0 and 5.0 < dist < 50.0:
                reward += 20.0

            if 8.0 < dist < 65.0:
                reward += 5.0

            if dist < 5.0:
                reward -= 10.0

        elif mode == self.MODE_PURSUIT:
            reward += 10.0 * heading_align
            reward += 6.0 * los_reward
            reward += 6.0 * progress
            reward += 0.04 * closing_speed
            reward -= 0.020 * dist

            if 15.0 < dist < 100.0:
                reward += 2.0

            reward += 80.0 * agent_damage_rate
            reward -= 80.0 * enemy_damage_rate

        return reward

    def step(self, action):
        mode = self._select_mode()
        self.current_mode = mode

        agent_action = np.asarray(action, dtype=np.float32)
        enemy_action = self._enemy_constant_turn_action()

        prev_dist = float(np.linalg.norm(self.pos[1] - self.pos[0]))

        self._apply_guidance_action(0, agent_action)
        self._apply_guidance_action(1, enemy_action)

        self.step_count += 1

        dist = float(np.linalg.norm(self.pos[1] - self.pos[0]))
        progress = prev_dist - dist

        agent_los_deg, agent_dist = self._los_angle_deg(0, 1)
        enemy_los_deg, enemy_dist = self._los_angle_deg(1, 0)

        agent_damage_rate = self._compute_damage_rate(agent_dist, agent_los_deg)
        enemy_damage_rate = self._compute_damage_rate(enemy_dist, enemy_los_deg)

        agent_damage_to_hp = agent_damage_rate * self.damage_scale
        enemy_damage_to_hp = enemy_damage_rate * self.damage_scale

        self.hp[1] -= agent_damage_to_hp
        self.hp[0] -= enemy_damage_to_hp
        self.hp = np.clip(self.hp, 0.0, 100.0)

        speed = self._speed(0)
        height = float(self.pos[0, 2])
        enemy_height = float(self.pos[1, 2])

        rel = self.pos[1] - self.pos[0]
        to_enemy = rel / (np.linalg.norm(rel) + 1e-6)

        forward = self._forward(0)
        right = self._right(0)
        up = self._up(0)

        heading_align = float(np.dot(forward, to_enemy))
        closing_speed = float(np.dot(self.vel[0], to_enemy))
        lateral_error = float(np.dot(right, to_enemy))
        vertical_error = float(np.dot(up, to_enemy))

        enemy_behind_agent = self._is_enemy_behind(0, 1)
        danger_score = self._enemy_on_my_six_score(0, 1)

        reward = self._mode_reward(
            mode=mode,
            dist=dist,
            progress=progress,
            heading_align=heading_align,
            closing_speed=closing_speed,
            lateral_error=lateral_error,
            vertical_error=vertical_error,
            dir_cmd=agent_action[:3],
            agent_los_deg=agent_los_deg,
            enemy_damage_rate=enemy_damage_rate,
            agent_damage_rate=agent_damage_rate,
            height=height,
            speed=speed,
            danger_score=danger_score,
            enemy_behind_agent=enemy_behind_agent,
        )

        vertical_speed = float(self.vel[0, 2])

        altitude_error = max(0.0, self.safe_altitude - height)
        reward -= 8.0 * altitude_error

        if height >= self.safe_altitude:
            reward += 5.0

        if height >= self.safe_altitude + 10.0:
            reward += 2.0

        if height < self.min_combat_altitude + 5.0:
            reward -= 100.0

        if vertical_speed < -3.0:
            reward -= 2.0 * abs(vertical_speed)

        if speed < 10.0:
            reward -= 8.0

        if speed > 160.0:
            reward -= 12.0

        reward -= 0.02

        terminated = False
        truncated = False

        if self.hp[1] <= 0.0:
            reward += 300.0
            terminated = True

        if self.hp[0] <= 0.0:
            reward -= 300.0
            terminated = True

        if height < self.min_combat_altitude:
            reward -= 300.0
            terminated = True

        if enemy_height < self.min_combat_altitude:
            reward += 300.0
            terminated = True

        if self.step_count >= self.max_steps:
            truncated = True

        info = {
            "mode": int(mode),
            "dist": dist,
            "agent_hp": float(self.hp[0]),
            "enemy_hp": float(self.hp[1]),
            "agent_damage_rate": float(agent_damage_rate),
            "enemy_damage_rate": float(enemy_damage_rate),
            "agent_damage_to_hp": float(agent_damage_to_hp),
            "enemy_damage_to_hp": float(enemy_damage_to_hp),
            "agent_damage": float(agent_damage_rate),
            "enemy_damage": float(enemy_damage_rate),
            "agent_los_deg": float(agent_los_deg),
            "enemy_los_deg": float(enemy_los_deg),
            "agent_energy": float(self._energy(0)),
            "enemy_energy": float(self._energy(1)),
            "agent_speed": float(speed),
            "enemy_speed": float(self._speed(1)),
            "agent_pos": self.pos[0].copy(),
            "enemy_pos": self.pos[1].copy(),
            "agent_action": agent_action.copy(),
            "enemy_action": enemy_action.copy(),
            "danger_score": float(danger_score),
            "agent_vertical_speed": vertical_speed,
            "agent_altitude": float(height),
            "enemy_altitude": float(enemy_height),
            "min_combat_altitude": float(self.min_combat_altitude),
            "safe_altitude": float(self.safe_altitude),
        }

        return self._get_obs(0), float(reward), terminated, truncated, info
