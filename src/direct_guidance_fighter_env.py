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

        self.base_obs_dim = 29
        self.obs_dim = self.base_obs_dim + self.NUM_MODES
        self.act_dim = 4

        self.damage_scale = 8.0
        self.projectile_speed = 95.0
        self.max_lead_time = 1.5

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
        self.enemy_bt_mode = "pursuit"
        self.enemy_maneuver = "lead_pursuit"
        self.enemy_maneuver_steps = 0
        self.enemy_skill = 1.0

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

    def _angle_deg_between(self, a, b):
        a = self._normalize(a)
        b = self._normalize(b)
        cos_theta = np.clip(np.dot(a, b), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_theta)))

    def _los_angle_deg(self, shooter, target):
        rel = self.pos[target] - self.pos[shooter]
        dist = float(np.linalg.norm(rel))
        to_target = rel / (dist + 1e-6)

        forward = self._forward(shooter)
        cos_theta = np.clip(np.dot(forward, to_target), -1.0, 1.0)
        theta = float(np.degrees(np.arccos(cos_theta)))

        return theta, dist

    def _los_rate(self, shooter, target):
        rel_pos = self.pos[target] - self.pos[shooter]
        rel_vel = self.vel[target] - self.vel[shooter]
        r2 = max(float(np.dot(rel_pos, rel_pos)), 1e-6)
        return float(np.linalg.norm(np.cross(rel_pos, rel_vel)) / r2)

    def _closure_rate(self, shooter, target):
        rel_pos = self.pos[target] - self.pos[shooter]
        rel_vel = self.vel[target] - self.vel[shooter]
        to_target = self._normalize(rel_pos)
        return float(-np.dot(rel_vel, to_target))

    def _lead_point(self, shooter, target):
        rel = self.pos[target] - self.pos[shooter]
        dist = float(np.linalg.norm(rel))
        t_go = dist / max(self.projectile_speed, 1.0)
        t_go = float(np.clip(t_go, 0.05, self.max_lead_time))
        return self.pos[target] + self.vel[target] * t_go

    def _lead_dir(self, shooter, target):
        return self._normalize(self._lead_point(shooter, target) - self.pos[shooter])

    def _lead_angle_deg(self, shooter, target):
        return self._angle_deg_between(self._forward(shooter), self._lead_dir(shooter, target))

    def _aspect_angle_deg(self, shooter, target):
        to_shooter = self._normalize(self.pos[shooter] - self.pos[target])
        return self._angle_deg_between(self._forward(target), to_shooter)

    def _tail_advantage_score(self, shooter, target):
        aspect = self._aspect_angle_deg(shooter, target)
        return float(np.clip((aspect - 90.0) / 90.0, 0.0, 1.0))

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
        lead_angle_deg = self._lead_angle_deg(0, 1)

        enemy_behind = self._is_enemy_behind(0, 1)
        danger_score = self._enemy_on_my_six_score(0, 1)

        if height < self.safe_altitude + 5.0 or speed < 16.0:
            return self.MODE_RECOVER

        if enemy_behind and danger_score > 0.65:
            return self.MODE_EVADE

        if 5.0 < dist < 120.0 and (agent_los_deg < 70.0 or lead_angle_deg < 55.0):
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

        dist = float(np.linalg.norm(rel_pos))
        los_rate = self._los_rate(i, j)
        aspect_angle = self._aspect_angle_deg(i, j)
        closure_rate = self._closure_rate(i, j)
        lead_angle = self._lead_angle_deg(i, j)
        tail_advantage = self._tail_advantage_score(i, j)
        energy_advantage = my_energy - enemy_energy
        altitude_error = self.pos[i, 2] - self.safe_altitude

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
                dist / 150.0,
                los_rate * 10.0,
                aspect_angle / 180.0,
                closure_rate / 100.0,
                lead_angle / 180.0,
                tail_advantage,
                energy_advantage / 5000.0,
                altitude_error / 50.0,
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

    def _blend_action_with_lead_guidance(self, i, action):
        j = 1 - i
        action = np.asarray(action, dtype=np.float32)

        raw_dir = self._normalize(action[:3])
        throttle = float(action[3])

        lead_dir = self._lead_dir(i, j)
        pure_dir = self._normalize(self.pos[j] - self.pos[i])

        dist = float(np.linalg.norm(self.pos[j] - self.pos[i]))
        danger = self._enemy_on_my_six_score(i, j)
        height = float(self.pos[i, 2])
        speed = self._speed(i)

        if height < self.safe_altitude + 5.0 or speed < 16.0:
            guidance_dir = np.array([self._forward(i)[0], self._forward(i)[1], 0.75], dtype=np.float32)
            guidance_dir = self._normalize(guidance_dir)
            blend = 0.85
            throttle = 1.0
        elif danger > 0.65:
            away = self._normalize(self.pos[i] - self.pos[j])
            side = self._normalize(np.cross(np.array([0.0, 0.0, 1.0], dtype=np.float32), away))
            guidance_dir = self._normalize(
                0.55 * away + 0.45 * side + np.array([0.0, 0.0, 0.25], dtype=np.float32)
            )
            blend = 0.70
            throttle = max(throttle, 0.9)
        else:
            lead_weight = float(np.clip((140.0 - dist) / 120.0, 0.25, 0.85))
            guidance_dir = self._normalize(lead_weight * lead_dir + (1.0 - lead_weight) * pure_dir)
            blend = 0.45

        desired_dir = self._normalize((1.0 - blend) * raw_dir + blend * guidance_dir)

        return desired_dir, throttle

    def _apply_guidance_action(self, i, action):
        desired_dir, throttle = self._blend_action_with_lead_guidance(i, action)

        height = float(self.pos[i, 2])
        vertical_speed = float(self.vel[i, 2])

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

        speed = self._speed(i)
        turn_alpha = np.clip(0.16 - 0.0007 * speed, 0.07, 0.14)

        self.forward_vec[i] = self._normalize(
            (1.0 - turn_alpha) * self.forward_vec[i] + turn_alpha * desired_dir
        )

        throttle = 0.35 + 0.65 * throttle
        throttle = np.clip(throttle, 0.35, 1.0)

        thrust_accel = self.forward_vec[i] * throttle * 42.0
        gravity = np.array([0.0, 0.0, -self.g], dtype=np.float32)
        drag = -0.008 * self.vel[i] * speed

        accel = thrust_accel + gravity + drag

        self.vel[i] += accel * self.dt
        self.pos[i] += self.vel[i] * self.dt

    def _choose_enemy_maneuver(self):
        maneuvers = [
            "lead_pursuit",
            "hard_break_left",
            "hard_break_right",
            "high_yoyo",
            "low_yoyo",
            "scissors",
            "extend",
        ]

        probs = np.array([0.30, 0.15, 0.15, 0.14, 0.10, 0.10, 0.06], dtype=np.float32)
        probs = probs / probs.sum()

        self.enemy_maneuver = str(self.np_random.choice(maneuvers, p=probs))
        self.enemy_maneuver_steps = int(self.np_random.integers(18, 55))

    def _enemy_constant_turn_action(self):
        i = 1
        target = 0

        rel = self.pos[target] - self.pos[i]
        dist = float(np.linalg.norm(rel))
        to_agent = self._normalize(rel)

        height = float(self.pos[i, 2])
        speed = self._speed(i)

        enemy_los_deg, _ = self._los_angle_deg(i, target)
        agent_los_deg, _ = self._los_angle_deg(target, i)

        danger_score = self._enemy_on_my_six_score(i, target)
        agent_aspect = self._aspect_angle_deg(target, i)

        throttle = 0.75

        if height < self.safe_altitude + 5.0 or speed < 16.0:
            self.enemy_bt_mode = "recover"

            f = self._forward(i)
            desired_dir = self._normalize(np.array([f[0], f[1], 0.75], dtype=np.float32))

            return np.array([desired_dir[0], desired_dir[1], desired_dir[2], 1.0], dtype=np.float32)

        agent_has_tail = (
            agent_los_deg < 35.0
            and 5.0 < dist < 130.0
            and agent_aspect > 100.0
        )

        if danger_score > 0.45 or agent_has_tail:
            self.enemy_bt_mode = "defensive"

            away = self._normalize(self.pos[i] - self.pos[target])
            left = self._normalize(
                np.cross(np.array([0.0, 0.0, 1.0], dtype=np.float32), away)
            )

            if self.enemy_maneuver_steps <= 0:
                self.enemy_turn_dir = float(self.np_random.choice([-1.0, 1.0]))
                self.enemy_maneuver_steps = int(self.np_random.integers(20, 45))

            self.enemy_maneuver_steps -= 1

            vertical = 0.20
            if height < self.safe_altitude + 15.0:
                vertical = 0.45

            desired_dir = self._normalize(
                0.35 * away
                + 0.85 * self.enemy_turn_dir * left
                + np.array([0.0, 0.0, vertical], dtype=np.float32)
            )

            return np.array([desired_dir[0], desired_dir[1], desired_dir[2], 1.0], dtype=np.float32)

        if 5.0 < dist < 120.0 and enemy_los_deg < 65.0:
            self.enemy_bt_mode = "attack"

            lead_dir = self._lead_dir(i, target)
            desired_dir = self._normalize(0.85 * lead_dir + 0.15 * to_agent)

            return np.array([desired_dir[0], desired_dir[1], desired_dir[2], 0.85], dtype=np.float32)

        if self.enemy_maneuver_steps <= 0:
            self._choose_enemy_maneuver()

        self.enemy_maneuver_steps -= 1
        self.enemy_bt_mode = self.enemy_maneuver

        lead_dir = self._lead_dir(i, target)
        f = self._forward(i)

        side = self._normalize(
            np.cross(np.array([0.0, 0.0, 1.0], dtype=np.float32), to_agent)
        )

        if self.enemy_maneuver == "lead_pursuit":
            desired_dir = self._normalize(0.80 * lead_dir + 0.20 * to_agent)
            throttle = 0.82

        elif self.enemy_maneuver == "hard_break_left":
            desired_dir = self._normalize(
                0.25 * to_agent
                + 0.95 * side
                + np.array([0.0, 0.0, 0.08], dtype=np.float32)
            )
            throttle = 1.0

        elif self.enemy_maneuver == "hard_break_right":
            desired_dir = self._normalize(
                0.25 * to_agent
                - 0.95 * side
                + np.array([0.0, 0.0, 0.08], dtype=np.float32)
            )
            throttle = 1.0

        elif self.enemy_maneuver == "high_yoyo":
            desired_dir = self._normalize(
                0.70 * lead_dir
                + 0.25 * side * self.enemy_turn_dir
                + np.array([0.0, 0.0, 0.50], dtype=np.float32)
            )
            throttle = 0.90

        elif self.enemy_maneuver == "low_yoyo":
            desired_z = -0.25 if height > self.safe_altitude + 18.0 else 0.20

            desired_dir = self._normalize(
                0.75 * lead_dir
                + 0.25 * side * self.enemy_turn_dir
                + np.array([0.0, 0.0, desired_z], dtype=np.float32)
            )
            throttle = 1.0

        elif self.enemy_maneuver == "scissors":
            phase = 1.0 if (self.step_count // 20) % 2 == 0 else -1.0

            desired_dir = self._normalize(
                0.35 * f
                + 0.90 * phase * side
                + np.array([0.0, 0.0, 0.10], dtype=np.float32)
            )
            throttle = 0.65

        elif self.enemy_maneuver == "extend":
            away = self._normalize(self.pos[i] - self.pos[target])

            desired_dir = self._normalize(
                0.75 * away
                + 0.25 * f
                + np.array([0.0, 0.0, 0.15], dtype=np.float32)
            )
            throttle = 1.0

        else:
            desired_dir = lead_dir
            throttle = 0.80

        if height < self.safe_altitude + 8.0:
            desired_dir[2] = max(desired_dir[2], 0.35)
            throttle = max(throttle, 0.95)

        if height > 85.0:
            desired_dir[2] = min(desired_dir[2], -0.10)

        desired_dir = self._normalize(desired_dir)

        return np.array([desired_dir[0], desired_dir[1], desired_dir[2], throttle], dtype=np.float32)

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
        agent_lead_deg,
        agent_aspect_deg,
        los_rate,
        enemy_damage_rate,
        agent_damage_rate,
        height,
        speed,
        danger_score,
        enemy_behind_agent,
    ):
        reward = 0.0

        los_reward = max(0.0, 1.0 - agent_los_deg / 60.0)
        lead_reward = max(0.0, 1.0 - agent_lead_deg / 45.0)
        aspect_reward = max(0.0, (agent_aspect_deg - 80.0) / 100.0)
        stable_los_reward = max(0.0, 1.0 - los_rate * 8.0)

        desired_dir = self._normalize(dir_cmd)
        to_enemy = self._normalize(self.pos[1] - self.pos[0])
        lead_dir = self._lead_dir(0, 1)

        reward += 2.0 * float(np.dot(desired_dir, to_enemy))
        reward += 5.0 * float(np.dot(desired_dir, lead_dir))

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

            reward -= 120.0 * enemy_damage_rate

            if height >= self.safe_altitude:
                reward += 10.0

            if 30.0 < speed < 100.0:
                reward += 1.0

        elif mode == self.MODE_ATTACK:
            reward += 240.0 * agent_damage_rate
            reward -= 130.0 * enemy_damage_rate

            reward += 8.0 * heading_align
            reward += 10.0 * los_reward
            reward += 14.0 * lead_reward
            reward += 8.0 * aspect_reward
            reward += 4.0 * stable_los_reward
            reward += 2.0 * progress

            if agent_lead_deg < 35.0 and 5.0 < dist < 130.0:
                reward += 4.0

            if agent_los_deg < 45.0 and 5.0 < dist < 120.0:
                reward += 5.0

            if agent_los_deg < 25.0 and agent_lead_deg < 25.0 and 5.0 < dist < 90.0:
                reward += 14.0

            if agent_los_deg < 10.0 and agent_lead_deg < 12.0 and 5.0 < dist < 55.0:
                reward += 28.0

            if agent_aspect_deg > 110.0 and 8.0 < dist < 90.0:
                reward += 8.0

            if 8.0 < dist < 65.0:
                reward += 5.0

            if dist < 5.0:
                reward -= 10.0

        elif mode == self.MODE_PURSUIT:
            reward += 7.0 * heading_align
            reward += 7.0 * los_reward
            reward += 10.0 * lead_reward
            reward += 6.0 * aspect_reward
            reward += 4.0 * stable_los_reward
            reward += 6.0 * progress
            reward += 0.04 * closing_speed
            reward -= 0.018 * dist

            if 15.0 < dist < 100.0:
                reward += 2.0

            reward += 90.0 * agent_damage_rate
            reward -= 90.0 * enemy_damage_rate

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

        agent_lead_deg = self._lead_angle_deg(0, 1)
        enemy_lead_deg = self._lead_angle_deg(1, 0)

        agent_aspect_deg = self._aspect_angle_deg(0, 1)
        enemy_aspect_deg = self._aspect_angle_deg(1, 0)

        los_rate = self._los_rate(0, 1)
        closure_rate = self._closure_rate(0, 1)

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
            agent_lead_deg=agent_lead_deg,
            agent_aspect_deg=agent_aspect_deg,
            los_rate=los_rate,
            enemy_damage_rate=enemy_damage_rate,
            agent_damage_rate=agent_damage_rate,
            height=height,
            speed=speed,
            danger_score=danger_score,
            enemy_behind_agent=enemy_behind_agent,
        )

        vertical_speed = float(self.vel[0, 2])
        agent_energy = self._energy(0)
        enemy_energy = self._energy(1)
        energy_advantage = agent_energy - enemy_energy

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

        if 24.0 <= speed <= 95.0:
            reward += 2.0
        elif speed < 14.0:
            reward -= 12.0
        elif speed > 160.0:
            reward -= 12.0

        if energy_advantage > -800.0:
            reward += 0.5
        else:
            reward -= 0.001 * abs(energy_advantage + 800.0)

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
            "enemy_bt_mode": self.enemy_bt_mode,
            "enemy_maneuver": self.enemy_maneuver,
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
            "agent_lead_deg": float(agent_lead_deg),
            "enemy_lead_deg": float(enemy_lead_deg),
            "agent_aspect_deg": float(agent_aspect_deg),
            "enemy_aspect_deg": float(enemy_aspect_deg),
            "los_rate": float(los_rate),
            "closure_rate": float(closure_rate),
            "agent_energy": float(agent_energy),
            "enemy_energy": float(enemy_energy),
            "energy_advantage": float(energy_advantage),
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
