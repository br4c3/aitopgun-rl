import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


LOG_STD_MIN = -20
LOG_STD_MAX = 2


class ReplayBuffer:
    def __init__(self, capacity, n_agents, obs_dim, act_dim):
        self.capacity = capacity
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, n_agents), dtype=np.float32)
        self.next_obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)

    def push(self, obs, actions, rewards, next_obs, done):
        self.obs[self.ptr] = np.asarray(obs, dtype=np.float32)
        self.actions[self.ptr] = np.asarray(actions, dtype=np.float32)
        self.rewards[self.ptr] = np.asarray(rewards, dtype=np.float32)
        self.next_obs[self.ptr] = np.asarray(next_obs, dtype=np.float32)
        self.done[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.tensor(self.obs[idx], device=device),
            torch.tensor(self.actions[idx], device=device),
            torch.tensor(self.rewards[idx], device=device),
            torch.tensor(self.next_obs[idx], device=device),
            torch.tensor(self.done[idx], device=device),
        )


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs):
        x = self.net(obs)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self(obs)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()

        action = torch.tanh(z)

        log_prob = normal.log_prob(z)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # action range:
        # raw tanh: [-1, 1]
        # thrust should be [0, 1]
        action_scaled = action.clone()
        action_scaled[:, 3] = (action_scaled[:, 3] + 1.0) / 2.0

        return action_scaled, log_prob


class Critic(nn.Module):
    def __init__(self, global_obs_dim, global_act_dim, hidden_dim=256):
        super().__init__()

        self.q = nn.Sequential(
            nn.Linear(global_obs_dim + global_act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_obs, global_action):
        x = torch.cat([global_obs, global_action], dim=-1)
        return self.q(x)


class MASAC:
    def __init__(
        self,
        n_agents,
        obs_dim,
        act_dim,
        device="cpu",
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        lr=3e-4,
    ):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.global_obs_dim = n_agents * obs_dim
        self.global_act_dim = n_agents * act_dim

        self.actors = nn.ModuleList([
            Actor(obs_dim, act_dim).to(device)
            for _ in range(n_agents)
        ])

        self.critics1 = nn.ModuleList([
            Critic(self.global_obs_dim, self.global_act_dim).to(device)
            for _ in range(n_agents)
        ])

        self.critics2 = nn.ModuleList([
            Critic(self.global_obs_dim, self.global_act_dim).to(device)
            for _ in range(n_agents)
        ])

        self.target_critics1 = nn.ModuleList([
            Critic(self.global_obs_dim, self.global_act_dim).to(device)
            for _ in range(n_agents)
        ])

        self.target_critics2 = nn.ModuleList([
            Critic(self.global_obs_dim, self.global_act_dim).to(device)
            for _ in range(n_agents)
        ])

        for i in range(n_agents):
            self.target_critics1[i].load_state_dict(self.critics1[i].state_dict())
            self.target_critics2[i].load_state_dict(self.critics2[i].state_dict())

        self.actor_opts = [
            torch.optim.Adam(self.actors[i].parameters(), lr=lr)
            for i in range(n_agents)
        ]

        self.critic1_opts = [
            torch.optim.Adam(self.critics1[i].parameters(), lr=lr)
            for i in range(n_agents)
        ]

        self.critic2_opts = [
            torch.optim.Adam(self.critics2[i].parameters(), lr=lr)
            for i in range(n_agents)
        ]

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        actions = []

        for i in range(self.n_agents):
            o = torch.tensor(obs[i], dtype=torch.float32, device=self.device).unsqueeze(0)

            if deterministic:
                mean, _ = self.actors[i](o)
                a = torch.tanh(mean)
                a[:, 3] = (a[:, 3] + 1.0) / 2.0
            else:
                a, _ = self.actors[i].sample(o)

            actions.append(a.cpu().numpy()[0])

        return np.asarray(actions, dtype=np.float32)

    def update(self, replay, batch_size=256):
        obs, actions, rewards, next_obs, done = replay.sample(batch_size, self.device)

        global_obs = obs.reshape(batch_size, -1)
        global_actions = actions.reshape(batch_size, -1)
        global_next_obs = next_obs.reshape(batch_size, -1)

        losses = {}

        with torch.no_grad():
            next_actions = []
            next_log_probs = []

            for i in range(self.n_agents):
                a_i, logp_i = self.actors[i].sample(next_obs[:, i, :])
                next_actions.append(a_i)
                next_log_probs.append(logp_i)

            next_actions_cat = torch.cat(next_actions, dim=-1)

        for i in range(self.n_agents):
            with torch.no_grad():
                q1_next = self.target_critics1[i](global_next_obs, next_actions_cat)
                q2_next = self.target_critics2[i](global_next_obs, next_actions_cat)
                q_next = torch.min(q1_next, q2_next)

                target_q = rewards[:, i:i+1] + self.gamma * (1.0 - done) * (
                    q_next - self.alpha * next_log_probs[i]
                )

            q1 = self.critics1[i](global_obs, global_actions)
            q2 = self.critics2[i](global_obs, global_actions)

            critic1_loss = F.mse_loss(q1, target_q)
            critic2_loss = F.mse_loss(q2, target_q)

            self.critic1_opts[i].zero_grad()
            critic1_loss.backward()
            self.critic1_opts[i].step()

            self.critic2_opts[i].zero_grad()
            critic2_loss.backward()
            self.critic2_opts[i].step()

            # actor update
            new_actions = []
            log_probs = []

            for j in range(self.n_agents):
                a_j, logp_j = self.actors[j].sample(obs[:, j, :])

                if j != i:
                    a_j = a_j.detach()
                    logp_j = logp_j.detach()

                new_actions.append(a_j)
                log_probs.append(logp_j)

            new_actions_cat = torch.cat(new_actions, dim=-1)

            q1_pi = self.critics1[i](global_obs, new_actions_cat)
            q2_pi = self.critics2[i](global_obs, new_actions_cat)
            q_pi = torch.min(q1_pi, q2_pi)

            actor_loss = (self.alpha * log_probs[i] - q_pi).mean()

            self.actor_opts[i].zero_grad()
            actor_loss.backward()
            self.actor_opts[i].step()

            self._soft_update(self.critics1[i], self.target_critics1[i])
            self._soft_update(self.critics2[i], self.target_critics2[i])

            losses[f"agent_{i}_actor_loss"] = actor_loss.item()
            losses[f"agent_{i}_critic_loss"] = (critic1_loss.item() + critic2_loss.item()) / 2.0

        return losses

    def _soft_update(self, net, target_net):
        for p, tp in zip(net.parameters(), target_net.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    def save(self, path):
        torch.save({
            "actors": [a.state_dict() for a in self.actors],
            "critics1": [c.state_dict() for c in self.critics1],
            "critics2": [c.state_dict() for c in self.critics2],
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)

        for i in range(self.n_agents):
            self.actors[i].load_state_dict(ckpt["actors"][i])
            self.critics1[i].load_state_dict(ckpt["critics1"][i])
            self.critics2[i].load_state_dict(ckpt["critics2"][i])
