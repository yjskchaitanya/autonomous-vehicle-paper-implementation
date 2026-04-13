from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PathDQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 24) -> None:
        super().__init__()
        self.state_fc1 = nn.Linear(state_dim, hidden_dim)
        self.state_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_embed = nn.Embedding(action_dim, hidden_dim)
        self.common_fc = nn.Linear(hidden_dim, hidden_dim)
        self.output_fc = nn.Linear(hidden_dim, 1)
        self.action_dim = action_dim

    def q_values(self, state: torch.Tensor) -> torch.Tensor:
        batch = state.shape[0]
        state_features = F.relu(self.state_fc1(state))
        state_features = self.state_fc2(state_features)
        actions = torch.arange(self.action_dim, device=state.device).unsqueeze(0).expand(batch, -1)
        action_features = self.action_embed(actions)
        state_features = state_features.unsqueeze(1).expand_as(action_features)
        joint = F.relu(self.common_fc(state_features + action_features))
        q = self.output_fc(joint).squeeze(-1)
        return q

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.q_values(state)


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, *args) -> None:
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    def __init__(self, config: dict, device: str = "cpu") -> None:
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.action_dim = int(config["action_dim"])
        self.gamma = float(config["gamma"])
        self.batch_size = int(config["batch_size"])
        self.target_update_frequency = int(config["target_update_frequency"])
        self.gradient_clip = float(config["gradient_clip"])
        self.epsilon = float(config["epsilon_start"])
        self.epsilon_end = float(config["epsilon_end"])
        self.epsilon_decay = float(config["epsilon_decay"])
        self.steps = 0

        self.policy_net = PathDQN(
            state_dim=int(config["state_dim"]),
            action_dim=self.action_dim,
            hidden_dim=int(config["hidden_dim"]),
        ).to(self.device)
        self.target_net = PathDQN(
            state_dim=int(config["state_dim"]),
            action_dim=self.action_dim,
            hidden_dim=int(config["hidden_dim"]),
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=float(config["learning_rate"]))
        self.buffer = ReplayBuffer(int(config["buffer_size"]))

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        if (not greedy) and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return int(q_values.argmax(dim=1).item())

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size)
        states = torch.tensor(np.stack([x.state for x in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([x.action for x in batch], dtype=torch.int64, device=self.device)
        rewards = torch.tensor([x.reward for x in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.stack([x.next_state for x in batch]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([x.done for x in batch], dtype=torch.float32, device=self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1).values
            target = rewards + (1.0 - dones) * self.gamma * next_q

        loss = F.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return float(loss.item())
