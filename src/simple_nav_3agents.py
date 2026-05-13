# src/simple_nav_3agents.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from pettingzoo.mpe import simple_spread_v3
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import os

class PlanRecognitionLSTM(nn.Module):
    def __init__(self, obs_dim=18, action_dim=5, hidden_dim=64, num_goals=3):
        super().__init__()
        self.lstm = nn.LSTM(obs_dim + action_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_goals)

    def forward(self, seq_obs, seq_actions):
        x = torch.cat([seq_obs, seq_actions], dim=-1)
        lstm_out, _ = self.lstm(x)
        return torch.softmax(self.fc(lstm_out[:, -1, :]), dim=-1)

class DQNAgent:
    def __init__(self, obs_dim, action_dim, lr=1e-3, gamma=0.95, epsilon=0.1):
        self.model = nn.Sequential(
            nn.Linear(obs_dim + 3, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.target_model = nn.Sequential(*self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_dim = action_dim

    def act(self, obs, goal_onehot):
        state = np.concatenate([obs, goal_onehot])
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        return self.model(state_t).argmax().item()

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        q_values = self.model(states)
        next_q = self.target_model(next_states).max(1)[0].detach()
        target = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(q_values.gather(1, actions.unsqueeze(1)).squeeze(), target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def train_simple_nav_3agents(episodes=15000, beta=0.01, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = simple_spread_v3.parallel_env(N=3, max_cycles=100, continuous_actions=False)
    num_goals = 3
    agent = DQNAgent(obs_dim=env.observation_space('agent_0').shape[0], action_dim=5)
    lstm = PlanRecognitionLSTM(obs_dim=18, action_dim=5, num_goals=num_goals)

    rewards_history, pcrs, ptrs = [], [], []

    for ep in tqdm(range(episodes), desc=f"SimpleNav3 β={beta}"):
        observations, infos = env.reset()                    # ← SỬA Ở ĐÂY
        leader_goal = random.randint(0, num_goals-1)
        leader_goal_onehot = np.zeros(num_goals)
        leader_goal_onehot[leader_goal] = 1.0

        episode_reward = 0
        belief = np.ones(num_goals) / num_goals
        correct_steps = 0
        total_steps = 0
        seq_obs = []
        seq_act = []

        for step in range(100):
            total_steps += 1
            actions = {}

            leader_obs = observations['agent_0']
            leader_a = agent.act(leader_obs, leader_goal_onehot)
            actions['agent_0'] = leader_a

            for i in [1, 2]:
                follower_obs = observations[f'agent_{i}']
                follower_goal_idx = np.argmax(belief)
                follower_onehot = np.zeros(num_goals)
                follower_onehot[follower_goal_idx] = 1.0
                follower_a = agent.act(follower_obs, follower_onehot)
                actions[f'agent_{i}'] = follower_a

            next_observations, rewards, terminations, truncations, infos = env.step(actions)  # ← SỬA Ở ĐÂY

            # Update belief
            seq_obs.append(leader_obs)
            seq_act.append(np.eye(5)[leader_a])
            if len(seq_obs) >= 10:
                seq_o = torch.FloatTensor(np.array(seq_obs[-10:])).unsqueeze(0)
                seq_a = torch.FloatTensor(np.array(seq_act[-10:])).unsqueeze(0)
                belief = lstm(seq_o, seq_a).detach().numpy()[0]

            predicted_goal = np.argmax(belief)
            if predicted_goal == leader_goal:
                correct_steps += 1

            # KL Gain
            true_dist = np.zeros(num_goals)
            true_dist[leader_goal] = 1.0
            kl_before = -np.log(belief[leader_goal] + 1e-10)
            kl_after = -np.log(belief[leader_goal] + 1e-10)
            delta_kl = kl_before - kl_after

            r_tilde = rewards['agent_0'] + beta * delta_kl
            episode_reward += r_tilde

            state = np.concatenate([leader_obs, leader_goal_onehot])
            next_state = np.concatenate([next_observations['agent_0'], leader_goal_onehot])
            agent.store(state, leader_a, r_tilde, next_state, terminations['agent_0'])
            agent.update()

            observations = next_observations
            if all(terminations.values()) or all(truncations.values()):
                break

        rewards_history.append(episode_reward)
        pcrs.append(1.0 if predicted_goal == leader_goal else 0.0)
        ptrs.append(correct_steps / total_steps if total_steps > 0 else 0)

        agent.epsilon = max(0.01, agent.epsilon * 0.995)

    # Lưu kết quả
    os.makedirs("results/simple_nav_3agents", exist_ok=True)
    os.makedirs("plots/simple_nav_3agents", exist_ok=True)
    df = pd.DataFrame({"reward": rewards_history, "pcr": pcrs, "ptr": ptrs})
    df.to_csv(f"results/simple_nav_3agents/results_beta{beta}.csv", index=False)

    plt.figure(figsize=(15,4))
    plt.subplot(1,3,1); plt.plot(np.convolve(rewards_history, np.ones(100)/100, mode='valid')); plt.title('Episode Reward')
    plt.subplot(1,3,2); plt.plot(np.convolve(pcrs, np.ones(100)/100, mode='valid')); plt.title('PCR')
    plt.subplot(1,3,3); plt.plot(np.convolve(ptrs, np.ones(100)/100, mode='valid')); plt.title('PTR')
    plt.savefig(f"plots/simple_nav_3agents/plot_beta{beta}.png")
    plt.close()

    print(f"✅ HOÀN THÀNH SimpleNav3 β={beta} | Reward: {np.mean(rewards_history[-1000:]):.3f} | PCR: {np.mean(pcrs[-1000:]):.3f} | PTR: {np.mean(ptrs[-1000:]):.3f}")

if __name__ == "__main__":
    for beta in [0.0, 0.01]:
        train_simple_nav_3agents(beta=beta)