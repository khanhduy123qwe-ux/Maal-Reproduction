# src/lfm_maal.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import os

class LeadFollowMaze(gym.Env):
    def __init__(self):
        super().__init__()
        self.grid_size = (10, 16)
        self.exits = [(2,13), (7,13), (2,2), (7,2)]
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=15, shape=(4,), dtype=int)

    def reset(self, seed=None):
        self.leader_pos = np.array([4, 4])
        self.follower_pos = np.array([5, 5])
        self.goal_idx = random.randint(0, 3)
        self.true_goal = np.array(self.exits[self.goal_idx])
        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([self.leader_pos, self.follower_pos])

    def step(self, actions):
        leader_a, follower_a = actions
        self.leader_pos = self._move(self.leader_pos, leader_a)
        self.follower_pos = self._move(self.follower_pos, follower_a)
        reward = -0.1
        done = False
        if np.all(self.leader_pos == self.true_goal) and np.all(self.follower_pos == self.true_goal):
            reward += 1.0
            done = True
        return self._get_obs(), reward, done, False, {}

    def _move(self, pos, action):
        moves = [(-1,0),(1,0),(0,-1),(0,1),(0,0)]
        new_pos = pos + np.array(moves[action])
        return np.clip(new_pos, [0,0], [9,15])

class QLearningMAAL:
    def __init__(self, env, beta=0.01, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(5))
        self.num_goals = 4

    def get_state_key(self, obs, goal_idx):
        return tuple(obs) + (goal_idx,)

    def choose_action(self, state_key):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.Q[state_key])

    def bayesian_update(self, belief, obs_leader, action_leader):
        likelihood = np.ones(self.num_goals) * 0.01
        for g_idx, g_pos in enumerate(self.env.exits):
            dist = np.linalg.norm(obs_leader[:2] - g_pos) + 1e-6
            likelihood[g_idx] = 1.0 / dist
        belief = belief * likelihood
        belief /= belief.sum() + 1e-8
        return belief

def train_lfm(episodes=100000, beta=0.01, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    env = LeadFollowMaze()
    agent = QLearningMAAL(env, beta=beta)
    
    rewards, pcrs, ptrs = [], [], []
    MAX_STEPS_PER_EPISODE = 200   # Giới hạn để progress bar chạy mượt
    
    for ep in tqdm(range(episodes), desc=f"LFM β={beta}"):
        obs, _ = env.reset()
        leader_goal_idx = env.goal_idx
        belief = np.ones(4) / 4
        
        episode_reward = 0
        correct_steps = 0
        total_steps = 0
        predicted_goal = -1
        
        done = False
        step_count = 0
        
        while not done and step_count < MAX_STEPS_PER_EPISODE:
            step_count += 1
            total_steps += 1
            
            leader_state = agent.get_state_key(obs, leader_goal_idx)
            leader_a = agent.choose_action(leader_state)
            
            follower_state = agent.get_state_key(obs, np.argmax(belief))
            follower_a = agent.choose_action(follower_state)
            
            next_obs, r, done, _, _ = env.step([leader_a, follower_a])
            
            # Tính KL an toàn (không còn warning)
            true_dist = np.zeros(4)
            true_dist[leader_goal_idx] = 1.0
            kl_before = -np.log(belief[leader_goal_idx] + 1e-10)
            
            belief = agent.bayesian_update(belief, obs, leader_a)
            
            kl_after = -np.log(belief[leader_goal_idx] + 1e-10)
            delta_kl = kl_before - kl_after
            
            r_tilde = r + beta * delta_kl
            episode_reward += r
            
            next_state = agent.get_state_key(next_obs, leader_goal_idx)
            best_next = np.max(agent.Q[next_state])
            agent.Q[leader_state][leader_a] += agent.alpha * (r_tilde + agent.gamma * best_next - agent.Q[leader_state][leader_a])
            
            obs = next_obs
            predicted_goal = np.argmax(belief)
            if predicted_goal == leader_goal_idx:
                correct_steps += 1
        
        rewards.append(episode_reward)
        pcrs.append(1.0 if predicted_goal == leader_goal_idx else 0.0)
        ptrs.append(correct_steps / total_steps if total_steps > 0 else 0)
        
        agent.epsilon = max(0.01, agent.epsilon * 0.9995)
    
    # Lưu kết quả
    os.makedirs("results/lfm", exist_ok=True)
    os.makedirs("plots/lfm", exist_ok=True)
    df = pd.DataFrame({"reward": rewards, "pcr": pcrs, "ptr": ptrs})
    df.to_csv(f"results/lfm/results_beta{beta}.csv", index=False)
    
    plt.figure(figsize=(15,4))
    plt.subplot(1,3,1); plt.plot(np.convolve(rewards, np.ones(500)/500, mode='valid')); plt.title('Episode Reward')
    plt.subplot(1,3,2); plt.plot(np.convolve(pcrs, np.ones(500)/500, mode='valid')); plt.title('PCR')
    plt.subplot(1,3,3); plt.plot(np.convolve(ptrs, np.ones(500)/500, mode='valid')); plt.title('PTR')
    plt.savefig(f"plots/lfm/plot_beta{beta}.png")
    plt.close()
    
    print(f" HOÀN THÀNH LFM β={beta} | Reward: {np.mean(rewards[-3000:]):.3f} | PCR: {np.mean(pcrs[-3000:]):.3f} | PTR: {np.mean(ptrs[-3000:]):.3f}")

if __name__ == "__main__":
    for beta in [0.0, 0.01]:
        train_lfm(beta=beta)