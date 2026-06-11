"""
Lead-Follow Maze (LFM) — MAAL Reproduction
Fixes:
  1. Bayesian update dùng action-alignment thay vì 1/dist
  2. Tách Q_leader và Q_follower riêng biệt
  3. Cập nhật Q_follower mỗi bước (follower thực sự học)
  4. Epsilon bắt đầu từ 1.0, decay chậm hơn (0.9999)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import os


# ─────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────

class LeadFollowMaze(gym.Env):
    def __init__(self):
        super().__init__()
        self.grid_size = (10, 16)
        self.exits = [(2, 13), (7, 13), (2, 2), (7, 2)]
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=15, shape=(4,), dtype=int)
        # Định nghĩa vector di chuyển cho từng action
        self.moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]

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

        if (np.all(self.leader_pos == self.true_goal)
                and np.all(self.follower_pos == self.true_goal)):
            reward += 1.0
            done = True

        return self._get_obs(), reward, done, False, {}

    def _move(self, pos, action):
        new_pos = pos + np.array(self.moves[action])
        return np.clip(new_pos, [0, 0], [9, 15])


# ─────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────

class QLearningMAAL:
    """
    Hai Q-table hoàn toàn tách biệt:
      - Q_leader  : học policy để đi tới goal + phát tín hiệu legibility
      - Q_follower: học policy theo belief về goal của leader
    """

    def __init__(self, env, beta=0.01, alpha=0.1, gamma=0.9, epsilon=1.0):
        self.env = env
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_goals = 4

        # FIX 2: Tách riêng Q-table cho leader và follower
        self.Q_leader   = defaultdict(lambda: np.zeros(5))
        self.Q_follower = defaultdict(lambda: np.zeros(5))

    # ── Key helpers ──────────────────────────────────────────────────────────

    def get_leader_key(self, obs, goal_idx):
        """State của leader = (pos_leader, pos_follower, goal_idx)"""
        return tuple(obs) + (goal_idx,)

    def get_follower_key(self, obs, believed_goal_idx):
        """State của follower = (pos_leader, pos_follower, believed_goal_idx)"""
        return tuple(obs) + (believed_goal_idx,)

    # ── Action selection ─────────────────────────────────────────────────────

    def choose_leader_action(self, state_key):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        return int(np.argmax(self.Q_leader[state_key]))

    def choose_follower_action(self, state_key):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        return int(np.argmax(self.Q_follower[state_key]))

    # ── FIX 1: Bayesian update dùng action-alignment ─────────────────────────
    #
    #   P(action | goal_g) ∝ exp( temperature × cos(move, dir_to_goal) )
    #
    #   Lý do: action của leader hướng đến goal nào thì likelihood goal đó cao.
    #   Đây mới là tín hiệu "legibility" đúng nghĩa, không phải 1/dist(pos).

    def bayesian_update(self, belief, leader_pos, action, temperature=3.0):
        move_vec = np.array(self.env.moves[action], dtype=float)
        move_norm = np.linalg.norm(move_vec)

        likelihood = np.ones(self.num_goals) * 1e-6

        for g_idx, g_pos in enumerate(self.env.exits):
            direction = np.array(g_pos, dtype=float) - leader_pos
            dir_norm = np.linalg.norm(direction)

            if dir_norm < 1e-6:
                # Đã đứng tại goal → likelihood tối đa
                likelihood[g_idx] = np.exp(temperature)
                continue

            if move_norm < 1e-6:
                # Action = STAY → ít thông tin, likelihood đều
                likelihood[g_idx] = 1.0
            else:
                cos_sim = np.dot(move_vec, direction) / (move_norm * dir_norm)
                likelihood[g_idx] = np.exp(temperature * cos_sim)

        posterior = belief * likelihood
        total = posterior.sum()
        if total < 1e-10:
            return np.ones(self.num_goals) / self.num_goals
        return posterior / total

    # ── Q-update helpers ─────────────────────────────────────────────────────

    def update_leader(self, state_key, action, reward, next_state_key):
        best_next = np.max(self.Q_leader[next_state_key])
        td_error = reward + self.gamma * best_next - self.Q_leader[state_key][action]
        self.Q_leader[state_key][action] += self.alpha * td_error

    # FIX 3: Follower Q-update — được gọi mỗi bước
    def update_follower(self, state_key, action, reward, next_state_key):
        best_next = np.max(self.Q_follower[next_state_key])
        td_error = reward + self.gamma * best_next - self.Q_follower[state_key][action]
        self.Q_follower[state_key][action] += self.alpha * td_error


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────

def train_lfm(episodes=100_000, beta=0.01, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    env = LeadFollowMaze()

    # FIX 4: Epsilon bắt đầu 1.0, decay chậm hơn để đủ exploration
    agent = QLearningMAAL(env, beta=beta, epsilon=1.0)

    rewards, pcrs, ptrs = [], [], []
    MAX_STEPS = 200

    for ep in tqdm(range(episodes), desc=f"LFM β={beta}"):
        obs, _ = env.reset()
        leader_goal_idx = env.goal_idx
        belief = np.ones(4) / 4.0

        episode_reward = 0.0
        correct_steps = 0
        total_steps = 0
        predicted_goal = -1

        done = False
        step_count = 0

        while not done and step_count < MAX_STEPS:
            step_count += 1
            total_steps += 1

            # ── Leader chọn action ────────────────────────────────────────────
            leader_sk = agent.get_leader_key(obs, leader_goal_idx)
            leader_a  = agent.choose_leader_action(leader_sk)

            # ── Follower chọn action theo belief hiện tại ─────────────────────
            believed_goal  = int(np.argmax(belief))
            follower_sk    = agent.get_follower_key(obs, believed_goal)
            follower_a     = agent.choose_follower_action(follower_sk)

            # ── Bước môi trường ───────────────────────────────────────────────
            next_obs, r, done, _, _ = env.step([leader_a, follower_a])

            # ── FIX 1: Bayesian update dùng action-alignment ──────────────────
            log_p_before = np.log(belief[leader_goal_idx] + 1e-10)
            belief = agent.bayesian_update(belief, obs[:2], leader_a)
            log_p_after  = np.log(belief[leader_goal_idx] + 1e-10)

            # Legibility bonus: KL giảm = belief đúng tăng → thưởng dương
            delta_kl = log_p_after - log_p_before   # > 0 khi action legible
            r_leader   = r + beta * delta_kl
            r_follower = r  # Follower tối ưu hoá reward thô

            episode_reward += r_leader

            # ── Cập nhật Q_leader ─────────────────────────────────────────────
            next_leader_sk = agent.get_leader_key(next_obs, leader_goal_idx)
            agent.update_leader(leader_sk, leader_a, r_leader, next_leader_sk)

            # ── FIX 3: Cập nhật Q_follower mỗi bước ──────────────────────────
            next_believed_goal = int(np.argmax(belief))
            next_follower_sk   = agent.get_follower_key(next_obs, next_believed_goal)
            agent.update_follower(follower_sk, follower_a, r_follower, next_follower_sk)

            obs = next_obs
            predicted_goal = int(np.argmax(belief))
            if predicted_goal == leader_goal_idx:
                correct_steps += 1

        rewards.append(episode_reward)
        pcrs.append(1.0 if predicted_goal == leader_goal_idx else 0.0)
        ptrs.append(correct_steps / total_steps if total_steps > 0 else 0.0)

        # FIX 4: Decay chậm — đạt epsilon_min ~0.01 sau khoảng ep 50k
        agent.epsilon = max(0.01, agent.epsilon * 0.9999)

    # ── Lưu kết quả ──────────────────────────────────────────────────────────
    os.makedirs("results/lfm", exist_ok=True)
    os.makedirs("plots/lfm",   exist_ok=True)

    df = pd.DataFrame({"reward": rewards, "pcr": pcrs, "ptr": ptrs})
    df.to_csv(f"results/lfm/results_beta{beta}.csv", index=False)

    # ── Plot ─────────────────────────────────────────────────────────────────
    window = 500
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"LFM  β={beta}", fontsize=12)

    for ax, data, title in zip(
        axes,
        [rewards, pcrs, ptrs],
        ["Episode Reward (leader)", "PCR — Prediction Correct Rate", "PTR — Prediction Tracking Rate"],
    ):
        smoothed = np.convolve(data, np.ones(window) / window, mode="valid")
        ax.plot(smoothed)
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"plots/lfm/plot_beta{beta}.png", dpi=120)
    plt.close()

    last = slice(-3000, None)
    print(
        f"\n✓ LFM β={beta} | "
        f"Reward: {np.mean(rewards[last]):.3f} | "
        f"PCR: {np.mean(pcrs[last]):.3f} | "
        f"PTR: {np.mean(ptrs[last]):.3f}"
    )


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    for beta in [0.0, 0.01]:
        train_lfm(episodes=100_000, beta=beta)