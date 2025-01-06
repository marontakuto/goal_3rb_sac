import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pfrl
from pfrl.agents import SoftActorCritic
from pfrl.replay_buffers import ReplayBuffer
import math
from pfrl.utils import clip_l2_grad_norm_

class CustomSoftActorCritic(SoftActorCritic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_q_func(self, batch):
        """Compute loss for a given Q-function."""

        batch_next_state = batch["next_state"]
        batch_rewards = batch["reward"]
        batch_terminal = batch["is_state_terminal"]
        batch_state = batch["state"]
        batch_actions = batch["action"]
        batch_discount = batch["discount"]

        with torch.no_grad(), pfrl.utils.evaluating(self.policy), pfrl.utils.evaluating(
            self.target_q_func1
        ), pfrl.utils.evaluating(self.target_q_func2):
            # 次状態からアクション分布を取得
            next_action_distrib = self.policy(batch_next_state)

            # サンプルされたアクション
            next_actions = next_action_distrib.sample()

            # ログ確率を計算
            next_log_prob = next_action_distrib.log_prob(next_actions)

            # 次状態でのQ値を計算
            next_q1 = self.target_q_func1((batch_next_state, next_actions))
            next_q2 = self.target_q_func2((batch_next_state, next_actions))
            next_q = torch.min(next_q1, next_q2)

            # エントロピー項を計算し形状を調整
            next_log_prob = next_log_prob.mean(dim=1, keepdim=True)  # アクション次元を平均化
            entropy_term = self.temperature * next_log_prob  # 形状: [64, 1]

            # 確認
            assert next_q.shape == entropy_term.shape

            target_q = batch_rewards + batch_discount * (
                1.0 - batch_terminal
            ) * torch.flatten(next_q - entropy_term)

        predict_q1 = torch.flatten(self.q_func1((batch_state, batch_actions)))
        predict_q2 = torch.flatten(self.q_func2((batch_state, batch_actions)))

        loss1 = 0.5 * F.mse_loss(target_q, predict_q1)
        loss2 = 0.5 * F.mse_loss(target_q, predict_q2)

        # Update stats
        self.q1_record.extend(predict_q1.detach().cpu().numpy())
        self.q2_record.extend(predict_q2.detach().cpu().numpy())
        self.q_func1_loss_record.append(float(loss1))
        self.q_func2_loss_record.append(float(loss2))

        self.q_func1_optimizer.zero_grad()
        loss1.backward()
        if self.max_grad_norm is not None:
            clip_l2_grad_norm_(self.q_func1.parameters(), self.max_grad_norm)
        self.q_func1_optimizer.step()

        self.q_func2_optimizer.zero_grad()
        loss2.backward()
        if self.max_grad_norm is not None:
            clip_l2_grad_norm_(self.q_func2.parameters(), self.max_grad_norm)
        self.q_func2_optimizer.step()
    
    def update_policy_and_temperature(self, batch):
        """Compute loss for actor."""

        batch_state = batch["state"]

        # 現在の状態でのアクション分布を計算
        action_distrib = self.policy(batch_state)
        sampled_actions = action_distrib.sample()
        log_prob = action_distrib.log_prob(sampled_actions)

        # Q値の計算
        q1 = self.q_func1((batch_state, sampled_actions))
        q2 = self.q_func2((batch_state, sampled_actions))
        q = torch.min(q1, q2)  # 最小値を使用

        # エントロピー項の計算
        entropy_term = self.temperature * log_prob.mean(dim=1, keepdim=True)

        # 確認
        assert q.shape == entropy_term.shape, f"q.shape={q.shape}, entropy_term.shape={entropy_term.shape}"
        loss = torch.mean(entropy_term - q)

        self.policy_optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            clip_l2_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()

        self.n_policy_updates += 1

        if self.entropy_target is not None:
            self.update_temperature(log_prob.detach())

        # Record entropy
        with torch.no_grad():
            try:
                self.entropy_record.extend(
                    action_distrib.entropy().detach().cpu().numpy()
                )
            except NotImplementedError:
                # Record - log p(x) instead
                self.entropy_record.extend(-log_prob.detach().cpu().numpy())

# Define example policy and Q-functions
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, mid_units):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, mid_units)
        self.fc2 = nn.Linear(mid_units, mid_units)
        self.fc3 = nn.Linear(mid_units, action_dim)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.distributions.Normal(x, 1.0)

class QFunction(nn.Module):
    def __init__(self, state_dim, action_dim, mid_units):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, mid_units)
        self.fc2 = nn.Linear(mid_units, mid_units)
        self.fc3 = nn.Linear(mid_units, 1)
    
    def forward(self, stat_act):
        state, action = stat_act
        print(state, action, stat_act)
        stat_act = torch.cat([state, action], dim=-1)  # 状態と行動を結合
        x = F.relu(self.fc1(stat_act))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

# 畳み込み・プーリングを終えた画像の1次元入力数の計算
def calculate(img_width, img_height, channels, kernels, pool_info):
    cnv_num = len(channels) # 畳み込み回数
    pool_interval = pool_info[0] # プーリングする間隔(何回の畳み込みごとか)
    for i in range(cnv_num):
        img_width = img_width - (kernels[i] - 1)
        img_height = img_height - (kernels[i] - 1)
        if (i + 1) % pool_interval == 0:
            img_width = math.ceil(img_width / 2)
            img_height = math.ceil(img_height / 2)
    img_input = img_width * img_height * channels[-1]
    # print(img_width, img_height, img_input)
    return img_input

# Parameters
state_dim = 4  # Example observation size
action_dim = 2  # Example action size
mid_units = 256
gamma = 0.99
replay_start_size = 64
replay_buffer_size = 100
minibatch_size = 64
gpu_id = -1  # Use CPU; set to 0 or above for GPU

# Instantiate policy, Q-functions, and optimizers
policy = Policy(state_dim, action_dim, mid_units)
q_func1 = QFunction(state_dim, action_dim, mid_units)
q_func2 = QFunction(state_dim, action_dim, mid_units)

policy_optimizer = optim.Adam(policy.parameters(), lr=3e-4)
q_func1_optimizer = optim.Adam(q_func1.parameters(), lr=3e-4)
q_func2_optimizer = optim.Adam(q_func2.parameters(), lr=3e-4)

# Replay buffer
replay_buffer = ReplayBuffer(replay_buffer_size)

# Instantiate SAC agent
agent = CustomSoftActorCritic(
    policy=policy,
    q_func1=q_func1,
    q_func2=q_func2,
    policy_optimizer=policy_optimizer,
    q_func1_optimizer=q_func1_optimizer,
    q_func2_optimizer=q_func2_optimizer,
    replay_buffer=replay_buffer,
    gamma=gamma,
    gpu=gpu_id,
    replay_start_size=replay_start_size,
    minibatch_size=minibatch_size,
)

# Training loop example
n_episodes = 10
for episode in range(n_episodes):
    obs = torch.randn(state_dim)  # Example initial observation
    done = False
    episode_reward = 0
    
    while not done:
        # Select action
        action = agent.batch_act([obs.numpy()])[0]

        # Convert action to PyTorch tensor
        action = torch.tensor(action, dtype=torch.float32)
        
        # Scale the actions
        action_0 = torch.sigmoid(action[0]) * 2  # Scale to 0~1 range
        action_1 = torch.tanh(action[1]) * (3 * torch.pi / 4)  # Scale to -pi*3/4~pi*3/4 range
        
        print(f"Obs: {obs}, Action: [{action_0:.4f}, {action_1:.4f}]")
        
        # Simulate environment step (replace this with actual environment)
        next_obs = torch.randn(state_dim)  # Example next observation
        reward = torch.randn(1).item()  # Example reward
        done = torch.rand(1).item() > 0.9  # Example done condition
        reset = False

        # Observe the step
        agent.batch_observe([next_obs.numpy()], [reward], [done], [reset])
        
        # Update observation and accumulate reward
        obs = next_obs
        episode_reward += reward
    
    print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

# Save agent parameters
# agent.save('sac_agent')
