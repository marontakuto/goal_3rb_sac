import torch
import torch.nn as nn
import torch.optim as optim
from pfrl.agents import SoftActorCritic
from pfrl.replay_buffers import ReplayBuffer
from pfrl.utils.batch_states import batch_states

# Define example policy and Q-functions
class Policy(nn.Module):
    def __init__(self, obs_size, action_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )
    
    def forward(self, x):
        return torch.distributions.Normal(self.model(x), 1.0)

class QFunction(nn.Module):
    def __init__(self, obs_size, action_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_size + action_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        return self.model(x)

# Parameters
obs_size = 4  # Example observation size
action_size = 2  # Example action size
gamma = 0.99
replay_start_size = 1000
replay_buffer_size = 100000
minibatch_size = 64
gpu_id = -1  # Use CPU; set to 0 or above for GPU

# Instantiate policy, Q-functions, and optimizers
policy = Policy(obs_size, action_size)
q_func1 = QFunction(obs_size, action_size)
q_func2 = QFunction(obs_size, action_size)

policy_optimizer = optim.Adam(policy.parameters(), lr=3e-4)
q_func1_optimizer = optim.Adam(q_func1.parameters(), lr=3e-4)
q_func2_optimizer = optim.Adam(q_func2.parameters(), lr=3e-4)

# Replay buffer
replay_buffer = ReplayBuffer(replay_buffer_size)

# Instantiate SAC agent
agent = SoftActorCritic(
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
n_episodes = 5
for episode in range(n_episodes):
    obs = torch.randn(obs_size)  # Example initial observation
    done = False
    episode_reward = 0
    
    while not done:
        # Select action
        action = agent.batch_act([obs.numpy()])[0]

        # Convert action to PyTorch tensor
        action = torch.tensor(agent.batch_act([obs.numpy()])[0], dtype=torch.float32)
        
        # Scale the actions
        action_0 = torch.sigmoid(action[0])  # Scale to 0~1 range
        action_1 = torch.tanh(action[1]) * (3 * torch.pi / 4)  # Scale to -pi*3/4~pi*3/4 range
        
        print(f"Obs: {obs}, Action: [{action_0:.4f}, {action_1:.4f}]")
        
        # Simulate environment step (replace this with actual environment)
        next_obs = torch.randn(obs_size)  # Example next observation
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
