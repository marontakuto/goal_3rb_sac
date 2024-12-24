import gymnasium as gym
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# 環境の作成（dtype の指定は不要）
env = Monitor(gym.make("Pendulum-v1", render_mode="human"))
eval_env = Monitor(gym.make("Pendulum-v1", render_mode="human"))

# SACモデルの作成
model = SAC("MlpPolicy", env, verbose=1, learning_rate=0.001, gamma=0.99, buffer_size=100000)

# 評価用コールバック
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/",
    log_path="./logs/",
    eval_freq=1000,
    deterministic=True,
)

# モデルの学習
print("Training the SAC model...")
model.learn(total_timesteps=20000, callback=eval_callback)

# モデルの保存
model.save("sac_pendulum")

# 学習後の結果をテスト
obs, _ = eval_env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = eval_env.step(action)
    eval_env.render()
    if done:
        obs, _ = eval_env.reset()

eval_env.close()
