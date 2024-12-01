import os

from chessrl.cnnextractor import CustomCNNExtractor
from chess_gymnasium_env.envs.chess import ChessEnv
import gymnasium as gym
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib import MaskablePPO

from chessrl.utils import make_env
from stable_baselines3.common.vec_env import VecMonitor

if __name__=="__main__":
    
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    # Number of parallel environments
    n_envs = 12

    # Create a vectorized environment
    # env = DummyVecEnv([make_env() for _ in range(n_envs)])

    # OR for SubprocVecEnv (more efficient for computationally intensive tasks)
    env = VecMonitor(SubprocVecEnv([make_env() for _ in range(n_envs)]))
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # setting policy
    policy_kwargs = dict(
        features_extractor_class=CustomCNNExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[dict(pi=[256, 256], vf=[512, 256])]  # Proper syntax
    )
    # Create a directory for TensorBoard logs
    log_dir = './tb_logs'

    model = MaskablePPO(
        "MlpPolicy", 
        env,
        policy_kwargs=policy_kwargs,
        gamma=0.995,              # Default gamma for PPO is 0.99, higher discount factor for long-term rewards
        learning_rate=0.0001,
        n_steps=2048,
        batch_size=1024,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        max_grad_norm=0.5,
        vf_coef=0.5,
        n_epochs=10,  # More epochs per update
        target_kl=0.01,  # KL divergence target for stability
        seed=32,                # Random seed for reproducibility
        verbose=1,              # Set verbosity level to 1 for progress logging
        tensorboard_log=log_dir
    )
    print(model.policy)
    print(f"Model is running on device: {model.policy.device}")
    
    try:
        model.learn(5_000_000)
    except KeyboardInterrupt:
        pass
    
    os.makedirs("models")
    model.save("models/ppo_mask")