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
    n_envs = 10

    # Create a vectorized environment
    # env = DummyVecEnv([make_env() for _ in range(n_envs)])

    # OR for SubprocVecEnv (more efficient for computationally intensive tasks)
    env = VecMonitor(SubprocVecEnv([make_env() for _ in range(n_envs)]))
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # setting policy
    policy_kwargs = dict(
        features_extractor_class=CustomCNNExtractor,
        features_extractor_kwargs=dict(features_dim=1024),
        net_arch=dict(pi=[1024, 1024], vf=[1024, 1024])  # Proper syntax
    )
    # Create a directory for TensorBoard logs
    log_dir = './tb_logs'

    model = MaskablePPO(
        "MlpPolicy", 
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=5e-5,  # Lower learning rate for smoother updates
        n_steps=4096,  # Increase timesteps per update for more stable gradients
        batch_size=1024,  # Increase batch size for more stable training
        gamma=0.99,  # Default gamma value for discounting
        gae_lambda=0.95,  # Default GAE Lambda for advantage estimation
        ent_coef=0.05,  # Entropy coefficient to encourage exploration
        clip_range=0.05,  # Smaller clip range for conservative updates
        target_kl=0.05,  # Allow larger policy updates before early stopping
        vf_coef=0.3,  # Lower value loss coefficient for more focus on policy updates
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
    
    if not os.path.isdir("data"):
        os.makedirs("data")
    model.save("data/ppo_mask")