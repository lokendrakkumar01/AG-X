"""
Reinforcement Learning Optimizer
=================================

RL agents for optimizing hypothetical anti-gravity efficiency metrics.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field
import gymnasium as gym
from gymnasium import spaces


@dataclass
class OptimizationResult:
    """Result from RL optimization."""
    best_action: np.ndarray
    best_reward: float
    reward_history: np.ndarray
    policy_params: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class GravityOptimizationEnv(gym.Env):
    """
    Custom Gym environment for gravity simulation optimization.
    
    The agent learns to adjust simulation parameters to maximize
    a hypothetical anti-gravity efficiency metric.
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, 
                 param_dim: int = 10,
                 objective_fn: Optional[Callable] = None,
                 max_steps: int = 100):
        super().__init__()
        
        self.param_dim = param_dim
        self.max_steps = max_steps
        self.objective_fn = objective_fn or self._default_objective
        
        # Action space: parameter adjustments
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(param_dim,), dtype=np.float32
        )
        
        # Observation space: current parameters + metrics
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(param_dim + 5,), dtype=np.float32
        )
        
        self.current_params = None
        self.step_count = 0
        self.best_reward = float('-inf')
    
    def _default_objective(self, params: np.ndarray) -> float:
        """Default objective: maximize some complex function."""
        # Example: Rastrigin-like function (inverted for maximization)
        A = 10
        n = len(params)
        value = A * n + np.sum(params**2 - A * np.cos(2 * np.pi * params))
        return -value  # Maximize by returning negative
    
    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        metrics = np.array([
            self.step_count / self.max_steps,
            self.best_reward if np.isfinite(self.best_reward) else 0,
            np.mean(self.current_params),
            np.std(self.current_params),
            np.max(np.abs(self.current_params)),
        ])
        return np.concatenate([self.current_params, metrics]).astype(np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_params = self.np_random.uniform(-1, 1, self.param_dim)
        self.step_count = 0
        self.best_reward = float('-inf')
        return self._get_obs(), {}
    
    def step(self, action: np.ndarray):
        self.step_count += 1
        
        # Apply action as parameter adjustment
        self.current_params = np.clip(
            self.current_params + action * 0.1,
            -5.0, 5.0
        )
        
        # Calculate reward
        reward = self.objective_fn(self.current_params)
        self.best_reward = max(self.best_reward, reward)
        
        # Termination
        terminated = self.step_count >= self.max_steps
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, {
            "best_reward": self.best_reward,
            "params": self.current_params.copy(),
        }


class RLOptimizer:
    """
    Reinforcement Learning Optimizer for Physics Parameters.
    
    Uses stable-baselines3 PPO or SAC to learn optimal parameter settings.
    """
    
    def __init__(self, algorithm: str = "PPO", device: str = "auto"):
        self.algorithm = algorithm
        self.device = device
        self.model = None
        self.env = None
        self.training_history = []
    
    def create_env(self, param_dim: int, objective_fn: Callable) -> GravityOptimizationEnv:
        """Create optimization environment."""
        self.env = GravityOptimizationEnv(param_dim=param_dim, objective_fn=objective_fn)
        return self.env
    
    def train(self, 
              objective_fn: Callable[[np.ndarray], float],
              param_dim: int = 10,
              total_timesteps: int = 10000,
              learning_rate: float = 3e-4) -> OptimizationResult:
        """
        Train RL agent to optimize objective function.
        
        Args:
            objective_fn: Function to maximize
            param_dim: Number of parameters
            total_timesteps: Training steps
            learning_rate: Learning rate
        """
        try:
            from stable_baselines3 import PPO, SAC
            from stable_baselines3.common.callbacks import BaseCallback
            
            env = self.create_env(param_dim, objective_fn)
            
            # Create model
            if self.algorithm == "SAC":
                self.model = SAC("MlpPolicy", env, learning_rate=learning_rate,
                                verbose=0, device=self.device)
            else:
                self.model = PPO("MlpPolicy", env, learning_rate=learning_rate,
                                verbose=0, device=self.device)
            
            # Training callback to track rewards
            class RewardCallback(BaseCallback):
                def __init__(self):
                    super().__init__()
                    self.rewards = []
                
                def _on_step(self):
                    if self.locals.get("rewards"):
                        self.rewards.extend(self.locals["rewards"])
                    return True
            
            callback = RewardCallback()
            self.model.learn(total_timesteps=total_timesteps, callback=callback)
            
            # Get best action
            obs, _ = env.reset()
            best_action = None
            best_reward = float('-inf')
            
            for _ in range(100):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
                if reward > best_reward:
                    best_reward = reward
                    best_action = info["params"]
                if done:
                    obs, _ = env.reset()
            
            return OptimizationResult(
                best_action=best_action,
                best_reward=best_reward,
                reward_history=np.array(callback.rewards),
                metadata={"algorithm": self.algorithm, "timesteps": total_timesteps}
            )
            
        except ImportError:
            # Fallback without stable-baselines3
            return self._random_search(objective_fn, param_dim, total_timesteps)
    
    def _random_search(self, objective_fn: Callable, 
                       param_dim: int, n_samples: int) -> OptimizationResult:
        """Fallback random search optimization."""
        best_params = None
        best_reward = float('-inf')
        rewards = []
        
        for _ in range(n_samples):
            params = np.random.uniform(-5, 5, param_dim)
            reward = objective_fn(params)
            rewards.append(reward)
            
            if reward > best_reward:
                best_reward = reward
                best_params = params
        
        return OptimizationResult(
            best_action=best_params,
            best_reward=best_reward,
            reward_history=np.array(rewards),
            metadata={"algorithm": "random_search", "samples": n_samples}
        )
    
    def optimize(self, objective_fn: Callable[[np.ndarray], float],
                 param_dim: int = 10,
                 **kwargs) -> OptimizationResult:
        """Convenience method for training."""
        return self.train(objective_fn, param_dim, **kwargs)
