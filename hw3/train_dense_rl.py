"""
HW3 Part 1: Train a dense MLP policy with PPO on privileged state.

Usage:
    python hw3/train_dense_rl.py \
        experiment.name=hw3_dense_ppo_seed0 \
        r_seed=0 \
        sim.task_set=libero_spatial \
        sim.eval_tasks=[9] \
        training.total_env_steps=200000 \
        training.rollout_length=128 \
        training.ppo_epochs=10 \
        training.minibatch_size=256
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from hw3.libero_env_fast import FastLIBEROEnv


# ---------------------------------------------------------------------------
# Policy and value networks
# ---------------------------------------------------------------------------

class DensePolicy(nn.Module):
    """
    MLP policy that maps privileged state observations to action distributions.
    Outputs a Gaussian distribution (mean + log_std) over the 7-DoF action space.
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()
        # TODO: Build the policy network layers and output heads.
        pass

    def forward(self, obs: torch.Tensor):
        """
        Args:
            obs: (B, obs_dim) float tensor
        Returns:
            dist: torch.distributions.Normal over actions
        """
        # TODO: Return a Normal distribution over actions given obs.
        pass

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """Sample an action and return (action, log_prob, entropy)."""
        dist = self.forward(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        action = action.clamp(-1.0, 1.0)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy


class DenseValueFunction(nn.Module):
    """MLP value function V(s) for PPO critic."""
    def __init__(self, obs_dim: int, hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()
        # TODO: Build the value network layers.
        pass

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns scalar value estimate of shape (B,)."""
        return self.value_head(self.net(obs)).squeeze(-1)


# ---------------------------------------------------------------------------
# PPO rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Stores a fixed-length on-policy rollout for PPO updates."""

    def __init__(self, rollout_length: int, obs_dim: int, action_dim: int, device: torch.device):
        self.rollout_length = rollout_length
        self.device = device
        self.obs = torch.zeros(rollout_length, obs_dim, device=device)
        self.actions = torch.zeros(rollout_length, action_dim, device=device)
        self.log_probs = torch.zeros(rollout_length, device=device)
        self.rewards = torch.zeros(rollout_length, device=device)
        self.values = torch.zeros(rollout_length, device=device)
        self.dones = torch.zeros(rollout_length, device=device)
        self.ptr = 0

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = done
        self.ptr += 1

    def full(self):
        return self.ptr >= self.rollout_length

    def reset(self):
        self.ptr = 0

    def compute_returns_and_advantages(self, last_value: torch.Tensor, gamma: float, gae_lambda: float):
        """
        Compute discounted returns and GAE advantages.

        Args:
            last_value: bootstrap value V(s_T) of shape ()
            gamma: discount factor
            gae_lambda: GAE lambda
        Returns:
            returns: (rollout_length,) tensor
            advantages: (rollout_length,) tensor
        """
        # TODO: Compute GAE advantages and discounted returns.
        returns = torch.zeros_like(self.rewards)
        advantages = torch.zeros_like(self.rewards)
        pass


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(policy: DensePolicy,
               value_fn: DenseValueFunction,
               optimizer: torch.optim.Optimizer,
               buffer: RolloutBuffer,
               returns: torch.Tensor,
               advantages: torch.Tensor,
               cfg: DictConfig):
    """
    Perform `ppo_epochs` passes of minibatch PPO updates on the stored rollout.

    Returns a dict of mean losses for logging.
    """
    # TODO: Implement PPO minibatch updates over ppo_epochs.
    return {}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@hydra.main(config_path="conf", config_name="dense_ppo", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.r_seed)
    np.random.seed(cfg.r_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Logging ---
    wandb.init(
        project=cfg.experiment.project,
        name=cfg.experiment.name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # --- Environment ---
    task_id = int(cfg.sim.eval_tasks[0])
    env = FastLIBEROEnv(
        task_id=task_id,
        max_episode_steps=cfg.sim.episode_length,
        cfg=cfg,
    )

    obs_dim = cfg.policy.obs_dim
    action_dim = cfg.policy.action_dim

    # --- Models ---
    policy = DensePolicy(obs_dim, action_dim, cfg.policy.hidden_dim, cfg.policy.n_layers).to(device)
    value_fn = DenseValueFunction(obs_dim, cfg.policy.hidden_dim, cfg.policy.n_layers).to(device)

    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value_fn.parameters()),
        lr=cfg.training.learning_rate,
    )

    buffer = RolloutBuffer(cfg.training.rollout_length, obs_dim, action_dim, device)

    # --- Rollout state ---
    obs, _ = env.reset()
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
    episode_return = 0.0
    episode_steps = 0
    total_steps = 0
    episode_returns = []
    episode_successes = []

    # --- Main loop ---
    while total_steps < cfg.training.total_env_steps:
        buffer.reset()

        # Collect one rollout
        with torch.no_grad():
            for _ in range(cfg.training.rollout_length):
                action, log_prob, _ = policy.get_action(obs_tensor.unsqueeze(0))
                value = value_fn(obs_tensor.unsqueeze(0))
                action_np = action.squeeze(0).cpu().numpy()

                next_obs, reward, done, truncated, info = env.step(action_np)
                episode_return += reward
                episode_steps += 1
                total_steps += 1

                buffer.add(
                    obs_tensor,
                    action.squeeze(0),
                    log_prob.squeeze(0),
                    torch.tensor(reward, device=device),
                    value.squeeze(0),
                    torch.tensor(float(done or truncated), device=device),
                )

                if done or truncated:
                    episode_returns.append(episode_return)
                    episode_successes.append(float(info.get("success_placed", 0.0)))
                    episode_return = 0.0
                    episode_steps = 0
                    obs, _ = env.reset()
                else:
                    obs = next_obs
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)

                if buffer.full():
                    break

            # Bootstrap last value
            last_value = value_fn(obs_tensor.unsqueeze(0)).squeeze(0)

        returns, advantages = buffer.compute_returns_and_advantages(
            last_value, cfg.training.gamma, cfg.training.gae_lambda
        )

        # PPO update
        update_info = ppo_update(policy, value_fn, optimizer, buffer, returns, advantages, cfg)

        # Logging
        if total_steps % cfg.log_interval < cfg.training.rollout_length:
            log_dict = {
                "train/total_steps": total_steps,
                **{f"train/{k}": v for k, v in update_info.items()},
            }
            if episode_returns:
                log_dict["train/episode_return"] = np.mean(episode_returns[-10:])
                log_dict["train/success_rate"] = np.mean(episode_successes[-10:])
            wandb.log(log_dict, step=total_steps)
            print(f"[{total_steps}/{cfg.training.total_env_steps}] "
                  f"return={log_dict.get('train/episode_return', float('nan')):.3f} "
                  f"policy_loss={update_info['policy_loss']:.4f}")

        # Checkpoint
        if total_steps % cfg.save_interval < cfg.training.rollout_length:
            ckpt = {
                "policy": policy.state_dict(),
                "value_fn": value_fn.state_dict(),
                "optimizer": optimizer.state_dict(),
                "total_steps": total_steps,
                "cfg": OmegaConf.to_container(cfg),
            }
            torch.save(ckpt, f"dense_ppo_{total_steps}.pth")

    # Final save
    torch.save({"policy": policy.state_dict(), "cfg": OmegaConf.to_container(cfg)},
               "dense_ppo_final.pth")
    env.close()
    wandb.finish()


if __name__ == "__main__":
    main()
