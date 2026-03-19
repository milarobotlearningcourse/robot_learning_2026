"""
HW3 Part 2: Fine-tune a transformer policy from HW1 with PPO or GRPO.

Usage (PPO):
    python hw3/train_transformer_rl.py \
        experiment.name=hw3_transformer_ppo_seed0 \
        r_seed=0 \
        init_checkpoint=/path/to/hw1/miniGRP.pth \
        rl.algorithm=ppo \
        sim.task_set=libero_spatial \
        sim.eval_tasks=[9]

Usage (GRPO with ground-truth resets):
    python hw3/train_transformer_rl.py \
        experiment.name=hw3_transformer_grpo_seed0 \
        r_seed=0 \
        init_checkpoint=/path/to/hw1/miniGRP.pth \
        rl.algorithm=grpo \
        sim.task_set=libero_spatial \
        sim.eval_tasks=[9]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../mini-grp'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from hw3.libero_env_fast import FastLIBEROEnv
from hw3.train_dense_rl import RolloutBuffer, ppo_update


# ---------------------------------------------------------------------------
# Separate value network (used with transformer policy)
# ---------------------------------------------------------------------------

class ValueFunction(nn.Module):
    """
    Separate MLP value network V(s).
    Keep this separate from the transformer policy as required by hw3.md.
    """
    def __init__(self, obs_dim: int, hidden_dim: int = 256, n_layers: int = 2):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(obs)).squeeze(-1)


# ---------------------------------------------------------------------------
# Transformer policy wrapper
# ---------------------------------------------------------------------------

class TransformerPolicyWrapper:
    """
    Wraps the HW1 GRP transformer model to provide a gym-style action interface.

    The transformer policy expects a history of observations and actions;
    this wrapper maintains the required context window internally.
    """

    def __init__(self, checkpoint_path: str, device: torch.device, cfg: DictConfig):
        # TODO: Load the HW1 transformer checkpoint and reconstruct the model.
        self.model = None
        self.device = device
        self.cfg = cfg
        pass

    def reset_context(self):
        self._context = []

    def get_action(self, obs: np.ndarray, deterministic: bool = False):
        """
        Query the transformer for an action given the current observation.

        Args:
            obs: (obs_dim,) numpy array
            deterministic: if True return mean, else sample
        Returns:
            action: (action_dim,) numpy array
            log_prob: scalar tensor
            entropy: scalar tensor
        """
        # TODO: Run a forward pass through the transformer and return (action, log_prob, entropy).
        raise NotImplementedError

    def parameters(self):
        return self.model.parameters()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()


# ---------------------------------------------------------------------------
# GRPO helpers
# ---------------------------------------------------------------------------

def collect_grpo_group(env: FastLIBEROEnv,
                       policy: TransformerPolicyWrapper,
                       init_state,
                       group_size: int,
                       max_steps: int,
                       device: torch.device):
    """
    Reset to the same initial state and collect `group_size` trajectories.

    Returns a list of trajectory dicts, each containing:
        obs, actions, log_probs, rewards, dones, total_return
    """
    # TODO: Collect group_size trajectories all starting from init_state.
    trajectories = []
    return trajectories


def grpo_update(policy: TransformerPolicyWrapper,
                value_fn: ValueFunction,
                policy_optimizer: torch.optim.Optimizer,
                trajectories_per_group: list,
                cfg: DictConfig,
                device: torch.device):
    """
    GRPO update: compute group-relative advantages and update policy.

    Args:
        trajectories_per_group: list of lists; each inner list is a group of
            trajectory dicts collected from the same initial state.
    Returns:
        dict with "policy_loss", "mean_return"
    """
    # TODO: Compute group-relative advantages and apply a clipped surrogate loss.
    return {"policy_loss": 0.0, "mean_return": 0.0}


# ---------------------------------------------------------------------------
# GRPO with world model (Part 2d)
# ---------------------------------------------------------------------------

def grpo_worldmodel_update(policy: TransformerPolicyWrapper,
                            world_model,
                            current_obs: np.ndarray,
                            group_size: int,
                            horizon: int,
                            cfg: DictConfig,
                            device: torch.device):
    """
    GRPO using the HW2 world model to generate imagined trajectories.

    Args:
        world_model: trained HW2 world model (SimpleWorldModel or DreamerV3)
        current_obs: (obs_dim,) current real observation used as rollout start
        group_size: number of imagined trajectories per state
        horizon: number of imagination steps
    Returns:
        dict with "policy_loss", "mean_imagined_return"
    """
    # TODO: Roll out imagined trajectories using the world model and apply GRPO.
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@hydra.main(config_path="conf", config_name="transformer_rl", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.r_seed)
    np.random.seed(cfg.r_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project=cfg.experiment.project,
        name=cfg.experiment.name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    task_id = int(cfg.sim.eval_tasks[0])
    env = FastLIBEROEnv(task_id=task_id, max_episode_steps=cfg.sim.episode_length, cfg=cfg)

    obs_dim = env.obs_dim
    action_dim = env._action_dim

    # Load transformer policy from HW1 checkpoint
    policy = TransformerPolicyWrapper(cfg.init_checkpoint, device, cfg)

    # Separate value function (required by hw3.md)
    value_fn = ValueFunction(obs_dim, cfg.value.hidden_dim, cfg.value.n_layers).to(device)

    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.training.learning_rate)
    value_optimizer = torch.optim.Adam(value_fn.parameters(), lr=cfg.value.learning_rate)

    algorithm = cfg.rl.algorithm.lower()

    if algorithm == "ppo":
        # ------------------------------------------------------------------
        # PPO loop (reuses the buffer + update from Part 1)
        # ------------------------------------------------------------------
        buffer = RolloutBuffer(cfg.training.rollout_length, obs_dim, action_dim, device)
        optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(value_fn.parameters()),
            lr=cfg.training.learning_rate,
        )

        obs, _ = env.reset()
        policy.reset_context()
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        total_steps = 0
        episode_returns, episode_successes = [], []
        ep_ret = 0.0

        while total_steps < cfg.training.total_env_steps:
            buffer.reset()
            with torch.no_grad():
                for _ in range(cfg.training.rollout_length):
                    action_np, log_prob, _ = policy.get_action(obs)
                    value = value_fn(obs_t.unsqueeze(0))
                    next_obs, reward, done, truncated, info = env.step(action_np)
                    ep_ret += reward
                    total_steps += 1
                    buffer.add(
                        obs_t,
                        torch.tensor(action_np, device=device),
                        log_prob if isinstance(log_prob, torch.Tensor) else torch.tensor(log_prob, device=device),
                        torch.tensor(reward, device=device),
                        value.squeeze(0),
                        torch.tensor(float(done or truncated), device=device),
                    )
                    if done or truncated:
                        episode_returns.append(ep_ret)
                        episode_successes.append(float(info.get("success_placed", 0.0)))
                        ep_ret = 0.0
                        obs, _ = env.reset()
                        policy.reset_context()
                    else:
                        obs = next_obs
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
                    if buffer.full():
                        break
                last_value = value_fn(obs_t.unsqueeze(0)).squeeze(0)

            returns, advantages = buffer.compute_returns_and_advantages(
                last_value, cfg.training.gamma, cfg.training.gae_lambda
            )
            update_info = ppo_update(policy, value_fn, optimizer, buffer, returns, advantages, cfg)

            if total_steps % cfg.log_interval < cfg.training.rollout_length:
                log = {"train/total_steps": total_steps, **{f"train/{k}": v for k, v in update_info.items()}}
                if episode_returns:
                    log["train/episode_return"] = np.mean(episode_returns[-10:])
                    log["train/success_rate"] = np.mean(episode_successes[-10:])
                wandb.log(log, step=total_steps)
                print(f"[PPO {total_steps}] return={log.get('train/episode_return', float('nan')):.3f}")

    elif algorithm == "grpo":
        # ------------------------------------------------------------------
        # GRPO loop with ground-truth resets (Part 2c)
        # ------------------------------------------------------------------
        total_steps = 0
        update_count = 0
        all_returns = []

        while total_steps < cfg.training.total_env_steps:
            # Collect groups: reset to different initial states
            # TODO: Reset env, capture init_state, call collect_grpo_group() num_groups times.
            trajectories_per_group = []

            update_info = grpo_update(policy, value_fn, policy_optimizer,
                                      trajectories_per_group, cfg, device)
            update_count += 1
            all_returns.extend([t["total_return"] for g in trajectories_per_group for t in g])

            log = {
                "train/total_steps": total_steps,
                "train/update": update_count,
                **{f"train/{k}": v for k, v in update_info.items()},
                "train/episode_return": np.mean(all_returns[-50:]) if all_returns else 0.0,
            }
            wandb.log(log, step=total_steps)
            print(f"[GRPO {total_steps}] return={log['train/episode_return']:.3f} "
                  f"policy_loss={update_info['policy_loss']:.4f}")

    else:
        raise ValueError(f"Unknown rl.algorithm: {algorithm}. Choose 'ppo' or 'grpo'.")

    # Save final checkpoint
    torch.save({
        "policy": {k: v for k, v in policy.model.state_dict().items()},
        "value_fn": value_fn.state_dict(),
        "cfg": OmegaConf.to_container(cfg),
    }, "transformer_rl_final.pth")

    env.close()
    wandb.finish()


if __name__ == "__main__":
    main()
