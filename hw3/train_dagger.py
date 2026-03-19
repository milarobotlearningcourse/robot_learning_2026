"""
HW3 Part 3 (Optional): DAgger — distill a dense teacher into a transformer student,
then fine-tune the student with RL.

Usage:
    python hw3/train_dagger.py \
        experiment.name=hw3_dagger_seed0 \
        r_seed=0 \
        teacher_checkpoint=/path/to/dense_policy.pth \
        student_init_checkpoint=/path/to/hw1/miniGRP.pth \
        dagger.num_rounds=10 \
        dagger.rollouts_per_round=20
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../mini-grp'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from hw3.libero_env_fast import FastLIBEROEnv
from hw3.train_dense_rl import DensePolicy
from hw3.train_transformer_rl import TransformerPolicyWrapper


# ---------------------------------------------------------------------------
# DAgger dataset
# ---------------------------------------------------------------------------

class DAggerDataset(torch.utils.data.Dataset):
    """
    Aggregated dataset for DAgger.
    Each entry is an (obs, teacher_action) pair collected across all rounds.
    """

    def __init__(self):
        self.obs_list = []
        self.action_list = []

    def add_rollout(self, obs_seq: list, actions_seq: list):
        """Append a rollout's (obs, teacher_action) pairs to the dataset."""
        self.obs_list.extend(obs_seq)
        self.action_list.extend(actions_seq)

    def __len__(self):
        return len(self.obs_list)

    def __getitem__(self, idx):
        obs = torch.tensor(self.obs_list[idx], dtype=torch.float32)
        action = torch.tensor(self.action_list[idx], dtype=torch.float32)
        return obs, action

    def save(self, path: str):
        torch.save({"obs": self.obs_list, "actions": self.action_list}, path)

    def load(self, path: str):
        data = torch.load(path)
        self.obs_list = data["obs"]
        self.action_list = data["actions"]


# ---------------------------------------------------------------------------
# Teacher wrapper
# ---------------------------------------------------------------------------

class DensePolicyTeacher:
    """
    Wraps the trained DensePolicy checkpoint for action labeling.
    """

    def __init__(self, checkpoint_path: str, obs_dim: int, action_dim: int,
                 hidden_dim: int, n_layers: int, device: torch.device):
        ckpt = torch.load(checkpoint_path, map_location=device)
        self.policy = DensePolicy(obs_dim, action_dim, hidden_dim, n_layers).to(device)
        self.policy.load_state_dict(ckpt["policy"])
        self.policy.eval()
        self.device = device

    def get_action(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Query teacher for a deterministic action label."""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = self.policy.get_action(obs_t, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()


# ---------------------------------------------------------------------------
# DAgger rollout collection
# ---------------------------------------------------------------------------

def collect_dagger_rollout(env: FastLIBEROEnv,
                            student: TransformerPolicyWrapper,
                            teacher: DensePolicyTeacher,
                            beta: float,
                            max_steps: int):
    """
    Collect one DAgger rollout.

    At each step:
      - With probability beta, follow the teacher; else follow the student.
      - Always label the visited state with the teacher's action.

    Args:
        beta: teacher mixing probability (1 = always teacher, 0 = always student)
    Returns:
        obs_list: list of obs arrays visited by the student
        teacher_actions: list of teacher-labeled actions at each obs
        total_return: sum of environment rewards
        success: whether the task was completed
    """
    # TODO: Roll out the student, label each visited state with the teacher, mix using beta.
    obs_list = []
    teacher_actions = []
    total_return = 0.0
    success = False
    return obs_list, teacher_actions, total_return, success


# ---------------------------------------------------------------------------
# Behavior cloning update step
# ---------------------------------------------------------------------------

def bc_update(student: TransformerPolicyWrapper,
              dataset: DAggerDataset,
              optimizer: torch.optim.Optimizer,
              cfg: DictConfig,
              device: torch.device):
    """
    Run `bc_epochs_per_round` supervised epochs on the aggregated DAgger dataset.

    Returns dict with "bc_loss".
    """
    # TODO: Supervised update on the aggregated dataset for bc_epochs_per_round epochs.
    return {"bc_loss": 0.0}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@hydra.main(config_path="conf", config_name="dagger", version_base=None)
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

    # Teacher: dense PPO policy from Part 1
    teacher = DensePolicyTeacher(
        cfg.teacher_checkpoint, obs_dim, action_dim,
        hidden_dim=256, n_layers=3, device=device
    )

    # Student: transformer policy initialized from HW1
    student = TransformerPolicyWrapper(cfg.student_init_checkpoint, device, cfg)
    student_optimizer = torch.optim.Adam(student.parameters(), lr=cfg.training.learning_rate)

    # Aggregated DAgger dataset
    dataset = DAggerDataset()
    os.makedirs(cfg.dagger.dataset_save_dir, exist_ok=True)

    all_returns = []
    all_successes = []

    # --- DAgger rounds ---
    for round_idx in range(cfg.dagger.num_rounds):

        # Compute beta (teacher mixing coefficient)
        # TODO: Compute beta for this round according to cfg.dagger.beta_schedule.
        beta = cfg.dagger.beta_init

        print(f"\n=== DAgger Round {round_idx + 1}/{cfg.dagger.num_rounds} | beta={beta:.3f} ===")

        # Collect rollouts and label with teacher
        round_returns = []
        round_successes = []
        for _ in range(cfg.dagger.rollouts_per_round):
            obs_seq, teacher_acts, ret, success = collect_dagger_rollout(
                env, student, teacher, beta=beta, max_steps=cfg.sim.episode_length
            )
            dataset.add_rollout(obs_seq, teacher_acts)
            round_returns.append(ret)
            round_successes.append(float(success))

        all_returns.extend(round_returns)
        all_successes.extend(round_successes)

        # Save dataset for this round
        dataset.save(os.path.join(cfg.dagger.dataset_save_dir, f"dagger_round_{round_idx:03d}.pth"))

        # Behavior cloning update on aggregated dataset
        bc_info = bc_update(student, dataset, student_optimizer, cfg, device)

        log = {
            "dagger/round": round_idx,
            "dagger/beta": beta,
            "dagger/dataset_size": len(dataset),
            "dagger/episode_return": np.mean(round_returns),
            "dagger/success_rate": np.mean(round_successes),
            **{f"dagger/{k}": v for k, v in bc_info.items()},
        }
        wandb.log(log, step=round_idx)
        print(f"  return={np.mean(round_returns):.3f}  "
              f"success={np.mean(round_successes):.2f}  "
              f"bc_loss={bc_info['bc_loss']:.4f}  "
              f"dataset_size={len(dataset)}")

        # Periodic checkpoint
        if (round_idx + 1) % cfg.save_interval == 0:
            torch.save({
                "student": {k: v for k, v in student.model.state_dict().items()},
                "round": round_idx,
                "cfg": OmegaConf.to_container(cfg),
            }, f"dagger_student_round{round_idx:03d}.pth")

    # Save final student
    torch.save({
        "student": {k: v for k, v in student.model.state_dict().items()},
        "cfg": OmegaConf.to_container(cfg),
    }, "dagger_student_final.pth")
    print("\nDAgger training complete. Final student saved to dagger_student_final.pth")
    print("You can now fine-tune with RL using train_transformer_rl.py and "
          "init_checkpoint=dagger_student_final.pth")

    env.close()
    wandb.finish()


if __name__ == "__main__":
    main()
