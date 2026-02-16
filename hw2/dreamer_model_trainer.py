import sys

import dill
from omegaconf import DictConfig, OmegaConf
import hydra
import torch
from torchvision import transforms
import os
import h5py
import numpy as np
from dreamerV3 import DreamerV3
from simple_world_model import SimpleWorldModel
from planning import CEMPlanner, PolicyPlanner, RandomPlanner
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv, DenseRewardEnv
import random
from sim_eval import eval_libero
from collections import deque


# Factory function to instantiate the correct model
def create_model(model_type, img_shape, action_dim, device, cfg):
    """
    Factory function to create a world model based on the specified type.
    
    Args:
        model_type: 'dreamer' or 'simple' 
        img_shape: Image shape [C, H, W]
        action_dim: Dimensionality of actions
        device: torch device
        cfg: Configuration object
        
    Returns:
        model: Instantiated model
    """
    if model_type.lower() == 'dreamer':
        model = DreamerV3(obs_shape=img_shape, action_dim=action_dim, cfg=cfg).to(device)
    elif model_type.lower() == 'simple':
        model = SimpleWorldModel(action_dim=action_dim, pose_dim=7, hidden_dim=256, cfg=cfg).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'dreamer' or 'simple'.")
    
    return model


class ModelTrainingWrapper:
    """
    Wrapper to provide unified interface for training different world models.
    Handles differences in forward passes and loss computation between models.
    """
    def __init__(self, model, model_type, device):
        self.model = model
        self.model_type = model_type.lower()
        self.device = device
        
    def forward_pass(self, images, poses, actions):
        """
        Unified forward pass that works with both model types.
        
        Args:
            images: Image tensor (B, T, H, W, C) or None for simple model
            poses: Pose tensor (B, T, 7)
            actions: Action tensor (B, T, 7)
            
        Returns:
            output: Model output (format depends on model type)
        """
        if self.model_type == 'dreamer':
            return self.model(images, actions)
        elif self.model_type == 'simple':
            # SimpleWorldModel expects normalized inputs
            return self.model(poses, actions)
    
    def compute_loss(self, output, images, rewards, dones, poses, actions ):
        """
        Compute loss in a way that works for both model types.
        
        Args:
            output: Output from forward_pass
            images: Image tensor
            rewards: Reward tensor
            dones: Done tensor
            poses: Pose tensor (used for SimpleWorldModel)
            actions: Action tensor (used for SimpleWorldModel)
            pred_coeff, dyn_coeff, rep_coeff: Loss coefficients (used for DreamerV3)
            
        Returns:
            losses: Dictionary with loss information
        """
        if self.model_type == 'dreamer':
            # Use DreamerV3 loss computation
            return self.model.compute_loss(
                output, images, rewards, dones, self.device
                
            )
        elif self.model_type == 'simple':
            # TODO: Part 1.2 - Implement SimpleWorldModel training loss
            ## Compute MSE loss between predicted and target poses/rewards
            pass


class LIBERODataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # crawl the data_dir and build the index map for h5py files
        self.index_map = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.hdf5') or file.endswith('.h5'):
                    file_path = os.path.join(root, file)
                    with h5py.File(file_path, 'r') as f:
                        for demo_key in f['data'].keys():
                            self.index_map.append((file_path, demo_key))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        # Load your data here
        # data_path = os.path.join(self.data_dir, self.data_files[idx])
        file_path, demo_key = self.index_map[idx]
        # data_list = []
        with h5py.File(file_path, 'r') as f:
            # for demo in f['data'].keys():
            demo = f['data'][demo_key]
            image = torch.from_numpy(f['data'][demo_key]['obs']['agentview_rgb'][()])
            action = torch.from_numpy(f['data'][demo_key]['actions'][()])
            dones = torch.from_numpy(f['data'][demo_key]['dones'][()])
            rewards = torch.from_numpy(f['data'][demo_key]['rewards'][()])
            # poses = torch.from_numpy(f['data'][demo_key]['robot_states'][()])
            poses = torch.from_numpy(np.concatenate( (f['data'][demo_key]['obs']["ee_pos"], 
                                        f['data'][demo_key]['obs']["ee_ori"][:,:3],
                                        (f['data'][demo_key]['obs']["gripper_states"][:,:1])), axis=-1))
            # Note: Images are returned in channel-last format (T, H, W, C)
            # Conversion to channel-first (T, C, H, W) happens in the training loop
        return image, action, rewards, dones, poses  # Return the image and label if needed


class CircularBufferDataset(torch.utils.data.Dataset):
    """Circular buffer dataset that holds up to max_trajectories.
    When full, oldest trajectories are overwritten.
    """
    def __init__(self, cfg=None, data_dir=None):
        self.trajectories = []
        self.write_idx = 0
        self._cfg=cfg

        if data_dir is None:
            data_dir = getattr(cfg, 'data_dir', None)
            if data_dir is None and cfg is not None:
                data_dir = getattr(getattr(cfg, 'dataset', None), 'data_dir', None)
            if data_dir is None:
                data_dir = '/network/projects/real-g-grp/libero/targets_clean/'

        if cfg.dataset.load_dataset:
            dataset = LIBERODatasetLeRobot(
                repo_id=cfg.dataset.to_name,
                transform=transforms.ToTensor(),
                cfg=cfg
            )
        else:
            data_dir = getattr(cfg.dataset, 'data_dir', '/network/projects/real-g-grp/libero/targets_clean/')
            dataset = LIBERODataset(data_dir, transform=transforms.ToTensor())
        num_to_load = min(len(dataset), self._cfg.dataset.buffer_size)
        if num_to_load == 0:
            return

        indices = np.random.choice(len(dataset), size=num_to_load, replace=False)
        for idx in range(num_to_load):
            images, actions, rewards, dones, poses = dataset[idx]

            # dones = np.zeros_like(rewards)
            # dones[-1] = 1

            self.add_trajectory(
                np.array(images),
                np.array(actions),
                np.array(rewards),
                np.array(dones),
                np.array(poses)
            )
        
    def add_trajectory(self, images, actions, rewards, dones, poses):
        """Add a trajectory to the buffer. Overwrites oldest if full."""
        trajectory = {
            'images': torch.from_numpy(images),
            'actions': torch.from_numpy(actions),
            'rewards': torch.from_numpy(rewards),
            'dones': torch.from_numpy(dones),
            'poses': torch.from_numpy(poses)
        }
        
        if len(self.trajectories) < self._cfg.dataset.buffer_size:
            self.trajectories.append(trajectory)
        else:
            # Overwrite oldest trajectory
            self.trajectories[self.write_idx] = trajectory
            self.write_idx = (self.write_idx + 1) % self._cfg.dataset.buffer_size
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        return traj['images'], traj['actions'], traj['rewards'], traj['dones'], traj['poses']

from datasets import load_dataset
import datasets
class LIBERODatasetLeRobot(torch.utils.data.Dataset):


    """A dataset class for loading LIBERO data from the LeRobot repository."""

    def __init__(self, repo_id, transform=None, cfg=None):
        # super().__init__(repo_id, transform)
        self.repo_id = repo_id
        self.transform = transform
        self._dataset = datasets.load_dataset(repo_id, split='train[:{}]'.format(cfg.dataset.buffer_size), keep_in_memory=True)


    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        # Load trajectory data from LeRobot dataset
        sample = self._dataset[idx]
        
        # Extract trajectory components
        images = torch.from_numpy(np.array(sample['img'])).float()
        actions = torch.from_numpy(np.array(sample['action'])).float()
        rewards = torch.from_numpy(np.array(sample['rewards'])).float() if 'rewards' in sample else torch.zeros(len(actions))
        dones = torch.from_numpy(np.array(sample['terminated'])).float() if 'terminated' in sample else torch.zeros(len(actions))
        poses = torch.from_numpy(np.array(sample['poses'])).float() if 'poses' in sample else torch.zeros(len(actions), 7)
        
        # Note: Images are returned in channel-last format (T, H, W, C)
        # Conversion to channel-first (T, C, H, W) happens in the training loop
        
        return images, actions, rewards, dones, poses


@hydra.main(config_path="./conf", config_name="64pix-pose")
def my_main(cfg: DictConfig):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wandb = None
    if not cfg.testing:
        import wandb
        # start a new wandb run to track this script
        wandb.init(
            project=cfg.experiment.project,
            # track hyperparameters and run metadata
            config= OmegaConf.to_container(cfg),
            name=cfg.experiment.name,
        )
        wandb.run.log_code(".")

    # Get model type from config or default to 'dreamer'
    model_type = getattr(cfg, 'model_type', 'dreamer')
    print(f"[info] Using model type: {model_type}")

    # Initialize the model using factory
    img_shape = [3, 64, 64]
    model = create_model(model_type, img_shape, action_dim=7, device=device, cfg=cfg)
    
    # Wrap model for unified training interface
    model_wrapper = ModelTrainingWrapper(model, model_type, device)
    
    # Initialize planner (works with both model types through the model interface)
    if cfg.use_policy:
        print("[info] Using policy-based planner (CEMPlanner with policy)")
        import torch.nn as nn
        # Stochastic policy that outputs both mean and log_std for Gaussian distribution
        policy = nn.Sequential(
            nn.Linear(7, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 14)  # Output 14: 7 for mean, 7 for log_std
        )
        policy.to(device)
        planner = PolicyPlanner(
            model, 
            policy_model=policy,
            action_dim=7,
            cfg=cfg
        )
    else:
        planner = CEMPlanner(
            model, 
            action_dim=7,
            cfg=cfg
        )

    # Initialize circular buffer dataset
    if cfg.use_random_data:
        print("[info] Using CircularBufferDataset with random data collection")
        dataset = CircularBufferDataset(cfg=cfg)
        print(f"[info] Initialized buffer with {len(dataset)} trajectories")
    else:
        # Use Hugging Face dataset by default for portability; fall back to local HDF5 if requested.
        if cfg.dataset.load_dataset:
            dataset = LIBERODatasetLeRobot(
                repo_id=cfg.dataset.to_name,
                transform=transforms.ToTensor(),
                cfg=cfg
            )
        else:
            data_dir = getattr(cfg.dataset, 'data_dir', '/network/projects/real-g-grp/libero/targets_clean/')
            dataset = LIBERODataset(data_dir, transform=transforms.ToTensor())

    batch_size = 32
    cfg.policy.sequence_length = 16
    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Add linear learning rate scheduler that decays to 0 over training
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=1.0,  # Start at full learning rate
        end_factor=0.01,    # End at 0 learning rate
        total_iters=cfg.max_iters     # Decay over num_epochs
    )
    policy_loss = 0

    # Training loop
    for epoch in range(cfg.max_iters):
        
        num_idx = np.arange(len(dataset))
        np.random.shuffle(num_idx)
        loss = 0.0
        batch_counter = 0
        
        # Process data in batches
        for batch_start in range(0, len(num_idx), batch_size):
            batch_end = min(batch_start + batch_size, len(num_idx))
            batch_indices = num_idx[batch_start:batch_end]
            # Collect sequences for the batch
            # [TODO]
            # TODO:    
            ## Implement data loading and training step for the batch
            print(f'Epoch [{epoch+1}/{cfg.max_iters }], Batch [{batch_counter}/{(len(dataset) + batch_size - 1) // batch_size}], Loss: {batch_loss.item():.4f}, policy_loss: {policy_loss:.4f}')

        # save the model checkpoint
        if epoch % cfg.eval_vid_iters == 0:
            torch.save(model.state_dict(), f'model_epoch_{epoch+1}_batch_{batch_counter}.pth', pickle_module=dill)
                # Evaluate the model using eval_libero from sim_eval
            print("[info] Starting evaluation on LIBERO tasks...")
            data = eval_libero(planner, device, cfg, iter_=epoch, log_dir="./", 
                               wandb=wandb)
            if cfg.use_random_data:
                # Add new random trajectories to the buffer
                for traj in data['traj']:
                    dones = np.zeros_like(traj['rewards'])
                    dones[-1] = 1
                    ## observations need to be changed to channel first
                    observations = np.array(traj['observations'])  # (T, 1, H, W, C) -> (T, H, W, C)
                    observations = np.transpose(observations, (0, 3, 1, 2))  # (T, H, W, C) -> (T, C, H, W)
                    dataset.add_trajectory(observations, np.array(traj['actions']),
                                           np.array(traj['rewards']), np.array(dones), np.array(traj['poses']))
                print(f"[info] Added new random trajectories to buffer. Current buffer size: {len(dataset)}")
        
        # Step the learning rate scheduler after each epoch
        scheduler.step()
        print(f'Learning rate after epoch {epoch+1}: {scheduler.get_last_lr()[0]:.6f}')



if __name__ == '__main__':
    my_main()