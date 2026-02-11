"""
Mock GRP Model that replays trajectories instead of learning a model.
This mock replays the first trajectory from the dataset, returning actions
sequentially until terminated == True.
"""

import numpy as np
import torch
from torch import nn
from grp_model import GRP


class ReplayModel(nn.Module):
    """
    A mock GRP model that replays trajectories from the dataset.
    Instead of learning to predict actions, it stores and replays a trajectory
    sequentially, returning the next action each time it's called.
    """
    
    def __init__(self, cfg, dataset=None):
        """
        Initialize the ReplayModel.
        
        Args:
            cfg: Configuration object
            dataset: Dataset object containing trajectories
        """
        super(ReplayModel, self).__init__()
        self._cfg = cfg
        self.dataset = dataset
        
        # Trajectory storage
        self.trajectory = None
        self.current_step = 0
        self.trajectory_loaded = False
        
        # Load the first trajectory if dataset is provided
        if dataset is not None:
            self._load_first_trajectory()

    def set_dataset(self, dataset):
        """
        Set the dataset for the model.
        
        Args:
            dataset: Dataset object containing trajectories
        """
        self.dataset = dataset
        self._load_first_trajectory()   
    
    def _load_first_trajectory(self):
        """
        Load the first trajectory from the dataset.
        Extracts states, actions, and termination flags up to the first terminated=True.
        """
        if self.dataset is None:
            raise ValueError("Dataset must be provided to load trajectories")
        
        # Try to get the first trajectory from the dataset
        # This assumes the dataset has a method to access trajectories
        # Adjust based on your actual dataset structure
        trajectory = self.dataset.get_trajectory(0)  # Get first trajectory
        
        self.trajectory = trajectory
        self.current_step = 0
        self.trajectory_loaded = True
        
        # Find the step where terminated == True
        self.terminal_step = 0
        for i, step_data in enumerate(trajectory):
            if step_data.get('terminated', False) or step_data.get('done', False):
                self.terminal_step = i
                break
        else:
            # If no terminal step found, use the entire trajectory
            self.terminal_step = len(trajectory) - 1
    
    def load_trajectory(self, trajectory_data):
        """
        Manually load a trajectory.
        
        Args:
            trajectory_data: List of step dictionaries containing 'observation', 'action', 'terminated', etc.
        """
        self.trajectory = trajectory_data
        self.current_step = 0
        self.trajectory_loaded = True
        
        # Find the terminal step
        self.terminal_step = len(trajectory_data) - 1
        for i, step_data in enumerate(trajectory_data):
            if step_data.get('terminated', False) or step_data.get('done', False):
                self.terminal_step = i
                break
    
    def reset(self):
        """Reset the current step to the beginning of the trajectory."""
        self.current_step = 0
        return self.trajectory[0]['init_state'] if self.trajectory_loaded else None
    
    def forward(self, images, goals_txt=None, goal_imgs=None, targets=None, pose=None, mask_=False, last_action=None):
        """
        Forward pass that returns the next action in the replay trajectory.
        
        Args:
            images: Current observation images (batch_size, channels, height, width)
            goals_txt: Goal text embeddings (optional, not used in replay)
            goal_imgs: Goal images (optional, not used in replay)
            targets: Target actions (optional, not used in replay)
            pose: Pose data (optional, not used in replay)
            mask_: Mask flag (optional, not used in replay)
            last_action: Last action taken (optional, not used in replay)
        
        Returns:
            (action, loss): 
                - action: The next action from the trajectory
                - loss: None (no loss in replay mode)
        """
        if not self.trajectory_loaded:
            raise RuntimeError("No trajectory loaded. Call load_trajectory() or provide a dataset to __init__")
        
        if self.current_step > self.terminal_step:
            # Episode is over, return zeros
            batch_size = images.shape[0] if images is not None else 1
            action_dim = self._cfg.action_dim * self._cfg.policy.action_stacking
            zero_action = torch.zeros((batch_size, action_dim), 
                                     dtype=torch.float32, 
                                     device=self._cfg.device)
            return (zero_action, None)
        
        # Get the action from the current step
        step_data = self.trajectory[self.current_step]
        action = step_data.get('action', None)
        
        if action is None:
            batch_size = images.shape[0] if images is not None else 1
            action_dim = self._cfg.action_dim * self._cfg.policy.action_stacking
            zero_action = torch.zeros((batch_size, action_dim), 
                                     dtype=torch.float32, 
                                     device=self._cfg.device)
            action = zero_action
        else:
            # Convert action to tensor if needed
            if not isinstance(action, torch.Tensor):
                action = torch.tensor(action, dtype=torch.float32, device=self._cfg.device)
            
            # Add batch dimension if needed
            if action.dim() == 1:
                action = action.unsqueeze(0)
        
        # Increment step counter
        self.current_step += 1
        
        return (action, None)
    
    def get_trajectory_info(self):
        """
        Get information about the currently loaded trajectory.
        
        Returns:
            dict: Information about the trajectory including length and terminal step
        """
        if not self.trajectory_loaded:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "trajectory_length": len(self.trajectory),
            "terminal_step": self.terminal_step,
            "current_step": self.current_step,
            "episode_complete": self.current_step > self.terminal_step,
        }
    
    def is_episode_complete(self):
        """Check if the trajectory replay has reached the terminal state."""
        return self.current_step > self.terminal_step
    
    def encode_text_goal(self, goal, tokenizer=None, text_model=None):
        return 0
    
    def preprocess_state(self, state):
        return state
    
    def preprocess_goal_image(self, goal_img): 
        return goal_img
    def decode_action(self, action):
        return action
