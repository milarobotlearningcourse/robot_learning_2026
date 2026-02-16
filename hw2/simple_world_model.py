import torch
import torch.nn as nn
import numpy as np
from dreamerV3 import GRPBase


class SimpleWorldModel(GRPBase):
    """
    Simple world model that predicts the next pose and reward given current pose and action.
    
    Architecture:
    - Takes current pose (7-d) + normalized action
    - Simple MLP to predict next pose (7-d) and reward (scalar)
    """
    
    def __init__(self, 
                 action_dim=7,
                 pose_dim=7,
                 hidden_dim=256,
                 cfg=None):
        # TODO: Part 1.1 - Initialize SimpleWorldModel architecture
        ## Define the feature network and output heads (pose and reward)
        pass
    
    def forward(self, pose, action):
        """
        Forward pass to predict next pose and reward.
        
        Args:
            pose: Pose tensor of shape (B, pose_dim) or (B, T, pose_dim), normalized
            action: Action tensor of shape (B, action_dim) or (B, T, action_dim), normalized
        
        Returns:
            next_pose_pred: Predicted normalized pose (B, pose_dim) or (B, T, pose_dim)
            reward_pred: Predicted reward (B, 1) or (B, T, 1)
        """
        # TODO: Part 1.1 - Implement forward pass
        ## Concatenate pose and action, pass through feature network and output heads
        pass
    
    def predict_next_pose(self, pose, action):
        """
        Convenience method to predict next pose and reward, decoding pose to original space.
        
        Args:
            pose: Normalized pose (7-d vector or batch)
            action: Action in original space (will be encoded)
        
        Returns:
            next_pose: Pose in original space
            reward: Predicted reward
        """
        # TODO: Part 1.1 - Implement prediction method
        ## Encode action, call forward, and decode pose to original space
        pass
    
    def compute_loss(self, pose, action, target_pose, target_reward=None):
        """
        Compute MSE loss between predicted and target pose and reward.
        
        Args:
            pose: Current pose tensor (B, pose_dim) or (B, T, pose_dim), normalized
            action: Action tensor (B, action_dim) or (B, T, action_dim), normalized
            target_pose: Target pose tensor (B, pose_dim) or (B, T, pose_dim), normalized
            target_reward: Target reward tensor (B, 1) or (B, T, 1), optional
        
        Returns:
            loss: Total MSE loss (pose + reward if target_reward is provided)
        """
        # TODO: Part 1.2 - Implement SimpleWorldModel loss computation
        ## Compute MSE loss for pose and reward predictions
        pass
