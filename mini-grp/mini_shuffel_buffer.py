## Code to fetch data and create an easy dataset.

import hydra, json
from omegaconf import DictConfig, OmegaConf
from transformers import T5Tokenizer, T5ForConditionalGeneration
## import python garbage collector
import gc
gc.enable()
import numpy as np
import torch
import cv2
import time
import torch.profiler
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

# Set the max size to 10GB (10 * 1024 * 1024 * 1024 bytes)
size = 20 * 1024 * 1024 * 1024
os.environ["HF_DATASETS_IN_MEMORY_MAX_SIZE"] = str(size)

# def apply_transforms(episode, cfg, dataset_name):
#     """
#     Apply the necessary transforms to the episode data.
#     This function is a placeholder for any transformations that need to be applied.
#     """
# import prismatic.vla.datasets.rlds.oxe.transforms as rlds_transforms

#     # Example transformation: resize images, normalize actions, etc.
#     # episode = [rlds_transforms.OXE_STANDARDIZATION_TRANSFORMS[cfg.dataset.dataset_indicies[dataset_name]["dataset_key"]](x) for x in episode]
#     return episode

# === Bridge-V2 =>> Dataset-Specific Transform ===
def relabel_bridge_actions(traj: Dict[str, Any]) -> Dict[str, Any]:
    """Relabels actions to use reached proprioceptive state; discards last timestep (no-action)."""
    movement_actions = traj["observation"]["state"][:6] - traj["observation"]["state"][:6]
    traj_truncated = tf.nest.map_structure(lambda x: x[:-1], traj)
    traj_truncated["action"] = tf.concat([movement_actions, traj["action"][:-1, -1:]], axis=1)

    return traj_truncated

def bridge_oxe_dataset_transform(trajectory):
    """
    Applies to version of Bridge V2 in Open X-Embodiment mixture.

    Note =>> In original Bridge V2 dataset, the first timestep has an all-zero action, so we remove it!
    """
    # trajectory = trajectory[1:]  # Remove the first timestep with all-zero action

    for i in range(0, len(trajectory)-1):
        trajectory[i]["action"] = np.concatenate((trajectory[i]['action']['world_vector'], 
                                                    trajectory[i]['action']['rotation_delta'], 
                                                        [trajectory[i]['action']['open_gripper']], 
                                                        ), axis=-1
                                                    ).astype(np.float32)
        trajectory[i]["language_instruction"] = trajectory[i]["observation"]["natural_language_instruction"]
        # trajectory[i] = relabel_bridge_actions(trajectory[i])
        trajectory[i]["action"][:6] = trajectory[i+1]["observation"]["state"][:6] - trajectory[i]["observation"]["state"][:6] ## Get realtive change action
        ## Clip action to be between -1 and 1
        trajectory[i]["action"] = np.clip(trajectory[i]["action"], -1.0, 1.0)
        trajectory[i]["observation"]["eef_state"] = trajectory[i]["observation"]["state"][:6]
        trajectory[i]["observation"]["gripper_state"] = trajectory[i]["observation"]["state"][-1:]
    return trajectory[:-1] ## Remove last timestep with no action

def maniskill_dataset_transform(trajectory):
    for i in range(0, len(trajectory)):
        trajectory[i]["observation"]["gripper_state"] = trajectory[i]["observation"]["state"][7:8]
        trajectory[i]["observation"]["eef_state"] = trajectory[i]["observation"]["state"][1:7] ## TODO: Not sure if this is the information for wrist pos and rotation
        trajectory[i]["action"] = trajectory[i]["action"].numpy()
        trajectory[i]['observation']["natural_language_instruction"] = trajectory[i]["language_instruction"]
    return trajectory

def libero_dataset_transform(trajectory):
    for i in range(0, len(trajectory)):
        trajectory[i]["observation"]["gripper_state"] = trajectory[i]["observation"]["state"][6:7]
        # trajectory[i]["observation"]["gripper_state"] = trajectory[i]["observation"]["state"][-2:]  # 2D gripper state
        trajectory[i]["observation"]["eef_state"] = trajectory[i]["observation"]["state"][:6] ## TODO: Not sure if this is the information for wrist pos and rotation
        trajectory[i]["action"] = trajectory[i]["action"].numpy()
        trajectory[i]["action"][-1:] = 1 - np.clip(trajectory[i]["action"][-1:], 0, 1)
        # libero actions are from [-1 , 1], and inverted for the gripper, bring this to the [0, 1] range
        # trajectory[i]["action"][6] = ((trajectory[i]["action"][6] + 1.0) / 2.0) * -1
        trajectory[i]['observation']["natural_language_instruction"] = trajectory[i]["language_instruction"]
    return trajectory

def robocook_dataset_transform(trajectory):
    for i in range(0, len(trajectory)):
        trajectory[i]["observation"]["eef_state"] = trajectory[i]["observation"]["state"][:6]
        trajectory[i]["action"] = trajectory[i]["action"].numpy()
        trajectory[i]['observation']["natural_language_instruction"] = trajectory[i]["language_instruction"]
        trajectory[i]["observation"]["gripper_state"] = trajectory[i]["observation"]["state"][-1:]
        trajectory[i]["observation"]["image"] = trajectory[i]["observation"]["image_1"]
    return trajectory

def saytap_transform(trajectory):
    for i in range(0, len(trajectory)):
        trajectory[i]["observation"]["eef_state"] = np.concatenate((trajectory[i]["observation"]["desired_vel"].numpy(),
                                                                trajectory[i]["observation"]["proj_grav_vec"].numpy() ),
                                                                  axis=-1).astype(np.float32)
        trajectory[i]["observation"]["gripper_state"] = 0 ## No gripper state in SayTap
        trajectory[i]["action"] = trajectory[i]["action"].numpy()
        trajectory[i]["state"] = trajectory[i]["observation"]["state"].numpy()
        trajectory[i]['observation']["natural_language_instruction"] = trajectory[i]["language_instruction"]
        # trajectory[i]["observation"]["gripper_state"] = trajectory[i]["observation"]["state"][-1:]
        # trajectory[i]["observation"]["image"] = trajectory[i]["observation"]["image_1"]
    return trajectory


def apply_transforms(episode, cfg, dataset_name):
    """
    Apply the necessary transforms to the episode data.
    This function is a placeholder for any transformations that need to be applied.
    """
    TRANSFORMS = {
        "bridge_oxe": bridge_oxe_dataset_transform,
        "stanford_robocook_converted_externally_to_rlds": robocook_dataset_transform,
        "maniskill_dataset_converted_externally_to_rlds": maniskill_dataset_transform,
        "saytap": saytap_transform,
        "libero_dataset_transform": libero_dataset_transform,
        # Add other dataset specific transforms here if needed
    }
    # Example transformation: resize images, normalize actions, etc.
    episode = TRANSFORMS[cfg.dataset.dataset_indicies[dataset_name]["dataset_key"]](episode)
    return episode

def get_total_dict_size(d):
    import sys
    import numpy as np
    # Start with the size of the dictionary object itself
    total_size = sys.getsizeof(d)
    
    for key, value in d.items():
        # Add the size of the key (usually a string)
        total_size += sys.getsizeof(key)
        
        # Add the size of the NumPy object + its internal data buffer
        if isinstance(value, np.ndarray):
            total_size += sys.getsizeof(value) + value.nbytes
        else:
            total_size += sys.getsizeof(value)
            
    return total_size

def convert_numpy_arrays_to_pil(dataset_dict):
    """
    Convert numpy arrays in dataset to PIL images for Hugging Face Hub visualization.
    Supports both torch tensors and numpy arrays.
    """
    from PIL import Image
    import numpy as np
    import torch
    
    converted_dict = {}
    for key, value in dataset_dict.items():
        if key in ["img", "goal_img"]:
            # Convert image arrays to PIL images for visualization on HF Hub
            pil_images = []
            if isinstance(value, torch.Tensor):
                value = value.cpu().numpy()
            
            # Handle batch of images
            if len(value.shape) == 4:  # [B, H, W, C]
                for img_array in value:
                    # Ensure values are in 0-255 range
                    if img_array.dtype == np.float32 or img_array.dtype == np.float64:
                        img_array = (img_array * 255).astype(np.uint8)
                    else:
                        img_array = img_array.astype(np.uint8)
                    # Convert BGR to RGB if needed (common with OpenCV)
                    if img_array.shape[2] == 3:
                        img_pil = Image.fromarray(img_array[:, :, ::-1], mode='RGB')
                    else:
                        img_pil = Image.fromarray(img_array, mode='RGB')
                    pil_images.append(img_pil)
                converted_dict[key] = pil_images
            else:
                converted_dict[key] = value
        else:
            converted_dict[key] = value
    
    return converted_dict

class CircularBuffer:
    """ A circular buffer implemented using a collection of numpy arrays.
    The buffer stores images, actions, goals, goal images, rotation deltas, and open gripper states.
    The buffer has a fixed size and overwrites old data when full.
    The buffer is initialized with a size and a configuration object.
    """
    def __init__(self, size, cfg, model):
        from cProfile import Profile
        from pstats import SortKey, Stats
        import tensorflow_datasets as tfds

        # with Profile() as profile:
        self._cfg = cfg
        self._model = model
        self._index = 0
        self._count = 0
        
        self._dataset_tmp = self.update_internal_dataset(size, old_data=None) 
                    
        if self._cfg.dataset.encode_with_t5:
            self._tokenizer = T5Tokenizer.from_pretrained(self._cfg.dataset.t5_version)
            self._text_model = T5ForConditionalGeneration.from_pretrained(self._cfg.dataset.t5_version)
            # self._dataset_tmp["t5_language_embedding"] = torch.tensor(np.zeros(shape=(self._size, self._cfg.max_block_size, self._cfg.n_embd)), dtype=torch.float, device=self._cfg.device)[0],  

        self._builders = {}
        if self._cfg.dataset.num_episodes > 0:
            for dataset_name in self._cfg.dataset.dataset_indicies:
                self._builders[dataset_name] = tfds.builder_from_directory(builder_dir=dataset_name)
                print("dataset size:", self._builders[dataset_name].info.splits["train"].num_examples)

        chars = cfg.dataset.chars_list
        cfg.vocab_size = len(chars)
        # create a mapping from characters to integers
        stoi = { ch:i for i,ch in enumerate(chars) }
        itos = { i:ch for i,ch in enumerate(chars) }
        self._encode_txt = lambda s: [stoi[c] for c in s] # text encoder to tokens: 
        self._decode_txy = lambda l: ''.join([itos[i] for i in l]) # token decoder to text: 
        print("vocab_size:", cfg.vocab_size)

        cfg.action_dim = len(cfg.dataset.action_mean)

        self._dataset_indicies = self._cfg.dataset.dataset_indicies
        start_ = time.time()
        if self._cfg.dataset.load_dataset is True:
            # Load the dataset from a file
            import datasets
            # with torch.profiler.record_function("Load huggingface dataset"):
            start__ = time.time()
            ## Load huggingface dataset into memory
            ## Only load first 200 samples for debugging
            dataset = datasets.load_dataset(self._cfg.dataset.to_name, split='train[:{}]'.format(self._cfg.dataset.buffer_size), keep_in_memory=True)
            print("Time to load huggingface dataset:", time.time() - start_)
            dataset_tmp = {
                "img": dataset["img"][:self._cfg.dataset.buffer_size], ## Some loading optimizations to improve debugging
                "action": dataset["action"][:self._cfg.dataset.buffer_size],
                "goal_img": dataset["goal_img"][:self._cfg.dataset.buffer_size],
                "goal_text_full": dataset["goal_text_full"][:self._cfg.dataset.buffer_size],
                "t5_language_embedding": dataset["t5_language_embedding"][:self._cfg.dataset.buffer_size] if self._cfg.dataset.encode_with_t5 else None,
                "pose": dataset["pose"][:self._cfg.dataset.buffer_size],
                "terminated": dataset["terminated"][:self._cfg.dataset.buffer_size],
                "init_state": dataset["init_state"][:self._cfg.dataset.buffer_size] if "init_state" in dataset.column_names else None,
            }
            print("Time to load huggingface data and copy: ", time.time() - start__)
            for i in range(len(dataset_tmp["img"])):
                # if len(dataset_tmp["action"][i:i+self._cfg.policy.action_stacking]) < self._cfg.policy.action_stacking:
                #     print("Skipping index", i, "because action length is less than", self._cfg.policy.action_stacking)
                #     continue
                pose = dataset_tmp["pose"][i]
                action = np.array(dataset_tmp["action"][i])
                self.add(
                        dataset_tmp["img"][i], 
                        action,
                        dataset_tmp["goal_text_full"][i], 
                        dataset_tmp["goal_img"][i],
                        language_instruction=dataset_tmp["t5_language_embedding"][i] if self._cfg.dataset.encode_with_t5 else None,
                        terminated=dataset_tmp["terminated"][i],
                        pose=np.concatenate([pose[2:5], pose[5:8], pose[:1]]),  # Rearranged to match eef pos, eef quat, gripper state
                        init_state=dataset_tmp["init_state"][i] if "init_state" in dataset_tmp else None,   
                        )
            if self._cfg.dataset.recompute_normalizations:
                scale = 1.4
                a_std, a_mean = ((self._dataset_tmp["action"][:self._count]).std(axis=0) + 1e-8) * scale, self._dataset_tmp["action"][:self._count].mean(axis=0)
                self._cfg.dataset.action_std = [float(x) for x in a_std]
                self._cfg.dataset.action_mean = [float(x) for x in a_mean   ]
                print("Recomputed action normalizations: mean:", self._cfg.dataset.action_mean, " std:", self._cfg.dataset.action_std)
                s_std, s_mean = ((self._dataset_tmp["pose"][:self._count]).std(axis=0) + 1e-8) * scale, self._dataset_tmp["pose"][:self._count].mean(axis=0)
                self._cfg.dataset.pose_mean = [float(x) for x in s_mean]
                self._cfg.dataset.pose_std = [float(x) for x in s_std]
            print("Loaded dataset with size:", self._count)
        elif self._cfg.dataset.load_dataset == "skip":
            pass
        else:
            
            get_multi_dataset_portion(self._builders, self, self._cfg)
            print("Time to load full dataset:", time.time() - start_)

    def update_internal_dataset(self, size, old_data=None):
        self._size = size
        _dataset_tmp = {
                    "img": torch.tensor(np.zeros(shape=(size, self._cfg.image_shape[0], self._cfg.image_shape[0], 3)), dtype=torch.uint8, device=self._cfg.device), 
                    "pose": torch.tensor(np.zeros(shape=(size, len(self._cfg.dataset.action_std)),), dtype=torch.float, device=self._cfg.device),
                    "action": torch.tensor(np.zeros(shape=(size, len(self._cfg.dataset.action_std)),), dtype=torch.float, device=self._cfg.device),
                    "goal": torch.tensor(np.zeros(shape=(size, self._cfg.max_block_size)), dtype=torch.long, device=self._cfg.device), 
                    "goal_text_full": ["" for _ in range(size)], # This is a list of strings, not a tensor
                    "goal_img": torch.tensor(np.zeros(shape=(size, self._cfg.image_shape[0], self._cfg.image_shape[0], 3)), dtype=torch.uint8, device=self._cfg.device),
                    # "rotation_delta": [], "open_gripper": [] 
                    "t5_language_embedding": torch.tensor(np.zeros(shape=(size, self._cfg.max_block_size, self._cfg.n_embd)), dtype=torch.float, device=self._cfg.device) if self._cfg.dataset.encode_with_t5 else None,
                    "terminated": torch.tensor(np.zeros(shape=(size, 1)), dtype=torch.uint8, device=self._cfg.device),
                    "init_state": torch.tensor(np.zeros(shape=(size, 92)), dtype=torch.float, device=self._cfg.device) if self._cfg.dataset.use_init_state else None,
                    } 
        if old_data is not None:
            for key in _dataset_tmp:
                if key == "t5_language_embedding" and self._cfg.dataset.encode_with_t5 is False:
                    continue
                _dataset_tmp[key][:len(old_data[key])] = old_data[key]

        return _dataset_tmp



    def print_mem_footprint(self):
        from pympler import asizeof
        print("Memory used by the dataset cBuffer:", get_total_dict_size(self._dataset_tmp) / 1e6, "MB")
        ## Compute the memory use of the datset part self._dataset_tmp["img"]
        print("Memory used by the dataset cBuffer image:", asizeof.asizeof(self._dataset_tmp["img"]) / 1e6, "MB")
        print("Memory used by the dataset cBuffer goal image:", asizeof.asizeof(self._dataset_tmp["goal_img"]) / 1e6, "MB")
        print("Memory used by the dataset cBuffer: t5_language_embedding", asizeof.asizeof(self._dataset_tmp["t5_language_embedding"]) / 1e6, "MB")

    def add(self, obs, action, goal, goal_img, language_instruction=None, pose=None, terminated=0, 
            morphology=0, init_state=None):
        """ Add an observation, action, goal, goal image, rotation delta, and open gripper state to the buffer."""
    
        self._dataset_tmp["img"][self._index] = torch.tensor(np.array(obs), dtype=torch.uint8, device=self._cfg.device)
        self._dataset_tmp["goal_img"][self._index] = torch.tensor(np.array(goal_img), dtype=torch.uint8, device=self._cfg.device)
        self._dataset_tmp["action"][self._index] = torch.tensor(action, dtype=torch.float, device=self._cfg.device)
        if pose is not None:
            self._dataset_tmp["pose"][self._index] = torch.tensor(pose, dtype=torch.float32, device=self._cfg.device)  # Store robot pose
            
        self._dataset_tmp["goal_text_full"][self._index] = goal  # Store the full goal text
        if init_state is not None:
            self._dataset_tmp["init_state"][self._index] = torch.tensor(init_state, dtype=torch.float32, device=self._cfg.device)   
        self._dataset_tmp["terminated"][self._index] = torch.tensor(terminated, dtype=torch.uint8, device=self._cfg.device)
        
        ## Make goal embeddings of a fixed length and fill in the earlier chunks with the true goal data
        if self._cfg.dataset.encode_with_t5:
            if language_instruction is not None:
                ##TODO: This does not work if the original language instrictuion size is less than the new max_block_size
                self._dataset_tmp["t5_language_embedding"][self._index] = torch.tensor(language_instruction[:self._cfg.max_block_size], dtype=torch.float, device=self._cfg.device)
            else:
                with torch.profiler.record_function("Process goal text with T5"):
                    goal_ = self._model.process_text_embedding_for_buffer(goal, tokenizer=self._tokenizer, text_model=self._text_model)
                    self._dataset_tmp["t5_language_embedding"][self._index] = torch.tensor(goal_, dtype=torch.float, device=self._cfg.device)
        
        goal_ = " " * self._cfg.max_block_size
        goal_ = goal[:self._cfg.max_block_size] + goal_[len(goal):self._cfg.max_block_size] 
        # assert len(goal_) == self._cfg.max_block_size
        self._dataset_tmp["goal"][self._index] = torch.tensor(self._encode_txt(goal_), dtype=torch.float, device=self._cfg.device)
        self._count += 1
        if self._cfg.dataset.download_all is True and (self._count >= self._size):
            ## Double the buffer sizes to fit all the data.
            print("Doubling buffer size from", self._size, "to", self._size * 2, " to fit incomming data.")
            self.print_mem_footprint()
            old_data = self._dataset_tmp
            self._dataset_tmp = self.update_internal_dataset(self._size * 2, old_data=old_data)
        else:
            self._index = (self._index + 1) % self._size

    def get_batch_grp(self, split, cfg, batch_size, morphology=0):
        # from torchvision import transforms
        from torchvision.transforms import v2 # Recommend v2 for new code
        from einops import rearrange
        if self._cfg.policy.use_image_augmentations:
            # Add image Augmentations to improve performance
            transform_crop_scale = v2.Compose([
                v2.RandomResizedCrop(size=(cfg.image_shape[0], cfg.image_shape[1]), scale=(0.8, 1.0), antialias=True),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                v2.ToDtype(torch.float32)
            ])
        else:
            transform_crop_scale = v2.Compose([
                v2.ToDtype(torch.float32) # Convert to float [0,1] after crop/resize
            ])
        # generate a small batch of inputs x and targets y
        # data = dataset['train'] if split == 'train' else dataset['test']
        data = self._dataset_tmp
        
        ## Randomly sample indices for the batch but account for action stacking past the index and obs_stacking before the index.
        ix = np.random.randint(min(self._count, self._size)-((cfg.policy.action_stacking + cfg.policy.obs_stacking)-1), size=(batch_size,))
        ix = torch.tensor(ix).to(cfg.device)
        
        obs_ = data["img"][ix-cfg.policy.obs_stacking+1].to(torch.float).unsqueeze(1) # Convert to [B, T, C, H, W] format for torchvision transforms, and back.
        for i in range(1, cfg.policy.obs_stacking): ## This is slow but works.
            obs_ = torch.cat((obs_, data["img"][ix-cfg.policy.obs_stacking+1+i].unsqueeze(1)), axis=1) ## concatenate along the time dimension 
        obs_ = transform_crop_scale(obs_) # Convert to [B, T, C, H, W] format for torchvision transforms, and back.
        x = self._model.normalize_state(rearrange(obs_, 'b t h w c -> b h w (c t)', c=3, t=cfg.policy.obs_stacking)) ## Rearranging the image to have the stacked history in the last channel dimension)  # Flatten the time dimension for batching
    
        pose = self._model.encode_pose(data["pose"][ix].to(torch.float32)) # Convert to [B, T, C]
        ## Add noise to the pose for data augmentation
        if cfg.policy.add_noise_to_state:
            noise = torch.randn_like(pose) * cfg.policy.state_noise_std
            pose = pose + noise

        if cfg.dataset.encode_with_t5:
            x_goal = torch.tensor(data["t5_language_embedding"][ix], dtype=torch.float, device=cfg.device)
        else:
            x_goal = data["goal"][ix]
    
        x_goal_img = self._model.normalize_state(transform_crop_scale(data["goal_img"][ix].to(torch.float))) ## [B, C, H,  W]
        x_goal_img = x_goal_img # Convert to [B, H, W, C] format from torchvision.
    
        # TODO: 
        ## Provide the block masking logic for the attention head
        y = 0 ## discrete or continuous actions
        
        # Get last action (action at timestep before current observation)
        ## Zero out the last action if ix == 0
        row_mask = torch.logical_and(ix > 0, (data["terminated"][ix - 1][:,0] - 1) * -1) # if the previous step was terminated, we also zero out the last action
        last_action = torch.zeros_like(y[:, :cfg.action_dim])
        last_action[row_mask] = data["action"][ix - 1][row_mask]
        last_action = self._model.encode_action(last_action)
        # last_action = self._model.encode_action(data["action"][ix - 1]) if ix > 0 else torch.zeros_like(y[:, :cfg.action_dim])
        ## Add noise to the last_action for data augmentation
        if cfg.policy.add_noise_to_state:
            noise = torch.randn_like(last_action) * cfg.policy.state_noise_std
            last_action = last_action + noise
        
        return x, pose, x_goal, x_goal_img, y, last_action
    
    def get_trajectory(self, trajectory_index=0):
        """
        Extract a trajectory from the buffer based on trajectory index.
        Trajectories are separated by the 'terminated' indicator.
        
        Args:
            trajectory_index (int): Which trajectory to retrieve (0 for first, 1 for second, etc.)
        
        Returns:
            list: A list of dictionaries, each containing a step with keys:
                - 'observation': image observation
                - 'action': action taken
                - 'goal': goal text
                - 'goal_img': goal image
                - 'pose': pose/state information
                - 't5_language_embedding': T5 embedding (if applicable)
                - 'terminated': boolean indicating if episode ended
        """
        data = self._dataset_tmp
        trajectory_count = 0
        trajectory_start = None
        trajectory = []
        
        # Find the start of the requested trajectory
        for i in range(min(self._count, self._size)):
            if trajectory_count == trajectory_index:
                trajectory_start = i
                break
            
            # Check if this step is terminated
            if data["terminated"][i].item() == 1:
                trajectory_count += 1
        
        if trajectory_start is None:
            raise IndexError(f"Trajectory index {trajectory_index} not found in buffer")
        
        # Extract steps until we hit a terminated state
        for i in range(trajectory_start, min(self._count, self._size)):
            step_dict = {
                'observation': data["img"][i].detach().cpu().numpy(),
                'action': data["action"][i].detach().cpu().numpy(),
                'goal': data["goal_text_full"][i],
                'goal_img': data["goal_img"][i].detach().cpu().numpy(),
                'pose': data["pose"][i].detach().cpu().numpy(),
                'terminated': bool(data["terminated"][i].item()),
                'init_state': data["init_state"][i].detach().cpu().numpy() if self._cfg.dataset.use_init_state else None,
            }
            
            # Add T5 embedding if available
            if self._cfg.dataset.encode_with_t5 and data["t5_language_embedding"] is not None:
                step_dict['t5_language_embedding'] = data["t5_language_embedding"][i].detach().cpu().numpy()
            
            trajectory.append(step_dict)
            
            # Stop if we've reached a terminated state
            if data["terminated"][i].item() == 1:
                break
        
        return trajectory
    
    def shuffle(self, shared_queue):
        print("num", shared_queue)
        while True:
            data = shared_queue.get() ## Update the data when messaged from the Queue
            if data is None:
                break
            ## Call function to swap out a portion of data.
            get_multi_dataset_portion(self._builders, self, self._cfg)

    def save(self, path):
        """
        Save the dataset to a file.
        """
        ## Prepare dataset for push to huggingface
        from datasets import Dataset
        import datasets
        from PIL import Image

        save_count = self._count if self._count < self._size else self._size
        data_to_save = {}
        for k, v in self._dataset_tmp.items():
            if v is None: continue
            val = v[:save_count]
            if isinstance(val, torch.Tensor):
                val = val.cpu().numpy()
            data_to_save[k] = val

        ## Trim the dataset to the actual count
        self._dataset_tmp = {k: v for k, v in self._dataset_tmp.items() if not (k == "t5_language_embedding" and self._cfg.dataset.encode_with_t5 is False)}
        for key in self._dataset_tmp:
            ## check if the key is a list of strings
            if isinstance(self._dataset_tmp[key], list): ## Text data
                self._dataset_tmp[key] = self._dataset_tmp[key][:self._count]
            else:
                self._dataset_tmp[key] = self._dataset_tmp[key][:self._count].detach().cpu().numpy()
        print("Trimmed data size to remove zeros")

        ## Convert numpy arrays to PIL images for HF Hub visualization
        print("Converting images to PIL format for Hugging Face Hub...")
        self._dataset_tmp = convert_numpy_arrays_to_pil(self._dataset_tmp)

        ds = Dataset.from_dict(self._dataset_tmp)
        print("Converted data to huggingface dataset.")
        ## create a normal distribution in torch
        a_std, a_mean = (self._dataset_tmp["action"]).std(axis=0) + 1e-8, self._dataset_tmp["action"].mean(axis=0)
        self._cfg.dataset.action_std = [float(x) for x in a_std]
        self._cfg.dataset.action_mean = [float(x) for x in a_mean   ]

        with open('./config.json', 'w') as f:
            json.dump(OmegaConf.to_container(self._cfg, resolve=True), f, indent=2)
        print("Saved config file.")
        new_features = ds.features.copy()
        # new_features["img"] = Image()
        ds.cast(new_features)
        print('Features:', ds.features)
        ds.push_to_hub(self._cfg.dataset.to_name)
    
def get_dataset_portion(builder, cbuffer, cfg, list_, dataset_name=None):
    """
    Helper function to get a portion of the dataset.
    """
    import cv2
    import numpy as np
    from datasets import load_dataset
    # ------------
    # Train and test splits
    # Loading data
    # create RLDS dataset builder
    for c in list(list_):
        datasetRemote = builder.as_dataset(split='train[' + str(c) + ':' + str(c+cfg.dataset.chunk_size) + ']') ## Most likely a very slow way to get data from the dataset, but it is a better mix
        gc.collect()
        for episode in datasetRemote:
            episode = list(episode['steps'])
            ## https://github.com/openvla/openvla/blob/main/prismatic/vla/datasets/rlds/oxe/transforms.py
            episode = apply_transforms(episode, cfg, dataset_name)
            goal_img = cv2.resize(np.array(episode[-1]['observation']["image"], dtype=np.float32), (cfg.image_shape[0], cfg.image_shape[1]))  
            for i in range(len(episode)): ## Resize images to reduce computation
                obs = cv2.resize(np.array(episode[i]['observation']["image"], dtype=np.float32), (cfg.image_shape[0], cfg.image_shape[1]))
                pose = np.concatenate([episode[i]["observation"]["eef_state"].numpy(), episode[i]["observation"]["gripper_state"].numpy()])
                ## Apply any additional transformations like https://github.com/openvla/openvla/blob/main/prismatic/vla/datasets/rlds/dataset.py#L39
                cbuffer.add(obs = obs, 
                            action = episode[i]['action'],
                            goal= episode[i]['observation']["natural_language_instruction"].numpy().decode(),
                            goal_img=goal_img,
                            terminated = 1 if (i == (len(episode) - 1)) or episode[i]['terminated'] else 0,
                            pose = pose,
                            # morphology = cfg.dataset.dataset_indicies[dataset_name]["morphology"],
                            )
    print("A terminé le mélange.")
    return cbuffer

def _create_dataset_generator(builder, cfg, dataset_name, ix):
    """
    Generator function for creating datasets using from_generator.
    Yields individual samples from the dataset.
    """
    import cv2
    import numpy as np
    import gc
    from PIL import Image
    
    for c in ix:
        datasetRemote = builder.as_dataset(split='train[' + str(c) + ':' + str(c+cfg.dataset.chunk_size) + ']')
        gc.collect()
        for episode in datasetRemote:
            episode = list(episode['steps'])
            episode = apply_transforms(episode, cfg, dataset_name)
            goal_img = cv2.resize(np.array(episode[-1]['observation']["image"], dtype=np.float32), 
                                  (cfg.image_shape[0], cfg.image_shape[1]))
            
            for i in range(len(episode)):
                obs = cv2.resize(np.array(episode[i]['observation']["image"], dtype=np.float32), 
                                (cfg.image_shape[0], cfg.image_shape[1]))
                pose = np.concatenate([episode[i]["observation"]["eef_state"].numpy(), 
                                      episode[i]["observation"]["gripper_state"].numpy()])
                
                yield {
                    "img": Image.fromarray(obs.astype(np.uint8), mode='RGB'),
                    "action": episode[i]['action'],
                    "goal_text_full": episode[i]['observation']["natural_language_instruction"].numpy().decode(),
                    "goal_img": Image.fromarray(goal_img.astype(np.uint8), mode='RGB'),
                    "terminated": 1 if (i == (len(episode) - 1)) or episode[i]['terminated'] else 0,
                    "pose": pose,
                }

def get_multi_dataset_portion(builders, cbuffer, cfg):
    """
    Helper function to get a portion of the dataset.
    Supports both direct loading and huggingface from_generator option.
    """
    import tensorflow_datasets as tfds
    import numpy as np
    from tqdm import tqdm, trange
    import cv2
    from datasets import load_dataset, Dataset
    
    # Check if we should use from_generator approach
    use_generator = getattr(cfg.dataset, 'use_generator', False)
    
    for dataset_name, builder in builders.items():
        print("Loading dataset:", dataset_name)
        ## Get the number of items in the dataset
        samples_ = (int(cfg.dataset.num_episodes * 
                        cfg.dataset.dataset_indicies[dataset_name]["weight"]))/cfg.dataset.chunk_size
        print(" size_ ", builder.info.splits["train"].num_examples
                , " total samples to fetch", int(samples_),
                 " chunk_size ", cfg.dataset.chunk_size)
        if cfg.dataset.download_all is True:
            ## Most likely grabbing the entire dataset
            ## create a list that contains the indicies to grab data, where each index is the start of a chunk
            ix = list(range(0, cfg.dataset.num_episodes, cfg.dataset.chunk_size))
        else:
            ix = np.random.randint(builder.info.splits["train"].num_examples-cfg.dataset.chunk_size, size=int(samples_))
        
        if use_generator:
            # Use from_generator for memory-efficient loading
            print(f"Using from_generator approach for {dataset_name}")
            gen_func = lambda: _create_dataset_generator(builder, cfg, dataset_name, ix)
            generated_dataset = Dataset.from_generator(gen_func)
            
            # This sends the finalized data to the Hugging Face servers
            generated_dataset.push_to_hub(cfg.dataset.to_name)
            # Add samples from generated dataset to circular buffer
            # for sample in generated_dataset:
            #     cbuffer.add(
            #         obs=sample["obs"],
            #         action=sample["action"],
            #         goal=sample["goal"],
            #         goal_img=sample["goal_img"],
            #         terminated=sample["terminated"],
            #         pose=sample["pose"],
            #     )
        else:
            # Use original direct loading approach
            get_dataset_portion(builder, cbuffer, cfg, dataset_name=dataset_name, list_=ix)

@hydra.main(config_path="./conf", config_name="64pix-pose")
def my_main(cfg: DictConfig):
    import numpy as np
    # ------------
    # Train and test splits
    # Loading data
    # create RLDS dataset builder
    cfg.dataset.save_initial_dataset = True
    cfg.dataset.load_dataset = False
    # cfg.dataset.encode_with_t5 = True  # Also encode the text for the dataset.
    cfg.n_embd = 512  # T5 small embedding size
    cfg.device = "cpu"  # Use CPU for dataset creation to avoid GPU memory issues.
    from grp_model import GRP
    model = GRP(cfg)
    model.to(cfg.device)
    np.random.seed(cfg.r_seed)
    ## Prompt to make sure user wants to overwrite existing dataset

    response = input(f"Dataset {cfg.dataset.to_name} already exists. Overwrite? (y/n): ")
    if response.lower() != 'y':
        print("Exiting without overwriting dataset.")
        return
    cbuffer = CircularBuffer(cfg.dataset.buffer_size, cfg, model)

    print("Dataset shape:", len(cbuffer._dataset_tmp["img"]))
    print("Dataset len:", cbuffer._count)

    if cfg.dataset.save_initial_dataset and not cfg.dataset.use_generator:
        print("Saving dataset to:", cfg.dataset.to_name)
        ############# Save the dataset to a file
        cbuffer.save(cfg.dataset.to_name)

if __name__ == "__main__":
    results = my_main()
    print("results:", results)