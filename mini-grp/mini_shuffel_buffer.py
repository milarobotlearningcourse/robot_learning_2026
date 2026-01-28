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

def bridge_oxe_dataset_transform(trajectory):
    """
    Applies to version of Bridge V2 in Open X-Embodiment mixture.

    Note =>> In original Bridge V2 dataset, the first timestep has an all-zero action, so we remove it!
    """
    trajectory = trajectory[1:]  # Remove the first timestep with all-zero action

    for i in range(0, len(trajectory)):
        trajectory[i]["action"] = np.concatenate((trajectory[i]['action']['world_vector'], 
                                                    trajectory[i]['action']['rotation_delta'], 
                                                        [trajectory[i]['action']['open_gripper']], 
                                                        ), axis=-1
                                                    ).astype(np.float32),
        trajectory[i]["language_instruction"] = trajectory[i]["observation"]["natural_language_instruction"]
        # trajectory = relabel_bridge_actions(trajectory)
        trajectory[i]["observation"]["eef_state"] = trajectory[i]["observation"]["state"][:6]
        trajectory[i]["observation"]["gripper_state"] = trajectory[i]["observation"]["state"][-1:]
    return trajectory

def maniskill_dataset_transform(trajectory):
    for i in range(0, len(trajectory)):
        trajectory[i]["observation"]["gripper_state"] = trajectory[i]["observation"]["state"][7:8]
        trajectory[i]["observation"]["eef_state"] = trajectory[i]["observation"]["state"][1:7] ## TODO: Not sure if this is the information for wrist pos and rotation
        trajectory[i]["action"] = trajectory[i]["action"].numpy()
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
        # Add other dataset specific transforms here if needed
    }
    # Example transformation: resize images, normalize actions, etc.
    episode = TRANSFORMS[cfg.dataset.dataset_indicies[dataset_name]["dataset_key"]](episode)
    return episode

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
        self._size = size
        self._cfg = cfg
        self._model = model
        self._index = 0
        self._count = 0
        self._dataset_tmp = {
                            "img": torch.tensor(np.zeros(shape=(self._size, self._cfg.image_shape[0], self._cfg.image_shape[0], 3)), dtype=torch.uint8, device=self._cfg.device), 
                            "pose": torch.tensor(np.zeros(shape=(self._size, len(self._cfg.env.action_std)),), dtype=torch.float, device=self._cfg.device),
                            "action": torch.tensor(np.zeros(shape=(self._size, len(self._cfg.env.action_std)),), dtype=torch.float, device=self._cfg.device),
                            "goal": torch.tensor(np.zeros(shape=(self._size, self._cfg.max_block_size)), dtype=torch.long, device=self._cfg.device), 
                            "goal_text_full": ["" for _ in range(self._size)], # This is a list of strings, not a tensor
                            "goal_img": torch.tensor(np.zeros(shape=(self._size, self._cfg.image_shape[0], self._cfg.image_shape[0], 3)), dtype=torch.uint8, device=self._cfg.device),
                            # "rotation_delta": [], "open_gripper": [] 
                            "t5_language_embedding": torch.tensor(np.zeros(shape=(self._size, cfg.max_block_size, self._cfg.n_embd)), dtype=torch.float, device=self._cfg.device) if self._cfg.dataset.encode_with_t5 else None,
                            "terminal": torch.tensor(np.zeros(shape=(self._size, 1)), dtype=torch.uint8, device=self._cfg.device),
                            } 
                    
        if self._cfg.dataset.encode_with_t5:
            self._tokenizer = T5Tokenizer.from_pretrained(self._cfg.dataset.t5_version)
            self._text_model = T5ForConditionalGeneration.from_pretrained(self._cfg.dataset.t5_version)
            # self._dataset_tmp["t5_language_embedding"] = torch.tensor(np.zeros(shape=(self._size, self._cfg.max_block_size, self._cfg.n_embd)), dtype=torch.float, device=self._cfg.device)[0],  

        self._builders = {}
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

        cfg.action_dim = len(cfg.env.action_mean)

        self._dataset_indecies = self._cfg.dataset.dataset_indicies
        start_ = time.time()
        if self._cfg.dataset.load_dataset is True:
            # Load the dataset from a file
            import datasets
            # with torch.profiler.record_function("Load huggingface dataset"):
            start__ = time.time()
            dataset = datasets.load_dataset(self._cfg.dataset.to_name, split='train')
            print("Time to load huggingface dataset:", time.time() - start_)
            dataset_tmp = {
                "img": dataset["img"][:self._cfg.dataset.buffer_size], ## Some loading optimizations to improve debugging
                "action": dataset["action"][:self._cfg.dataset.buffer_size],
                "goal_img": dataset["goal_img"][:self._cfg.dataset.buffer_size],
                "goal_text_full": dataset["goal_text_full"][:self._cfg.dataset.buffer_size],
                "t5_language_embedding": dataset["t5_language_embedding"][:self._cfg.dataset.buffer_size],
                "pose": dataset["pose"][:self._cfg.dataset.buffer_size],
            }
            print("Time to load huggingface data and copy: ", time.time() - start__)
            for i in range(len(dataset_tmp["img"])):
                if len(dataset_tmp["action"][i:i+self._cfg.policy.action_stacking]) < self._cfg.policy.action_stacking:
                    print("Skipping index", i, "because action length is less than", self._cfg.policy.action_stacking)
                    continue
                pose = dataset_tmp["pose"][i]
                action = dataset_tmp["action"][i]
                self.add(
                        dataset_tmp["img"][i], 
                        action,
                        dataset_tmp["goal_text_full"][i], 
                        dataset_tmp["goal_img"][i],
                        language_instruction=dataset_tmp["t5_language_embedding"][i] if cfg.dataset.encode_with_t5 else None,
                        terminal=0,
                        pose=pose,
                        )
            print("Loaded dataset with size:", self._count)
        elif self._cfg.dataset.load_dataset == "skip":
            pass
        else:
            
            get_multi_dataset_portion(self._builders, self, self._cfg)
            print("Time to load full dataset:", time.time() - start_)

    def print_mem_footprint(self):
        from pympler import asizeof
        print("Memory used by the dataset cBuffer:", asizeof.asizeof(self._dataset_tmp) / 1e6, "MB")
        print("Memory used by the dataset cBuffer image:", asizeof.asizeof(self._dataset_tmp["img"]) / 1e6, "MB")
        print("Memory used by the dataset cBuffer goal image:", asizeof.asizeof(self._dataset_tmp["goal_img"]) / 1e6, "MB")
        print("Memory used by the dataset cBuffer: t5_language_embedding", asizeof.asizeof(self._dataset_tmp["t5_language_embedding"]) / 1e6, "MB")

    def add(self, obs, action, goal, goal_img, language_instruction=None, pose=None, terminal=0, 
            morphology=0):
        """ Add an observation, action, goal, goal image, rotation delta, and open gripper state to the buffer."""
    
        self._dataset_tmp["img"][self._index] = torch.tensor(np.array(obs), dtype=torch.uint8, device=self._cfg.device)
        self._dataset_tmp["goal_img"][self._index] = torch.tensor(np.array(goal_img), dtype=torch.uint8, device=self._cfg.device)
        self._dataset_tmp["action"][self._index] = torch.tensor(action, dtype=torch.float, device=self._cfg.device)
        if pose is not None:
            self._dataset_tmp["pose"][self._index] = torch.tensor(pose, dtype=torch.float32, device=self._cfg.device)  # Store robot pose
            
        self._dataset_tmp["goal_text_full"][self._index] = goal  # Store the full goal text
        self._dataset_tmp["terminal"][self._index] = torch.tensor(terminal, dtype=torch.uint8, device=self._cfg.device)
        
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
        ix = np.random.randint(min(self._count, self._size)-((cfg.policy.action_stacking + cfg.policy.obs_stacking)-1), size=(batch_size,))
        with torch.profiler.record_function("Get batch from circular buffer and process obs image"):
            obs_ = data["img"][ix].to(torch.float).unsqueeze(1).permute(0, 1, 4, 2, 3) # Convert to [B, T, C, H, W] format for torchvision transforms, and back.
            for i in range(1, cfg.policy.obs_stacking): ## This is slow but works.
                obs_ = torch.concatenate((obs_, data["img"][ix+i].unsqueeze(1).permute(0, 1, 4, 2, 3)), axis=1) ## concatenate along the time dimension 
            obs_ = transform_crop_scale(obs_).permute(0, 1, 3, 4, 2) # Convert to [B, T, C, H, W] format for torchvision transforms, and back.
            x = self._model.normalize_state(rearrange(obs_, 'b t h w c -> b h w (c t)', c=3, t=cfg.policy.obs_stacking)) ## Rearranging the image to have the stacked history in the last channel dimension)  # Flatten the time dimension for batching
        
        pose = data["pose"][ix].to(torch.float32).unsqueeze(1) # Convert to [B, T, C]

        if cfg.dataset.encode_with_t5:
            x_goal = torch.tensor(data["t5_language_embedding"][ix], dtype=torch.float, device=cfg.device)
        else:
            x_goal = data["goal"][ix]
        x_goal_img = self._model.normalize_state(transform_crop_scale(data["goal_img"][ix].permute(0,3,1,2).to(torch.float))) ## [B, C, H,  W]
        x_goal_img = x_goal_img.permute(0, 2, 3, 1) # Convert to [B, H, W, C] format from torchvision.
        # TODO: 
        ## Provide the block masking logic for the attention head
        y = self._model.encode_action(data["action"][ix + cfg.policy.obs_stacking - 1])
        if cfg.policy.action_stacking > 1:
            ## Stack the next cfg.policy.action_stacking actions together
            for i in range(1, cfg.policy.action_stacking): ## This is slow but works.
                y_next = self._model.encode_action(data["action"][ix +cfg.policy.obs_stacking - 1 +i])
                y = torch.cat((y, y_next), dim=1) ## stack on time timension. 
        return x, pose, x_goal, x_goal_img, y
    
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

        ds = Dataset.from_dict(data_to_save)
        ## create a normal distribution in torch
        a_std, a_mean = (self._dataset_tmp["action"][:self._count] + torch.randn(self._dataset_tmp["action"][:self._count].size(), device=self._cfg.device) * 0.001).std(axis=0) * 1.2, self._dataset_tmp["action"][:self._count].mean(axis=0)
        self._cfg.env.action_std = a_std.cpu().numpy().tolist()
        self._cfg.env.action_mean = a_mean.cpu().numpy().tolist()

        with open('./config.json', 'w') as f:
            json.dump(OmegaConf.to_container(self._cfg, resolve=True), f, indent=2)
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
        # datasetRemote = builder.as_dataset(split='train[3:4]')
        gc.collect()
        for episode in datasetRemote:
            episode = list(episode['steps'])
            ## https://github.com/openvla/openvla/blob/main/prismatic/vla/datasets/rlds/oxe/transforms.py
            episode = apply_transforms(episode, cfg, dataset_name)
            goal_img = cv2.resize(np.array(episode[-1]['observation']["image"], dtype=np.float32), (cfg.image_shape[0], cfg.image_shape[1]))  
            # print("Ajout de", len(episode), "données à la buffer circular.")
            for i in range(len(episode)): ## Resize images to reduce computation
                if (i+cfg.policy.action_stacking > len(episode)):
                    # print("Skipping index", i, "because action length is less than", cfg.policy.action_stacking)
                    continue
                obs = cv2.resize(np.array(episode[i]['observation']["image"], dtype=np.float32), (cfg.image_shape[0], cfg.image_shape[1]))
                pose = episode[i]["observation"]["state"].numpy()
                cbuffer.add(obs = obs, 
                            action = episode[i]['action'],
                            goal= episode[i]['observation']["natural_language_instruction"].numpy().decode(),
                            goal_img=goal_img,
                            terminal = 1 if i == len(episode) - 1 else 0,
                            pose = pose,
                            # morphology = cfg.dataset.dataset_indicies[dataset_name]["morphology"],
                            )
    print("A terminé le mélange.")
    return cbuffer

def get_multi_dataset_portion(builders, cbuffer, cfg):
    """
    Helper function to get a portion of the dataset.
    """
    import tensorflow_datasets as tfds
    import numpy as np
    from tqdm import tqdm, trange
    import cv2
    # from PIL import Image
    from datasets import load_dataset
    # ------------
    # Train and test splits
    # Loading data
    # create RLDS dataset builder
    for dataset_name, builder in builders.items():
        print("Loading dataset:", dataset_name)
        ## Get the number of items in the dataset
        samples_ = (int(cfg.dataset.num_episodes * 
                        cfg.dataset.dataset_indicies[dataset_name]["weight"]))/cfg.dataset.chunk_size
        print(" size_ ", builder.info.splits["train"].num_examples
                , " total samples to fetch", int(samples_),
                 " chunk_size ", cfg.dataset.chunk_size)
        if cfg.dataset.chunk_size == builder.info.splits["train"].num_examples:
            ## Most likely grabbing the entire dataset
            ix = [0]
        else:
            ix = np.random.randint(builder.info.splits["train"].num_examples-cfg.dataset.chunk_size, size=int(samples_))
        get_dataset_portion(builder, cbuffer, cfg, dataset_name=dataset_name, list_=ix)

@hydra.main(config_path="./conf", config_name="simpleEnv-64pix-pose")
def my_main(cfg: DictConfig):
    import numpy as np
    # ------------
    # Train and test splits
    # Loading data
    # create RLDS dataset builder
    cfg.dataset.save_initial_dataset = True
    cfg.dataset.load_dataset = False
    cfg.dataset.encode_with_t5 = True  # Also encode the text for the dataset.
    cfg.n_embd = 512  # T5 small embedding size
    from grp_model import GRP
    model = GRP(cfg)
    model.to(cfg.device)
    np.random.seed(cfg.r_seed)
    cbuffer = CircularBuffer(cfg.dataset.buffer_size, cfg, model)

    print("Dataset shape:", len(cbuffer._dataset_tmp["img"]))
    print("Dataset len:", cbuffer._count)

    if cfg.dataset.save_initial_dataset:
        print("Saving dataset to:", cfg.dataset.to_name)
        ############# Save the dataset to a file
        cbuffer.save(cfg.dataset.to_name)
    # cbuffer.save(cfg.dataset.to_name)


if __name__ == "__main__":
    results = my_main()
    print("results:", results)