import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def get_patches_fast(images, cfg):
    from einops import rearrange
    batch_size, height, width, channels = images.shape
    patch_size = cfg.patch_size ## n_patches = 8

    patches = rearrange(images[:,:,:,:3], 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size)
    if channels > 3:
        ## History stacking in the channel dimension for observations only, not goal images.
        patches = rearrange(images, 'b (h p1) (w p2) (c hs) -> b (h w hs) (p1 p2 c)', p1 = patch_size, p2 = patch_size, hs=cfg.policy.obs_stacking) ## Stack the history in the channel dimension
    return patches


def calc_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B,T,C = x.shape
        # TODO: 
        ## Provide the block masking logic for the attention head
        if mask is None: 
            mask = torch.ones(T, device=x.device, dtype=torch.bool)
        else: 
            mask = mask.to(x.device)
        mask = mask.unsqueeze(0).unsqueeze(0)
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(mask == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd=n_embd, dropout=dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        with torch.profiler.record_function("Self-Attention"):
            out = torch.cat([h(x, mask) for h in self.heads], dim=-1)
            out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd=n_embd, dropout=dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, mask=None):
        x = x + self.sa(self.ln1(x), mask)
        x = x + self.ffwd(self.ln2(x))
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768):
        self.patch_size = patch_size
        super().__init__()
        # Conv layer to split image into patches and embed them
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e h w -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))

    def forward(self, x):
        b = x.shape[0]
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # Prepend [CLS] token
        x = torch.cat([cls_tokens, x], dim=1)
        return x


class GRP(nn.Module):
    def __init__(self, cfg, mlp_ratio=4):
        super(GRP, self).__init__()
        self._cfg = cfg

        # ------------------------------------------------------------------
        # Vocabulary / text configuration
        # ------------------------------------------------------------------
        chars = cfg.dataset.chars_list
        cfg.vocab_size = len(chars)

        # ------------------------------------------------------------------
        # 1) Patch embedding: map flattened image patches -> n_embd
        #    get_patches_fast produces patches of size (patch_size * patch_size * 3)
        #    for both observations and goal images.
        # ------------------------------------------------------------------
        patch_dim = cfg.patch_size * cfg.patch_size * 3
        self.patch_embedding = nn.Linear(patch_dim, cfg.n_embd)

        # ------------------------------------------------------------------
        # 2) Text embedding (when not using T5)
        #    goals_txt is [B, T] of token ids -> [B, T, n_embd]
        # ------------------------------------------------------------------
        if not cfg.dataset.encode_with_t5:
            self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embd)

        # ------------------------------------------------------------------
        # 3) Positional embeddings
        #    Compute an upper bound on sequence length:
        #    CLS + obs_patches + GOAL_IMG + goal_patches + text_tokens
        # ------------------------------------------------------------------
        h, w, _ = cfg.image_shape
        patches_per_image = (h // cfg.patch_size) * (w // cfg.patch_size)
        obs_patches_max = patches_per_image * cfg.policy.obs_stacking
        goal_patches = patches_per_image
        max_seq_len = 1 + obs_patches_max + 1 + goal_patches + cfg.max_block_size
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, cfg.n_embd)
        )

        # ------------------------------------------------------------------
        # 4) Special tokens: CLS and GOAL_IMG
        # ------------------------------------------------------------------
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.n_embd))
        self.goal_img_token = nn.Parameter(torch.randn(1, 1, cfg.n_embd))

        # ------------------------------------------------------------------
        # 5) Transformer encoder blocks
        # ------------------------------------------------------------------
        self.blocks = nn.ModuleList(
            [Block(cfg.n_embd, cfg.n_head, cfg.dropout) for _ in range(cfg.n_blocks)]
        )

        # ------------------------------------------------------------------
        # 6) Action head: map CLS embedding -> action vector
        # ------------------------------------------------------------------
        self.action_head = nn.Linear(
            cfg.n_embd, cfg.action_dim * cfg.policy.action_stacking
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, images, goals_txt, goal_imgs, targets=None, pose=None, mask_=False, last_action=None):
        """
        images:    [B, H, W, C_obs]
        goals_txt: [B, T] (token ids) or [B, T, n_embd] if encode_with_t5
        goal_imgs: [B, H, W, 3]
        """
        B, _, _, _ = images.shape

        # ------------------------------------------------------------------
        # 1) Extract patches from observation and goal images
        # ------------------------------------------------------------------
        obs_patches = get_patches_fast(images, self._cfg)      # [B, N_obs, patch_dim]
        goal_patches = get_patches_fast(goal_imgs, self._cfg)  # [B, N_goal, patch_dim]

        # ------------------------------------------------------------------
        # 2) Encode text goals
        # ------------------------------------------------------------------
        if self._cfg.dataset.encode_with_t5:
            # goals_txt is already [B, T, n_embd]
            goals_e = goals_txt
            _, T, _ = goals_txt.shape
        else:
            # goals_txt is [B, T] of token ids -> [B, T, n_embd]
            goals_e = self.token_embedding_table(goals_txt)
            B_txt, T = goals_txt.shape
            assert B_txt == B, "Batch size mismatch between images and text goals"

        # ------------------------------------------------------------------
        # 3) Project patches to embedding space
        # ------------------------------------------------------------------
        obs_token = self.patch_embedding(obs_patches)   # [B, N_obs, n_embd]
        goal_token = self.patch_embedding(goal_patches) # [B, N_goal, n_embd]

        # ------------------------------------------------------------------
        # 4) Special tokens
        # ------------------------------------------------------------------
        cls_token = self.cls_token.expand(B, -1, -1)          # [B, 1, n_embd]
        goal_img_token = self.goal_img_token.expand(B, -1, -1)  # [B, 1, n_embd]

        # ------------------------------------------------------------------
        # 5) Concatenate tokens in sequence:
        #    [CLS, obs_token, GOAL_IMG, goal_token, goals_e]
        # ------------------------------------------------------------------
        tokens = torch.cat(
            [cls_token, obs_token, goal_img_token, goal_token, goals_e],
            dim=1,
        )  # [B, seq_len, n_embd]
        seq_len = tokens.size(1)

        # ------------------------------------------------------------------
        # 6) Add positional embeddings
        # ------------------------------------------------------------------
        pos = self.pos_embedding[:, :seq_len, :]
        tokens = tokens + pos

        # ------------------------------------------------------------------
        # 7) Build a simple block mask to switch between text and image goals
        #    mask_ == True : mask goal image tokens (use text goal)
        #    mask_ == False: mask text tokens      (use image goal)
        # ------------------------------------------------------------------
        mask = None
        if mask_ is not None:
            # Start / end indices for segments in the concatenated sequence
            n_cls = 1
            n_obs = obs_token.size(1)
            n_goal_img = 1
            n_goal = goal_token.size(1)

            cls_idx = 0
            obs_start = cls_idx + n_cls
            obs_end = obs_start + n_obs
            goal_img_idx = obs_end
            goal_start = goal_img_idx + n_goal_img
            goal_end = goal_start + n_goal
            text_start = goal_end
            text_end = text_start + T  # should equal seq_len

            mask = torch.ones(seq_len, dtype=torch.bool, device=tokens.device)
            if mask_:
                # Mask out goal image tokens
                mask[goal_img_idx:goal_end] = False
            else:
                # Mask out text tokens
                mask[text_start:text_end] = False

        # ------------------------------------------------------------------
        # 8) Transformer blocks
        # ------------------------------------------------------------------
        for block in self.blocks:
            tokens = block(tokens, mask=mask)

        # ------------------------------------------------------------------
        # 9) Classification token and action head
        # ------------------------------------------------------------------
        cls_output = tokens[:, 0, :]           # [B, n_embd]
        out = self.action_head(cls_output)     # [B, action_dim * action_stacking]

        # ------------------------------------------------------------------
        # 10) Loss (continuous actions by default, using MSE)
        # ------------------------------------------------------------------
        if targets is not None:
            loss = F.mse_loss(out, targets)
        else:
            loss = torch.tensor(0.0, device=out.device)

        return out, loss
    
    def resize_image(self, image):
        """
        Docstring for resize_image
        
        :param self: Description
        :param image: Description
        self._resize_state = lambda sf:   cv2.resize(np.array(sf, dtype=np.float32), (cfg.image_shape[0], cfg.image_shape[1]))  # resize state
        """
        import cv2
        import numpy as _np
        img = _np.array(image, dtype=_np.float32)
        img = cv2.resize(img, (self._cfg.image_shape[0], self._cfg.image_shape[1]))
        return img

    def normalize_state(self, image):
        """
        Docstring for preprocess_state
        
        :param self: Description
        :param image: Description
        self._encode_state = lambda af:   ((af/(255.0)*2.0)-1.0) # encoder: take a float, output an integer
        self._resize_state = lambda sf:   cv2.resize(np.array(sf, dtype=np.float32), (cfg.image_shape[0], cfg.image_shape[1]))  # resize state
        """
        # img = _np.array(image, dtype=_np.float32)
        # img = cv2.resize(img, (self._cfg.image_shape[0], self._cfg.image_shape[1]))
        enc = ((image / 255.0) * 2.0) - 1.0
        # t = _torch.tensor(enc, dtype=_torch.float32, device=self._cfg.device)
        return enc
    
    def preprocess_state(self, image):
        img = self.resize_image(image)
        img = self.normalize_state(img)
        return img

    def preprocess_goal_image(self, image):
        return self.preprocess_state(image)
    
    def reset(self):
        """
        Reset the model's internal state if needed.
        """
        return None

    def encode_text_goal(self, goal, tokenizer=None, text_model=None):
        import numpy as _np
        import torch as _torch
        if self._cfg.dataset.encode_with_t5:
            if tokenizer is None or text_model is None:
                raise ValueError("tokenizer and text_model must be provided when using T5 encoding")
            # TODO:    
            ## Provide the logic converting text goal to T5 embedding tensor
            pass
        else:
            pad = " " * self._cfg.max_block_size
            goal_ = goal[:self._cfg.max_block_size] + pad[len(goal):self._cfg.max_block_size]
            try:
                stoi = {c: i for i, c in enumerate(self._cfg.dataset.chars_list)}
                ids = [stoi.get(c, 0) for c in goal_]
            except Exception:
                ids = [0] * self._cfg.max_block_size
            return _torch.tensor(_np.expand_dims(_np.array(ids, dtype=_np.int64), axis=0), dtype=_torch.long, device=self._cfg.device)

    def process_text_embedding_for_buffer(self, goal, tokenizer=None, text_model=None):
        """
        Process text goal embedding for storing in the circular buffer.
        Returns a numpy array of shape (max_block_size, n_embd) without batch dimension.
        """
        import numpy as _np
        if tokenizer is None or text_model is None:
            raise ValueError("tokenizer and text_model must be provided when using T5 encoding")
        
        goal_ = _np.zeros((self._cfg.max_block_size, self._cfg.n_embd), dtype=_np.float32)
        input_ids = tokenizer(goal, return_tensors="pt").input_ids
        goal_t = text_model.encoder(input_ids).last_hidden_state.detach().cpu().numpy()
        goal_[:len(goal_t[0]), :] = goal_t[0][:self._cfg.max_block_size]
        return goal_

    def decode_action(self, action_tensor):
        """Decode normalized actions to original action space"""
        import torch as _torch
        action_mean = _torch.tensor(np.repeat([self._cfg.dataset.action_mean], self._cfg.policy.action_stacking, axis=0).flatten(), 
                                   dtype=action_tensor.dtype, device=action_tensor.device)
        action_std = _torch.tensor(np.repeat([self._cfg.dataset.action_std], self._cfg.policy.action_stacking, axis=0).flatten(), 
                                  dtype=action_tensor.dtype, device=action_tensor.device)
        return (action_tensor * (action_std)) + action_mean

    def encode_action(self, action_float):
        """Encode actions to normalized space [-1, 1]"""
        import torch as _torch
        ## If the action_float has length greater than action_dim then use stacking otherwise just use normal standardiaztion vectors
        if action_float.shape[1] == len(self._cfg.dataset.action_mean):
            action_mean = _torch.tensor(self._cfg.dataset.action_mean, dtype=action_float.dtype, device=action_float.device)
            action_std = _torch.tensor(self._cfg.dataset.action_std, dtype=action_float.dtype, device=action_float.device)
            return (action_float - action_mean) / (action_std)  

        action_mean = _torch.tensor(np.repeat([self._cfg.dataset.action_mean], self._cfg.policy.action_stacking, axis=0).flatten(), 
                                   dtype=action_float.dtype, device=action_float.device)
        action_std = _torch.tensor(np.repeat([self._cfg.dataset.action_std], self._cfg.policy.action_stacking, axis=0).flatten(), 
                                  dtype=action_float.dtype, device=action_float.device)
        return (action_float - action_mean) / (action_std)
    
    def decode_state(self, state_tensor):
        """
        Docstring for decode_state
        
        :param self: Description
        :param state_tensor: Description
        self._decode_state = lambda sinN: (sinN * state_std) + state_mean  # Undo mapping to [-1, 1]
        """
        import torch as _torch
        state_mean = _torch.tensor(self._cfg.dataset.state_mean, dtype=state_tensor.dtype, device=state_tensor.device)
        state_std = _torch.tensor(self._cfg.dataset.state_std, dtype=state_tensor.dtype, device=state_tensor.device)
        return (state_tensor * (state_std)) + state_mean
    
    def encode_state(self, state_float):
        """
        Docstring for encode_state
        
        :param self: Description
        :param state_float: Description
        self._encode_state = lambda sf:   (sf - state_mean)/(state_std) # encoder: take a float, output an integer
        """
        import torch as _torch
        state_mean = _torch.tensor(self._cfg.dataset.state_mean, dtype=state_float.dtype, device=state_float.device)
        state_std = _torch.tensor(self._cfg.dataset.state_std, dtype=state_float.dtype, device=state_float.device)
        return (state_float - state_mean) / (state_std)


@torch.no_grad()
def estimate_loss(model, dataset):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(model._cfg.eval_iters)
        for k in range(model._cfg.eval_iters):
            X, x_pose, x_goal, x_goal_img, Y, last_action = dataset.get_batch_grp(split, model._cfg, model._cfg.batch_size)
            logits, loss = model(X, x_goal, x_goal_img, Y, pose=x_pose, last_action=last_action)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
