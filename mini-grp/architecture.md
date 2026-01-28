# Complete GRP Architecture Guide

This document shows how all components fit together in the GRP model.

## 🏗️ Complete Architecture Overview

```
INPUT
  ↓
[Images] [Goal Text] [Goal Images]
  ↓         ↓            ↓
┌─────────────────────────────────────┐
│ 1. PATCH EXTRACTION                 │
│    - obs_patches: [B, n_patches, patch_dim] │
│    - patches_g: [B, n_patches, patch_dim]  │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ 2. EMBEDDING PROJECTION             │
│    - obs_embd: [B, n_patches, n_embd]      │
│    - goal_embd: [B, n_patches, n_embd]     │
│    - text_embd: [B, T, n_embd]      │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ 3. TOKEN ASSEMBLY                   │
│    [CLS, obs_embd, GOAL_IMG, goal_embd, text_embd] │
│    → tokens: [B, seq_len, n_embd]   │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ 4. POSITIONAL EMBEDDINGS            │
│    tokens = tokens + pos_embedding  │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ 5. MASK CREATION                    │
│    mask = create_mask(mask_, ...)   │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ 6. TRANSFORMER BLOCKS (x4)          │
│    for block in self.blocks:        │
│      tokens = block(tokens, mask)   │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ 7. CLS TOKEN EXTRACTION             │
│    cls_output = tokens[:, 0, :]     │
│    → [B, n_embd]                    │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ 8. ACTION PREDICTION                │
│    out = self.mlp_head(cls_output)  │
│    → [B, action_dim * stacking]     │
└─────────────────────────────────────┘
  ↓
OUTPUT: (actions, loss)
```

## 📦 What Goes in `__init__`

### Required Components:

```python
def __init__(self, cfg, mlp_ratio=4):
    super(GRP, self).__init__()
    self._cfg = cfg
    
    # 1. PATCH EMBEDDING
    # Projects patches to embedding space
    # Input: [B, n_patches, patch_dim]
    # Output: [B, n_patches, n_embd]
    patch_dim = ???  # Calculate: patch_size * patch_size * channels
    self.patch_embedding = nn.Linear(patch_dim, cfg.n_embd)
    
    # 2. TEXT EMBEDDING (only if not using T5)
    # Converts token IDs to embeddings
    # Input: [B, max_block_size] (token IDs)
    # Output: [B, max_block_size, n_embd]
    if not cfg.dataset.encode_with_t5:
        self.token_embedding_table = nn.Embedding(
            cfg.vocab_size,  # vocabulary size
            cfg.n_embd       # embedding dimension
        )
    
    # 3. SPECIAL TOKENS (learnable parameters)
    # CLS token: used for final prediction
    # GOAL_IMG token: marks start of goal image patches
    self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.n_embd))
    self.goal_img_token = nn.Parameter(torch.randn(1, 1, cfg.n_embd))
    
    # 4. POSITIONAL EMBEDDINGS
    # Learnable position encodings
    # Shape: [1, max_seq_len, n_embd]
    max_seq_len = ???  # Calculate: CLS + obs_patches + GOAL_IMG + goal_patches + text
    self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, cfg.n_embd))
    
    # 5. TRANSFORMER BLOCKS
    # Multiple encoder blocks for processing
    self.blocks = nn.ModuleList([
        Block(cfg.n_embd, cfg.n_head, cfg.dropout)
        for _ in range(cfg.n_blocks)
    ])
    
    # 6. ACTION HEAD
    # Final layer to predict actions
    # Input: [B, n_embd]
    # Output: [B, action_dim * action_stacking]
    self.mlp_head = nn.Linear(
        cfg.n_embd, 
        cfg.action_dim * cfg.policy.action_stacking
    )
```

## 🔄 What Goes in `forward`

### Complete Flow:

```python
def forward(self, images, goals_txt, goal_imgs, targets=None, pose=None, mask_=False):
    B = images.shape[0]  # Batch size
    
    # ============================================
    # STEP 1: Extract Patches
    # ============================================
    obs_patches = get_patches_fast(images, self._cfg)
    # Shape: [B, n_obs_patches, patch_dim]
    
    patches_g = get_patches_fast(goal_imgs, self._cfg)
    # Shape: [B, n_goal_patches, patch_dim]
    
    # ============================================
    # STEP 2: Encode Text Goals
    # ============================================
    if self._cfg.dataset.encode_with_t5:
        goals_e = goals_txt  # Already embeddings
        # Shape: [B, T, n_embd]
        B, T, E = goals_txt.shape
    else:
        goals_e = self.token_embedding_table(goals_txt)
        # Shape: [B, max_block_size, n_embd]
        B, E = goals_txt.shape
        T = self._cfg.max_block_size
    
    # ============================================
    # STEP 3: Project Patches to Embeddings
    # ============================================
    obs_embd = self.patch_embedding(obs_patches)
    # Shape: [B, n_obs_patches, n_embd]
    
    goal_embd = self.patch_embedding(patches_g)
    # Shape: [B, n_goal_patches, n_embd]
    
    # ============================================
    # STEP 4: Get Special Tokens
    # ============================================
    cls_token = self.cls_token.expand(B, -1, -1)
    # Shape: [B, 1, n_embd]
    
    goal_img_token = self.goal_img_token.expand(B, -1, -1)
    # Shape: [B, 1, n_embd]
    
    # ============================================
    # STEP 5: Concatenate All Tokens
    # ============================================
    # Order: [CLS, obs_patches, GOAL_IMG, goal_patches, text]
    tokens = torch.cat([
        cls_token,      # [B, 1, n_embd]
        obs_embd,       # [B, n_obs_patches, n_embd]
        goal_img_token, # [B, 1, n_embd]
        goal_embd,      # [B, n_goal_patches, n_embd]
        goals_e         # [B, T, n_embd]
    ], dim=1)  # Concatenate along sequence dimension
    # Shape: [B, seq_len, n_embd]
    # where seq_len = 1 + n_obs_patches + 1 + n_goal_patches + T
    
    # ============================================
    # STEP 6: Add Positional Embeddings
    # ============================================
    seq_len = tokens.shape[1]
    tokens = tokens + self.pos_embedding[:, :seq_len, :]
    # Shape: [B, seq_len, n_embd] (unchanged)
    
    # ============================================
    # STEP 7: Create Block Mask
    # ============================================
    # mask_ is a boolean flag:
    # - True: mask goal image patches (use text goal)
    # - False: mask text tokens (use image goal)
    mask = create_block_mask(mask_, seq_len, ...)  # You need to implement this
    # Shape: [seq_len] (boolean tensor)
    
    # ============================================
    # STEP 8: Pass Through Transformer Blocks
    # ============================================
    for block in self.blocks:
        tokens = block(tokens, mask=mask)
    # Shape: [B, seq_len, n_embd] (unchanged, but content transformed)
    
    # ============================================
    # STEP 9: Extract CLS Token
    # ============================================
    cls_output = tokens[:, 0, :]
    # Shape: [B, n_embd]
    
    # ============================================
    # STEP 10: Predict Actions
    # ============================================
    out = self.mlp_head(cls_output)
    # Shape: [B, action_dim * action_stacking]
    
    # ============================================
    # STEP 11: Compute Loss
    # ============================================
    if targets is not None:
        # For continuous actions: MSE loss
        loss = F.mse_loss(out, targets)
        # For discrete actions: Cross-entropy loss
        # loss = F.cross_entropy(out.view(-1, ...), targets.view(-1))
    else:
        loss = torch.tensor(0.0, device=out.device)
    
    return (out, loss)
```

## 📐 Shape Flow Through the Model

```
Input Shapes:
  images:      [B, H, W, C]           # e.g., [256, 64, 64, 6]
  goals_txt:   [B, max_block_size]     # or [B, T, n_embd] if T5
  goal_imgs:   [B, H, W, 3]           # e.g., [256, 64, 64, 3]

After Patch Extraction:
  obs_patches: [B, n_obs_patches, patch_dim]    # e.g., [256, 128, 192]
  patches_g:   [B, n_goal_patches, patch_dim]    # e.g., [256, 64, 192]

After Embedding Projection:
  obs_embd:    [B, n_obs_patches, n_embd]       # e.g., [256, 128, 128]
  goal_embd:   [B, n_goal_patches, n_embd]      # e.g., [256, 64, 128]
  goals_e:     [B, T, n_embd]                   # e.g., [256, 12, 128]

After Special Tokens:
  cls_token:      [B, 1, n_embd]                # e.g., [256, 1, 128]
  goal_img_token: [B, 1, n_embd]                 # e.g., [256, 1, 128]

After Concatenation:
  tokens: [B, seq_len, n_embd]
  # seq_len = 1 + 128 + 1 + 64 + 12 = 206
  # e.g., [256, 206, 128]

After Transformer Blocks:
  tokens: [B, seq_len, n_embd]  # Same shape, transformed content
  # e.g., [256, 206, 128]

After CLS Extraction:
  cls_output: [B, n_embd]  # e.g., [256, 128]

After Action Head:
  out: [B, action_dim * stacking]  # e.g., [256, 7]
```

## 🔑 Key Calculations You Need

### 1. Patch Dimension
```python
# For observation images with history stacking:
patch_dim = cfg.patch_size * cfg.patch_size * (3 * cfg.policy.obs_stacking)
# Example: 8 * 8 * (3 * 2) = 8 * 8 * 6 = 384

# For goal images (no stacking):
patch_dim = cfg.patch_size * cfg.patch_size * 3
# Example: 8 * 8 * 3 = 192
```

### 2. Number of Patches
```python
# Images are 64x64, patch_size is 8
n_patches_per_image = (64 // 8) * (64 // 8) = 8 * 8 = 64

# With obs_stacking=2, patches are stacked in channel dimension
# So you get: 64 * 2 = 128 patches for observations
n_obs_patches = 64 * cfg.policy.obs_stacking
```

### 3. Sequence Length
```python
seq_len = (
    1 +                    # CLS token
    n_obs_patches +        # Observation patches
    1 +                    # GOAL_IMG token
    n_goal_patches +       # Goal image patches
    cfg.max_block_size     # Text tokens
)
```

## 🎯 What You Still Need to Implement

1. **Calculate dimensions** in `__init__`:
   - `patch_dim` (for patch_embedding input)
   - `max_seq_len` (for positional embeddings)

2. **Create block mask function**:
   - Takes `mask_` boolean flag
   - Returns mask tensor indicating which tokens to attend to
   - Masks either goal image patches OR text tokens based on `mask_`

3. **Handle edge cases**:
   - What if sequence length > max_seq_len? (truncate positional embeddings)
   - What if sequence length < max_seq_len? (pad or use subset)

## 🧩 How Components Connect

```
__init__ creates:
  ├── patch_embedding ──────┐
  ├── token_embedding_table─┤
  ├── cls_token ────────────┤
  ├── goal_img_token ───────┤
  ├── pos_embedding ────────┤
  ├── blocks ───────────────┤
  └── mlp_head ─────────────┘
         │
         │ All used in forward()
         ↓
forward() uses them in sequence:
  1. patch_embedding → project patches
  2. token_embedding_table → encode text (if not T5)
  3. cls_token, goal_img_token → special tokens
  4. pos_embedding → add positions
  5. blocks → transform tokens
  6. mlp_head → predict actions
```

## 💡 Final Checklist

- [ ] Calculate patch_dim correctly
- [ ] Calculate max_seq_len correctly
- [ ] Create all layers in __init__ (not None!)
- [ ] Project both obs and goal patches
- [ ] Include goal_img_token in concatenation
- [ ] Add positional embeddings correctly
- [ ] Create and pass mask to blocks
- [ ] Iterate through blocks (not call directly)
- [ ] Extract CLS token correctly
- [ ] Use mlp_head (not action_head)
- [ ] Compute loss if targets provided

This is the complete picture! Now you can see how everything connects. 🚀
