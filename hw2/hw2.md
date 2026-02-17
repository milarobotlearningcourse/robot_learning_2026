# Homework 2: World Models and Model-Based Planning

## Introduction

In this assignment, you will implement model-based reinforcement learning for robotic manipulation tasks in the LIBERO benchmark. Unlike purely behavior cloning approaches, model-based methods learn a **world model** that predicts future states, and use this model to **plan** action sequences that maximize reward.

You will progressively build up your understanding by:
1. Training a simple pose-based world model
2. Integrating a policy to improve planning efficiency
3. Upgrading to DreamerV3, a state-of-the-art world model with image observations
4. Training an image-based policy that works with DreamerV3

This writeup intentionally avoids step-by-step recipes. Your job is to read the provided code, identify the missing pieces, and justify your design choices in the report.

This assignment teaches you how model-based planning can complement learned policies and how to scale from simple state representations (poses) to high-dimensional observations (images).

---

## Background: Model-Based RL

### World Models

A **world model** learns to predict how the environment evolves:
$$\hat{s}_{t+1} = f_\theta(s_t, a_t)$$

Where:
- $s_t$ is the current state (pose or image latent)
- $a_t$ is the action taken
- $\hat{s}_{t+1}$ is the predicted next state

In this assignment, you'll work with two types of world models:

1. **SimpleWorldModel**: Predicts next robot pose (7-D) given current pose and action
   - Input: Current pose + action → Output: Next pose + reward
   - Simple MLP architecture, fast training

2. **DreamerV3**: Advanced world model with image observations
   - Encodes images to latent states using a CNN encoder
   - Uses Recurrent State-Space Model (RSSM) for temporal dynamics
   - Predicts rewards, continues (episode termination), and reconstructed images

### Planning with Cross-Entropy Method (CEM)

Given a world model, we can plan actions by:
1. Sampling $K$ random action sequences of length $H$ (the planning horizon)
2. Rolling out each sequence in the world model to predict cumulative reward
3. Selecting the top $M$ "elite" sequences with highest predicted reward
4. Fitting a Gaussian distribution to the elite actions
5. Repeating for $L$ iterations to refine the plan

This is called the **Cross-Entropy Method (CEM)**, a gradient-free optimization approach for action selection.

### Policy-Guided Planning

Random sampling in CEM can be inefficient. We can improve it by using a learned **policy** to initialize the action distribution:
- Instead of starting with mean=0, std=1, we start with the policy's predicted mean and std
- CEM then refines this initialization to find better action sequences
- This combines the benefits of learned behavior (from the policy) with model-based planning

---

## Part 1: Simple World Model Planning

**Goal**: Implement and train a simple pose-based world model that predicts the next robot pose given the current pose and action.

### Part 1.1: Understanding the SimpleWorldModel

First, examine [simple_world_model.py](simple_world_model.py). The `SimpleWorldModel` class:
- Takes normalized pose (7-D) + normalized action (7-D) as input
- Uses a simple MLP to predict the next normalized pose and reward

**Key methods**:
- `forward(pose, action)`: Returns predicted next pose and reward
- `compute_loss(pose, action, target_pose, target_reward)`: MSE loss for training
- `predict_next_pose(pose, action)`: Convenience method that handles encoding/decoding

### Part 1.2: Training the Simple World Model

**What to implement**: Complete the training loop in [dreamer_model_trainer.py](dreamer_model_trainer.py)

You need to implement the `compute_loss` method in the `ModelTrainingWrapper` class for the `'simple'` model type. Define and justify the loss terms you choose (pose, reward, and any regularization). You should be able to explain why your implementation is numerically stable.

### Part 1.3: Implementing CEM Planning

**What to implement**: Complete the CEM planner in [planning.py](planning.py)

The `CEMPlanner` class implements the Cross-Entropy Method. You need to:
- Decide how to initialize the action distribution and justify it
- Decide how to handle action bounds during sampling and explain your choice
- Implement a stopping or convergence condition (even if you keep a fixed iteration count, justify why)

### Part 1.4: Running Experiments

**Commands to run**:

```bash
# Train SimpleWorldModel and evaluate with CEM planning
python dreamer_model_trainer.py \
    model_type=simple \
    planner.type=cem \
    planner.horizon=10 \
    planner.num_samples=100 \
    planner.num_elites=10 \
    training.num_epochs=50 \
    exp_name=q1_simple_cem
```

**What to analyze**:
- Plot training loss curves (pose_loss and reward_loss over epochs)
- Evaluate success rate in LIBERO environment
- Visualize planned action sequences (optional: modify code to save these)
- Compare model prediction error on validation data

**Required reflection**:
- Identify a failure case from your evaluation and explain what part of the model or planner caused it
- Propose one concrete change that could fix the failure, and test it

**Expected results**:
- Training loss should decrease and converge
- Success rate should be > 0 (random policy gets ~0%)
- Model predictions should reasonably match ground truth poses

### Questions for Part 1:

1. How does the planning horizon affect success rate? Try `horizon=[5, 10, 15]`
2. How does the number of CEM samples affect planning quality? Try `num_samples=[50, 100, 200]`
3. What is the trade-off between planning time and success rate?
4. Explain one design decision you made in the CEM implementation and why you rejected at least one alternative.

---

## Part 2: Policy-Guided Planning

**Goal**: Improve CEM planning efficiency by using a learned policy to initialize the action distribution.

### Part 2.1: Understanding Policy Initialization

The `PolicyPlanner` class (see [planning.py](planning.py)) uses a trained policy model to:
1. Predict an action sequence given the current state
2. Use this as the initial mean for CEM
3. CEM then refines this initialization

This is more efficient than random initialization because:
- The policy has learned useful behaviors from data
- CEM only needs to make local improvements
- Fewer samples are needed to find good plans

### Part 2.2: Implementing Policy Training

**What to implement**: Complete the policy training in [planning.py](planning.py)

The `PolicyPlanner` has an `update` method that trains the policy using data collected from the environment. You need to:
- Decide how you batch and normalize the data (justify your choices)
- Implement a training loop with a loss that you can explain and defend
- Add at least one sanity check or diagnostic metric (for example, action distribution statistics or prediction error on a held-out subset)

### Part 2.3: Integrating Policy with CEM

Once the policy is trained, integrate it with CEM. Decide how the policy influences the initial distribution and justify any fixed hyperparameters (like initial std) you choose. Explain how your choice affects exploration vs exploitation.

### Part 2.4: Running Experiments

**Commands to run**:

```bash
# First, train policy from offline data
python dreamer_model_trainer.py \
    model_type=simple \
    planner.type=policy \
    planner.horizon=10 \
    training.num_epochs=50 \
    exp_name=q2_policy_training

# Then, use policy-guided CEM planning
python dreamer_model_trainer.py \
    model_type=simple \
    planner.type=policy_guided_cem \
    planner.horizon=10 \
    planner.num_samples=50 \
    planner.num_elites=5 \
    load_policy=outputs/q2_policy_training/policy.pth \
    exp_name=q2_policy_cem
```

**What to analyze**:
- Compare success rates: Random CEM vs Policy-guided CEM
- Compare planning time (fewer samples needed with policy guidance)
- Visualize how CEM refines the policy's initial predictions
- Analyze cases where policy alone fails but policy+CEM succeeds

**Required ablation**:
- Change one design choice in your policy training (optimizer, loss, normalization, or architecture) and report its impact

### Questions for Part 2:

1. How much does the policy improve CEM efficiency? Compare success rate with 50 samples (policy-guided) vs 100 samples (random)
2. Can the policy alone (without CEM refinement) solve the task? Why or why not?
3. What are the benefits of combining learned policies with model-based planning?
4. Provide one plot or table that demonstrates a non-obvious failure mode of the policy or planner.

---

## Part 3: DreamerV3 World Model

**Goal**: Replace the simple pose-based model with DreamerV3, a sophisticated world model that works with image observations.

### Part 3.1: Understanding DreamerV3

DreamerV3 (see [dreamerV3.py](dreamerV3.py)) is a state-of-the-art world model with:

**Architecture**:
1. **Encoder**: CNN that maps images to embedding vectors
   ```
   Image (C, H, W) → CNN → Embedding (e.g., 256-D)
   ```

2. **RSSM (Recurrent State-Space Model)**: Maintains latent world state
   ```
   - Deterministic state (h): Recurrent hidden state (GRU)
   - Stochastic state (z): Discrete latent variables (categorical distribution)
   - Combines: h_{t+1} = f(h_t, z_t, a_t)
   ```

3. **Decoder**: Reconstructs images from latent state
   ```
   (h, z) → Decoder CNN → Reconstructed Image
   ```

4. **Prediction Heads**:
   - Reward predictor: (h, z) → predicted reward
   - Continue predictor: (h, z) → probability of episode continuation
   - Representation model: Encodes observations
   - Dynamics model: Predicts next state without observations

### Part 3.2: Training DreamerV3

**What to implement**: The DreamerV3 training

You must identify which losses are needed (reconstruction, reward, continue, and dynamics/representation) and explain how they are weighted. If you change any default weights, justify them with a small ablation.

### Part 3.3: CEM Planning with DreamerV3

**What to implement**: Complete the `_evaluate_sequences_dreamer` method in [planning.py](planning.py)

You should be able to explain how imagined rollouts are generated and how rewards are accumulated. Include a brief discussion of any numerical or stability issues you encountered.

### Part 3.4: Running Experiments

**Commands to run**:

```bash
# Train DreamerV3 world model
python dreamer_model_trainer.py \
    model_type=dreamer \
    planner.type=cem \
    planner.horizon=10 \
    planner.num_samples=100 \
    training.num_epochs=100 \
    exp_name=q3_dreamer_cem
```

**What to analyze**:
- Compare training time: DreamerV3 vs SimpleWorldModel (DreamerV3 should be slower)
- Compare success rates: Image-based (DreamerV3) vs Pose-based (SimpleWorldModel)
- Visualize reconstructed images to verify the model learned visual features
- Analyze the three loss components over training

**Required diagnostic**:
- Show at least one reconstruction example that fails and explain why

### Questions for Part 3:

1. How does DreamerV3 compare to SimpleWorldModel in terms of:
   - Training time
   - Sample efficiency
   - Final success rate
2. Why might image-based models be beneficial even if pose is available?
3. Visualize image reconstructions - what features did the model learn?
4. Describe one implementation detail that you had to discover by reading the code rather than the writeup.

---

## Part 4: Image-Based Policy

**Goal**: Train a policy that works directly with images and integrates with DreamerV3.

### Part 4.1: Policy Architecture for Images

**What to implement**: Modify the `PolicyPlanner` to work with images

You must choose an image encoder or reuse parts of DreamerV3. Explain why your choice is appropriate and what alternatives you considered.

### Part 4.3: Policy-Guided CEM with DreamerV3

Combine the image-based policy with CEM planning. Clearly describe how information flows from image to RSSM state to action distribution, and explain any assumptions you make about temporal alignment or horizon length.

### Part 4.4: Running Experiments

**Commands to run**:

```bash
# Train image-based policy
python dreamer_model_trainer.py \
    model_type=dreamer \
    planner.type=policy \
    planner.horizon=10 \
    training.num_epochs=100 \
    exp_name=q4_dreamer_policy

# Evaluate policy-guided CEM with DreamerV3
python dreamer_model_trainer.py \
    model_type=dreamer \
    planner.type=policy_guided_cem \
    planner.horizon=10 \
    planner.num_samples=50 \
    load_policy=outputs/q4_dreamer_policy/policy.pth \
    load_world_model=outputs/q3_dreamer_cem/world_model.pth \
    exp_name=q4_dreamer_policy_cem
```

**What to analyze**:
- Compare all four approaches:
  1. SimpleWorldModel + Random CEM
  2. SimpleWorldModel + Policy-guided CEM
  3. DreamerV3 + Random CEM
  4. DreamerV3 + Policy-guided CEM
- Create a table showing success rates and planning times
- Analyze which approach works best and why

**Required robustness check**:
- Evaluate one additional seed or environment variation and discuss variability

### Questions for Part 4:

1. Create a comparison table of all methods. Which performs best?
2. What are the trade-offs between model complexity and performance?
3. When would you use each approach in practice?
4. How does sample efficiency compare between the methods?
5. Describe one result that surprised you and provide a plausible explanation.

---

## Submission Requirements

### 1. Code Submission

Create a submission folder containing:

**Required directory structure**:
```
submit/
├── hw2/
│   ├── dreamer_model_trainer.py
│   ├── simple_world_model.py
│   ├── planning.py
│   ├── dreamerV3.py
│   ├── world_model.py
│   └── sim_eval.py
├── outputs/
│   ├── q1_simple_cem/
│   │   ├── miniGRP.pth
│   │   ├── training_logs.csv
│   │   └── .hydra/
│   ├── q2_policy_training/
│   │   ├── miniGRP.pth
│   │   ├── policy.pth
│   │   └── training_logs.csv
│   ├── q2_policy_cem/
│   │   └── evaluation_results.json
│   ├── q3_dreamer_cem/
│   │   ├── miniGRP.pth
│   │   └── training_logs.csv
│   ├── q4_dreamer_policy/
│   │   ├── miniGRP.pth
│   │   └── policy.pth
│   └── q4_dreamer_policy_cem/
│       └── evaluation_results.json
└── README.md
```

**Important notes**:
- Keep the experiment names exactly as specified (`q1_simple_cem`, etc.)
- Each trained model should be saved as `miniGRP.pth`
- Keep the `.hydra/` folder for model configuration
- The submission zip should be under 50MB (models are small)
- **Mac users**: Use `zip -vr submit.zip submit -x "*.DS_Store"` to avoid compression artifacts

**Upload to Gradescope**:
- Code + logs → **HW2 Code**
- Written report (PDF) → **HW2 Written**

### 2. Written Report

Submit a PDF report addressing all questions from each part. Your report should include:

**Part 1: Simple World Model**
- Loss curves (pose loss and reward loss over epochs)
- Success rate vs planning horizon plot (horizon = 5, 10, 15)
- Success rate vs number of CEM samples (samples = 50, 100, 200)
- Discussion: How does planning horizon affect performance? What is the trade-off between sample count and computation time?

**Additional report requirements**:
- Include a short "design decisions" section that explains at least three choices you made by reading the code (not the writeup)
- Include at least two failure-case analyses with screenshots or plots
- Cite the exact experiment name and random seed for every plot or table

**Part 2: Policy-Guided Planning**  
- Comparison table:
  ```
  | Method                    | Success Rate | Avg Planning Time |
  |---------------------------|--------------|-------------------|
  | Random CEM (100 samples)  | X%           | Y ms              |
  | Policy-guided CEM (50)    | X%           | Y ms              |
  | Policy alone (no CEM)     | X%           | Y ms              |
  ```
- Case study: Show 2-3 examples where policy+CEM succeeds but policy alone fails
- Discussion: Why is combining policies with planning beneficial?

**Part 3: DreamerV3**
- Training comparison table:
  ```
  | Model Type      | Training Time | Final Loss | Success Rate |
  |-----------------|---------------|------------|--------------|
  | SimpleWorldModel| X min         | Y          | Z%           |
  | DreamerV3      | X min         | Y          | Z%           |
  ```
- Visualizations:
  - Sample reconstructed images from DreamerV3 decoder
  - Loss curves for all three DreamerV3 loss components (reconstruction, dynamics, representation)
- Discussion: What visual features did DreamerV3 learn? Why might images be better than poses?

**Part 4: Image-Based Policy**
- Final comparison of all four approaches:
  ```
  | Method                           | Success Rate | Planning Time | Model Params |
  |----------------------------------|--------------|---------------|--------------|
  | Simple + Random CEM              | X%           | Y ms          | Z M          |
  | Simple + Policy-guided CEM       | X%           | Y ms          | Z M          |
  | DreamerV3 + Random CEM           | X%           | Y ms          | Z M          |
  | DreamerV3 + Policy-guided CEM    | X%           | Y ms          | Z M          |
  ```
- Analysis: Which method would you deploy in these scenarios?
  1. Real-time control (< 50ms per action)
  2. Maximum success rate (no time constraint)
  3. Limited training data (< 1000 demonstrations)
  4. Sim-to-real transfer (domain adaptation needed)

**General Requirements**:
- Clear, well-labeled plots with legends
- Concise but complete answers to all questions
- Include at least one surprising or interesting finding from your experiments
- Total length: 6-10 pages (including figures)

### 3. Leaderboard Submission (Optional Extra Credit)

Submit your best model to the [course leaderboard](https://huggingface.co/spaces/gberseth/mila-robot-learning-course) for autonomous evaluation.

**How to submit**:
```bash
# Navigate to your best model's output folder
cd outputs/q4_dreamer_policy_cem/

# Upload to Hugging Face (requires HF CLI and authentication)
huggingface-cli upload gberseth/mini-grp-hw2-{your_name} .
```

**Required files in the upload**:
- `miniGRP.pth` - Your trained model weights
- `policy.pth` - Your policy weights (if applicable)
- `.hydra/` - Model configuration folder
- `simple_world_model.py` or `dreamerV3.py` - Model architecture file


## Submitting Model to Leaderboard

There is a [leaderboard](https://huggingface.co/spaces/gberseth/mila-robot-learning-course) that can be used to evaluate your model. If you don't have LIBERO working locally or SimplerEnv, you can submit your trained policies to this page for evaluation. **Everyone must submit their best model to this leaderboard for autonomous evaluation.**

### How to Submit

Navigate to your model's output folder and upload to Hugging Face:

```bash
# Example: Submit your best model from Part 4
cd outputs/q4_dreamer_policy_cem
hf upload gberseth/mini-grp .
```

**Required files in your output folder**:
```
outputs/q4_dreamer_policy_cem/
├── miniGRP.pth              # Your trained model weights
├── policy.pth               # Policy weights (if using policy planner)
├── simple_world_model.py    # OR dreamerV3.py - model architecture
├── .hydra/                  # Hydra config folder (auto-generated)
│   ├── config.yaml
│   └── overrides.yaml
```

The evaluation code will:
1. Load your model using `miniGRP.pth` and the config in `.hydra/`
2. Import your model architecture from `simple_world_model.py` or `dreamerV3.py`
3. Run episodes on held-out LIBERO tasks
4. Report success rate, average reward, and inference time

**Leaderboard scoring**:
- Success rate (primary metric)
- Planning/inference time (secondary, for tie-breaking)
- Robustness across different tasks

---

## Frequently Asked Questions


**Q: How long should training take?**
A: SimpleWorldModel should train in 10-20 minutes. DreamerV3 may take 2-4 hours depending on your GPU. Start with smaller models (`hidden_dim=64`) for debugging.

**Q: My CEM planner is very slow. Is this normal?**
A: CEM with 100 samples and horizon 10 should take less than 1 second per planning step. If it's slower, check:
- Are you computing gradients unnecessarily? Use `torch.no_grad()`
- Are you moving tensors to CPU repeatedly? Keep everything on GPU
- Is your world model in eval mode? Call `model.eval()`

**Q: The policy doesn't improve the CEM planning. Why?**
A: Common issues:
- Policy not trained on good quality data (should train after collecting CEM-planned trajectories)
- Policy overfitting to training data (collect more diverse demonstrations)
- CEM temperature too low (policy produces same actions every time)
- Not enough policy training iterations

**Q: DreamerV3 reconstructed images look blurry. Is something wrong?**
A: No, this is expected. The RSSM learns to capture task-relevant features, not pixel-perfect reconstructions. As long as you can recognize objects and robot position, it's fine.

**Q: Can I use a pre-trained policy from HW1?**
A: Yes! In fact, this is encouraged for Part 2. You can load your HW1 policy and use it to initialize CEM. But is can be slow.

**Q: Can I modify the provided code beyond the TODO sections?**
A: Yes, as long as the core functionality remains. You can add helper functions, logging, visualizations, etc. Document any major changes in your README.

---

## Debugging Guide

### Issue: Model loss doesn't decrease

**Potential causes**:
1. **Learning rate too high/low**: Try [1e-5, 1e-4, 1e-3]
2. **Incorrect normalization**: Check that poses/actions are normalized before feeding to model
3. **Data loading error**: Verify dataset shapes with `print(batch['pose'].shape)`
4. **Gradient explosion**: Add gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`

**Debug steps**:
```python
# Check if model can overfit to small batch
small_batch = next(iter(train_loader))
for i in range(1000):
    loss = model.compute_loss(small_batch)
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        print(f"Step {i}: Loss {loss.item()}")
# Should see loss decrease significantly
```

### Issue: CEM planning gives random actions

**Potential causes**:
1. **World model not trained**: Verify model weights are loaded
2. **Incorrect state format**: Check that initial_state has correct keys and shapes
3. **Temperature too high**: Lower `planner.temperature` to 0.5 or 0.3
4. **Not enough iterations**: Try `num_iterations=10` instead of 5

**Debug steps**:
```python
# Check if world model makes reasonable predictions
test_pose = torch.randn(1, 7).to(device)
test_action = torch.randn(1, 7).to(device)
pred_pose, pred_reward = world_model(test_pose, test_action)
print(f"Prediction shape: {pred_pose.shape}, Reward: {pred_reward.item()}")
```

### Issue: DreamerV3 training crashes with CUDA OOM

**Solutions**:
1. Reduce batch size: `training.batch_size=16` → `training.batch_size=8`
2. Reduce sequence length: `training.seq_length=50` → `training.seq_length=20`
3. Reduce model size: `model.hidden_dim=512` → `model.hidden_dim=256`
4. Use gradient checkpointing (advanced)

### Issue: Policy-guided CEM worse than random CEM

**Potential causes**:
1. **Policy trained on poor data**: Should train on successful CEM trajectories, not random data
2. **Policy-CEM std too low**: Actions can't deviate from policy. Increase `action_std`
3. **Policy model capacity**: Try larger MLP [512, 512, 256]

**Debug steps**:
```python
# Compare policy predictions with ground-truth actions
policy_actions = policy_model(test_states)
print("Policy actions mean:", policy_actions.mean(dim=0))
print("True actions mean:", true_actions.mean(dim=0))
# Should be similar
```

---

## Extended Challenges (Optional)

If you finish the main assignment and want to explore further:
  
2. **Ensemble Models**: Train multiple world models and use ensemble for planning (average predictions or use disagreement for exploration)

5. **Multi-Task**: Train a single world model on multiple LIBERO tasks with task conditioning

6. **Real Robot**: If you have access to a robot, try sim-to-real transfer with domain randomization

These are not required for the assignment but can earn extra credit and make for interesting project extensions!

---

## Dataset

Use the LIBERO dataset with spatial tasks (no no-ops):
**Hugging Face Dataset**: [gberseth/libero_spatial_sequence](https://huggingface.co/datasets/gberseth/libero_spatial_sequence)

This dataset contains:
- 7-DOF robot arm demonstrations
- RGB images (128x128 or 64x64)
- Robot end-effector poses (7-D: position + quaternion)
- Actions (7-D: delta end-effector pose)
- Task descriptions (text goals)

**Loading the dataset**:
```python
from datasets import load_dataset

dataset = load_dataset("gberseth/libero_spatial_sequence")
train_data = dataset["train"]
val_data = dataset["validation"]

# Access data
for sample in train_data:
    image = sample["image"]          # RGB image
    pose = sample["pose"]             # 7-D pose
    action = sample["action"]         # 7-D action
    reward = sample["reward"]         # Scalar reward
    text_goal = sample["text"]        # Task description
```

**Data preprocessing** (already implemented in `dreamer_model_trainer.py`):
- Images: Resize to model input size, normalize to [-1, 1]
- Poses: Normalize using dataset statistics
- Actions: Normalize using dataset statistics
- Text: Encode using T5 (optional) or character-level encoding

---

## Grading Rubric (100 points)

### Code Implementation (50 points)
- **Part 1**: SimpleWorldModel training and CEM planning (10 points)
  - Correct loss computation in `ModelTrainingWrapper` (4 pts)
  - Correct CEM implementation in `CEMPlanner` (6 pts)
  
- **Part 2**: Policy training and integration (12 points)
  - Policy training loop implementation (5 pts)
  - Policy-guided CEM initialization (4 pts)
  - Correct integration and evaluation (3 pts)
  
- **Part 3**: DreamerV3 integration (14 points)
  - Correct `_evaluate_sequences_dreamer` implementation (8 pts)
  - Understanding of RSSM state and dynamics (4 pts)
  - Proper handling of image inputs (2 pts)
  
- **Part 4**: Image-based policy (14 points)
  - Policy architecture for images (5 pts)
  - Training with RSSM states (5 pts)
  - Integration with DreamerV3 CEM (4 pts)

### Written Report (40 points)
- **Part 1 Analysis** (8 points)
  - Loss curves and experimental results (4 pts)
  - Thoughtful discussion of planning parameters (4 pts)
  
- **Part 2 Analysis** (10 points)
  - Comprehensive comparison (5 pts)
  - Insightful case studies (3 pts)
  - Clear explanation of policy+planning benefits (2 pts)
  
- **Part 3 Analysis** (10 points)
  - Training comparison and visualizations (5 pts)
  - Image reconstruction analysis (3 pts)
  - Discussion of image vs pose representations (2 pts)
  
- **Part 4 Analysis** (12 points)
  - Complete comparison table (5 pts)
  - Scenario-based analysis (4 pts)
  - Novel insights or interesting findings (3 pts)

### Code Quality and Documentation (10 points)
- Clean, readable code with comments (4 pts)
- Proper experiment tracking and logging (3 pts)
- README with clear instructions (3 pts)

### Extra Credit (up to 10 points)
- Leaderboard top 3: 5%, 3%, 2% bonus
- Additional experiments beyond requirements (up to 3 pts)
- Creative visualizations or analysis (up to 2 pts)

---

## Tips for Success

1. **Start early**: DreamerV3 training can take several hours
2. **Debug with small models**: Test your code with `hidden_dim=64` before full training
3. **Visualize often**: Plot predictions, reconstructions, and planned trajectories
4. **Use smaller horizons for debugging**: Start with `horizon=3` to verify correctness
5. **Save checkpoints**: Training crashes happen - save intermediate models
6. **Compare implementations**: Reference the provided code structure carefully
7. **Ask questions**: Use office hours if you're stuck on architecture details

**Common pitfalls**:
- Forgetting to normalize poses/actions before feeding to models
- Incorrect tensor shapes when switching between sequence and batch dimensions
- Not properly handling the RSSM state dictionary in DreamerV3
- CEM temperature too low (no exploration) or too high (too random)  
- Training policy without sufficient data (collect more demonstrations first)

---

## Additional Resources

**Papers**:
- [DreamerV3](https://arxiv.org/abs/2301.04104): Mastering Diverse Domains through World Models
- [World Models](https://arxiv.org/abs/1803.10122): Original world models paper
- [PlaNet](https://arxiv.org/abs/1811.04551): Cross-Entropy Method for planning
- [LIBERO](https://arxiv.org/abs/2305.17031): Benchmark for lifelong robot learning

**Code References**:
- Official DreamerV3 implementation: [https://github.com/danijar/dreamerv3](https://github.com/danijar/dreamerv3)
- LIBERO benchmark: [https://github.com/Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)

**Conceptual videos**:
- Pieter Abbeel's lecture on Model-Based RL
- Danijar Hafner's talk on DreamerV3

Good luck, and have fun building world models!

---

## Submitting Model to Leader Board

There is a [leaderboard](https://huggingface.co/spaces/gberseth/mila-robot-learning-course) that can be used to evalaute model. If you don't have LIBERO working or SimpleEnv you can submit your trained policies to this page for evalaution. Also, everyone must submit their model to this leaderboard for autonomous evalaution. 


Submit your model by uploading the files in your output folder for the model you want evaluated. 
```
/playground/mini-grp/outputs/2026-01-28/10-24-45$ hf upload gberseth/mini-grp .
```

Make sure these files exist in this folder
```
-rw-rw-r-- 1 gberseth gberseth   14251 Jan 28 10:25 grp_model.py
drwxrwxr-x 2 gberseth gberseth    4096 Jan 28 10:24 .hydra
-rw-rw-r-- 1 gberseth gberseth 1112018 Jan 28 10:25 miniGRP.pth
```

The evalaution code will look for your .pth file, which needs you grp_model.py file to understand how to use your model, and the .hydra folder to load your model configuration.
