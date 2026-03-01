# Tutorial: Vision-Language-Action Models — From Absolute Zero

> **Prerequisites**: Read the [RL tutorial](tutorial-rl-robotics-from-scratch.md) first, OR just know basic Python. This tutorial is self-contained for the VLA concepts.

This tutorial explains how a robot can **see a camera image**, **read a text instruction** like "pick up the red block," and **output motor commands** to do it. This is a **Vision-Language-Action (VLA)** model — the architecture behind frontier robotics at Google (RT-2), Stanford (OpenVLA), and Amazon.

---

## Part 1: The Big Picture — Three Types of Robot Learning

Before diving into VLA, let's understand where it sits:

### Approach 1: Reinforcement Learning (RL)

The robot learns by trial and error in simulation. No human demonstrations needed.

```
Robot tries random actions → gets reward → improves → repeat 1M times
```

- **Input**: Robot's own body state (joint angles, velocities)
- **Output**: Motor torques
- **Learns from**: Reward signal (a score)
- **Used for**: Locomotion (walking), simple tasks
- **Limitation**: Can't understand language, can't use camera images efficiently

### Approach 2: Behavior Cloning (BC)

A human demonstrates the task. The robot watches and copies.

```
Human demos → Dataset of (observation, action) pairs → Train neural network to copy
```

- **Input**: Robot state (joint angles, object positions)
- **Output**: Actions (move end-effector by [dx, dy, dz, gripper])
- **Learns from**: Expert demonstrations
- **Used for**: Manipulation (pick, place, fold)
- **Limitation**: Can only do the ONE task it was trained on. Can't generalize to new instructions.

### Approach 3: VLA (Vision-Language-Action) — The Frontier

Combines vision (cameras) + language (instructions) + action prediction in one model. A single model can understand **any task described in words** and execute it.

```
Camera image + "pick up the red block" → Robot motor commands
Camera image + "open the drawer" → Different motor commands
```

- **Input**: Camera image + text instruction + robot state
- **Output**: Actions
- **Learns from**: Demonstrations labeled with language
- **Used for**: Multi-task manipulation, zero-shot generalization
- **This is what Google RT-2, OpenVLA, and Amazon Robotics are building**

### Why VLA Matters

| Feature | RL (PPO) | BC | VLA |
|---------|----------|----|-----|
| Needs demonstrations? | No | Yes | Yes |
| Understands language? | No | No | **Yes** |
| Uses camera? | Usually no | Sometimes | **Yes** |
| Multi-task? | No (1 reward) | No (1 task) | **Yes (any instruction)** |
| State of the art? | For locomotion | For single tasks | **For general manipulation** |

---

## Part 2: What the VLA Sees, Reads, and Does

### Input 1: Camera Image

A 64×64 RGB image from the robot's wrist camera (or an overhead camera):

```
image shape: (3, 64, 64)
  3 = Red, Green, Blue color channels
  64 × 64 = pixel resolution

What it shows: the table, the red block, the green target, the robot arm
```

In our code:
```python
image = torch.randn(1, 3, 64, 64)  # Batch of 1 image, 3 channels, 64×64
```

In a real system, this comes from an actual camera. In our project, we generate dummy images (random noise) because the important learning happens through the state and language.

### Input 2: Language Instruction

A natural language sentence describing what to do:

```python
instructions = [
    "pick up the red block and place it at the green target",
    "grasp the red cube and move it to the target location",
    "lift the red block then set it on the target",
]
```

The model learns that all of these mean the same thing. In a full-scale system, different instructions could produce completely different behaviors:
- "pick up the red block" → reach, grasp, lift
- "push the red block to the left" → reach, push sideways
- "stack the blue cube on the red cube" → different sequence entirely

### Input 3: Robot State

The robot's own body knowledge — same as in RL/BC:

```python
state = [
    joint_positions...,    # 7 joint angles of the arm
    ee_position...,        # end-effector x, y, z
    gripper_position...,   # how open is the gripper
    target_position...,    # where the goal is
    object_position...,    # where the block is
]
# Total: ~20 numbers
```

### Output: Actions

Motor commands for the robot arm:

```python
action = [dx, dy, dz, gripper]
# dx, dy, dz = move end-effector by this much in x, y, z
# gripper = open (1.0) or close (-1.0) the gripper
# All values in [-1, 1]
```

### The Complete Flow

```
Camera image (64×64×3)──────────────►┐
                                      │
Language: "pick up the red block"───►├──► VLA Model ──► [dx, dy, dz, gripper]
                                      │
Robot state (20 numbers)────────────►┘
```

One model, three inputs, one output. The magic is in how these are fused.

---

## Part 3: How the VLA Model Works — Architecture

The VLA model has four main components, each processing a different type of input, then fusing them together. Think of it as a team of specialists collaborating:

### Component 1: Vision Encoder (ViT) — "The Eyes"

**Purpose**: Convert a raw pixel image into a set of meaningful feature vectors.

**How**: Split the image into patches, then process them with a transformer.

```
Step 1: Split image into patches
  64×64 image ÷ 8×8 patch size = 8×8 = 64 patches
  Each patch is 8×8×3 = 192 raw numbers

Step 2: Project each patch to an embedding
  192 numbers → 128-dim vector (via a learned linear projection)
  Now we have 64 vectors, each 128-dim

Step 3: Add position information
  Patch at top-left should mean something different from patch at bottom-right
  Add a learned "position embedding" to each patch vector

Step 4: Add a CLS token
  Prepend a special token that will aggregate information from ALL patches
  Now we have 65 vectors (1 CLS + 64 patches)

Step 5: Pass through transformer layers
  Self-attention lets each patch "look at" every other patch
  After 2-4 layers: each vector now knows about the full image
  CLS token captures a global summary
```

In code:

```python
class PatchEmbedding(nn.Module):
    def __init__(self, image_size=64, patch_size=8, embed_dim=128):
        self.num_patches = (64 // 8) ** 2  # = 64 patches
        self.proj = nn.Conv2d(3, 128, kernel_size=8, stride=8)  # Projects each patch
        self.pos_embed = nn.Parameter(torch.randn(1, 64, 128))  # Learned positions

    def forward(self, image):
        x = self.proj(image)           # (B, 128, 8, 8)
        x = x.flatten(2).transpose(1,2)  # (B, 64, 128) — 64 patches, 128 dim each
        x = x + self.pos_embed        # Add position info
        return x

class VisionEncoder(nn.Module):
    def __init__(self):
        self.patch_embed = PatchEmbedding()
        self.cls_token = nn.Parameter(torch.randn(1, 1, 128))  # Global summary token
        self.encoder = TransformerEncoder(num_layers=2)

    def forward(self, images):
        patches = self.patch_embed(images)       # (B, 64, 128)
        cls = self.cls_token.expand(B, -1, -1)   # (B, 1, 128)
        tokens = torch.cat([cls, patches], dim=1) # (B, 65, 128)
        return self.encoder(tokens)               # (B, 65, 128) — enriched
```

**Output**: 65 vectors, each 128-dimensional. The first (CLS) is a summary. The rest represent spatial regions of the image.

### Component 2: Language Encoder — "The Ears"

**Purpose**: Convert a text instruction into feature vectors that capture meaning.

**How**: Character-level tokenization → embedding → transformer.

```
Step 1: Tokenize the text
  "pick up the red block" → [16, 9, 3, 11, 0, 21, 16, ...]
  Each character → a number (a=1, b=2, ... space=0)
  Pad to fixed length (64 characters)

Step 2: Embed each character
  Each number → a 128-dim learned vector
  64 characters → 64 vectors of dim 128

Step 3: Add position embeddings
  Same idea as patches — position matters in text

Step 4: Pass through transformer layers
  Self-attention lets each character "see" the full sentence
  After 2 layers: vectors capture word-level and phrase-level meaning
```

In code:

```python
class SimpleLanguageEncoder(nn.Module):
    def __init__(self, vocab_size=1000, max_len=64, embed_dim=128):
        self.token_embed = nn.Embedding(vocab_size, 128)  # Char → vector
        self.pos_embed = nn.Parameter(torch.randn(1, 64, 128))
        self.encoder = TransformerEncoder(num_layers=2)

    def tokenize(self, texts):
        # "pick up" → [16, 9, 3, 11, 0, 21, 16, 0, 0, ...]  (padded to 64)
        char_to_idx = {' ':0, 'a':1, 'b':2, 'c':3, ...}
        tokens = [[char_to_idx[c] for c in text] for text in texts]
        return pad_to_length(tokens, 64)

    def forward(self, token_ids):
        x = self.token_embed(token_ids)  # (B, 64, 128)
        x = x + self.pos_embed           # Add position
        return self.encoder(x)            # (B, 64, 128)
```

**Output**: 64 vectors, each 128-dimensional. Together they represent the meaning of the instruction.

**Note**: In production (RT-2, OpenVLA), this would be a pretrained large language model like PaLM or LLaMA. Our character-level encoder is a lightweight demo that demonstrates the architecture.

### Component 3: State Projection — "The Body Sense"

**Purpose**: Convert robot state numbers into the same 128-dim space as vision and language.

```python
class StateProjection(nn.Module):
    def __init__(self, state_dim=20, embed_dim=128):
        self.proj = nn.Sequential(
            nn.Linear(20, 128),   # 20 state numbers → 128-dim
            nn.GELU(),
            nn.Linear(128, 128),  # Refine
        )
    
    def forward(self, state):
        return self.proj(state).unsqueeze(1)  # (B, 1, 128) — a single token
```

**Output**: 1 vector of dim 128. The robot's body state compressed into the same space.

### Component 4: Fusion Transformer — "The Brain"

**Purpose**: Combine vision + language + state and make a decision.

This is the key insight: all three inputs are now vectors in the same 128-dimensional space. We concatenate them into one long sequence and let a transformer attend across everything:

```
Visual tokens:   65 vectors × 128-dim  (what the robot sees)
Language tokens: 64 vectors × 128-dim  (what the human said)
State token:      1 vector  × 128-dim  (what the robot feels)
───────────────────────────────────────
Total:          130 tokens  × 128-dim  → Fusion Transformer
```

```python
# In the VLA forward pass:
all_tokens = torch.cat([
    visual_tokens,   # (B, 65, 128)
    language_tokens,  # (B, 64, 128)
    state_token,      # (B, 1, 128)
], dim=1)             # (B, 130, 128)

fused = self.fusion_transformer(all_tokens)  # (B, 130, 128)
```

Inside the fusion transformer, **self-attention** lets every token attend to every other token:
- Visual tokens learn which image regions match the mentioned "red block"
- Language tokens learn which words correspond to visible objects
- The state token grounds everything in the robot's current configuration

### Component 5: Action Head — "The Hands"

**Purpose**: Convert the fused representation into motor commands.

We take the CLS token (the first visual token, which now contains information from ALL inputs) and pass it through an MLP:

```python
self.action_head = nn.Sequential(
    nn.Linear(128, 128),     # 128 → 128
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(128, 64),      # 128 → 64
    nn.GELU(),
    nn.Linear(64, 4),        # 64 → 4 (dx, dy, dz, gripper)
    nn.Tanh(),               # Squash to [-1, 1]
)

cls_token = fused[:, 0]           # (B, 128) — the summary token
predicted_action = self.action_head(cls_token)  # (B, 4)
```

### Complete Architecture Diagram

```
Camera Image (3×64×64)                "pick up the red block"          Robot State (20 floats)
      │                                        │                              │
      ▼                                        ▼                              ▼
┌─────────────┐                    ┌───────────────────┐            ┌─────────────────┐
│ Patch Embed │                    │ Char Tokenization │            │ Linear + GELU   │
│ Conv2d 8×8  │                    │ + Embedding       │            │ 20 → 128        │
└──────┬──────┘                    └────────┬──────────┘            └────────┬────────┘
       │ 64 patches                         │ 64 char tokens                │ 1 token
       ▼                                    ▼                               │
┌──────────────┐                  ┌──────────────┐                          │
│ + CLS token  │                  │  Transformer │                          │
│ + Pos Embed  │                  │  (2 layers)  │                          │
│  Transformer │                  └──────┬───────┘                          │
│  (2 layers)  │                         │                                  │
└──────┬───────┘                         │                                  │
       │ 65 visual tokens                │ 64 lang tokens                   │
       └────────────────┬────────────────┴──────────────────┬───────────────┘
                        │                                   │
                        ▼                                   │
              ┌───────────────────┐                         │
              │  Concatenate all  │◄────────────────────────┘
              │  130 tokens ×128  │
              └────────┬──────────┘
                       │
                       ▼
              ┌───────────────────┐
              │ Fusion Transformer│
              │   (2 layers)      │
              └────────┬──────────┘
                       │
                       ▼
              CLS token (128-dim)
                       │
                       ▼
              ┌───────────────────┐
              │  Action Head MLP  │
              │  128→128→64→4     │
              │  + Tanh           │
              └────────┬──────────┘
                       │
                       ▼
              [dx, dy, dz, gripper]
              Action output (4-dim)
```

### Model Size

```
Vision Encoder:   ~530K parameters
Language Encoder: ~400K parameters  
State Projection:  ~20K parameters
Fusion Transformer:~330K parameters
Action Head:       ~25K parameters
────────────────────────────────────
Total:            ~1.4M parameters   (5.5 MB on disk)
```

For comparison:
- RT-2 (Google): **55 billion** parameters
- OpenVLA (Stanford): **7 billion** parameters
- Our SimScaleAI VLA: **1.4 million** parameters (a working demo of the same architecture)

---

## Part 4: The Training Data — From Demonstrations to Datasets

### Where Does Training Data Come From?

Unlike RL (which generates data by trial and error), VLA uses **supervised learning** from expert demonstrations. The pipeline:

```
Step 1: Run a scripted expert policy in simulation
Step 2: Record everything it does → HDF5 file
Step 3: Label each episode with a language instruction
Step 4: Train the VLA to predict the expert's actions
```

### Step 1: Data Generation

A scripted policy (hand-coded rules) performs the pick-and-place task many times:

```python
# Generate 200 expert demonstrations
# simscaleai/datagen/generator.py

for episode in range(200):
    obs, info = env.reset()              # Random block + target positions
    
    for step in range(300):
        action = scripted_policy(obs)     # Expert decides what to do
        
        # RECORD everything:
        save(obs["joint_pos"])            # 7 joint angles
        save(obs["ee_pos"])              # end-effector position  
        save(obs["target_pos"])          # target position
        save(obs["object_pos"])          # block position
        save(action)                     # [dx, dy, dz, gripper]
        save(reward)                     # score
        
        obs, reward, terminated, truncated, info = env.step(action)
```

This produces an HDF5 file (~2 MB) with 200 episodes × ~190 steps = ~38,000 data points.

### Step 2: The HDF5 File Structure

```
data/pick_place.h5
├── episode_0/
│   ├── observations/
│   │   ├── joint_pos      (190, 7)    ← 190 steps, 7 joints
│   │   ├── ee_pos         (190, 3)    ← end-effector x,y,z
│   │   ├── target_pos     (190, 3)    ← target location
│   │   └── object_pos     (190, 3)    ← block location
│   ├── actions            (190, 4)    ← what the expert did
│   └── rewards            (190,)      ← score per step
├── episode_1/
│   ├── observations/...
│   ├── actions...
│   └── rewards...
├── ... (200 episodes total)
```

### Step 3: Adding Language

During VLA training, each data point is paired with a randomly chosen language instruction:

```python
PICK_PLACE_INSTRUCTIONS = [
    "pick up the red block and place it at the green target",
    "grasp the red cube and move it to the target location",
    "pick the block and put it on the target",
    "move the red object to the green marker",
    "grab the cube and place it at the goal",
    "lift the red block then set it on the target",
    "pick up the object and transport it to the target zone",
    "grasp and relocate the red block to the green spot",
]

# Every time we sample a training example:
instruction = random.choice(PICK_PLACE_INSTRUCTIONS)
```

**Why random instructions?** The model learns that "pick up the red block" and "grasp the red cube" mean the same thing — it learns **semantic equivalence**. In a full system with multiple tasks, different instructions would map to different behaviors.

### Step 4: The VLA Dataset

The dataset class loads HDF5 data and packages it for training:

```python
class VLAPickPlaceDataset(Dataset):
    def __init__(self, h5_path):
        with h5py.File(h5_path) as f:
            for episode in f.keys():
                states = f[episode]["observations"][...]  # All obs concatenated
                actions = f[episode]["actions"][...]
                for t in range(len(actions)):
                    self.samples.append({
                        "state": states[t],     # 20 numbers (joint_pos + ee + target + obj)
                        "action": actions[t],    # 4 numbers (dx, dy, dz, gripper)
                    })
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "image": torch.randn(3, 64, 64),       # Dummy image (placeholder)
            "state": torch.from_numpy(sample["state"]),
            "action": torch.from_numpy(sample["action"]),
            "language": random.choice(INSTRUCTIONS),  # Random instruction
        }
```

**Note about dummy images**: In our project, we use random noise images because we're demonstrating the architecture. In production (RT-2, OpenVLA), real camera images are used and the vision encoder is a pretrained CLIP/SigLIP model. The state and language pathways are fully functional.

---

## Part 5: Training — How the VLA Learns

### The Difference From RL

| Aspect | RL (PPO) | VLA Training |
|--------|----------|-------------|
| **Data source** | Generated on-the-fly by the agent | Pre-collected demonstrations |
| **Objective** | Maximize reward | Minimize action prediction error |
| **Data reuse** | Used once, then discarded | Reused many times (epochs) |
| **Training type** | On-policy (data from current brain) | Off-policy (data from expert) |
| **Supervision** | Reward signal (sparse, noisy) | Direct action labels (dense, clean) |
| **Algorithm** | PPO (policy gradient) | Standard gradient descent (MSE loss) |

### The Training Loop

VLA training is simple supervised learning — predict the expert's action, compute the error, update weights:

```python
def train_vla(dataset_path, max_steps=2000):
    dataset = VLAPickPlaceDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = VisionLanguageAction(
        image_size=64, patch_size=8, embed_dim=128,
        num_heads=4, num_layers=2,
        action_dim=4, state_dim=20,
    )  # 1.4M parameters
    
    optimizer = AdamW(model.parameters(), lr=1e-4)
    
    for step in range(2000):
        batch = next(loader)
        
        # Forward pass: image + language + state → predicted action
        result = model(batch)
        
        # Loss: how wrong were the predicted actions?
        loss = MSE(result["predicted_actions"], batch["actions"])
        # loss = mean( (predicted - expert)² )
        # If predicted = [0.3, -0.1, 0.5, -0.8]
        # and expert   = [0.2, -0.2, 0.4, -1.0]
        # then error² per dim = [0.01, 0.01, 0.01, 0.04]
        # loss = mean = 0.0175
        
        # Backward pass: compute gradients
        loss.backward()
        
        # Update weights
        optimizer.step()
```

### What the Model Learns Over Time

```
Step    0: loss = 0.300  (random predictions — actions are all wrong)
Step  200: loss = 0.180  (learning the general direction of movement)
Step  500: loss = 0.100  (learning the reach → grasp → lift sequence)
Step 1000: loss = 0.070  (fine-tuning timing and gripper control)
Step 2000: loss = 0.064  (converged — close to expert, but not perfect)
```

The loss measures average squared error per action dimension. A loss of 0.064 means the average prediction is off by √0.064 ≈ 0.25 per dimension (on a [-1, 1] scale). Not perfect, but captures the general behavior.

### Why VLA is Harder Than BC

VLA and BC solve the same problem (predict expert actions), but VLA has extra complexity:

```
BC:  state(20) → MLP → action(4)
     Simple. Fast. But can't see images or understand language.

VLA: image(3×64×64) + language(64 chars) + state(20) → Transformers → action(4)
     More expressive. Handles multi-modal input. But needs more data and compute.
```

In our evaluation:
- **BC**: reward 55.6 — learned the motion pattern
- **VLA**: reward -46.7 — struggles because (a) dummy images, (b) only 38K samples, (c) 1.4M params is tiny

With real images and 100× more data, VLA dramatically outperforms BC — that's why Google invested billions in RT-2.

---

## Part 6: The Forward Pass — Step by Step

Let me trace one complete forward pass with concrete numbers.

### Input

```python
image = camera.capture()     # Shape: (3, 64, 64) — a photo of the table
language = "pick up the red block and place it at the green target"
state = [0.1, -0.3, 0.5, ..., 0.2]  # 20 numbers — current joint angles, positions
```

### Step 1: Vision Encoding

```python
# Split image into 64 patches (8×8 grid)
patches = patch_embed(image)           # (64, 128) — 64 patches, 128-dim each

# Prepend CLS token
tokens = cat([cls_token, patches])     # (65, 128)

# Self-attention (2 layers)
# Each patch can attend to every other patch
# Patch #7 (top-right) might attend strongly to patch #35 (center)
# because both contain the "red block"
visual_tokens = transformer(tokens)     # (65, 128)
```

### Step 2: Language Encoding

```python
# Tokenize: each character → a number
"pick up the red block and place..." → [16,9,3,11,0,21,16,0,20,8,5,0,...]
# Pad to 64 characters

# Embed + transform
char_embeddings = embed(token_ids)     # (64, 128)
lang_tokens = transformer(char_embeddings)  # (64, 128)
```

### Step 3: State Projection

```python
# Simple MLP: 20 → 128
state_token = linear(state)            # (1, 128)
```

### Step 4: Fusion

```python
# Concatenate everything
all_tokens = cat([
    visual_tokens,    # (65, 128) — what the robot sees
    lang_tokens,      # (64, 128) — what the human said
    state_token,      # (1, 128)  — what the robot feels
])                    # (130, 128) total

# Fusion transformer — this is where the magic happens
# Visual token for "red block region" attends to language token for "red block"
# Language token "place" attends to visual token for "green target region"
# State token attends to everything for context
fused = fusion_transformer(all_tokens)  # (130, 128)
```

### Step 5: Action Prediction

```python
# Take the CLS token — it's absorbed info from everything
cls = fused[0]                         # (128,)

# Pass through action head MLP
action = action_head(cls)              # (4,)
# action = [0.15, -0.22, 0.08, 0.95]
#           ↑      ↑      ↑      ↑
#         move   move   move    open
#         right  back    up    gripper
```

### Step 6: Loss (During Training)

```python
# Expert action for this timestep was: [0.12, -0.20, 0.10, 1.00]
# Our prediction was:                  [0.15, -0.22, 0.08, 0.95]

loss = MSE([0.15,-0.22,0.08,0.95], [0.12,-0.20,0.10,0.95])
     = mean([0.0009, 0.0004, 0.0004, 0.0025])
     = 0.00105

# This is good! Average error per dimension: √0.001 ≈ 0.03
# The prediction is very close to the expert's action
```

---

## Part 7: Self-Attention — The Key Mechanism

Self-attention is the thing that makes transformers (and VLA) work. Let me explain it intuitively.

### The Problem

We have 130 tokens. Each one knows only about itself:
- Visual token #23 knows it's a "red blob in the center"
- Language token #5 knows it's the character "r" after "the "
- But they don't know about each other

### The Solution: "Who should I pay attention to?"

Each token creates three vectors:
- **Query** (Q): "What am I looking for?"
- **Key** (K): "What do I have to offer?"
- **Value** (V): "What information do I carry?"

Then every token computes attention scores with every other token:

```
Attention(token_i, token_j) = softmax(Q_i · K_j / √dim)

High score = "these two tokens are relevant to each other"
Low score  = "these tokens have nothing to do with each other"
```

### Concrete Example

```
Token: visual patch showing "red block"
  Query: "I contain a red object — who mentions 'red'?"
  
  Attends strongly to:
    ✓ Language token "r" in "red"     (score 0.31)
    ✓ Language token "e" in "red"     (score 0.28)
    ✓ Language token "b" in "block"   (score 0.15)
  
  Attends weakly to:
    ✗ Visual token showing empty table (score 0.01)
    ✗ Language token "a" in "and"      (score 0.02)
```

After attention, the "red block" visual token now **contains information about the language instruction**. It "knows" that the user wants to interact with the red block.

### Why This Is Powerful

Without self-attention:
- Vision encoder sees a red block and a green target but doesn't know which matters
- Language encoder understands "pick up" but doesn't know what to pick up
- State encoder knows joint angles but doesn't know the goal

With self-attention across all modalities:
- Vision token for "red block" binds to language token for "red block" → "THIS is the object to pick up"
- Vision token for "green target" binds to language token for "place it at" → "THIS is where it goes"
- State token binds to both → "I need to move FROM my current position TO the red block THEN to the green target"

This cross-modal binding is what makes VLA models fundamentally more capable than separate vision/language/action systems.

---

## Part 8: Inference — Using the Trained Model

After training, deployment is simple:

```python
# Load the trained model
model = VisionLanguageAction(...)
model.load_state_dict(torch.load("checkpoints/pick_place_vla/model.pt"))
model.eval()

# Robot control loop (runs at 50 Hz on real hardware)
instruction = "pick up the red block and place it at the green target"

while not done:
    # Capture camera image
    image = camera.get_frame()                # (3, 64, 64)
    
    # Read robot state
    state = robot.get_joint_positions()        # (20,)
    
    # Predict action — ONE forward pass, < 5ms
    obs = {"image": image, "state": state}
    action = model.predict(obs, language=instruction)  # (4,)
    
    # Execute on robot
    robot.move(action)                         # Send to motors
```

### What Changes Between Training and Inference

| Aspect | Training | Inference |
|--------|----------|-----------|
| Batch size | 32 (parallel) | 1 (one robot) |
| Gradient computation | Yes (backward pass) | No (forward only) |
| Dropout | Active (regularization) | Disabled (deterministic) |
| Data source | HDF5 file | Live camera + sensors |
| Speed | ~100 ms / batch | < 5 ms / frame |
| Language | Random from template list | Fixed instruction from human |

---

## Part 9: How VLA Compares to BC — In Our Project

Our project trains both BC and VLA on the same pick-and-place task. Here's how they compare:

### Architecture Comparison

```
BC (Behavior Cloning):
  state(20) → Linear(256) → ReLU → Linear(256) → ReLU → Linear(256) → Linear(4)
  Parameters: ~200K
  Inputs: state only
  Cannot: see images, understand language

VLA (Vision-Language-Action):
  image(3×64×64) → ViT(2 layers) ──┐
  language(64 chars) → Transformer ─┼─→ Fusion(2 layers) → MLP → action(4)
  state(20) → Linear ──────────────┘
  Parameters: ~1.4M
  Inputs: image + language + state
  Can: see images, understand language, multi-task (in theory)
```

### Training Comparison

| Metric | BC | VLA |
|--------|-----|-----|
| Training data | 200 eps, 37.8K steps | Same data + language labels |
| Training steps | 3,000 | 2,000 |
| Final loss | 0.010 | 0.064 |
| Training time | ~30 seconds | ~2 minutes |
| Parameters | 200K | 1.4M |

### Evaluation Results (50 episodes)

| Policy | Reward | Success |
|--------|--------|---------|
| Scripted expert | 145.1 ± 148.9 | 20% |
| BC | 55.6 ± 76.0 | 0% |
| VLA | -46.7 ± 50.0 | 0% |

**Why does VLA perform worse than BC here?**

1. **Dummy images**: Our VLA gets random noise instead of real camera images — the vision encoder learns nothing useful. It's carrying 530K parameters of dead weight.
2. **Too little data**: 38K samples is barely enough for a 200K BC model, let alone a 1.4M VLA.
3. **Char-level tokenization**: Our language encoder processes individual characters. Production VLAs use pretrained LLMs that already understand language.
4. **This is a demo**: The goal is to demonstrate the architecture works. With real images and 100× data, VLA outperforms BC significantly — that's the entire point of RT-2/OpenVLA.

### When VLA Wins (At Scale)

```
Our scale:     38K samples,  1.4M params, dummy images → BC wins
Google scale: 130K episodes, 55B params, real cameras → VLA dominates

Key insight: VLA is a SCALING bet.
  - More data → VLA learns better visual grounding
  - More parameters → VLA captures more complex relationships
  - Pretrained components → VLA starts with vision + language understanding
  - Multiple tasks → VLA shares knowledge, BC can't
```

---

## Part 10: The Complete Data Flow — VLA vs RL

### VLA Data Flow (Supervised Learning)

```
OFFLINE (runs once):
  Scripted policy plays pick-and-place 200 times
      │
      ▼
  Record (obs, actions) → data/pick_place.h5  (HDF5 on disk, reusable)
      │
      ▼
  VLAPickPlaceDataset loads → adds random language instructions
      │
      ▼
  DataLoader samples batches of 32
      │
      ▼
  ┌──── TRAINING LOOP (2,000 steps) ────────────────────────┐
  │ For each batch:                                          │
  │   image + language + state → VLA model → predicted_action│
  │   loss = MSE(predicted, expert_action)                   │
  │   loss.backward() → optimizer.step()                     │
  │   (Same batch can be reused — data is always valid)      │
  └──────────────────────────────────────────────────────────┘
      │
      ▼
  Save model → checkpoints/pick_place_vla/model.pt  (5.5 MB)
      │
      ▼
  Deploy: image + "pick up the red block" → action in <5ms
```

### RL Data Flow (For Comparison)

```
ONLINE (runs continuously):
  ┌──── TRAINING LOOP (1M steps) ───────────────────────────┐
  │ Agent picks action → env simulates → gets reward         │
  │ Store in RAM buffer (2,048 steps)                        │
  │ Compute advantages (GAE)                                 │
  │ PPO update (10 epochs)                                   │
  │ DISCARD buffer (data is stale — policy changed)          │
  │ Repeat with updated brain                                │
  └──────────────────────────────────────────────────────────┘
```

### Key Difference

| | VLA | RL |
|-----|-----|-----|
| **Data collection** | Once, offline | Continuously during training |
| **Data storage** | HDF5 on disk (persists) | RAM buffer (discarded) |
| **Data reuse** | Many times (shuffle + re-epoch) | Once (then thrown away) |
| **Label source** | Expert demonstrations | Reward function |
| **Language** | Yes — drives behavior | No |
| **Camera** | Yes — visual input | Usually state only |

---

## Part 11: The Bigger Picture — Why This Matters for Amazon Robotics

### The Industry Trajectory

```
2019: BC on single tasks (basic pick and place)
      ↓
2022: RT-1 (Google) — one model, 700 tasks, real robot
      ↓
2023: RT-2 (Google) — VLA with 55B params, language generalization
      ↓
2024: OpenVLA (Stanford) — open-source 7B VLA, fine-tunable
      ↓
2025: Amazon Robotics — VLA for warehouse manipulation at scale
      ↓
2026: Foundation robot models — one model for any manipulation task
```

### What Amazon Needs (and What We Demonstrate)

| Capability | Industry Need | SimScaleAI Coverage |
|-----------|--------------|-------------------|
| Multi-modal perception | Camera + depth + state → unified representation | ViT vision encoder + state projection |
| Language conditioning | "Pick the shampoo bottle from aisle 3" | Char-level encoder + fusion transformer |
| Multi-task learning | One model for pick, place, sort, pack | Architecture supports it (need more tasks/data) |
| Sim-to-real transfer | Train in sim, deploy on real robots | Domain randomization + transfer benchmark |
| Data pipeline | Generate millions of demonstrations | Parallel data generation (8 workers, 72 ep/s) |
| Safety constraints | Don't crush objects, respect joint limits | Action clamping (Tanh to [-1,1]) |

### Our VLA in Context

We built a **1.4M parameter proof-of-concept** that demonstrates every component of the RT-2/OpenVLA architecture:
- ViT vision encoder (functionally equivalent to CLIP/SigLIP, just smaller)
- Language encoder (char-level instead of LLM, same interface)
- Cross-modal fusion transformer (identical architecture)
- MLP action head with Tanh clamping (standard)

To make it production-grade:
1. Replace dummy images with real camera renders → vision encoder learns
2. Replace char tokenizer with pretrained LLM (LLaMA-7B) → language encoder understands
3. Scale to millions of demonstrations across dozens of tasks → multi-task generalization
4. Add action chunking (predict 8 future actions, not 1) → smoother execution

The architecture is the same. The difference is scale.

---

## Glossary (VLA-Specific Terms)

| Term | Definition |
|------|-----------|
| **VLA** | Vision-Language-Action model — takes image + text → outputs robot actions |
| **ViT** | Vision Transformer — processes images by splitting into patches |
| **Patch** | A small square region of an image (e.g., 8×8 pixels) |
| **CLS token** | A special token that aggregates information from all patches |
| **Embedding** | Converting raw data (pixels, characters) into a dense vector |
| **Tokenization** | Breaking text into discrete units (characters, words, subwords) |
| **Self-attention** | Mechanism where each token computes relevance to every other token |
| **Cross-modal fusion** | Combining information from different modalities (vision + language) |
| **Action head** | Final MLP that converts representations into motor commands |
| **Action chunking** | Predicting multiple future actions at once (smoother control) |
| **MSE loss** | Mean Squared Error — average of (prediction − target)² |
| **HDF5** | Binary file format for storing large arrays (observations, actions) |
| **Off-policy** | Training on data from a different policy (the expert, not the current model) |
| **Behavior Cloning** | Simpler predecessor to VLA — state → action, no vision/language |
| **RT-2** | Google's 55B parameter VLA — the first VLA to work on real robots at scale |
| **OpenVLA** | Stanford's open-source 7B VLA — fine-tunable for custom tasks |
| **Foundation model** | One large model trained on diverse data, adaptable to many tasks |
| **CLIP/SigLIP** | Pretrained vision models that understand images + text jointly |
| **Domain randomization** | Varying simulation parameters so policies transfer to the real world |
| **Pretrained** | A model already trained on large generic data, then fine-tuned for your task |
