import wandb
from wandb.util import np
import diffuser.utils as utils
import pdb
import torch
from datetime import datetime
import os
import numpy as np

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'pose-trajectory'
    config: str = 'config.cfm_pose'

args = Parser().parse_args('diffusion')

#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

import pandas as pd

def load_npz_to_dataframe(filename: str, include_targets: bool = True):
    ''' Reconstructs a DataFrame from a saved npz file '''
    data = np.load(filename)
    df = pd.DataFrame(data['features'], columns=data['feature_names'])
    df['targets'] = [t for t in data['targets']]
    
    return df

poseTraject_df = load_npz_to_dataframe("/home/bubble/PhysicalAI_flowmatching/data/pose_traject_dataset.npz")

# 1. One-hot encode : This creates the 3D one-hot vector part = {Accel, Const, Decel} [2]
part_onehot = pd.get_dummies(poseTraject_df['part_enum'].astype(int), prefix='part').values

# 2. Assemble the final Context Vector C (9 scalars total) [2]
# C = [s_goal_x, s_goal_y, v_const, accel, part_0, part_1, part_2, q_init_x, q_init_y]
C = np.concatenate([
    poseTraject_df[['s_goal_x', 's_goal_y', 'v_const', 'accel']].values,
    part_onehot,
    poseTraject_df[['q_init_x', 'q_init_y']].values
], axis=1)

# 3. Process the Action Vector A (2D trajectories) [1]
# The 'targets' column contains arrays of shape (50, 7) [5]
# We stack them and keep only (x, y) which are indices 0 and 1 [6]
targets_raw = np.stack(poseTraject_df['targets'].values) 
A_2d = targets_raw[:, :, 0:2] # Shape: (batch_size, 50 steps, 2 variables)

# 4. Prepare PyTorch Tensors for Flow Matching
context_tensor = torch.tensor(C, dtype=torch.float32)  # (N, 9)

# Pad trajectory from 50 to 64 steps (power of 2 required by UNet down/upsample)
HORIZON = 64
A_padded = np.pad(A_2d, ((0,0), (0, HORIZON - A_2d.shape[1]), (0,0)), mode='edge')
A_for_model = torch.tensor(A_padded, dtype=torch.float32)  # (N, 64, 2)

print(f"Context Vector C shape: {context_tensor.shape}")  # (N, 9)
print(f"Action Vector A shape: {A_for_model.shape}")       # (N, 64, 2)

#-----------------------------------------------------------------------------#
#------------------------------- dataset wrapper -----------------------------#
#-----------------------------------------------------------------------------#

from torch.utils.data import Dataset

class PoseTrajectoryDataset(Dataset):
    """Wraps pose trajectory data for T-CFM training."""
    def __init__(self, actions, contexts):
        self.actions = actions    # (N, 50, 2)
        self.contexts = contexts  # (N, 9)
        self.observation_dim = actions.shape[-1]  # 2
        self.packed_dim = 0

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        action = self.actions[idx]       # (50, 2)
        context = self.contexts[idx]     # (9,)
        cond = (np.array([]), np.array([]))
        return action, context, cond

    def collate_fn(self, batch):
        actions = torch.stack([b[0] for b in batch])
        contexts = torch.stack([b[1] for b in batch])
        conds = [b[2] for b in batch]
        global_cond = {'hideouts': contexts}
        return actions, global_cond, conds

    def collate_fn_repeat(self, batch, num_samples):
        actions = torch.stack([b[0] for b in batch]).repeat_interleave(num_samples, dim=0)
        contexts = torch.stack([b[1] for b in batch]).repeat_interleave(num_samples, dim=0)
        conds = [b[2] for b in batch for _ in range(num_samples)]
        global_cond = {'hideouts': contexts}
        return actions, global_cond, conds

    def unnormalize(self, data):
        return data  # No normalization applied for now

# Train/test split (90/10)
N = len(context_tensor)
split = int(0.9 * N)
indices = torch.randperm(N)
train_idx, test_idx = indices[:split], indices[split:]

dataset = PoseTrajectoryDataset(A_for_model[train_idx], context_tensor[train_idx])
test_dataset = PoseTrajectoryDataset(A_for_model[test_idx], context_tensor[test_idx])

observation_dim = 2   # x, y pose
action_dim = 0
context_dim = context_tensor.shape[1]  # 9

print(f"Train: {len(dataset)}, Test: {len(test_dataset)}")
print(f"observation_dim={observation_dim}, context_dim={context_dim}")

#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#

model_config = utils.Config(
    args.model,
    savepath=(args.savepath, 'model_config.pkl'),
    horizon=HORIZON,                                # 64 (padded from 50, power of 2 for UNet)
    transition_dim=observation_dim + action_dim,   # 2 (x, y)
    lstm_in_dim=None,                              # no sequential history encoder
    lstm_out_dim=None,                             # no LSTM output
    global_cond_dim=context_dim,                   # 9 (context vector C)
    cond_dim=observation_dim,
    dim_mults=args.dim_mults,
    device=args.device
)

diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, 'diffusion_config.pkl'),
    horizon=HORIZON,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    clip_denoised=args.clip_denoised,
    predict_epsilon=args.predict_epsilon,
    ## loss weighting
    action_weight=args.action_weight,
    loss_weights=args.loss_weights,
    loss_discount=args.loss_discount,
    device=args.device,
    use_wavelet=args.use_wavelet
)

trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_evP" ery,
    ema_decay=args.ema_decay,
    sample_freq=0,  # disable rendering (no renderer for this dataset)
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    bucket=args.bucket,
    n_reference=args.n_reference,
    n_samples=args.n_samples,
)

#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

model = model_config()
diffusion = diffusion_config(model)
trainer = trainer_config(diffusion, dataset, test_dataset, None)

if args.cont is not None:
    trainer.load_model(args.cont)

utils.report_parameters(model)

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)
# n_epochs = 1

for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs} | {args.savepath}')
    trainer.train(n_train_steps=args.n_steps_per_epoch)

