import torch
from collections import Counter

sat_threshold = 0.95

total_steps = 0
sat_count = 0

sign_pattern_counter = Counter()

action_sum = None
action_sq_sum = None

# actions shape: [num_envs, 4]
a = actions.detach().cpu()

num_envs = a.shape[0]
total_steps += num_envs

# -------- Saturation --------
sat = (a.abs() > sat_threshold)
sat_count += sat.sum().item()

# -------- Mean / Std --------
if action_sum is None:
    action_sum = torch.zeros_like(a.sum(dim=0))
    action_sq_sum = torch.zeros_like(a.sum(dim=0))

action_sum += a.sum(dim=0)
action_sq_sum += (a ** 2).sum(dim=0)

# -------- Sign patterns --------
# Convert actions to sign pattern: -1, 0, 1
signs = torch.sign(a)

# Optional: treat near zero as 0
signs[torch.abs(a) < 0.05] = 0

for row in signs:
    key = tuple(row.int().tolist())
    sign_pattern_counter[key] += 1