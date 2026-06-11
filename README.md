# HOP-CPP

Code for the HOP-CPP paper: a constrained reinforcement-learning agent
(TD3-Lagrangian) for online coverage path planning. The agent combines a
fisheye height map observation (HM), an optimal-value heuristic (OPTV), and a
visit-frequency map (VF). `parameters_default.yaml` is the full HOP-CPP
configuration used in the paper.

The simulation environment lives in `class_gymenv.py`. On collision the agent
glides along obstacle walls; by default the glide geometry is reconstructed
online from lidar points (`glide_use_reconstructed_lidar_borders: true`), so
the agent can only glide along walls it has already sensed.

## Installation

Requires Python 3.10 on Linux (or WSL).

```bash
git clone https://github.com/JohannBlake/HOP-CPP.git
cd HOP-CPP
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train an agent

```bash
python train.py
```

This trains the HOP-CPP agent with the paper configuration (25.6M environment
steps; algorithm hyperparameters in `omnisafe/configs/off-policy/TD3Lag.yaml`,
environment parameters in `parameters_default.yaml`). Checkpoints are saved
every epoch (256k steps) to:

```
misc/logs/runs/TD3Lag-{GymEnvOmniSafe-v0}/seed-.../torch_save/epoch-*.pt
```

Useful overrides:

```bash
# quick functional test on CPU (2 epochs of 2000 steps)
python train.py --total-steps 4000 --steps-per-epoch 2000 --device cpu

# choose seed / device
python train.py --seed 7 --device cuda:0
```

Training metrics are always written to `progress.csv` next to the checkpoints.
To additionally log to Weights & Biases, run `wandb login` once and pass
`--use-wandb`.

## Visualize a trained agent

```bash
python vis.py --model misc/logs/runs/<run>/torch_save/epoch-100.pt --episodes 3
```

This rolls out deterministic episodes, prints per-episode metrics (return,
cost, collisions, coverage), and saves one trajectory image per episode to
`vis_output/`. Use `--device cpu` on machines without a GPU.

No pretrained checkpoint is shipped with the repository — train an agent
first, then visualize it.

## Wall gliding modes

`parameters_default.yaml` → `glide_use_reconstructed_lidar_borders`:

- `true` (default, used in the paper): glide along obstacle borders
  reconstructed from accumulated lidar hits. Walls that have not been sensed
  yet stop the agent instead.
- `false`: glide along the ground-truth obstacle polygons (upper-bound /
  debugging mode).
