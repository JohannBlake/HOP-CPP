# Train the HOP-CPP agent (TD3-Lagrangian via the vendored omnisafe library)
# on the coverage environment defined in class_gymenv.py.
#
# All environment / experiment parameters live in parameters_default.yaml.
# Algorithm hyperparameters live in omnisafe/configs/off-policy/TD3Lag.yaml.
# Checkpoints are written to ./misc/logs/runs/.../torch_save/epoch-*.pt.
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np
import yaml

from get_parameters import para


def load_vendored_omnisafe():
    # The repo ships a modified omnisafe; load it explicitly so a pip-installed
    # omnisafe can never shadow it.
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "omnisafe")
    spec = importlib.util.spec_from_file_location("omnisafe", os.path.join(path, "__init__.py"))
    omnisafe = importlib.util.module_from_spec(spec)
    sys.modules["omnisafe"] = omnisafe
    spec.loader.exec_module(omnisafe)
    return omnisafe


def parse_args():
    parser = argparse.ArgumentParser(description="Train the HOP-CPP agent.")
    parser.add_argument("--total-steps", type=int, default=None,
                        help="Override total training steps (default: omnisafe/configs/off-policy/TD3Lag.yaml).")
    parser.add_argument("--steps-per-epoch", type=int, default=None,
                        help="Override steps per epoch; a checkpoint is saved every epoch.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override the random seed (default: parameters_default.yaml).")
    parser.add_argument("--device", type=str, default=None,
                        help="Training device, e.g. cpu or cuda:0 (default: omnisafe config).")
    parser.add_argument("--use-wandb", action="store_true",
                        help="Enable Weights & Biases logging (run `wandb login` first).")
    return parser.parse_args()


def main():
    args = parse_args()

    if para.surface_grid_creation_type == "no_height_map":
        minimal_height_data = np.array([
            [7.18375, 48.60152778, 0.0]
        ])
        file_path = os.path.join(os.getcwd(), "misc", "geo_data", "height_data", "height_data.npy")
        np.save(file_path, minimal_height_data)

    omnisafe = load_vendored_omnisafe()
    import class_gymenv  # noqa: E402  (registers GymEnvOmniSafe-v0; must follow omnisafe load)

    custom_cfgs = {"seed": args.seed if args.seed is not None else getattr(para, "seed", 0)}
    train_cfgs = {}
    if args.total_steps is not None:
        train_cfgs["total_steps"] = args.total_steps
    if args.device is not None:
        train_cfgs["device"] = args.device
    if train_cfgs:
        custom_cfgs["train_cfgs"] = train_cfgs
    if args.steps_per_epoch is not None:
        custom_cfgs["algo_cfgs"] = {"steps_per_epoch": args.steps_per_epoch}
    if args.use_wandb:
        custom_cfgs["logger_cfgs"] = {"use_wandb": True}

    agent = omnisafe.Agent(algo=para.omnisafe_alg, env_id="GymEnvOmniSafe-v0", custom_cfgs=custom_cfgs)
    env_to_discard = class_gymenv.GymEnvOmniSafe(radiation_grid_visualization=False)  # noqa: F841  (warms env caches)

    # Mirror the full parameter file into the wandb run config when logging is on.
    if agent.cfgs.logger_cfgs.use_wandb:
        with open("parameters_default.yaml", "r") as file:
            yaml_parameter_defaults_config = yaml.safe_load(file)
        try:
            import wandb
            wandb.config.update(yaml_parameter_defaults_config)
        except Exception as e:
            print(f"Warning: Could not update wandb config with parameters_default.yaml: {e}")

    try:
        agent.learn()
    except Exception:
        import traceback
        print(traceback.format_exc(), flush=True)
        raise


if __name__ == "__main__":
    main()
